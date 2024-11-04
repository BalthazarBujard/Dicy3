import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim import Optimizer
from MusicDataset.MusicDataset_v2 import Fetcher
from typing import Callable, Tuple
from tqdm import tqdm
from architecture.Seq2Seq import Seq2SeqCoupling,Seq2SeqBase
from abc import ABC, abstractmethod
from utils.utils import *
from utils.metrics import compute_accuracy
import matplotlib.pyplot as plt
import numpy as np
import time
import optuna
from munch import Munch 
from typing import List,Union

# TODO : IMPLEMENT ABSTRACT BASE TRAINER CLASS FOR GENRAL PURPOSES
class Trainer(ABC):
    pass


#CHANGE TO ENTROPY
def compute_codebook_usage(idxs,vocab_size,pad_idx):
    idxs_nopad = [idx.numpy(force=True) for idx in idxs if idx!=pad_idx]
    
    counts = np.bincount(idxs_nopad,minlength=vocab_size)
    usage = np.count_nonzero(counts)/len(counts)
    
    return usage

class Seq2SeqTrainer(nn.Module):
    def __init__(self, 
                 model : Seq2SeqBase, 
                 gpu_id : int,
                 criterion : Callable,
                 optimizer : List[Optimizer],
                 trainer_name : str,
                 segmentation : str,
                 save_ckp : bool = True,
                 grad_accum_steps : int = 1,
                 codebook_loss_weight : float = 1.,
                 k : int = None,
                 chunk_size : float = 0.5,
                 track_size : float = 30,
                 resume_epoch : int =0,
                 init_sample_temperature : float = 2.,
                 min_temperature : float = 0.5,
                 with_decay : bool = False):
        
        super().__init__()
        self.model=model
        self.gpu_id = gpu_id
        self.criterion = criterion
        self.optimizer = optimizer
        self.grad_accum_steps = grad_accum_steps
        #add some options for checkpoint saving etc
        self.trainer_name = trainer_name
        self.segmentation = segmentation
        self.save_ckp=save_ckp
        self.resume_epoch = resume_epoch
        
        #VQ parameters (not used if kmeans vq)
        self.codebook_loss_alpha = codebook_loss_weight
        assert init_sample_temperature>=min_temperature, "init sample temp should be higher than min temperature"
        self.codebook_sample_temperature = init_sample_temperature
        self.min_temperature = min_temperature
        self.with_decay = with_decay
        
        #topK prediction
        if k == None : k = min(max(int(0.1*self.model.vocab_size),5),100) #10% du vocab - seuil a 5 et 100
        self.k = k
        self.chunk_size=chunk_size
        self.track_size=track_size
        
    def save_checkpoint(self, ckp_name):
        if not any(ckp_name.endswith(ext) for ext in (".pt",".pth")):
            raise ValueError(f"checkpoint filename must end with .pt or .pth")
        
        model = self.model.module if isinstance(self.model,DDP) else self.model #if ddp
        
        model_params = {"backbone_type":self.model.encoder.encoder.type,
                        "dim":model.dim,
                        "vocab_size":self.model.codebook_size,
                        "learnable_codebook" : self.model.encoder.quantizer.learnable_codebook,
                        "chunk_size" : self.chunk_size,
                        "tracks_size" : self.track_size,
                        "max_len":self.model.pe.pe.size(0),
                        "encoder_head":self.model.encoder.head_module,
                        "condense_type":self.model.encoder.condense_type,
                        "use_special_tokens":self.model.use_special_tokens,
                        "has_masking" : self.model.has_masking,
                        "mask_prob" : self.train_fetcher.loader.collate_fn.mask_prob,
                        "mask_len" : self.train_fetcher.loader.collate_fn.mask_len,
                        "task" : "coupling" if type(model)==Seq2SeqCoupling else "completion",
                        "decoder_only":self.model.decision.decoder_only,
                        "transformer_layers":self.model.decision.layers,
                        "dropout":self.model.decision.dropout,
                        "inner_dim":self.model.decision.inner_dim,
                        "heads":self.model.decision.heads,
                        "norm_first":self.model.decision.norm_first,
                        "segmentation": self.segmentation,
                        "top-K" : self.k
                        
                  } 
        state_dict =self.model.state_dict() if isinstance(self.model,Seq2SeqBase) else self.model.module.state_dict() #if DDP model.module
        optim_state_dict = [optim.state_dict() for optim in self.optimizer]
        torch.save({
            "model_class":self.model.__class__,
            "state_dict":state_dict,
            "optimizer":optim_state_dict,
            "model_params":model_params,
            },"runs/coupling/"+ckp_name)
    
    def _forward(self,inputs:Munch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        model = self.model.module if isinstance(self.model, DDP) else self.model
        
        if type(model)==Seq2SeqCoupling:
            src, tgt, src_pad_mask, tgt_pad_mask, src_mask_indices = inputs.values()
            #compute output
            logits, tgt, tgt_idx, codebook_loss = self.model(src, tgt, src_pad_mask, tgt_pad_mask,
                                                        sample_codebook_temp=self.codebook_sample_temperature)
            
        elif type(model)==Seq2SeqBase: #for autocompletion
            src, src_pad_mask, src_mask_indices, label = inputs.values() 
            #compute output
            logits, tgt, tgt_idx, codebook_loss = self.model(src, src_pad_mask, sample_codebook_temp=self.codebook_sample_temperature)
            
        return logits,tgt,tgt_idx,codebook_loss

    @torch.no_grad    
    def evaluate(self,eval_fetcher):
        prYellow("Evaluation...")
        loss=0
        acc=0
        cb_usage=0
        self.model.eval()
        for _ in range(len(eval_fetcher)):
            inputs = next(eval_fetcher)
            
            logits,tgt,tgt_idx,codebook_loss = self._forward(inputs)
            
            tgt_out = tgt_idx[:,1:] #ground truth
            
            loss_ce=self.criterion(logits.reshape(-1,logits.size(-1)), tgt_out.reshape(-1)).item() #reshqaped as (B*T,vocab_size) and (B*T,)
            
            loss += loss_ce + self.codebook_loss_alpha*codebook_loss
            
            #topK search
            preds = predict_topK(self.k,logits,tgt_out)
            
            acc += compute_accuracy(preds,tgt_out.reshape(-1),pad_idx=self.model.special_tokens_idx["pad"])
            
            cb_usage += compute_codebook_usage(preds,self.model.vocab_size,pad_idx=self.model.special_tokens_idx["pad"])
        
        loss=loss/len(eval_fetcher)
        acc/=len(eval_fetcher)
        cb_usage/=len(eval_fetcher)
        
        return loss, acc, cb_usage
    
    def plot_loss(self, epoch, train_losses, val_losses):
        plt.figure(figsize=(10,10),dpi=150)
        plt.plot(range(1,epoch+2),train_losses,label="train loss")
        if len(val_losses) != 0:
            plt.plot(range(1,epoch+2),val_losses,label="val loss")
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("Cross Entropy")
        plt.savefig(f"runs/coupling/Loss_{self.trainer_name}.png")
        plt.show()
    
    
    #TODO : ADD reg_alpha to class attributes ?
    def _compute_loss(self,logits,tgt_out,reg_alpha,codebook_loss) -> torch.Tensor :
        
        loss_ce = self.criterion(logits.reshape(-1,logits.size(-1)), tgt_out.reshape(-1)) #reshqaped as (B*T,vocab_size) and (B*T,)
        loss_commit = self.codebook_loss_alpha*codebook_loss
        
        #add codebook diversity loss ?
        
        #entropy regularization --> maximize entropy = use most of vocabulary
        probs = F.softmax(logits,dim=-1)
        entropy = -1.*(torch.sum(probs * torch.log(probs + 1e-9), dim=-1).mean())
        loss_entropy = reg_alpha*entropy
        
        #totasl loss = crossentropy + codebook loss + (-entropy)
        loss = (loss_ce + loss_commit - loss_entropy)/self.grad_accum_steps 
        
        return loss
    
    def _run_batch(self,train_fetcher,reg_alpha,step,trial):    
        
        inputs = next(train_fetcher)
       
        logits,tgt,tgt_idx,codebook_loss = self._forward(inputs)
                
        #zero grad
        for optim in self.optimizer : optim.zero_grad()
        
        tgt_out = tgt_idx[:,1:] #ground truth outputs are the token indexes shifted to the right
        
        #logits : (B, T, vocab_size) and tgt_out : (B, T)
        
        loss_ce = self.criterion(logits.reshape(-1,logits.size(-1)), tgt_out.reshape(-1)) #reshqaped as (B*T,vocab_size) and (B*T,)
        loss_commit = self.codebook_loss_alpha*codebook_loss
        
        #add codebook diversity loss ?
        
        #entropy regularization --> maximize entropy = use most of vocabulary
        probs = F.softmax(logits,dim=-1)
        entropy = -1.*(torch.sum(probs * torch.log(probs + 1e-9), dim=-1).mean())
        loss_entropy = reg_alpha*entropy
        
        #totasl loss = crossentropy + codebook loss + (-entropy)
        loss = (loss_ce + loss_commit - loss_entropy)/self.grad_accum_steps 
        
        preds = predict_topK(self.k,logits,tgt_out)
        
        acc = compute_accuracy(preds,tgt_out.reshape(-1),pad_idx=self.model.special_tokens_idx["pad"])
        
        preds = preds.reshape(logits.shape[:-1]) #reshape to B,chunks
        
        if step%20==0 and trial==None:
            if self.gpu_id==0:
                prYellow(f"Pred {preds[0].numpy(force=True)}")
                prYellow(f"GT {tgt_out[0].numpy(force=True)}")
                prYellow(f"Pred {preds[1].numpy(force=True)}")
                prYellow(f"GT {tgt_out[1].numpy(force=True)}")
                prYellow(f"Pred {preds[2].numpy(force=True)}")
                prYellow(f"GT {tgt_out[2].numpy(force=True)}")
                prYellow(loss.item())
                
                prRed(f"Pred {torch.argmax(logits[0],-1).numpy(force=True)}")
                prGreen(f"Probs {torch.sort(probs[0][:10],dim=-1,descending=True)[0][:,:5].numpy(force=True)}") #show the 5 highest probabilities for 10 first tokens
        
        loss.backward()
        
        if step%self.grad_accum_steps == 0 or step == len(train_fetcher):
            for optim in self.optimizer : optim.step()
        
        return loss, acc
                
    def train(self, train_fetcher, val_fetcher,epochs, evaluate=True,reg_alpha=0.1, trial : optuna.trial.Trial = None):
        if evaluate :
            assert val_fetcher != None, "To evaluate the model a validation set is needed."
        
        train_iter=epochs*len(train_fetcher) #total iterations
        
        iter_count=0
        if self.gpu_id==0:
            progress_bar = tqdm(total=train_iter,initial=self.resume_epoch*len(train_fetcher))
        
        
        train_losses=[]
        val_losses=[]
        epoch_0=self.resume_epoch
        if self.resume_epoch>0:
            try :
                d=np.load(f"/data3/ansynth_nonbp/bujard/DICY2/runs/coupling/eval_{self.trainer_name}.npy",allow_pickle=True)
                train_losses=d['train_loss']
                val_losses=d['test_loss']
            except:
                pass
        
        best_loss = float('inf')
        best_codebook_usage=0
        
        init_temperature = self.codebook_sample_temperature
        
        self.train_fetcher = train_fetcher #to get params for saving
        
        for epoch in range(self.resume_epoch,epochs):
            train_loss=0
            train_acc=0
            val_loss=0
            self.model.train()
            
            #with ddp we need to set epoch on sampler before creating dataloader iterator (done in fetcher when restarting loader)
            try :
                train_fetcher.loader.sampler.set_epoch(epoch) 
            except:
                pass
            
            for step in range(len(train_fetcher)):
                if self.gpu_id==0:
                    progress_bar.update(1) #update progress bar
                iter_count+=1
                
                #get inputs and targets
                #inputs = next(train_fetcher)
                
                loss, acc = self._run_batch(train_fetcher,reg_alpha,step,trial)
                
                train_loss+=loss.item()
                
                train_acc+=acc
                
                #temperature annealing
                if self.with_decay:
                    new_temp = init_temperature - (init_temperature-self.min_temperature)/(train_iter) * iter_count
                    self.codebook_sample_temperature = min(new_temp,self.min_temperature)
                
                
            train_loss = train_loss/len(train_fetcher)
            train_losses.append(train_loss)
            train_acc/=len(train_fetcher)
            
            prGreen(f"Training loss at epoch {epoch+1}/{epochs} : {train_loss}. Accuracy = {train_acc}")
            #val loss
            if evaluate:
                val_loss, val_acc, codebook_usage = self.evaluate(val_fetcher)
                val_loss = val_loss.item()
                prGreen(f"Validation loss at epoch {epoch+1}/{epochs} : {val_loss}. Accuracy = {val_acc}")
            
            else : 
                val_loss = train_loss #for checkpooint saving
                val_acc = train_acc
            
            
            val_losses.append(val_loss)    
            
            if epoch-epoch_0>0 and trial==None:   
                if self.gpu_id==0:
                    self.plot_loss(epoch-epoch_0,train_losses, val_losses)
            
            if val_loss<best_loss:
                best_loss=val_loss
                best_codebook_usage = codebook_usage #like so they are not totally decorrelated during optim ?
                if self.save_ckp:
                    if self.gpu_id==0 : #only save rank 0 model
                        self.save_checkpoint(self.trainer_name+".pt") #save ckp as run name and add .pt extension
                
            
            if self.gpu_id==0: #check rank
                np.save(f"runs/coupling/eval_{self.trainer_name}.npy",{"train_loss":train_losses,"test_loss":val_losses,"train_acc":train_acc,"test_acc":val_acc},allow_pickle=True)
        
        if trial!=None:
            return best_loss,best_codebook_usage
            
                