from .Encoder import LocalEncoder,Backbone
from .Decision import Decision
from .VectorQuantizer import KmeansQuantizer # VectorQuantizer, GumbelVectorQuantizer, 
from .Seq2Seq import Seq2SeqBase, Seq2SeqCoupling 
import numpy as np
from fairseq.checkpoint_utils import load_model_ensemble_and_task
#from vector_quantize_pytorch import VectorQuantize # TODO : ADD OTHER OPTIONS OF VQ
import torch.nn as nn
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Tuple, Union
from pathlib import Path

#wrapper class to get eattributes without changing whole code
class myDDP(DDP):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

#script to build different models and load checkpopints

#TODO : NOW THE SEQ2SEQ BUILDER CAN TAKE **KWARGS FROM DICT
def load_model_checkpoint(ckp_path:Path, backbone_checkpoint="/data3/anasynth_nonbp/bujard/w2v_music_checkpoint.pt") -> Tuple[Union[Seq2SeqBase,Seq2SeqCoupling], dict, dict] :
    
    ckp = torch.load(ckp_path, map_location=torch.device("cpu"))
    model_class = ckp["model_class"]
    state_dict = ckp["state_dict"]
    model_params = ckp["model_params"]
    optimizer_state_dict = ckp['optimizer']
    
    #bb_ckp="../w2v_music_checkpoint.pt"

    bb_type=model_params["backbone_type"]
    max_len = model_params["max_len"]
    dim=model_params["dim"]
    if dim == 0 : dim = 768
    vocab_size=model_params["vocab_size"]
    encoder_head=model_params["encoder_head"]
    condense_type = model_params["condense_type"]
    use_special_tokens=model_params["use_special_tokens"]
    has_masking=model_params['has_masking']
    decoder_only=model_params["decoder_only"]
    transformer_layers=model_params["transformer_layers"]
    inner_dim=model_params["inner_dim"]
    heads = model_params["heads"]
    try:
        dropout = model_params["dropout"]
    except :
        dropout = 0.1 #default value for older models
    
    try :
        chunking = model_params['pre_post_chunking']
    except :
        chunking = 'pre' #old models have pre chunking
        
    try :
        special_vq = model_params['special_vq']
        data = model_params["vq_data"]
        assert data is not None, "If special VQ, specify which data is the VQ from..."
    except :
        special_vq = False
        data = None
    
    try :
        relative_pe = model_params["relative_pe"]
    except:
        relative_pe = False
    
    task = model_params["task"]
    
    #if issubclass(model_class,Seq2SeqBase):
    model = SimpleSeq2SeqModel(backbone_checkpoint,bb_type,dim,vocab_size,max_len,encoder_head,use_special_tokens,chunking=chunking,
                                   condense_type=condense_type,has_masking=has_masking,task=task,
                                   transformer_layers=transformer_layers,decoder_only=decoder_only, relative_pe=relative_pe,inner_dim=inner_dim,heads=heads,dropout=dropout,
                                   special_vq=special_vq,chunk_size = model_params["chunk_size"], data = data)
    
    #else : raise ValueError(f"the model class from the checkpoint is invalid. Should be an instance (or subclass) of 'Seq2SeqBase' but got {model_class}")
    
    model.load_state_dict(state_dict)
    
    segmentation_startegy = model_params["segmentation"]
    
    #information attributes
    model.segmentation = segmentation_startegy
    model.name = model_params["run_id"]
    
    return model, model_params, optimizer_state_dict

def build_backbone(checkpoint : Path, type : str, mean : bool, pooling : bool, output_final_proj : bool, fw : str = "fairseq") -> Backbone:
    #load pretrained backbone
    if fw=="fairseq":
        models, _, _ = load_model_ensemble_and_task([checkpoint])
        pretrained_backbone = models[0]
    
    else :
        NotImplementedError("Not implemented builder for other framework than fairseq")
    
    backbone = Backbone(pretrained_backbone,type,mean,pooling,output_final_proj)
    
    return backbone

def build_quantizer(dim : int, vocab_size : int, learnable_codebook : bool, restart : bool, is_special: bool, chunk_size : float = None, data : str = None)-> KmeansQuantizer:
    #vector quantizer  
    assert vocab_size in [16,32,64,128,256,512,1024]
    if is_special:
        print("Special VQ")
        centers=np.load(f"clustering/kmeans_centers_{vocab_size}_{chunk_size}s_{data}.npy",allow_pickle=True)
    else :
        centers=np.load(f"clustering/kmeans_centers_{vocab_size}_{dim}.npy",allow_pickle=True)
    centers=torch.from_numpy(centers)
    vq = KmeansQuantizer(centers,learnable_codebook,dim,restart,is_special, data)
    
    return vq
    

def build_localEncoder(backbone_ckp : Path, backbone_type : str, freeze_backbone : bool, dim : int, 
                       vocab_size : int, learnable_codebook : bool, restart_codebook : bool,
                       chunking : str = "post", encoder_head : str = "mean", condense_type : str = None, 
                       special_vq: bool = True, chunk_size : float = None, data : str = None) -> LocalEncoder:
    
    output_final_proj = dim==256 #if model dimension is 256 we want the final projection output, else 768 hidden layer output dim
    
    #load pretrained backbone
    backbone=build_backbone(backbone_ckp,backbone_type,
                            mean=False,pooling=False, 
                            output_final_proj=output_final_proj,
                            fw="fairseq") #no mean or pooling for backbone in seq2seq, collapse done in encoder
    
    if freeze_backbone:
        backbone.eval() # SI ON UNFREEZE BB IL FAUT TRAIN VQ
        backbone.freeze() #freeze backbone
    
    elif learnable_codebook == False:
        raise ValueError("Train VQ if backbone in learning.")
    
    else : #trainable bb and codebook -> only freeze feature extractor (CNN)
        backbone.freeze_feature_extractor()
       
    
    #vector quantizer  
    vq = build_quantizer(dim, vocab_size, learnable_codebook,restart_codebook, special_vq, chunk_size, data)
    
    localEncoder=LocalEncoder(backbone,vq,encoder_head,embed_dim=backbone.dim,condense_type=condense_type,chunking_pre_post_encoding=chunking)
    
    return localEncoder


#create class for decision module to handle forward call in seq2seq
def build_decision(dim : int, layers : int, vocab_size : int, 
                   inner_dim : int = 2048, heads : int = 8, dropout : float = 0.1, decoder_only : bool = False, 
                   norm_first : bool = True, relative_pe : bool = False) -> Decision:
    
    decisionModule = Decision(dim, layers, vocab_size, inner_dim, heads, dropout, decoder_only, norm_first, relative_pe)
    return decisionModule
    


def SimpleSeq2SeqModel(backbone_checkpoint : Path,
                       backbone_type : str, 
                       dim : int,
                       vocab_size : int,
                       max_len : int,
                       encoder_head : str,
                       use_special_tokens : bool,
                       task : str,
                       chunking : str,
                       restart_codebook : bool = False,
                       condense_type : str = None,
                       has_masking : bool = False,
                       freeze_backbone : bool = True,
                       learnable_codebook : bool = False,
                       transformer_layers : int = 12,
                       dropout : float = 0.1,
                       decoder_only : bool = True,
                       inner_dim : int = 2048,
                       heads : int = 12,
                       norm_first : bool = True,
                       special_vq : bool = True,
                       chunk_size : float = None,
                       data : str = None,
                       relative_pe : bool = False,
                       kmeans_init : bool = False,
                       threshold_ema_dead_code : float = 0,
                       commit_weight : float = 1.,
                       diversity_weight :  float = 0.1) -> Seq2SeqBase:
    
    assert task.lower() in ["coupling","completion"]
    assert chunking in ['pre','post']
    
    localEncoder=build_localEncoder(backbone_checkpoint,backbone_type, freeze_backbone, dim,
                                    vocab_size, learnable_codebook, restart_codebook, chunking,
                                    encoder_head, condense_type, 
                                    special_vq, chunk_size, data)
        
    decision_module = build_decision(localEncoder.dim,transformer_layers,
                                     vocab_size=vocab_size+3*use_special_tokens, #+ pad, sos, eos
                                     inner_dim=inner_dim,
                                     heads=heads,
                                     dropout=dropout,
                                     decoder_only=decoder_only,
                                     norm_first=norm_first,
                                     relative_pe=relative_pe)
    
    model_class = Seq2SeqCoupling if task == "coupling" else Seq2SeqBase
    
    seq2seq = model_class(localEncoder, decision_module, max_len, use_special_tokens=use_special_tokens,has_masking=has_masking)
    return seq2seq


"""
ema_update =  not learnable_codebook
vq = VectorQuantize(dim,vocab_size, 
                    commitment_weight=commit_weight,
                    diversity_weight=diversity_weight,
                    learnable_codebook=learnable_codebook,
                    ema_update=ema_update,
                    kmeans_init=kmeans_init,
                    threshold_ema_dead_code=threshold_ema_dead_code)


vq = GumbelVectorQuantizer(dim,
                            1,  #only one head for now
                            vocab_size,
                            diversity_weight)            
                                
vq = VectorQuantizer(dim, vocab_size,
                        learnable_codebook=learnable_codebook,
                        ema_update=ema_update,
                        kmeans_init=kmeans_init,
                        threshold_ema_dead_code=threshold_ema_dead_code)
"""