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

#wrapper class to get eattributes without changing whole code
class myDDP(DDP):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

#script to build different models and load checkpopints

#TODO : NOW THE SEQ2SEQ BUILDER CAN TAKE **KWARGS FROM DICT
def load_model_checkpoint(ckp_path:str, backbone_checkpoint="/data3/anasynth_nonbp/bujard/w2v_music_checkpoint.pt") -> Seq2SeqBase :
    
    ckp = torch.load(ckp_path, map_location=torch.device("cpu"))
    model_class = ckp["model_class"]
    state_dict = ckp["state_dict"]
    model_params = ckp["model_params"]
    
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
    
    task = model_params["task"]
    
    #if issubclass(model_class,Seq2SeqBase):
    model = SimpleSeq2SeqModel(backbone_checkpoint,bb_type,dim,vocab_size,max_len,encoder_head,use_special_tokens,
                                   condense_type=condense_type,has_masking=has_masking,task=task,transformer_layers=transformer_layers,decoder_only=decoder_only,inner_dim=inner_dim,heads=heads,dropout=dropout)
    
    #else : raise ValueError(f"the model class from the checkpoint is invalid. Should be an instance (or subclass) of 'Seq2SeqBase' but got {model_class}")
    
    model.load_state_dict(state_dict)
    
    segmentation_startegy = model_params["segmentation"]
    model.segmentation = segmentation_startegy
    
    return model, model_params

def build_backbone(checkpoint, type, mean, pooling, fw="fairseq"):
    #load pretrained backbone
    if fw=="fairseq":
        models, _, _ = load_model_ensemble_and_task([checkpoint])
        pretrained_backbone = models[0]
    
    else :
        NotImplementedError("Not implemented builder for other framework than fairseq")
    
    backbone = Backbone(pretrained_backbone,type,mean,pooling)
    
    return backbone

def build_quantizer(dim, vocab_size, learnable_codebook):
    #vector quantizer  
    assert vocab_size in [16,32,64,128,256,512,1024]
    centers=np.load(f"/data3/anasynth_nonbp/bujard/DICY2/clustering/kmeans_centers_{vocab_size}.npy",allow_pickle=True)
    centers=torch.from_numpy(centers)
    vq = KmeansQuantizer(centers,learnable_codebook,dim)
    
    return vq
    

def build_localEncoder(backbone: Backbone, quantizer : nn.Module, head_module : str = "mean", condense_type=None):
    encoder = LocalEncoder(backbone,quantizer,head_module,embed_dim=backbone.dim,condense_type=condense_type)
    return encoder

#create class for decision module to handle forward call in seq2seq
def build_decision(dim, layers, vocab_size , inner_dim=2048, heads=8, dropout=0.1, decoder_only=False, norm_first=True):
    decisionModule = Decision(dim, layers, vocab_size, inner_dim, heads, dropout, decoder_only, norm_first)
    return decisionModule
    


def SimpleSeq2SeqModel(backbone_checkpoint,
                       backbone_type, 
                       dim,
                       vocab_size,
                       max_len,
                       encoder_head,
                       use_special_tokens,
                       task,
                       restart_codebook=False,
                       condense_type=None,
                       has_masking=False,
                       freeze_backbone=True,
                       learnable_codebook=False,
                       transformer_layers=6,
                       dropout=0.1,
                       decoder_only=False,
                       inner_dim=2048,
                       heads=8,
                       norm_first=True,
                       kmeans_init=False,
                       threshold_ema_dead_code=0,
                       commit_weight=1.,
                       diversity_weight=0.1):
    
    assert task.lower() in ["coupling","completion"]
    
    #load pretrained backbone
    backbone=build_backbone(backbone_checkpoint,backbone_type,mean=False,pooling=False,fw="fairseq") #no mean or pooling for backbone in seq2seq, collapse done in encoder
    
    if freeze_backbone:
        backbone.eval() # SI ON UNFREEZE BB IL FAUT TRAIN VQ
        backbone.freeze() #freeze backbone
    
    elif learnable_codebook == False:
        raise ValueError("Train VQ if backbone in learning.")
    
    else : #trainable bb and codebook
        backbone.freeze_feature_extractor()
       
    
    #vector quantizer  
    assert vocab_size in [16,32,64,128,256,512,1024]
    centers=np.load(f"/data3/anasynth_nonbp/bujard/DICY2/clustering/kmeans_centers_{vocab_size}.npy",allow_pickle=True)
    centers=torch.from_numpy(centers)
    vq = KmeansQuantizer(centers,learnable_codebook,dim,restart=restart_codebook)
    #vq.eval()
    
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
    
    localEncoder=build_localEncoder(backbone, vq, encoder_head, condense_type)
    
    decision_module = build_decision(localEncoder.dim,transformer_layers,
                                     vocab_size=vocab_size+3*use_special_tokens, #+ pad, sos, eos
                                     inner_dim=inner_dim,
                                     heads=heads,
                                     dropout=dropout,
                                     decoder_only=decoder_only,
                                     norm_first=norm_first)
    
    model_class = Seq2SeqCoupling if task == "coupling" else Seq2SeqBase
    
    seq2seq = model_class(localEncoder, decision_module, max_len, use_special_tokens=use_special_tokens,has_masking=has_masking)
    return seq2seq

    