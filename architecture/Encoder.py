#%%
import torch
from torch.nn import MultiheadAttention
import torch.nn as nn
import torch.nn.functional as F
import transformers, fairseq
import math
from utils.utils import *
from typing import Union, List, Tuple

# TODO : AJOUTER UN ARGUMENT POUR LE CAS OU ON FERAIT DU PRETRAIN DU BACKBONE, DANS CE CAS L'ARGUMENT MASK DOIT ETRE A TRUE MAIS DANS LE CAS GENERAL DE NOTRE APPLICATION 
# ON NE PREVOIT PAS DE FAIRE D EPRE-TRAIN MAIS SIMPLEMENT DE L'ADAPTATION
class Backbone(nn.Module):
    """_summary_

    General class for pretrained backbones
    
    """
    def __init__(self, pretrained_model:nn.Module, type:str, mean=False, pooling=False, output_final_proj = False):
        super().__init__()
        self.backbone = pretrained_model
        self.type = type
        self.__mean=mean #private to not modify it after creation
        if pooling : 
            assert mean==False, "If pooling, no average should be done."
        self.pooling=pooling #if backbone should condense infor with pooling. used as an intermediate feature for latent space analysis, normaly done in loacalEncoder
        self.output_final_proj = output_final_proj #flag to use either hidden layer output or final projection output
        self.frozen = False
        
    @property
    def dim(self):
        if isinstance(self.backbone, transformers.PreTrainedModel):
            #seems like all hf pretrainedmodels share the same config file structure 
            if self.output_final_proj:
                embed_dim = self.backbone.config.classifier_proj_size
            
            else : embed_dim=self.backbone.config.hidden_size
        
        elif isinstance(self.backbone, fairseq.models.wav2vec.wav2vec2.Wav2Vec2Model):
            if self.output_final_proj :
                embed_dim = self.backbone.cfg["final_dim"]
            
            else : embed_dim= self.backbone.cfg["encoder_embed_dim"]
        
        else :
            raise NotImplementedError("No implementation for other than fairseq Wav2Vec2Model and HuggingFace PreTrainedModel(s)")
                   
        return embed_dim       
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    def freeze(self):
        self.backbone.requires_grad_(False)
        self.frozen = True
    
    def freeze_feature_extractor(self):
        if self.type=="w2v":
            if isinstance(self.backbone, transformers.Wav2Vec2ForPreTraining):
                self.backbone.wav2vec2.feature_extractor.requires_grad_(False)
            
            elif isinstance(self.backbone, fairseq.models.wav2vec.wav2vec2.Wav2Vec2Model):
                self.backbone.feature_extractor.requires_grad_(False)
            
            else:
                raise TypeError("Only HF or fairseq")
        else :
            raise NotImplementedError("No implementation for other that wav2vec2 backbone.")

        self.frozen=True
        
    #def train(self, mode=True):
    #    self.backbone.train(mode)
    
    #def _get_feat_extract_output_lengths(self,*args):
     #   self.backbone._get_feat_extract_output_lengths(*args)
    
    @property
    def mean(self):
        return self.__mean
    
    #TODO: DANS LA DOCUMENTATION DE HUGGING FACE ILS DISENT QUE LE WAV2VEC BASE A DES MEILEURS RESULTATS
    # SI EN INFERENCE ON DONNE PAS DE PAD MASK
    def forward(self, x : torch.Tensor, padding_mask : torch.Tensor = None):
        if self.type=="w2v":
            if isinstance(self.backbone, transformers.Wav2Vec2ForPreTraining):
                if padding_mask!=None :
                    raise RuntimeError("Not sure if padding mask should be given to HF model...")
                
                #on dirait que le wav2vec de HF ne fait pas directement dans le forward le maskage des timesteps donc pas besoin de modifier.
                if self.output_final_proj :
                    outputs = self.backbone(x, attention_mask = padding_mask, output_hidden_states=False)
                    z = outputs.projected_states 
                
                else :
                    outputs = self.backbone(x, attention_mask = padding_mask, output_hidden_states=True)
                    z = outputs.hidden_states[-1]
                
            
            elif isinstance(self.backbone, fairseq.models.wav2vec.wav2vec2.Wav2Vec2Model):
                
                outputs = self.backbone(x, features_only=True, padding_mask=padding_mask, mask=False) 
                z = outputs['x']
                
                #il faut proceder comme ca sinon le projected states c'est la sortie de la cosine sim entre y et x... (a priori en regardant le code)
                if self.output_final_proj:
                    z = self.backbone.final_proj(z)
            
            else :
                str="The wav2vec pretrained backbone of type" + str(type(self.backbone)) + "is not supported.\
                                Only backbones from HuggingFace or fairseq are valid"
                raise TypeError(str)
        
        else :
            raise NotImplementedError("No Backbone implementation for other models than wav2vec")

        #(B,L,D)
        
        if self.mean : z = torch.mean(z,dim=1) #remove time axis. no need for keepdim i think
        
        elif self.pooling:
            z = z.transpose(1,2) #swap L,D to D,L for max_pool over time
            z = F.max_pool1d(z,z.size(-1)) #max_pool over all timesteps
            z = z[...,0] #remove time dimension

        #(B,D) if mean or pooling, else (B,L,D)
        
        return z

#TODO : DOUBLE CHECK IF THIS IMPLEMENTATION IS CORRECT ESPECIALLY THE CREATE_MASK METHOD                
class TransformerEncoderBloc(nn.Module):
    def __init__(self,embed_dim=768, 
                 num_heads=12, dropout=0.2, inner_dim=2048,
                 condense_type='mask'):
        
        assert condense_type in ['mask','weighed']
        super().__init__()
        self.dim=embed_dim
        self.mha = MultiheadAttention(self.dim, num_heads,dropout,batch_first=True)
        self.dropout=nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(self.dim) #Layer Normalization following attention
        self.ln2 = nn.LayerNorm(self.dim)
        self.fc1 = nn.Linear(self.dim, inner_dim)
        self.fc2 = nn.Linear(inner_dim, self.dim)
        self.relu = nn.ReLU(inplace=True)
        self.condense_type = condense_type # what kind of information condensation to use
        
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    def _create_collapse_mask(self,S,T,idx=0):
        mask = torch.full(size=(T,S), fill_value=-torch.tensor(float("inf")),device=self.device)
        mask[idx,:]=0 #create mask to mask all timesteps expect last one that attends to all  previous steps
        mask = torch.diagonal_scatter(mask,torch.zeros(min(T,S))) #let self attention because otherwise error during attention's softmax
        
        #mask=mask.to(torch.bool)
        
        #save idx
        if not hasattr(self,"idx"):
            self.idx=idx
        elif self.idx!=idx:
            self.idx=idx
        
        return mask
    
    def forward(self,x:torch.Tensor, padding_mask : torch.Tensor = None): #x : (B,L,D)
        
        #compute self-attention mask
        mask=self._create_collapse_mask(x.shape[1],x.shape[1]) if self.condense_type=='mask' else None #CAREFUL THAT TAKING LAST STEP IS OKAY (MAYBE WITH THE PADDING THERE IS NO INFO BUT MHA SHOULD LEARN TO PUT IT THERE)

        #transformer block with pre-LN (better convergence cf "On Layer Normalization")
        x_norm = self.ln1(x)
        x_out, weights = self.mha(x_norm, x_norm, x_norm, attn_mask = mask, key_padding_mask = padding_mask) #self attention
        x = x + x_out 
        
        if self.condense_type=='mask':
            #keep only idx element in which all info is condensed due to masking in mha
            x = x[:,self.idx,:] # (B,D) just one vector of dim per batch element
        
        else :
            #collapse weights from BxLxL to BxLX1
            weights = weights.mean(dim=1).unsqueeze(-1) #mean or sum should be equivalent
            x = weights*x #multiply sequence steps by corresponding weight
            x = x.sum(dim=1) #sum weighed steps across time dimension
        
        #FFN and LN
        x_norm = self.ln2(x)
        x = x + self.dropout(self.fc2(self.relu(self.fc1(x_norm))))
        
        return x
        


class LocalEncoder(nn.Module):
    def __init__(self, pretrained_encoder : Backbone, quantizer : nn.Module,
                 head_module : str = "mean", condense_type=None, embed_dim : int = 768, 
                 num_heads : int = 8, dropout : float = 0.1, inner_dim : int = 2048, chunking_pre_post_encoding : str = "pre"):
        
        super().__init__()
        self.encoder = pretrained_encoder
        
        assert self.encoder.mean==False, "Backbone should return a sequence but backbone.mean=True !"
        assert head_module in ["attention", "pooling", "mean"], "head module accepts only 'attention' for MHA, 'pooling' for simple max pooling or 'mean'  as choices."
        assert chunking_pre_post_encoding in ["pre", "post"], "Wrong argument, choose between 'pre' and 'post"
        
        self.head_module=head_module
        self.condense_type = condense_type
        
        if head_module=="attention":
            if condense_type==None : raise ValueError("collapse module is attention, a condense type has to be specified : 'mask' or 'weighed'")
            
            if embed_dim != self.encoder.dim:
                prRed("For now this class doesnt accept embed_dim different than the one given by backbone.\
                    Later might implement adaptation layer to project to the correct embed_dim given as input")
                #self.adapt_layer=nn.Linear(self.encoder.get_embed_dim(),embed_dim)
                embed_dim=self.encoder.dim
            
            self.transformerbloc = TransformerEncoderBloc(embed_dim,num_heads,dropout,inner_dim,condense_type)
        
        self.chunking_pre_post_encoding = chunking_pre_post_encoding #order to follow for chunking -> before or after encoding
        
        self.embed_dim=embed_dim    
        
        self.quantizer = quantizer
        self.dim = quantizer.dim       
        
        #self.out_proj=nn.Linear(embed_dim,self.dim) if self.dim != embed_dim else nn.Identity()
    
    #collapse information accross time dimension
    def collapse(self, x : torch.Tensor, padding_mask : torch.Tensor) -> torch.Tensor:
        #expected x : (B,L,D)
        if self.head_module == 'attention':   
            #x = self.pe(x)
            x = self.transformerbloc(x,padding_mask)  
    
        elif self.head_module == "pooling":
            x = x.transpose(1,2) #swap L,D to D,L for max_pool over time
            x = F.max_pool1d(x,x.size(-1)) #max_pool over all timesteps
            x = x[...,0] #remove time dimension
        
        elif self.head_module == "mean":
            if padding_mask==None:
                x = torch.mean(x,dim=1) #mean accross time dimension    
            
            else :
                mask = ~padding_mask #true is padding and we sum accross not padded tokens
                x = torch.sum(x*mask.unsqueeze(-1),dim=1)/torch.sum(mask,dim=1,keepdim=True)
                
            
        return x
    
    # computes padding mask after encoding (subsampling)
    # works with fairseq
    #from https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/wav2vec/wav2vec2.py line 623
    def _process_padding_mask(self, x : torch.Tensor, padding_mask : torch.Tensor) -> torch.Tensor:
        if padding_mask is not None and padding_mask.any():
            input_lengths = (1 - padding_mask.long()).sum(-1)
            # apply conv formula to get real output_lengths
            output_lengths = self.encoder.backbone._get_feat_extract_output_lengths(input_lengths)

            padding_mask = torch.zeros(
                x.shape[:2], dtype=x.dtype, device=x.device
            )

            # these two operations makes sure that all values
            # before the output lengths indices are attended to
            padding_mask[
                (
                    torch.arange(padding_mask.shape[0], device=padding_mask.device),
                    output_lengths - 1,
                )
            ] = 1
            padding_mask = (1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])).bool()
            
        else :
            padding_mask=torch.zeros(x.shape[:2],device=x.device).bool()
        
        return padding_mask
    
    def forward(self, x : torch.Tensor,
                sample_codebook_temp : float = None, #not used at thge moment
                padding_mask : torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:        
        # x is expected to have shape B,chunks,samples
        if x.ndim!=3:
            raise ValueError(f"The input tensor x has not the expected shape of (batch,chunks,samples) but has shape {x.shape}")
        
        B,chunks,max_samples = x.shape
        
        if self.chunking_pre_post_encoding == "pre":
            x = x.contiguous().view(-1,max_samples) #first reshape as B*chunks, max_samples for backbone compatibility
            padding_mask = padding_mask.contiguous().view(-1,max_samples) if padding_mask!=None else None
            
            x = self.encoder(x, padding_mask) #contextualized representations from pretrained_backbone. (B*chunks,L,D) L<<max_samples
            
            #process padding mask
            padding_mask = self._process_padding_mask(x, padding_mask) #(B*chunks,L)
        
        
        else : #post-chunking
            x = x.contiguous().view(B,-1) #first reshape as B, chunks*max_samples = track_duration for backbone compatibility
            padding_mask = padding_mask.contiguous().view(B,-1) if padding_mask!=None else None
            
            x = self.encoder(x, padding_mask) #contextualized representations from pretrained_backbone. (B,L,D) L<<track_duration
            
            #reshape and process mask more complicated for post-chunking
            #we want to preserve the number of chunks --> pad to reshape as (B,chunks,-1,D)
            T = torch.ceil(torch.tensor(x.size(1)/chunks)).int() #duration of a chunk to have 'chunks'
            #the total duration of the sequence
            new_L = chunks*T
            #pad length to have new_L
            pad = new_L-x.size(1)
            x = torch.cat([x,torch.zeros((x.size(0),pad,x.size(-1)),device=x.device)],dim=1)
            #reshape as B*chunks, L', D
            x=x.view(B*chunks,-1,x.size(-1))


            #process mask with new padded x
            padding_mask = padding_mask.view(-1,max_samples) if padding_mask!=None else None #(B*chunks,max_samples)
            #process mask with x without the padding -> avoid appending True to mask where it shouldnt
            #and len of x[:,:-pad] is equivalent to the output length of max_samples
            padding_mask = self._process_padding_mask(x[:,:-pad], padding_mask) #(B*chunks,L-pad)
            padding_mask = padding_mask.view(B,chunks,-1) #reshape as (B,chunks,L-pad) for easier append of pad mask
            pad_step_mask = torch.zeros(padding_mask.shape[:2]+(pad,),device=padding_mask.device, dtype=torch.bool) #(B,chunks,pad_len) init as False
            pad_step_mask[:,-1]=True #the last 'pad' steps of the last chunk are padded
            padding_mask = torch.cat([padding_mask,pad_step_mask],dim=-1) #(B,chunks,L')
            padding_mask = padding_mask.view(B*chunks,-1) #final reshape as B*chunks, L'            

        
        
        x = self.collapse(x, padding_mask)
    
        # at this point x : (B*chunks,D) 
        
        #reshape at expected shape (B, chunks, D)
        x = x.view(B,chunks,-1) 
        
        #vector quantizer
        xq, indices, codebook_loss = self.quantizer(x, sample_codebook_temp=sample_codebook_temp)
        
        return xq, indices, codebook_loss #indices are needed for cressentropy loss in seq2seq model training

#take a sequence of local codes (quantized-->tokens) and outputs a single token
#needs the same vector quantizer as local encoder OR output probabilities accross vocabulary (is it differentiable? j'crois pas justement)
class GlobalEncoder(nn.Module):
    def __init__(self,quantizer : nn.Module, num_heads : int = 12, dropout :  float = 0.1, forward_expansion : int = 2): 
        super().__init__()
        #embed dim depends on quantizer
        self.dim=quantizer.dim
        self.quantizer=quantizer
        inner_dim=self.dim*forward_expansion
        self.transformerbloc = TransformerEncoderBloc(self.dim,num_heads,dropout,inner_dim=inner_dim)
    
    def forward(self,x):
        #x is a sequence of tokens from the quantizer codebook of context size T_context
        x = self.transformerbloc(x) #extract single code
        
        #quantize
        x_q, indices, commitment_loss = self.quantizer(x)
        
        return x_q
        


    
        

        
"""
# %%
from torch.utils.data import DataLoader
from MusicDataset.MusicDataset_v2 import MusicContainer
from wav2vec2.wav2vec2_utils import DataCollatorForWav2Vec2, Fetcher
from transformers import AutoFeatureExtractor, Wav2Vec2ForPreTraining
        
# %% Instanciate model and preprocessor


feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-base")

# %% Dataset, DataLoader, Fetcher

root = "/data3/anasynth_nonbp/bujard/data/Examples"
max_duration=5.0
sampling_rate=feature_extractor.sampling_rate
segmentation_strategy="uniform"

#dataset containing all the chunks
eval_ds = MusicContainer(root, max_duration, sampling_rate, segmentation_strategy)

#DataCollator
collate_fn = DataCollatorForWav2Vec2(model, feature_extractor, split="eval")

#dataloader
batch_size=8
eval_loader=DataLoader(eval_ds,batch_size,collate_fn=collate_fn)
eval_fetcher=Fetcher(eval_loader)
#%%
mha = MultiheadAttention(embed_dim=model.config.output_hidden_size,
                         num_heads=12,dropout=0.1,
                         batch_first=True)





# %%
inputs=next(eval_fetcher)
with torch.no_grad():
    c = model(inputs.x, output_hidden_states=True).hidden_states[-1]
print(c.shape)


# %%
mask = torch.ones(size=(c.shape[1],c.shape[1]))
mask[-1,:]=0 #create mask to mask all timesteps expect last one that attends to all  previous steps
mask=mask.to(torch.bool)
mask_per_head = []
#print(mask)
with torch.no_grad():
    y,attn_weights = mha(c,c,c, attn_mask=mask, 
                         need_weights=True, 
                         average_attn_weights=False) #average to false to get weights per head
# %%
mask = nn.Transformer().generate_square_subsequent_mask(10) #check to understand structure of masking
"""