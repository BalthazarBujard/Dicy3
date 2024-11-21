import torch
import torch.nn as nn

#decision module class : contains the transformer module (enc-dec or dec-only)
class Decision(nn.Module):
    def __init__(self, dim : int, layers : int, vocab_size : int, inner_dim : int = 2048, heads : int = 8, dropout=0.1,decoder_only : bool = False, norm_first : bool = True):
        
        super().__init__()
        
        self.dim = dim
        self.layers = layers
        self.inner_dim = inner_dim
        self.heads = heads
        self.decoder_only = decoder_only
        self.norm_first = norm_first
        self.dropout=dropout
        
        if decoder_only :
            decoder_layer = nn.TransformerDecoderLayer(dim,nhead=heads, dim_feedforward=inner_dim, batch_first=True,norm_first=norm_first,dropout=dropout)
            self.decision = nn.TransformerDecoder(decoder_layer,layers,norm=nn.LayerNorm(dim))
        
        else :
            #warning UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
            self.decision = nn.Transformer(dim,nhead=heads,num_encoder_layers=layers,num_decoder_layers=layers, 
                                           dim_feedforward=inner_dim, batch_first=True,norm_first=norm_first,dropout=dropout)
        
        
        self.output_layer = nn.Linear(self.dim,vocab_size)
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
        
        if self.decoder_only:
            memory = src
            out = self.decision(tgt,memory,tgt_mask=tgt_mask,memory_mask=src_mask,tgt_key_padding_mask=tgt_pad_mask,memory_key_padding_mask=src_pad_mask)
        
        else :
            out = self.decision(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask)
        
        # add here the output projection ? seems logical : decision "decides" what token to choose  
        out = self.output_layer(out)  
        
        return out

    def encode(self,src,src_mask=None,src_pad_mask=None):
        if not self.decoder_only:
            memory = self.decision.encoder(src,mask=src_mask,src_key_padding_mask=src_pad_mask)
        
        else : memory = src
        
        return memory
    
    def decode(self,tgt,memory,tgt_mask=None,memory_mask=None,tgt_pad_mask=None,memory_pad_mask=None):
        if not self.decoder_only:
            out = self.decision.decoder(tgt,memory,tgt_mask=tgt_mask,memory_mask=memory_mask,tgt_key_padding_mask=tgt_pad_mask,memory_key_padding_mask=memory_pad_mask)
        else : out = self.decision(tgt,memory,tgt_mask=tgt_mask,memory_mask=memory_mask,tgt_key_padding_mask=tgt_pad_mask,memory_key_padding_mask=memory_pad_mask)
        
        out = self.output_layer(out) #logits
        return out
    