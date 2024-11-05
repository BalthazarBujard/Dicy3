# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 15:58:14 2024

@author: balth
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices, _sample_negative_indices
from transformers import AutoFeatureExtractor, Wav2Vec2ForPreTraining, AutoModel
import os
import numpy as np
from pathlib import Path
from itertools import chain
from librosa import load
import matplotlib.pyplot as plt
from munch import Munch
import time 

#%%

#list every file in the folder dname
def listdir(dname):
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
                          for ext in ['wav','aif']]))
    return fnames

#as is this dataset is implemented for the wav2vec2 pretraining framework
#can be sued for HuBERT evaluation at the moment (no pretraining strategy implemented)
class MusicDataset(Dataset):
    """
    Dataset handling all the preprocessing necessary to convert an file input to
    the corresponfing normalized waveform.
    
    to do : 
        - make this class adapted for multiple pretrained models (HuBERT, EnCodec, ...)
        - create a way to segment the audio files into small chunks for the model (~10 sec)
            -> find a way to adapt the indexing to the correct audio file and timeframe
    """
    def __init__(self, path, feature_extractor, subsampling_fn, from_folder=True,
                 segmentation_strategy="uniform", max_duration=3.0,
                 pretraining_strategy="contrastive", num_negatives_samples=None, split="train"):
        """
        

        Parameters
        ----------
        path : str
            path to the folder containing the audio files
        
        feature_extractor : transformers.SequenceFeatureExtractor
            PreProcessing module corresponding to the used model (wav2vec2, HuBERT,...)
        
        subsampling_factor : int
            the subsampling factor due to the first stage cnn encoder . 
            can be computed using model._get_feat_extract_output_lengths
            
        max_duration : float, optional
            max duration of an audio input in seconds. The default is 10.0.
        
        num_negatives_samples : int, optional
            number of negative samples for the contrastive loss. The default is None.
            
        split : str, optional
            Which split is this dataset used for, e.g. train/eval. The default is "train".

        Returns
        -------
        None.

        """
        super(MusicDataset, self).__init__()
        self.path=path
        
        self.feature_extractor=feature_extractor
        self.sampling_rate=feature_extractor.sampling_rate
        self.subsampling_fn=subsampling_fn
        
        self.max_samples=int(self.sampling_rate*max_duration)
        
        if from_folder:
            self.audio_paths = listdir(path)
        else : self.audio_paths=path
        self.audio_chunks = self._create_chunks(self.audio_paths, segmentation_strategy)
        
        
        self.split=split
        
        self.pretraining_strategy=pretraining_strategy
        
        if split=="train":
            if pretraining_strategy=="contrastive":
                assert num_negatives_samples!=None, "number of negative samples has to be specified"
                assert num_negatives_samples>0, "number of negative samples has to be > 0"
            
            elif pretraining_strategy == "masked_prediction":
                raise NotImplementedError("masked prediction (HuBERT) pretraining strategy not implemented yet")
            
            else :
                raise ValueError(f"{pretraining_strategy} is not a valid argument")
                
        self.num_negatives_samples = num_negatives_samples
    
    def __len__(self):
        return len(self.audio_chunks)
    
    def __getitem__(self, index):

        #open corresponding audio file
        path, start_idx, end_idx = self.audio_chunks[index]
        
        #only load corresponding chunk
        start = start_idx/self.sampling_rate #seconds
        duration = (end_idx-start_idx)/self.sampling_rate #seconds
        chunk, _ = load(path, sr = self.sampling_rate, offset=start, duration=duration)
        
        
        #preprocess with feature extractor
        processed_chunk = self.feature_extractor(chunk, return_tensors="pt",
                                                 padding="max_length",
                                                 sampling_rate=self.sampling_rate,
                                                 max_length=self.max_samples).input_values
        
        #remove batch dim
        # processed_chunk=processed_chunk.squeeze(0)
        
        #to evaluate the power of pre-trained models on speech in a music context
        #we need to compute a set of mask indices and negative samples for contrastive loss (if training)
        
        #ATTENTION, IL FAUT LE MODELE POUR CONNAITRE LA TAILLE DE LA SEQUENCE !
        #SOIT LE FAIRE EN DEHORS DU DATASET (BOF...) OU PLUTOT AVOIR PARAMETRE QUI PEUT PERMETTRE DE CONNAITRE
        #LONGUEUR SEQUENCE APRES CNN -> subsampling function from model._get_...
        encoded_sequence_length = self.subsampling_fn(processed_chunk.shape[-1]).item() #int(len(chunk)//self.subsampling_factor)
        
        
        mask_time_indices = _compute_mask_indices(
            shape=(1, encoded_sequence_length), mask_prob=0.2, mask_length=2
        )
        
        
        sampled_negative_indices = []
        
        if self.split=="train":
            if self.pretraining_strategy=="contrastive":
                sampled_negative_indices = _sample_negative_indices(
                    features_shape=(1, encoded_sequence_length),
                    num_negatives=self.num_negatives_samples,
                    mask_time_indices=mask_time_indices,
                )
                
        sampled_negative_indices = torch.tensor(data=sampled_negative_indices,
                                                        dtype=torch.long)
        
        #to tensor
        mask_time_indices = torch.tensor(data=mask_time_indices, 
                                         dtype=torch.long)
        
        # print(encoded_sequence_length,processed_chunk.shape, mask_time_indices.shape, sampled_negative_indices.shape)
        
        inputs = Munch(x = processed_chunk.squeeze(0), 
                       mask_indices = mask_time_indices.squeeze(0),
                      negative_indices = sampled_negative_indices.squeeze(0))
        return inputs
    
    def _segment_track(self, track, strategy="uniform"):
        """

        Parameters
        ----------
        track : np.ndarray(float)
            input track to segment into chunck depending from the strategy argument
        strategy : str, optional
            strategy to use for segmenting the track into chunks.
            "uniform" uses the max_duration attribute to create N chunck of max_samples
            The default is "uniform".

        Returns
        -------
        chunks : List[Tuple(int,int)]
            list of tuple of frame indexes corresponding to individual chunks
            of the segmented track

        """
        if strategy == "uniform":
            #segment input track into chunck of max_samples
            N = len(track)//self.max_samples #number of max_samples chunks
            r = len(track)%self.max_samples #remainder 
            if N > 0:
                chunks = [[i*self.max_samples,(i+1)*self.max_samples] for i in range(N)]
                
                if r != 0 and self.subsampling_fn(r).item()>0:
                    #self.subsampling_fn(r).item()>0 is not necessary as the sequence is padded to 
                    #max_samples but better to discard those remainders because 
                    #there will be nothing but zeros (mostly)
                    chunks += [[N*self.max_samples,N*self.max_samples+r]]
            else :
                chunks = [[0, len(track)]]
        
        else :
            raise NotImplementedError("no implementation for other segemntation strategy than 'uniform'")
        
        return chunks
    
    def _create_chunks(self, audio_paths, segment_strategy="uniform"):
        audio_chunks = []
        for path in audio_paths:
            #open file
            track, _ = load(path, sr = self.sampling_rate)
            
            #get chunks
            chunks = self._segment_track(track, strategy=segment_strategy)
            
            #append path to the chunks
            chunks_with_path = [[path, start,end] for start, end in chunks]
            
            # path_to_append = np.repeat(path, len(chunks)).reshape(-1,1)
            # chunks_with_path =  np.concatenate((path_to_append,chunks),axis=1)
            
            audio_chunks += chunks_with_path#.tolist()
        
        return audio_chunks
    


pretrained_paths = {"wav2vec2" : "facebook/wav2vec2-base",
                    "hubert" : "facebook/hubert-base-ls960"}
pretraining_strategies = {"wav2vec2" : "contrastive",
                    "hubert" : "masked_prediction"}


def get_loader(path, model_name, batch_size=8, from_folder=True,
               segmentation_strategy='uniform', 
               max_duration=3.0, split="train"):
    """
    

    Parameters
    ----------
    path : str
        path to folder containing audio files.
    model_name : str
        base model name (wav2vec2, hubert, ...).
    batch_size : int, optional
        size of batch. The default is 8.
    segmentation_strategy : str, optional
        the segmentation strategy to use during track segmentation. The default is 'uniform'.
    max_duration : float, optional
        max duration of the uniform segmentation (in seconds). The default is 10.0.
    split : str, optional
        which split is this loader intended to : "train" or "eval". The default is "train".

    Returns
    -------
    loader : torch.utils.data.DataLoader
        dataloader adapted to the given model and split.

    """
    
    #create the model and preprocessor corresponding to the pretrained_model argument
    try :
        pretrained_model_name_or_path = pretrained_paths[model_name]
    except KeyError:
        print(f"{model_name} not in pretrained models dictionnary keys : {pretrained_paths.keys()}")
    
    
    #TODO : do we have to instanciate models in here ? 
    # Better to instanciate outside and giving corresponding arguments as for dataset ?
    model = AutoModel.from_pretrained(pretrained_model_name_or_path) #need for subsampling _fn
    feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_model_name_or_path) #preprocess input
    
    #get parameters from model base name
    pretraining_startegy=pretraining_strategies[model_name]
    
    num_negative_samples=None
    if model_name=="wav2vec2":
        num_negative_samples = model.config.num_negatives
        
    #TODO : HuBERT pretraining strategy
    
    #instanciate dataset
    dataset = MusicDataset(path, feature_extractor,
                           model._get_feat_extract_output_lengths,
                           from_folder,
                           segmentation_strategy,max_duration,
                           pretraining_startegy,num_negative_samples,split)
    
    
    shuffle=False
    if split=="train":
        shuffle=True
    
    #instanciate dataloader
    loader = DataLoader(dataset, batch_size, shuffle)
    
    #delete unuseful objects
    del model
    del feature_extractor
    
    return loader
        

class Fetcher:
    def __init__(self, loader):
        self.loader=loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _fetch_inputs(self):
        #method to fetch next set of inputs
        try:
            #try to fectch next inputs
            inputs = next(self.iter_loader)
        except (AttributeError, StopIteration):
            #if self.iter_loader not already instantiated or end of loader
            self.iter_loader = iter(self.loader)
            inputs = next(self.iter_loader)
        
        return inputs
    
    def __next__(self):
        inputs = self._fetch_inputs()
        
        #pass inputs to cuda
        return Munch({key : item.to(self.device) for key, item in inputs.items()}) 
        
            
    
#%%    
# path="../data/Examples"
# feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
# model = Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-base")
# #%%
# t0=time.time()
# ds = MusicDataset(path, feature_extractor, 
#                   subsampling_fn=model._get_feat_extract_output_lengths,
#                   num_negatives_samples=100)
# t=time.time()-t0
# idx = np.random.randint(0,len(ds),size=1).item()
# inputs=ds[idx]   
    
# #%% test segmentation

# track, _ = load(ds.audio_paths[0], sr = ds.sampling_rate) 
    
# chunks = ds._segment_track(track) 

# path_to_append=np.repeat("path", len(chunks)).reshape(-1,1)

# chunks_with_path = np.concatenate((chunks,path_to_append),axis=1)

# #%%

# loader = get_loader(path,"hubert", batch_size=16, split="eval")

# fetcher=Fetcher(loader)

# inputs=next(fetcher)

# inputs = next(iter(loader))
    
#%%
# for i in range(len(ds)):
#     inputs=ds[i]
#     if inputs.x.shape[1] != 160000:
#         print(inputs.x.shape)
    
# #%%
# loader_iter=iter(loader)
# for _ in range(len(loader)):
#     inputs = next(loader_iter)
    
    