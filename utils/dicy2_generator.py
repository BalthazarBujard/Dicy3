#%%
#%%
from architecture.Model import load_model_checkpoint
from architecture.Seq2Seq import Seq2SeqBase,Seq2SeqCoupling
from utils.utils import lock_gpu, prGreen, prRed, prYellow, detect_onsets, find_non_empty
import torch
import numpy as np
import sys, typing, os
from typing import Union, List
import scipy.io.wavfile as wav
import soundfile as sf
from librosa import resample
from munch import Munch

#for dicy2 library
sys.path.insert(0,"/data3/anasynth_nonbp/bujard/Dicy2-python")

from MusicDataset.MusicDataset_v2 import MusicContainer4dicy2,Fetcher,MusicDataCollator
from torch.utils.data import DataLoader

from dicy2.corpus_event import Dicy2CorpusEvent # type: ignore
from dicy2.generator import Dicy2Generator # type: ignore
from dicy2.label import ListLabel # type: ignore
from dicy2.prospector import FactorOracleProspector # type: ignore
from gig.main.corpus import GenericCorpus # type: ignore
from gig.main.influence import LabelInfluence # type: ignore
from gig.main.query import InfluenceQuery # type: ignore

#TODO : FIND BETTER WINDOW FOR FADE -> cos et sin -> cf crossfade
#generate fading window
def fade_inout(size,fade_t=0.005,sampling_rate=16000):
    t_in = int(fade_t*sampling_rate) 
    t_out=int(fade_t*sampling_rate)
    in_w=np.hamming(t_in)[:t_in//2]
    out_w = np.hamming(t_out)[t_out//2:]
    
    if size < len(in_w)+len(out_w) : #no fade in/out if chunk too small
        return np.array([1]*size) 
    
    fade_inout = np.concatenate([in_w,[1]*(size-(len(in_w)+len(out_w))),out_w])

    return fade_inout

def cross_fade_windows(fade_time,sampling_rate):
    f=1/(4*fade_time) #frequency of cos and sine windows (sin=1 and cos=0 at tmax=fade_time)
    t=np.linspace(0,fade_time,int(fade_time*sampling_rate))
    fade_in = np.sin(2*np.pi*f*t)
    fade_out = np.cos(2*np.pi*f*t)
    
    return fade_in,fade_out

def crossfade(memory_chunks,response,continous,fade_in_idx,fade_out_idx,fade_in,fade_out,r):
    
    #get delta/2 in front of response (last continous chunk) and delta/2 before continous (new bloc to fade in)
    #not first iteration -> response not empty
    if fade_out_idx!=None:
        prev = memory_chunks[fade_in_idx-1][-r:] if fade_in_idx>0 else [] #s0-1
        nextt = memory_chunks[fade_out_idx+1][:r] if fade_out_idx<len(memory_chunks)-1 else [] #prev s1
        
        r1,r2=r,r
        if len(prev)==0:
            r2=2*r
        if len(nextt)==0:
            r1=2*r
        
        #separate crossfade point from continous and response
        cf1 = np.concatenate([response[-r1-1:],nextt])*fade_out
        cf2 = np.concatenate([prev,continous[:r2+1]])*fade_in 
        cf = cf1+cf2 #crossfade
        
        #remove cf part from continous and response before concatenating
        response = response[:-r1] #VERIFIER SI IL FAUT PAS FAIRE -r1±1
        continous = continous[r2:] #IDEM
    
    else : #first iteration --> response is empty, we just need to fade_in first continous segment
        prev = memory_chunks[fade_in_idx-1][-r:] if fade_in_idx>0 else [] #s0-1
        
        r2=r
        if len(prev)==0:
            r2=2*r
        
        cf = np.concatenate([prev,continous[:r2+1]])*fade_in
        
        #remove cf part
        continous = continous[r2:]
    
    #concate evrything
    response = np.concatenate([response,cf,continous]).tolist()
    
    return response

def crossfadev2(memory,response,continous,fade_in_t,fade_out_t,fade_in,fade_out,r,x_l,x_r):
    #compute fade_in, fade_out, r here from fade_time
    # fade_in,fade_out = cross_fade_windows(fade_t,sampling_rate)
    # r = int((fade_t/2)*sampling_rate) #crossing point = half fade time
    
    #get delta/2 in front of response (last continous chunk) and delta/2 before continous (new bloc to fade in)
    #not first iteration -> response not empty
    if fade_out_t!=None and fade_in_t!=None:
        prev = memory[fade_in_t-r:fade_in_t] if fade_in_t>r else [] #s0-1
        nextt = memory[fade_out_t:fade_out_t+r-x_l] if fade_out_t+r<len(memory) else [] #prev s1, shift by x to sync with new segment
        
        r1=len(fade_out)-len(nextt)
        r2=len(fade_in)-len(prev)
        
        to_fade_in = np.concatenate([prev,continous[:r2]])
        to_fade_out = np.concatenate([response[-r1:],nextt])
                
        #separate crossfade point from continous and response
        cf1 = to_fade_out*fade_out
        cf2 = to_fade_in*fade_in 
        cf = cf1+cf2 #crossfade
        
        #remove cf part from continous and response before concatenating
        response = response[:-r1-1] #remove end
        continous = continous[r2+1:] #remove beginning
    
    #first iteration --> response is empty, we just need to fade_in first continous segment
    #or prev was silence
    elif fade_in_t!=None : 
        prev = memory[fade_in_t-r:fade_in_t] if fade_in_t>r else [] #s0-1
        
        r2=len(fade_in)-len(prev)
        
        to_fade_in = np.concatenate([prev,continous[:r2]])
        
        cf = to_fade_in*fade_in
        
        #remove cf from continous
        continous = continous[r2+1:] #verifier si ±1
    
    elif fade_out_t != None : #if new segment is silence just fade out prev segemnt 
        nextt = memory[fade_out_t:fade_out_t+r-x_l] if fade_out_t+r<len(memory) else []
        r1=len(fade_out)-len(nextt)
        to_fade_out = np.concatenate([response[-r1:],nextt])
        cf = to_fade_out*fade_out
    
    else : #if both segemnts were silence do nothing
        cf = []
    
    #concate evrything
    response = np.concatenate([response,cf,continous]).tolist()
    
    return response

def clean(memory_chunks,continous,s0,sampling_rate):
    
    r_in,r_out=0,0
    duration = len(continous)/sampling_rate #seconds
    onsets,backtrack = detect_onsets(continous.astype(np.float32),sampling_rate,True) #seconds
    
    #close to end of continous chunk
    if any(onsets>0.9*duration):
        #crop end of cotninous to backtrack timestamp
        t_lim = int(backtrack[onsets>0.9*duration][0]*sampling_rate) #get first element
        r_out = len(continous)-t_lim
        continous = continous[:t_lim]
    
    #to check if there is actually an onset close to the begin we prepend the previous index to the first chunk in continous
    if any(onsets<0.1*duration):
        if s0>0: #if not first chunk
            prev = memory_chunks[s0-1]
            prev_dur = len(prev)/sampling_rate
            new_continous = np.concatenate([prev,continous])
            new_dur = duration + prev_dur
            
            onsets,backtrack = detect_onsets(new_continous.astype(np.float32),sampling_rate,True) #seconds
            
            if any(onsets<(prev_dur+0.1*new_dur)) and any(onsets>prev_dur): #onsets in begining of original
                #print(onsets[onsets<prev_dur+0.1*new_dur])
                #print(onsets[onsets>prev_dur])
                back=backtrack[onsets<(0.1*new_dur+prev_dur)][-1]
                #print(back)
                if back >= prev_dur/2 : #we dont want to go back to far to find a backtrack
                    #find first backtrack before onset (even in prev chunk)
                    t_lim = int(back*sampling_rate)
                    r_in = t_lim - len(prev) #>0 if backtrack in original continous and <0 if in prev
                    continous = new_continous[t_lim:] #continous is new_c cropped at backtrack
                
                else :
                    #use energy enveloppe to find local minima 
                    pass
    
    #print("r_in",r_in/sampling_rate,"r_out",r_out/sampling_rate)
    return continous,r_in,r_out



def find_elems_in_range(array, lower, upper):
    elems=[]
    for elem in array:
        if lower<elem<upper:
            elems.append(elem)
    return elems

#uses the timesstamps from onsets to detect early or late onsets in continous
#onsets,back in seconds
#t0 and t1 in samples
def cleanv2(memory,t0,t1,sampling_rate,onsets,backtrack,max_backtrack):
    
    #to seconds
    t0 = t0/sampling_rate
    t1 = t1/sampling_rate
    
    pad=0 #for sync issues, we need to have the same continous len before and after cleaning
    
    #duration = t1-t0
    #close to end of continous chunk
    lower = t1 - max_backtrack #min(0.25,0.1*duration) #for onset close to end no sync problem
    onsets_ = find_elems_in_range(onsets,lower,t1)
    if len(onsets_)>0:
        onset = onsets_[0] #first onset above thresh
        #find backtrack before onset
        back = backtrack[onsets<=onset][-1] #get matching backtrack to onset as new end
        if abs(back-t1)<max_backtrack: #dont go too far away
            pad = t1-back
            t1 = back
    
    #close to beginning onset
    upper = t0 + max_backtrack#0.1*duration
    onsets_ = find_elems_in_range(onsets,t0,upper)
    if len(onsets_)>0:
        onset = onsets_[0] #first onset above thresh
        back = backtrack[onsets<=onset][-1] #get matching backtrack to onset as new end
        if abs(back-t0)<max_backtrack: #dont go too far away
            pad += back-t0 #take account shift at beginning 
            t0 = back
            
            
    #to samples
    t0 = int(t0*sampling_rate)
    t1 = int(t1*sampling_rate)
    pad = int(pad*sampling_rate)
    
    # continous = memory[t0:t1]
    # if pad>0:
    #     fill = np.full(pad,fill_value=continous[-1])
    #     w=cross_fade_windows(pad*sampling_rate,sampling_rate)[1] #get cos window
    #     fill *= w #fade from last value to zero
    #     #peut etre il faut concatenate continous[:-1], fill pck sinon on repete valeure de fin
    #     continous = np.concatenate([continous,fill]) #pad with last value and fade out
        
    # if pad<0 : #happens if no backtrack at end and backtrack < t0 at beginning --> we should crop the end of the continous to maintain synchroinicity
    #     continous = continous[:pad]
    
    # return continous,t0,t1
    return t0,t1

def generate_memory_corpus(memory_ds : MusicContainer4dicy2, model : Seq2SeqCoupling, chunk_segmentation : str):
    
    #dicy2 args
    max_continuity: int = 10000  # longest continuous sequence of the original text that is allowed before a jump is forced
    force_output: bool = True  # if no matches are found: output the next event (if True) or output None (if False)
    label_type = ListLabel
    
    collate_fn = MusicDataCollator(with_slices=True,unifrom_chunks=chunk_segmentation!="onset")
    memory_loader = DataLoader(memory_ds,1,False,collate_fn=collate_fn)
    memory_fetcher = Fetcher(memory_loader)
    memory_fetcher.device=model.device
    
    #generate the whole corpus data from memory chunks
    memory = []
    corpus_data=[]
    all_memory_chunks = []
    last_slice=0
    
    #IT WONT WORK LIKE THIS DUE TO THE SUBSAMPLING !!! COULD USE HOP RATIO ON OUTPUT LENGTH BUT CHECK IF IT WORKS !!
    """ if track_segmentation=='sliding':
        hop_size = int(max_track_duration*sampling_rate//memory_ds.hop_ratio) #same hop ratio for both memory and src """
    
    for i in range(len(memory_fetcher)):
        memory_data = next(memory_fetcher) #contains chunks and other data for model
        #encode memory -> extract labels for each slice/chunk
        z, memory_idx, _ = model.encoder(memory_data.src, padding_mask = memory_data.src_padding_masks[0])
        
        """ if track_segmentation == 'sliding':
            #here keep only the idxs after the hop
            if i != 0: #if first segment there is no context -> take all
                memory_idx = memory_idx[:,-hop_size:] #take last hop_size bit
                memory_data.slices = memory_data.slices[:,-hop_size:] """
                
        
        #corpus = memory as [(label, content)] where label is codebook index from encoding and content is the slice index
        corpus_data.extend([(label.item(), content.item()+last_slice) for label,content in zip(memory_idx[0],memory_data.slices[0])])
        last_slice = corpus_data[-1][1]+1 #slices only go from 0 to max so we need to update the real slice index basded on ietration
        
        #retrieve corresponding memory chunks
        memory_chunks=memory_ds.get_native_chunks(i) #unprocessed chunks with native sr
        all_memory_chunks.extend(memory_chunks) #append to list of all chunks
        
        #append to memory
        #memory.extend(np.concatenate(memory_chunks))
    
    memory_chunks = all_memory_chunks #rename for simplicity
    #memory = np.concatenate(memory_chunks)
    #dicy2 functions
    corpus = GenericCorpus([Dicy2CorpusEvent(content, i, label=label_type([label]))
                                                for (i, (label, content)) in enumerate(corpus_data)],
                                                label_types=[label_type])

    prospector = FactorOracleProspector(corpus, label_type, max_continuity=max_continuity)
    generator = Dicy2Generator(prospector, force_output=force_output)
    
    return memory_chunks, generator

def generate_response(src_ds : MusicContainer4dicy2, model : Seq2SeqCoupling,
                      chunk_segmentation : str, with_coupling : bool, k : int, 
                      generator : Dicy2Generator):

    label_type = ListLabel
    
    collate_fn = MusicDataCollator(with_slices=True,unifrom_chunks=chunk_segmentation!="onset")

    src_loader = DataLoader(src_ds,1,False,collate_fn=collate_fn)
    src_fetcher=Fetcher(src_loader)
    src_fetcher.device=model.device
    
    eos=model.special_tokens_idx['eos'].item()
    
    queries = []
    searches_for = []
    
    #now we iterate over all source chunks (sub-tracks) to create the whole response to input
    for i in range(len(src_fetcher)):
        src_data = next(src_fetcher)
                
        #generate response labels from source input : "What should be played given what i heard"
        #first encode source
        encoded_src, src_idx, src_pad_mask = model.encode(src_data.src, src_data.src_padding_masks) 
        
        if with_coupling: #if we want to generate a more complex response 
            _,tgt_idx = model.coupling(encoded_src, src_pad_mask, k, max_len=len(encoded_src[0]))
        
        else : tgt_idx = src_idx #for expermient purposes (identity matching with latent descriptors)

        search_for = np.array([label.item()  for label in tgt_idx[0][1:]]) #dont take sos

        #crop response to eos if early stop
        if any(search_for==eos):
            search_for = search_for[:np.where(search_for==eos)[0][0]]
        
        #with some models that collapsed there is only the eos token predicted and that riases error by influenceQuery
        if len(search_for)==0 :
            queries.extend([-1]*len(src_data.src[0])) #append as many silences as input chunks
            continue
        
        searches_for.extend(search_for)

        query = InfluenceQuery([LabelInfluence(label_type([v])) for v in search_for])

        output = generator.process_query(query)
        
        #memory slices index to retrieve
        slices=[typing.cast(Dicy2CorpusEvent, v.event).data if v is not None else None for v in output]
        
        if len(slices)<len(src_data.src[0]) :
            slices.extend([-1]*(len(src_data.src[0])-len(slices))) #add silences to match input chunks
        
        queries.extend(slices)
    
    return queries, searches_for

#TODO : modifier MusicContainer4dicy2 pour renvoyer des timestamps       
def concatenate(memory : np.ndarray, memory_chunks : List[np.ndarray], queries : List[int], 
                fade_t : float, sampling_rate : int, remove : bool, max_backtrack : float):
    response = []
    start=0
    stop=1
    continious_lens=[]
    
    #memory = np.concatenate(memory_chunks)
    onsets, backtrack = detect_onsets(memory.astype(np.float32),sampling_rate,True) #compute onsets and backtrack once over whole memeory
    
    #compute crossfade windows
    fade_in,fade_out = cross_fade_windows(fade_t,sampling_rate)
    r = int((fade_t/2)*sampling_rate) #crossing point = half fade time
    #fade_in_idx,fade_out_idx = None,None
    fade_in_t,fade_out_t = None,None #fade in and out timestamps (in samples)
    
    if max_backtrack==None : max_backtrack = fade_t/2 #si plus grand que fade_t/2 il faudrait recalculer la fenetre 
    
    memory_chunks.append(np.zeros_like(memory_chunks[0])) #comme ca le -1 prends du silence
    
    while stop < len(queries):
        s0 = queries[start]
        s1 = queries[stop]
        
        continous=[memory_chunks[s0].tolist()] #init with first slice
        
        #fade_in_idx = s0 #save first index of continous chunk
        fade_in_t = sum([len(memory_chunks[i]) for i in range(0,s0)]) if s0>0 else 0 #start time of first chunk (in samples)
        
        # -1 is silence
        if s0==-1 : 
            fade_in_t = None #no fade in for silence
            #find longest consecutive silences
            while s1 == -1:
                continous.append(memory_chunks[s1].tolist())
                s0 = s1
                stop+=1
                if stop == len(queries):break #finished with consecutive silence
                s1 = queries[stop]
        # non silent chunk
        else :    
            #find longest consecutive queries
            while s1 == s0+1 and s0!=-1:
                continous.append(memory_chunks[s1].tolist())
                s0 = s1
                stop += 1
                if stop == len(queries):break #finished with consecutive
                s1 = queries[stop]
        
        #compute continous len
        continious_lens.append(len(continous))
        
        #going out from here s0 and s1 are not neighbors
        #process continous chunks
        
        continous = np.concatenate(continous) #flatten list of chunks as one chunk
       
        #check if there is an onset too soon or too late in the continous segment
        #r_in,r_out=0,0 # reste de crop par onset a utiliser pour modifier fade_in/out_t
        x_l, x_r = 0,0 # crossing points shift after backtrack
        if remove and s0!=-1:
            #continous,r_in,r_out=clean(memory_chunks,continous,s0,sampling_rate)
            t0,t1 = fade_in_t,fade_in_t+len(continous)
            
            # continous,t0,t1=cleanv2(memory,t0,t1,sampling_rate,onsets,backtrack,
            #                         max_backtrack=max_backtrack)
            
            # detect onsets and move crossing points
            t0,t1 = cleanv2(memory,t0,t1,sampling_rate,onsets,backtrack,
                                    max_backtrack=max_backtrack)
            
            continous = memory[t0:t1]
            
            #decalage du point de montage ...
            x_l = fade_in_t - t0 #du debut >0 : decaler en arriere et <0 decaler en avant
            #x_r = fade_out_t - t1 doit etre calcule apres crossfade # de fin toujours >=0
            
            fade_in_t=t0
                    
        #update fade_in/out_t
        #fade_in_t += r_in 
        #fade_out_t = fade_out_t-r_out if fade_out_t!=None else None
        
        #response = crossfade(memory_chunks,response,continous,fade_in_idx,fade_out_idx,fade_in,fade_out,r)
        response = crossfadev2(memory,response,continous,fade_in_t,fade_out_t,fade_in,fade_out,r,x_l,x_r)
        
        #fade_out_idx = s1 #last continous index is the fade out index for the next iteration
        x_r = fade_out_t - t1
        fade_out_t = fade_in_t+len(continous) if s0!=-1 else None
        #response.extend(continous)
        
        #update
        start = stop
        stop = start+1
        
        #if last element in the list
        if start == len(queries)-1 : 
            s0 = queries[start]
            continous = memory_chunks[s0].tolist() #take last query
            #fade_in_idx=s0
            
            fade_in_t = sum([len(memory_chunks[i]) for i in range(0,s0)]) if s0>0 else 0 #start time of first chunk
            if s0==-1 : fade_in_t = None
            
            #response = crossfade(memory_chunks,response,continous,fade_in_idx,fade_out_idx,fade_in,fade_out,r)    
            response = crossfadev2(memory,response,continous,fade_in_t,fade_out_t,fade_in,fade_out,r,x=0)
            continious_lens.append(1) #+1   
    
    mean_len = np.mean(continious_lens)
    meadian_len = np.median(continious_lens)
    
    return response, mean_len, meadian_len

#pas fondamentalement plus rapide... ET EN RETARD PAR RAPPORT AUX UPDATES DE LA V1
def concatenatev2(memory : np.ndarray, memory_chunks : List[np.ndarray], queries : List[int], 
                fade_t : float, sampling_rate : int, remove : bool, max_backtrack : float):
    response = []
    start=0
    stop=1
    continious_lens=[]
    
    #memory = np.concatenate(memory_chunks)
    onsets, backtrack = detect_onsets(memory.astype(np.float32),sampling_rate,True) #compute onsets and backtrack once
    
    fade_in,fade_out = cross_fade_windows(fade_t,sampling_rate)
    r = int((fade_t/2)*sampling_rate) #half fade time
    fade_in_idx,fade_out_idx = None,None
    fade_in_t,fade_out_t = None,None
    x=0
    if max_backtrack==None : max_backtrack = fade_t/2 #si plus grand que fade_t/2 il faudrait recalculer la fenetre (pas le temps)
    
    #construct continous as [[t0,t1],...]
    continous_t = []
    continous_idx = [] #not used but for v1 compatibility
    while stop < len(queries):
        s0 = queries[start]
        s1 = queries[stop]
        
        t0_idx = s0 #save first index of continous chunk
        t1_idx = s0 #idx is the same but is the end of chunk
        t0 = sum([len(memory_chunks[i]) for i in range(0,s0)]) if s0>0 else 0 #start time of first chunk (in samples)
        t1 = sum([len(memory_chunks[i]) for i in range(0,s0+1)]) if s0+1<len(memory_chunks) else len(memory)
        
        #find longest consecutive queries
        while s1 == s0+1:
            s0 = s1
            stop += 1
            t1_idx = s1
            t1 = sum([len(memory_chunks[i]) for i in range(0,s0+1)]) if s0+1<len(memory_chunks) else len(memory)
            if stop == len(queries):break #finished with consecutive
            s1 = queries[stop]
        
        continous_t.append([t0,t1])
        continous_idx.append([t0_idx,t1_idx])
        
        #update
        start = stop
        stop = start+1
    
    
    #clean with onset/backtrack analysis
    if remove:
        continous_chunks = []
        continous_t_p = []
        for t0,t1 in continous_t:
            continous,t0,t1=cleanv2(memory,t0,t1,sampling_rate,onsets,backtrack,
                                        max_backtrack=max_backtrack)  
            
            continous_chunks.append(continous)
            continous_t_p.append([t0,t1])
    else :
        continous_chunks = [memory[t0:t1] for t0,t1 in continous_t]
        continous_t_p = continous_t
    
    #crossfade
    for i,(ts,continous) in enumerate(zip(continous_t_p,continous_chunks)):
        
        t0_p,t1_p = ts #new times after cleaning
        t0,t1 = continous_t[i] #get old times before cleaning
        
        #compute fade_in_t and x (shift du point de montage)
        fade_in_t = t0_p
        x = t0-t0_p
        #compute fade_out
        if i>0:
            fade_out_t = continous_t_p[i-1][1] #t1_p of previous segment
        else :
            fade_out_t = None
        
        response = crossfadev2(memory,response,continous,fade_in_t,fade_out_t,fade_in,fade_out,r,x)
        
    #compute analysis     
    continious_lens = [t1_idx-t0_idx+1 for t0_idx,t1_idx in continous_idx ]
    
    mean_len = np.mean(continious_lens)
    meadian_len = np.median(continious_lens)
    
    return response, mean_len, meadian_len

#TODO : USE SLIDING WINDOW TO GENERATE NEW CHUNKS LABELS WITH SOME CONTEXT
@torch.no_grad()
def generate(memory_path:str, src_path:Union[str,list[str]], model:Union[Seq2SeqBase,str],
                      k:int, with_coupling : bool,
                      max_track_duration:float,max_chunk_duration:float,
                      track_segmentation:str, chunk_segmentation:str,
                      concat_fade_time=0.2, remove=True, max_backtrack = None,
                      device=None,
                      sampling_rate=16000, tgt_sampling_rates : dict = {'solo':None,'mix':None},
                      max_output_duration=None, mix_channels=2, timestamps=None,
                      save_files=True,
                      save_dir='output'):
    
    if track_segmentation=='sliding':
        raise NotImplementedError("No concatenation of output for track chunking with sliding window...")

    if chunk_segmentation=='onset':
        raise ValueError("Concatenation algorithm not compatible with 'onset' segmentation")
    
    if device == None : device = lock_gpu()[0][0]
    
    prYellow("Creating data structure from src and memory paths...")
    #build data structure for model input
    #collate_fn = MusicDataCollator(with_slices=True,unifrom_chunks=chunk_segmentation!="onset")

    memory_ds = MusicContainer4dicy2(memory_path,max_track_duration,max_chunk_duration,sampling_rate,
                                    chunk_segmentation,pre_segemntation=track_segmentation,
                                    timestamps=timestamps)

    src_ds =  MusicContainer4dicy2(src_path,max_track_duration,max_chunk_duration,sampling_rate,
                                    chunk_segmentation,pre_segemntation=track_segmentation,
                                    timestamps=timestamps)
    
    #load model if checkpoint path is given
    if isinstance(model,str):
        prYellow("Loading model from checkpoint...")
        model = load_model_checkpoint(model)
        model.eval()
        model.to(device)
    
    
    prYellow("Generating memory corpus...")
    memory_chunks, generator = generate_memory_corpus(memory_ds,model,chunk_segmentation)
    memory = memory_ds.native_track
    
    
    prYellow("Generating reponse...")
    queries, searches_for = generate_response(src_ds, model, chunk_segmentation, with_coupling, k, generator)
    source = src_ds.native_track

    prYellow("Concatenate response...")
    #create response from queries by concatenating chunks from memory
    response, mean_len, median_len = concatenate(memory,memory_chunks, queries, fade_t=concat_fade_time, 
                                                 sampling_rate=memory_ds.native_sr,remove=remove,
                                                 max_backtrack=max_backtrack)
    
     
    memory = np.array(memory)
    source = np.array(source)
    response = np.array(response)
    
    #normalize to -1,1
    def normalize(arr):
        return np.interp(arr,(arr.min(),arr.max()),(-1,1))
    
    memory = normalize(memory)
    source = normalize(source)
    response = normalize(response)
    
    query = np.array(queries)
    search_for = np.array(searches_for)
    
    #compute entropy of labels = search for -> diversity
    counts = np.bincount(search_for,minlength=model.codebook_size)
    probs = counts/sum(counts)
    entropy = -sum([p*np.log2(p+1e-9) for p in probs]) #in bits
    
    pad = len(response)-len(source)
    if pad > 0 : #response > source
        source = np.concatenate([source,np.zeros(pad)])
    elif pad < 0 : # source > response
        response = np.concatenate([response, np.zeros(abs(pad))])
    
    if mix_channels==2:
        mix = np.concatenate([source[:,None],response[:,None]],axis=1)
    else :
        mix = np.mean([source,response],axis=0)
        mix = normalize(mix) #re-normalize after mean otherwise volume drop
    
    
    #needed when new memory given 
    pad = len(source)-len(memory)
    if pad > 0 : #src > memory
        memory = np.concatenate([memory, np.zeros(abs(pad))])
    elif pad < 0 : # memory > src
        memory = memory[:-abs(pad)] #crop memory to source
    
    
    if mix_channels==2:
        original = np.concatenate([source[:,None],memory[:,None]],axis=1)
    else :
        original = np.mean([source,memory],axis=0)
        original = normalize(original)
        
    
    
    
       
    os.makedirs(save_dir,exist_ok=True)
    
    #wav.write("output.wav",rate=16000,data=response)
    #wav.write("output_mix.wav",rate=16000,data=mix)
    
    if max_output_duration!=None:
        t0,t1 = find_non_empty(response,max_output_duration,memory_ds.native_sr,return_time=True)
        memory = memory [t0:t1]
        response = response[t0:t1]
        source = source[t0:t1]
        original=original[t0:t1,:] if original.ndim==2 else original[t0:t1]
        mix = mix[t0:t1,:] if mix.ndim ==2 else mix[t0:t1]
    
    if save_files:
        prYellow("Saving files...")
        
        #folder_trackname --> moises : 45273_..._voix et cannone : A{i}_Duo2_1_guitare
        if "moises" in memory_path:
            track_name = os.path.basename(os.path.dirname((os.path.dirname(memory_path)))) #track folder
            instrument_name = os.path.basename(os.path.dirname(memory_path))
            memory_name = f"{track_name}_{instrument_name}"
        else :
            A_name = os.path.basename(os.path.dirname(memory_path))
            memory_name = f"{A_name}_{os.path.basename(memory_path).split('.')[0]}"  
         
        save_file(save_dir,"memory",memory_name,memory,"wav",orig_rate=memory_ds.native_sr,tgt_rate=tgt_sampling_rates['solo'])

        #source_name = directory if moises, track name if canonne
        if "moises" in src_path[0]:
            track_name = os.path.basename(os.path.dirname((os.path.dirname(src_path[0])))) #track folder
            #instrument_name = os.path.basename(os.path.dirname(src_path[0]))
            source_name = f"{track_name}"
        else :
            source_name = f"{os.path.basename(src_path[0]).split('.')[0]}"  
              
        save_file(save_dir,"source",source_name,source,"wav",orig_rate=src_ds.native_sr,tgt_rate=tgt_sampling_rates['solo'])
        
        #use same name as memory for response --> crucial for evaluation
        save_file(save_dir,"response",memory_name,response,"wav",orig_rate=memory_ds.native_sr,tgt_rate=tgt_sampling_rates['solo'])
        
        mix_name = f"Cont_{source_name}_Mem_{memory_name}_A{model.codebook_size}_D{max_chunk_duration}_K{k}"
        save_file(save_dir,"mix",mix_name,mix,"wav",orig_rate=memory_ds.native_sr,tgt_rate=tgt_sampling_rates['mix'])
        
        save_file(save_dir,"original",source_name,original,"wav",orig_rate=memory_ds.native_sr,tgt_rate=tgt_sampling_rates['mix'])
        
        save_file(save_dir,"query",f"{mix_name}",query,"txt",orig_rate=None,tgt_rate=None)
        idx = save_file(save_dir,"search_for",f"{mix_name}",search_for,"txt",orig_rate=None,tgt_rate=None)
        
        write_info(model,memory_path, src_path, mix_name, k, with_coupling, 
                   remove, mean_len,median_len, entropy, w_size=max_chunk_duration,save_dir=save_dir)
    
    return Munch(memory = memory,
                 source = source,
                 response = response,
                 mix = mix,
                 original = original,
                 search_for = search_for,
                 query = query)

import torchaudio    
def save_file(dir, folder, fname, data, extension, orig_rate, tgt_rate):
    dir = os.path.join(dir,folder)
    os.makedirs(dir,exist_ok=True)
    idx=0
    while True:
        path = os.path.join(dir,f"{fname}_{idx}.{extension}")
        if not os.path.exists(path):
            break 
        idx+=1
    
    if extension == "wav":
        if tgt_rate != None and orig_rate != tgt_rate:
            #to (c,frames)
            if data.ndim==2:
                data = np.swapaxes(data,0,1) #(c,frames)
            data_tensor = torch.tensor(data)
            
            #resample
            data = torchaudio.functional.resample(data_tensor,orig_rate,tgt_rate).numpy(force=True)
            
            #to (frames,c)
            if data.ndim==2:
                data=np.swapaxes(data,0,1) #(frames,c)
            
            rate=tgt_rate 
        else : rate=orig_rate
        
        sf.write(path,samplerate=rate,data=data.astype(np.float32)) #wavfile expects -1,1 range and float32
        
    elif extension == "txt":
        np.savetxt(path,data,fmt='%d')
    
    return idx

def write_info(model,memory_path, source_paths, index, top_k, with_coupling, remove, mean_len, median_len, entropy, w_size, save_dir):
    # Ensure the info directory exists
    info_path = f"{save_dir}/info.txt"
    os.makedirs(os.path.dirname(info_path), exist_ok=True)
    
    if not isinstance(source_paths,list): source_paths = [source_paths] 
    
    # Prepare the content to write
    
    content = (
    f"Mix : {index} :\n"
    f"\tMemory: {memory_path}\n"
    f"\tSources:\n"
    + "\n".join(f"\t - {path}" for path in source_paths) + "\n"
    f"\tParams :\n"
    f"\t\tvocab_size = {model.codebook_size}, segmentation = {model.segmentation}, w_size = {w_size}[s], top-K = {top_k}, with_coupling = {with_coupling}, remove = {remove}\n"
    f"\tAnalysis :\n"
    f"\t\tmean_len = {mean_len:.2f}, median_len = {median_len:.2f}, entropy = {entropy:.2f} [Bits]\n\n")
    
    # Open the file in append mode and write the content
    with open(info_path, 'a') as file:
        file.write(content)
