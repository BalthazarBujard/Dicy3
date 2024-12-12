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
from concatenate import Concatenator, TimeStamp #custom library for concatenating audio chunks from time markers

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
    print("Corpus (= GT):\n",[label for label,_ in corpus_data])
    return memory_chunks, generator

def generate_response(src_ds : MusicContainer4dicy2, model : Seq2SeqCoupling,
                      chunk_segmentation : str, 
                      with_coupling : bool, k : int, decoding_type : str,
                      generator : Dicy2Generator):

    label_type = ListLabel
    
    collate_fn = MusicDataCollator(with_slices=True,unifrom_chunks=chunk_segmentation!="onset")

    src_loader = DataLoader(src_ds,1,False,collate_fn=collate_fn)
    src_fetcher=Fetcher(src_loader)
    src_fetcher.device=model.device
    
    eos=model.special_tokens_idx['eos'].item()
    
    queries = [] #slice indexes
    searches_for = [] #classes
    
    #now we iterate over all source chunks (sub-tracks) to create the whole response to input
    for i in range(len(src_fetcher)):
        src_data = next(src_fetcher)
                
        #generate response labels from source input : "What should be played given what i heard"
        #first encode source
        encoded_src, src_idx, src_pad_mask = model.encode(src_data.src, src_data.src_padding_masks) 
        
        if with_coupling: #if we want to generate a more complex response 
            _,tgt_idx = model.coupling(encoded_src, src_pad_mask, k, max_len=len(encoded_src[0]),decoding_type=decoding_type)
        
        else : tgt_idx = src_idx #for expermient purposes (identity matching with latent descriptors)
        
        search_for = np.array([label.item()  for label in tgt_idx[0][1:]]) #dont take sos

        #crop response to eos if early stop
        if any(search_for==eos):
            search_for = search_for[:np.where(search_for==eos)[0][0]]
        
        #with some models that collapsed there is only the eos token predicted and that riases error by influenceQuery
        if len(search_for)==0 :
            queries.extend([-1]*len(src_data.src[0])) #append as many silences as input chunks
            searches_for.extend([None]*len(src_data.src[0]))
            continue
        
        searches_for.extend(search_for)

        query = InfluenceQuery([LabelInfluence(label_type([v])) for v in search_for])

        output = generator.process_query(query)
        
        #memory slices index to retrieve
        slices=[typing.cast(Dicy2CorpusEvent, v.event).data if v is not None else None for v in output]
        
        #add silences to match input length
        if len(slices)<len(src_data.src[0]) :
            slices.extend([-1]*(len(src_data.src[0])-len(slices))) 
        
        queries.extend(slices)
    
    return queries, searches_for

def index_to_timestamp(index : int, chunks:np.ndarray):
    if index == -1: 
        t1 = len(np.reshape(chunks,-1))-1
        t0 = t1 - len(chunks[-1])
        return TimeStamp([t0,t1])
    
    t0 = sum([len(chunks[i]) for i in range(index)])
    t1 = t0+len(chunks[index])
    return TimeStamp([t0,t1])

def indexes_to_timestamps(indexes,chunks):
    markers = []
    for index in indexes:
        ts = index_to_timestamp(index,chunks)
        markers.append(ts)
    
    return markers

def compute_consecutive_lengths(idxs):
    if not idxs:
        return []
    
    lengths = []
    current_length = 1
    
    for i in range(1, len(idxs)):
        if idxs[i] == idxs[i - 1]+1 and idxs[i-1]!=-1:  # Same segment, increase length
            current_length += 1
        else:  # New segment, save current length and reset
            lengths.append(current_length)
            current_length = 1
    
    # Append the last segment length
    lengths.append(current_length)
    
    return lengths

def concatenate_response(memory:np.ndarray, memory_chunks:np.ndarray, queries:np.ndarray,
                         max_chunk_duration:float, sampling_rate:int, concat_fade_time:float, 
                         remove:bool, max_backtrack:float):
    #create concatenate object
    concatenate = Concatenator() 
    
    #append 2x chunk duration with zeros for silence handling. need 2 times for crossfade purposes
    silence = np.zeros(int(max_chunk_duration*sampling_rate))
    memory_with_silence = np.concatenate([memory,silence,silence]) 
    memory_chunks.extend([silence]*2)
    
    
    #convert queries to timestamps (markers)
    markers = indexes_to_timestamps(queries,memory_chunks)
    
    #create response from queries by concatenating chunks from memory
    response = concatenate(memory_with_silence, markers, sampling_rate,concat_fade_time,remove,max_backtrack)
    
    return response

#TODO : USE SLIDING WINDOW TO GENERATE NEW CHUNKS LABELS WITH SOME CONTEXT
@torch.no_grad()
def generate(memory_path:str, src_path:Union[str,list[str]], model:Union[Seq2SeqBase,str],
                      k:int, with_coupling : bool, decoding_type : str,
                      max_track_duration:float,max_chunk_duration:float,
                      track_segmentation:str, chunk_segmentation:str,
                      concat_fade_time=0.04, remove=True, max_backtrack = None,
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
    queries, searches_for = generate_response(src_ds, model, chunk_segmentation, with_coupling, k, decoding_type, generator)
    source = src_ds.native_track

    prYellow("Concatenate response...")
    response = concatenate_response(memory,memory_chunks,queries,max_chunk_duration,memory_ds.native_sr,concat_fade_time,remove,max_backtrack)
    
    prYellow("Computing statistics of consecutive segments...")
    lengths = compute_consecutive_lengths(queries)
    mean_len, median_len = np.mean(lengths), np.median(lengths)
     
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
        mix = np.sum([source,response],axis=0)
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
        original = np.sum([source,memory],axis=0)
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
                   remove, mean_len,median_len, entropy, w_size=max_chunk_duration,save_dir=save_dir, decoding_type=decoding_type)
    
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

def write_info(model: Seq2SeqBase, memory_path, source_paths, index, top_k, with_coupling, remove, mean_len, median_len, entropy, w_size, save_dir, decoding_type):
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
    f"\t\tvocab_size = {model.codebook_size}, segmentation = {model.segmentation}, w_size = {w_size}[s], top-K = {top_k}, with_coupling = {with_coupling}, remove = {remove}, decoding = {decoding_type}\n"
    f"\tAnalysis :\n"
    f"\t\tmean_len = {mean_len:.2f}, median_len = {median_len:.2f}, entropy = {entropy:.2f} [Bits]\n\n")
    
    # Open the file in append mode and write the content
    with open(info_path, 'a') as file:
        file.write(content)
