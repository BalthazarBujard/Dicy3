
import torch
import numpy as np
from librosa import time_to_frames,frames_to_time
from librosa.onset import onset_backtrack, onset_detect
from essentia.standard import Windowing,FFT,CartesianToPolar,FrameGenerator,Onsets,OnsetDetection
import essentia 

#utilitary functions
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk)) #function to print in green color in terminal
def prRed(skk): print("\033[91m {}\033[00m" .format(skk)) #red
def prYellow(skk): print("\033[93m {}\033[00m" .format(skk)) #yellow

def depth(L):
    if isinstance(L, list) or isinstance(L,tuple):
        return max(map(depth, L)) + 1
    return 0

def remove_duplicates(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def process_onsets(onsets,min_duration,max_duration): #in smaples or time
    processed_onsets = []
    
    for t0,t1 in zip(onsets[:-1],onsets[1:]):
        if t1-t0 <= max_duration:
            processed_onsets.extend([t0,t1])
        
        else :
            #rechunk too large chunks
            while t1-t0 > max_duration:
                processed_onsets.extend([t0,t0+max_duration])
                t0 = t0+max_duration
            
            processed_onsets.extend([t0,t1]) #append last element less than max_duration
    
    #remove duplicates
    processed_onsets = remove_duplicates(processed_onsets)
    return processed_onsets
                

""" def process_onsets(onsets, min_samples, max_samples):
    # Initialize an empty list to hold the processed segments
    processed_onsets = []
    i=0
    j=1
    while j <= len(onsets) - 1:
        t0,t1 = onsets[i],onsets[j]
        duration=t1-t0
        
        #if in the right range add and next
        if min_samples<=duration<=max_samples:
            processed_onsets.extend([t0,t1])
            i+=1
            j+=1
            continue
        
        #handle big onsets
        while duration > max_samples:
            #chunk by max_duration until duration is less than max
            processed_onsets.extend([t0,t0+max_samples])
            t0 = t0+max_samples
            duration = t1-t0
        
        #if at the end of chunking its more than min add the end of original onset and continue
        if duration>=min_samples:
            processed_onsets.extend([t0,t1])
            i+=1
            j+=1
            continue
        
        elif duration<min_samples and j < len(onsets)-1:
            j=i+1
            #retirer element qui a ete saute
            del onsets[i+1]
    
    processed_onsets = remove_duplicates(processed_onsets)        
    
    return processed_onsets """

def lock_gpu(num_devices=1):
    try :
        import manage_gpus as gpl
        manager=True
    except ModuleNotFoundError as e:
        manager=False
        
    devices=[]
    ids=[]
    if manager:
        for i in range(num_devices):
            try:
                    gpu_id_locked = gpl.obtain_lock_id(id=-1)
                    if gpu_id_locked!=-1:
                        device = torch.device(f"cuda:{i}")
                        prYellow(f"Locked GPU with ID {gpu_id_locked} on {device}")
                    else :
                        prRed("No GPU available.")
                        device=torch.device("cpu")
    
            except:
                prRed("Problem locking GPU. Send tensors to cpu")
                device=torch.device("cpu")
                gpu_id_locked=-1
            
            devices.append(device)
            ids.append(gpu_id_locked)
    else :
        device = torch.device("gpu" if torch.conda.is_available() else "cpu")
        devices = [device]
        ids=[-1]
    
    return devices,ids

def model_params(checkpoint):
    
    ckp = torch.load(checkpoint,map_location=torch.device('cpu'))
    params = ckp['model_params']
    for key, item in params.items():
        prYellow(f"{key} : {item}")


def load_trained_backbone_from_classifier(pretrained_file, backbone_checkpoint="/data3/anasynth_nonbp/bujard/w2v_music_checkpoint.pt"):
    from architecture.Encoder import Backbone
    from fairseq import checkpoint_utils
    import torch.nn as nn
    
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task([backbone_checkpoint])
    baseline_model=models[0]
    
    #w2v backbone adapted for classif
    state_dict=torch.load(pretrained_file,map_location=torch.device("cpu"))
    adapted_model = Backbone(baseline_model,"w2v",mean=True) #adapted model used mean of latents accross time
    
    num_classes=12 #individual instrument labels
    fc = nn.Linear(adapted_model.dim,num_classes)
    
    #list of state dicts if saved whole classifier
    if isinstance(state_dict,list):
        #si .pt contient model
        adapted_model.load_state_dict(state_dict[0])
        fc.load_state_dict(state_dict[1])
        
    
    else: #older trained models only had backbone in state dict
        #si .pt contient state_dict -> instancier model avant de load
        adapted_model.load_state_dict(state_dict)
        prYellow(f"Warning : there is not a checkpoint for the classification head. Returned random classif head.")
    
    classifier = nn.Sequential(
        adapted_model,
        fc
    )
        
    return adapted_model, classifier


def topK_search(k, logits,targets=None):
    B,C = logits.size()
    topK_idx = torch.topk(logits,k,dim=-1)[1] #(B,k)
    
    def random_sample(topK_idx):
        N,k = topK_idx.size()
        random_idx = torch.randint(0,k,size=(N,1),device=logits.device)
        sampled_idx = torch.gather(topK_idx, 1, random_idx).squeeze(1)
        
        return sampled_idx
    
    if targets != None :
        #find the corresct value in the topK
        #if there isnt then random choice
        sampled_idx=torch.empty(B, dtype=torch.long, device = logits.device) # init empty tenosr
        for i,(topK_sample,tgt) in enumerate(zip(topK_idx,targets)):
            
            if tgt in topK_sample:
                sampled_idx[i]=tgt
            
            else :
                #random sample from that list
                sampled_idx[i] = random_sample(topK_sample.unsqueeze(0)) #fct expects (N,k)-->(1,k)
    
    else :
        #return random sample among those values
        sampled_idx = random_sample(topK_idx)
    
    return sampled_idx

def predict_topK(k,logits,tgt=None):
    logits_rs = logits.reshape(-1,logits.size(-1))
    tgts_rs = tgt.reshape(-1) if tgt!=None else None
    preds = topK_search(k,logits_rs,tgts_rs)
    return preds

def build_coupling_ds(roots, BATCH_SIZE, MAX_TRACK_DURATION, MAX_CHUNK_DURATION,
                    segmentation_strategy="uniform",
                    pre_segmentation='sliding',
                    ignore=[],
                    direction="stem",
                    mask_prob=0.0,
                    mask_len=0,
                    SAMPLING_RATE=16000,
                    distributed=True):
    
    from MusicDataset.MusicDataset_v2 import MusicCouplingContainer, DataCollatorForCoupling, Fetcher
    from torch.utils.data import DataLoader,DistributedSampler
    import os
    
    #if mask_prob>0 or mask_len> 0 : raise NotImplementedError()

    collate_fn = DataCollatorForCoupling(unifrom_chunks=segmentation_strategy!="onset",sampling_rate=SAMPLING_RATE,mask_prob=mask_prob,mask_len=mask_len)
    
    ds = MusicCouplingContainer(roots, 
                            MAX_TRACK_DURATION, 
                            MAX_CHUNK_DURATION, 
                            SAMPLING_RATE,segmentation_strategy,
                            pre_segmentation=pre_segmentation,
                            ignore_instrument=ignore,
                            direction=direction) #for onset segmentation still needs upgrading to do (handle uneven number of chunks from input and target)
    sampler=None
    shuffle=True
    if distributed:
        sampler=DistributedSampler(ds)
        shuffle=False
    
    loader = DataLoader(ds, BATCH_SIZE, shuffle=shuffle,sampler=sampler, #with distributed shuffle = false
                        collate_fn=collate_fn,num_workers=2,pin_memory=True)

    fetcher = Fetcher(loader)
    
    return fetcher

#function to find the memory path in the info.txt file given a mix/response index
def extract_memory_path(file_path, index):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    mix_tag = f"Mix n. {index} :"
    memory_path = None
    
    for i, line in enumerate(lines):
        if mix_tag in line:
            # Look for the Memory line immediately following the mix_tag line
            for j in range(i + 1, len(lines)):
                if "Memory:" in lines[j]:
                    memory_path = lines[j].split("Memory:")[1].strip()
                    return memory_path
    return memory_path

def find_non_empty(track,max_duration,sampling_rate,return_time=False):
        max_samples=int(max_duration*sampling_rate) #convert max_duration in samples
        N=len(track)//max_samples #how many chunks
        if N>0:
            track=track[:N*max_samples] #remove last uneven section
            chunks = track.reshape(-1,max_samples) 
            chunks_norm = (chunks-np.mean(chunks,axis=-1,keepdims=True))/(np.std(chunks,axis=-1,keepdims=True)+1e-5) #Z-score normalize
            energies = np.sum(chunks_norm**2,axis=-1) #energies accross chunks
            #find first occurence of energy above threshold
            non_empty_chunk_idx = np.where(energies > 0.5)[0][0] if np.any(energies > 0.5) else None
            if non_empty_chunk_idx==None : 
                if not return_time:
                    return chunks[0]
                else :
                    len(chunks[0]),len(chunks[0])+max_samples
            
            non_empty_chunk = chunks[non_empty_chunk_idx]
            if not return_time:
                return non_empty_chunk
            else : return len(chunks[:non_empty_chunk_idx]),len(chunks[:non_empty_chunk_idx])+max_samples
        else : 
            if not return_time:
                return track
            else : return 0,len(track)

def normalize(arr):
    return np.interp(arr,(arr.min(),arr.max()),(-1,1))
    
""" 
def evaluate(model : torch.nn.Module, eval_fetcher, k: int, codebook_sample_temperature : float = 0):
    from architecture.Seq2Seq import Seq2SeqBase,Seq2SeqCoupling
    from utils.metrics import compute_accuracy,compute_entropy
    from tqdm import tqdm
    
    model.eval()
    acc=0
    diversity = 0 #predicted tokens entropy
    codebook_perplexity = 0 #entropy of encoded inputs
    with torch.no_grad():
        for _ in tqdm(range(len(eval_fetcher))):
            inputs = next(eval_fetcher)
            
            if type(model)==Seq2SeqCoupling:
                src, tgt, src_pad_mask, tgt_pad_mask, src_mask_indices = inputs.values()
                #compute output
                logits, tgt, tgt_idx, _ = model(src, tgt, src_pad_mask, tgt_pad_mask,
                                                            sample_codebook_temp=codebook_sample_temperature)
                
            elif type(model)==Seq2SeqBase: #for autocompletion
                src, src_pad_mask, src_mask_indices, label = inputs.values() 
                #compute output
                logits, tgt, tgt_idx, _ = model(src, src_pad_mask, sample_codebook_temp=codebook_sample_temperature)
            
            #encoded inputs
            encoded_src = model.encoder(src, padding_mask = src_pad_mask[0])[1] #only take cb indexes to compute entropy
                    
            tgt_out = tgt_idx[:,1:] #ground truth
            
            #topK search
            preds = predict_topK(k,logits,tgt_out)
            
            #accuracy
            acc += compute_accuracy(preds,tgt_out.reshape(-1),pad_idx=model.special_tokens_idx["pad"])
            
            #diversity
            diversity += compute_entropy(preds,min_length=model.vocab_size).item()
            
            
            codebook_perplexity += 2**compute_entropy(encoded_src.reshape(-1),min_length=model.codebook_size).item()
            
            #del src, tgt, src_pad_mask, tgt_pad_mask, src_mask_indices, logits, tgt_idx, tgt_out, preds
    
    acc /= len(eval_fetcher)
    diversity/=len(eval_fetcher)
    codebook_perplexity/=len(eval_fetcher)
    
    #torch.cuda.empty_cache()
    #del src, tgt, src_pad_mask, tgt_pad_mask, src_mask_indices, logits, tgt_idx, tgt_out, preds
    
    return acc, diversity, codebook_perplexity """


# def detect_onsets(track, sampling_rate, with_backtrack):
#     if sampling_rate<44100 : raise ValueError("The sampling rate for essentia onset detect is otpimized for 44.1kHz. For lower rates use librosa.")
    
#     od_complex = OnsetDetection(method='complex')

#     w = Windowing(type='hann')
#     fft = FFT() # Outputs a complex FFT vector.
#     c2p = CartesianToPolar() # Converts it into a pair of magnitude and phase vectors.

#     pool = essentia.Pool()
#     for frame in FrameGenerator(track, frameSize=1024, hopSize=512):
#         magnitude, phase = c2p(fft(w(frame)))
#         pool.add('odf.complex', od_complex(magnitude, phase))

#     # 2. Detect onset locations.
#     onsets = Onsets()
#     onsets_complex = onsets(essentia.array([pool['odf.complex']]), [1])
    
#     #3 post process onsets : if any onset detected after duration -> remove
#     onsets_complex = onsets_complex[onsets_complex<len(track)/sampling_rate]
    
#     if not with_backtrack:
#         return onsets_complex

#     onsets_backtrack=np.array([])
#     if len(onsets_complex)>0:
#         onset_frames = time_to_frames(onsets_complex,sr=sampling_rate,hop_length=512)

#         onsets_backtrack = onset_backtrack(onset_frames,pool['odf.complex'])
#         onsets_backtrack = frames_to_time(onsets_backtrack,sr=sampling_rate,hop_length=512)
    
#     return onsets_complex, onsets_backtrack

def detect_onsets(audio, sr,with_backtrack):
    onsets = onset_detect(y=audio,sr=sr,backtrack=False,units='time')
    if with_backtrack:
        backtrack = onset_detect(y=audio,sr=sr,backtrack=True,units='time')
        
        return onsets, backtrack
    return onsets
