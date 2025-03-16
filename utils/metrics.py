import torch
from pathlib import Path
#from frechet_audio_distance import FrechetAudioDistance
from fadtk import FrechetAudioDistance
from fadtk.model_loader import CLAPLaionModel,EncodecEmbModel,CLAPModel
from fadtk.fad_batch import cache_embedding_files
from utils.utils import compute_consecutive_lengths
from librosa.feature import mfcc
from sklearn.mixture import GaussianMixture
from librosa import load
import numpy as np
import glob
from torcheval.metrics import WordErrorRate  # type: ignore

def compute_WER(preds : torch.Tensor, gts : torch.Tensor, eos_idx : int) -> torch.Tensor:
    """_summary_
    Compute the Word Error Rate of sequences
    Args:
        preds (torch.Tensor): [batch size, sequence length] the predicted sequences
        gts (torch.Tensor): [batch size, sequence length] the ground truth sequences
    """
    
    wer = WordErrorRate()
    for pred, gt in zip(preds,gts):
        pred_stop = torch.argwhere(pred==eos_idx)[0,0]+1 if torch.isin(eos_idx,pred) else len(pred)
        tgt = " ".join([str(idx.item()) for idx in pred[:pred_stop]])
        
        gt_stop = torch.argwhere(gt==eos_idx)[0,0]+1 if torch.isin(eos_idx,gt) else len(gt)
        gt_ = " ".join([str(idx.item()) for idx in gt[:gt_stop]])

        wer.update([tgt],[gt_])
    
    return wer.compute()

def compute_accuracy(pred_sequence, gt_sequence, pad_idx):
    correct = sum(1 for gt,pred in zip(gt_sequence,pred_sequence) if (gt==pred and gt != pad_idx))
    total = len(gt_sequence[gt_sequence!=pad_idx])
    
    acc = correct/total
    return acc

def compute_entropy(input : torch.Tensor, min_length : int) -> torch.Tensor:
    counts = torch.bincount(input=input,minlength=min_length)
    probs = counts/torch.sum(counts)
    
    entropy = -torch.sum(probs*torch.log2(probs+1e-9))
    return entropy

#function to evaluate audio quality of predictions
#ref and tgt are paths to folders containing audio files
def evaluate_audio_quality(reference_dir : Path, target_dir : Path, fad_inf : bool, device : torch.device):
    
    # if device==None : device = lock_gpu()[0][0]
    
    model = EncodecEmbModel('48k')
    model.device=device
    
    #compute embeddings
    for d in [reference_dir, target_dir]:
        if Path(d).is_dir():
            cache_embedding_files(d, model, workers=1)
 
    fad = FrechetAudioDistance(model,audio_load_worker=1,load_model=False)
    
    if fad_inf:
        target_files = [Path(f) for f in glob.glob(target_dir+"/*.wav")]
        score = fad.score_inf(reference_dir,target_files).score

    else : score = fad.score(reference_dir,target_dir) 
    
    return score

def evaluate_APA(background_dir : Path, fake_background_dir : Path, target_dir : Path, embedding : str, fad_inf : bool, device :torch.device):

    #background is the folder containing true pairs
    #fake_background is the folder containing misaligned pairs = mix with a random accompaniement from random track
    #target is the folder containing the mix
    
    # if device==None : device = lock_gpu()[0][0]
    
    if embedding=="L-CLAP":
        model = CLAPLaionModel('music') 
    elif embedding == "CLAP":
        model = CLAPModel("2023")
    
    else : raise ValueError("'embedding' should be CLAP or L-CLAP")
    
    model.device=device
    
    #compute embeddings
    for d in [background_dir,fake_background_dir, target_dir]:
        if Path(d).is_dir():
            cache_embedding_files(d, model, workers=1)
    
    fad = FrechetAudioDistance(model,audio_load_worker=1,load_model=False)
    
    if fad_inf :
        target_files = [Path(f) for f in glob.glob(target_dir+"/*.wav")]
        fadYX_ = fad.score_inf(fake_background_dir,target_files).score #fad target and fake bg
        
        fadYX = fad.score_inf(background_dir,target_files).score
        
        fake_bg_files = [Path(f) for f in glob.glob(fake_background_dir+"/*.wav")]
        fadXX_ = fad.score_inf(background_dir,fake_bg_files).score
    
    else :
        fadYX_ = fad.score(fake_background_dir,target_dir) 
        fadYX = fad.score(background_dir,target_dir) 
        fadXX_ = fad.score(background_dir,fake_background_dir) 
    
    fads = {'XX_':fadXX_,"YX":fadYX,"YX_":fadYX_}
    
    #prGreen(f"{fadXX_},{fadYX},{fadYX_}")
    APA = 0.5 + (fadYX_ - fadYX)/fadXX_ 
    
    return APA, fads

def evaluate_similarity(target : np.ndarray, tgt_sr : int, gt : np.ndarray, gt_sr : int, w_size : float = 0.05):
    
    #compute MFCC for target and gt tracks with frame size of 50ms (cf "Music Similarity") with no overlaping frames
    N = int(w_size*tgt_sr)
    tgt_mfcc = mfcc(y=target,sr=tgt_sr,n_mfcc=8,n_fft=N,hop_length=N) #(8,#frames) 
    tgt_samples = np.swapaxes(tgt_mfcc,0,1) #(num samples = #frames, 8 = #features)
    
    N = int(w_size*gt_sr)
    gt_mfcc = mfcc(y=gt,sr=gt_sr,n_mfcc=8,n_fft=N,hop_length=N)
    gt_samples = np.swapaxes(gt_mfcc,0,1)
    
    
    #fit GMMs with mfcc features
    tgt_GMM = GaussianMixture(n_components=3,n_init=3,max_iter=300,random_state=42)
    tgt_GMM.fit(tgt_samples)
    
    gt_GMM = GaussianMixture(n_components=3,n_init=3,max_iter=300,random_state=42)
    gt_GMM.fit(gt_samples)
    
    #compute (log-)likelihood of "song A being generated from model B"
    score = gt_GMM.score(tgt_samples)
        
    return score

def compute_lengths_histogram(arr : np.ndarray, normalized : bool = False) -> np.ndarray :
    #compute consecutive lengths of the array
    lengths = compute_consecutive_lengths(arr)
    
    #compute histogram (=bincount)
    histogram = np.bincount(lengths)
    
    #probabilities rather than count
    if normalized : histogram/=sum(histogram)
    
    return histogram