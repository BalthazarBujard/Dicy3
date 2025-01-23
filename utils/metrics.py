import torch
from pathlib import Path
#from frechet_audio_distance import FrechetAudioDistance
from fadtk import FrechetAudioDistance
from fadtk.model_loader import CLAPLaionModel,EncodecEmbModel
from fadtk.fad_batch import cache_embedding_files

def compute_accuracy(pred_sequence, gt_sequence, pad_idx):
    correct = sum(1 for gt,pred in zip(gt_sequence,pred_sequence) if (gt==pred and gt != pad_idx))
    total = len(gt_sequence[gt_sequence!=pad_idx])
    
    acc = correct/total
    return acc

def compute_entropy(input,min_length):
    counts = torch.bincount(input=input,minlength=min_length)
    probs = counts/torch.sum(counts)
    
    entropy = -torch.sum(probs*torch.log2(probs+1e-9))
    return entropy

#function to evaluate audio quality of predictions
#ref and tgt are paths to folders containing audio files
def evaluate_audio_quality(reference_dir : Path, target_dir : Path):
    

    # fad = FrechetAudioDistance(
    #     model_name="encodec",
    #     sample_rate=48000,
    #     channels=2,
    #     verbose=True
    #     )
    
    model = EncodecEmbModel('48k')
    
    #compute embeddings
    for d in [reference_dir, target_dir]:
        if Path(d).is_dir():
            cache_embedding_files(d, model, workers=2)

    fad = FrechetAudioDistance(model,audio_load_worker=2,load_model=False)
    #fad.device=device
    score = fad.score(reference_dir,target_dir)
    return score

#PROBLEM WITH CLAP EMBEDDING
def evaluate_APA(background_dir : Path, fake_background_dir : Path, target_dir : Path):

    #background is the folder containing true pairs
    #fake_background is the folder containing misaligned pairs = mix with a random accompaniement from random track
    #target is the folder containing the mix 
    # fad = FrechetAudioDistance(
    #     model_name="vggish",
    #     sample_rate=16000,
    #     use_pca=False, 
    #     use_activation=False,
    #     verbose=True
    # )
    #fad.device=device
    
    model = CLAPLaionModel('music')
    
    #compute embeddings
    for d in [background_dir,fake_background_dir, target_dir]:
        if Path(d).is_dir():
            cache_embedding_files(d, model, workers=2)
    
    fad = FrechetAudioDistance(model,audio_load_worker=2,load_model=False)
    
    fadYX_ = fad.score(target_dir,fake_background_dir) #fad target and fake bg
    fadYX = fad.score(target_dir,background_dir)
    fadXX_ = fad.score(background_dir,fake_background_dir)
    
    fads = {'XX_':fadXX_,"YX":fadYX,"YX_":fadYX_}
    
    #prGreen(f"{fadXX_},{fadYX},{fadYX_}")
    APA = 0.5 + (fadYX_ - fadYX)/fadXX_ 
    
    return APA, fads