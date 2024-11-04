#%%
from utils.utils import lock_gpu
device = lock_gpu()[0][0]
import os
import torch
from architecture.Seq2Seq import Seq2SeqBase,Seq2SeqCoupling
from architecture.Model import load_model_checkpoint
from MusicDataset.MusicDataset_v2 import Fetcher
from utils.metrics import compute_accuracy,compute_entropy,evaluate_APA,evaluate_audio_quality
from utils.utils import predict_topK,prGreen,build_coupling_ds, extract_memory_path,prYellow, find_non_empty
from utils.coupling_ds_generator import extract_all_groups
from top_k_validity import compute_similarity
from tqdm import tqdm
import numpy as np 
from pathlib import Path
from typing import Union
from librosa import load
from librosa.feature import mfcc
from sklearn.mixture import GaussianMixture
from munch import Munch
import glob
from utils.dicy2_generator import generate
import argparse

def evaluate_similarity(target : Path, gt : Path):
    
    w_size = 0.05 # seconds
    
    target, tgt_sr= load(target,sr=None, mono=True)
    gt, gt_sr = load(gt,sr=None,mono=True)    
        
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
    

#this function evaluates the Decision module and the perception's quantized encoding
def evaluate_model(model : Seq2SeqBase, eval_fetcher : Fetcher, k: int):
    
    model.eval()
    acc=0
    diversity = 0 #predicted tokens entropy
    codebook_perplexity = 0 #entropy of encoded inputs
    cosine_sim = 0
    
    accs = []
    divs = []
    sims = []
    perplexs=[]
    
    eos_idx = model.special_tokens_idx["eos"].to(device)
    codebook = model.vocab_embedding_table
    
    with torch.no_grad():
        for _ in tqdm(range(len(eval_fetcher))):
            inputs = next(eval_fetcher)
            
            if type(model)==Seq2SeqCoupling:
                src, tgt, src_pad_mask, tgt_pad_mask, src_mask_indices = inputs.values()
                #compute output
                logits, tgt, tgt_idx, _ = model(src, tgt, src_pad_mask, tgt_pad_mask)
                
            elif type(model)==Seq2SeqBase: #for autocompletion
                src, src_pad_mask, src_mask_indices, label = inputs.values() 
                #compute output
                logits, tgt, tgt_idx, _ = model(src, src_pad_mask)
            
            #encoded inputs
            encoded_src = model.encoder(src, padding_mask = src_pad_mask[0])[1] #only take cb indexes to compute entropy
                    
            tgt_out = tgt_idx[:,1:] #ground truth
            
            #topK search
            preds = predict_topK(k,logits,tgt_out)
            
            #accuracy with topK
            accs.append(compute_accuracy(preds,tgt_out.reshape(-1),pad_idx=model.special_tokens_idx["pad"]))
            
            #diversity of pred
            divs.append(compute_entropy(preds,min_length=model.vocab_size).item())
            
            #perplexity -> cb usage
            perplexs.append(2**compute_entropy(encoded_src.reshape(-1),min_length=model.codebook_size).item())
            
            #topK validity
            sims.append(compute_similarity(tgt, tgt_idx, logits,
                                             codebook, eos_idx, k).numpy(force=True))
            
            #del src, tgt, src_pad_mask, tgt_pad_mask, src_mask_indices, logits, tgt_idx, tgt_out, preds
    
    acc = np.mean(accs)
    acc_std = np.std(accs)
    
    diversity = np.mean(divs)
    diversity_std = np.std(divs)
    
    codebook_perplexity = np.mean(perplexs)
    codebook_perplexity_std = np.std(perplexs)
    
    sims = np.array(sims) #convert to numpy
    cosine_sim = np.mean(sims[:,0])
    cosine_sim_std = np.mean(sims[:,1]) #mean std over top-K values (dont do another std otherwise its the std of the std)
    
    #torch.cuda.empty_cache()
    #del src, tgt, src_pad_mask, tgt_pad_mask, src_mask_indices, logits, tgt_idx, tgt_out, preds
    
    return Munch(accuracy={"mean":acc,"std":acc_std},
                 diversity={"mean":diversity,"std":diversity_std},
                 perplexity = {"mean":codebook_perplexity,"std":codebook_perplexity_std},
                 topK_sim = {"mean":cosine_sim,"std":cosine_sim_std})

def generate_examples(tracks_list,model,k,with_coupling,track_duration,chunk_duration,
                      segmentation_strategy,crossfade_time,max_duration,
                      save_dir,smaller=False):
    #tracks list is like [[t11,t12,...],...,[tm1,tm2,..,tmn]]
    bar = tqdm(range(len(tracks_list)))
    for tracks in tracks_list: #moises_tracks = [[t11,t12,...,t1N],...,[tM1,tM2,..,tMN']]
        #folder of multiple instruments
        #pick a source and memory
        memory = np.random.choice(tracks)
        src = [t for t in tracks if t!=memory]
        
        if smaller: #find small chunk in track
            y,sr = load(src[np.random.randint(0,len(src))],sr=None)
            t0,t1 = find_non_empty(y,max_duration,sr,return_time=True)
            timestamps = [t0/sr,t1/sr] #in seconds
        else : timestamps=None
        
        #generate
        output = generate(memory,src,model,k,with_coupling,track_duration,chunk_duration,
                        track_segmentation='uniform',
                        chunk_segmentation=segmentation_strategy,
                        concat_fade_time=crossfade_time,
                        remove=True,
                        save_dir=save_dir,
                        tgt_sampling_rates={'solo':48000,'mix':16000},
                        max_output_duration=max_duration, mix_channels=1, timestamps=timestamps, #mix in mono for evaluation
                        device=device)
        bar.update(1)
        
    
def save_to_file(data : dict, file : Path):
    with open(file,'a') as f:
        for key, value in data.items():
            f.write(f'{key}:{value}\n')

def parse_args():
        
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['all','model','quality','apa','similarity','none'],nargs="*")
    parser.add_argument('--model_ckp',nargs='*')
    parser.add_argument('--data',choices=['canonne','moises']) #canonne/moises
    parser.add_argument("--split", choices = ['val','test'])
    parser.add_argument('--k',type=float, default=0.1)
    #if already generated audio samples
    parser.add_argument("--generate",action='store_true')
    parser.add_argument("--with_coupling",action='store_true')
    parser.add_argument("--max_duration",type=float,default=60)
    parser.add_argument("--smaller",action='store_true')
    parser.add_argument("--crossfade_time",type=float,default=0.2)
    parser.add_argument("--quality_tgt_folder",default=None, help="target folder to generated samples to evaluate audio quality of the model") 
    parser.add_argument("--apa_tgt_folder",default=None, help="target folder to generated to evaluate apa")
    parser.add_argument("--similarity_tgt_folder",default=None, help="target folder to evaluate music similarity")
    parser.add_argument("--similarity_gt_folder",default=None, help="ground truth folder to evaluate music similarity")
    args = parser.parse_args()
    
    del parser 
    
    return args

           
def main():
    
    args = parse_args()
    if args.model_ckp == None:
        name = args.data
        root = f"/data3/anasynth_nonbp/bujard/DICY2/runs/coupling/{name}"
        if name=="canonne":
            models = [f"{name}_0.25_32",f"{name}_0.25_128",f"{name}_0.25_512",
                    f"{name}_0.5_32",f"{name}_0.5_128",f"{name}_0.5_512_1",
                    f"{name}_1_32",f"{name}_1_128",f"{name}_1_512"]
        else :
            models = [f"{name}_0.25_32_1",f"{name}_0.25_128",f"{name}_0.25_512",
                    f"{name}_0.5_32",f"{name}_0.5_128",f"{name}_0.5_512",
                    f"{name}_1_32",f"{name}_1_128",f"{name}_1_512"]
        
        model_ckps = [os.path.join(root,model)+'.pt' for model in models]
    else :
        model_ckps=args.model_ckp #with nargs=* --> always as list
    #model_ckps=[args.model_ckp]
    
    task = args.task
    #model_ckp = args.model_ckp
    
    k=int(args.k)
    
    for model_ckp in model_ckps:
        prYellow(os.path.basename(model_ckp))        
        
        if args.data == 'canonne':
            duos=f"/data3/anasynth_nonbp/bujard/data/BasesDeDonnees/ClementCannone_Duos/separate_and_csv/separate tracks/{args.split}"
            duos = [os.path.join(duos,A) for A in os.listdir(duos)]           
            
            trios = f"/data3/anasynth_nonbp/bujard/data/BasesDeDonnees/ClementCannone_Trios/4analysis_Exports_Impros_Coupees_Niveau/{args.split}"
            trios = [os.path.join(trios,A) for A in os.listdir(trios)]
        
        elif args.data == 'moises':
            moisesdb_val = f"../data/moisesdb_v2/{args.split}"
            moises_tracks = extract_all_groups(moisesdb_val,instruments_to_ignore=['drums','percussion','other'])
        
        if 'all' in args.task:
            args.task = ['model','quality','apa','similarity']
        
        #load model only for generation or model eval
        if 'model' in args.task or args.generate:
            model,params = load_model_checkpoint(model_ckp)
            model.to(device)
            
            #extract segmentation params
            track_duration = params['tracks_size']
            chunk_duration = params['chunk_size']
            segmentation_strategy = params['segmentation']
            
            if args.k<1: #portion of vocab size
                k = int(args.k*params['vocab_size'])
        
        
        path=os.path.abspath(__file__)
        dir = os.path.dirname(path)
        
        k_fname = f"k_{int(args.k)}" if args.k>=1 else f"k_{int(args.k*100)}%"
        
        save_dir = os.path.join(dir,f'evaluation/{os.path.basename(model_ckp).split(".pt")[0]}/{args.split}/{args.data}/{k_fname}')
        os.makedirs(save_dir,exist_ok=True)
        eval_file = os.path.join(save_dir,"eval.txt")
        
        #depends on dataset and split
        dataset = 'moisesdb_v2' if args.data == 'moises' else 'BasesDeDonnees'
        
        #generate audio from corresponding data folder
        if args.generate:
            prGreen("Generating audio...")
            args.crossfade_time = min(args.crossfade_time,chunk_duration/2) #if fade_t too big for single chunks

            if args.data=='moises':
                generate_examples(moises_tracks,model,k,args.with_coupling,track_duration,chunk_duration,
                                segmentation_strategy,args.crossfade_time,args.max_duration,save_dir,smaller=args.smaller)
                
            elif args.data == 'canonne':
                #get all tracks in each duo
                dA1_tracks = sorted(glob.glob(duos[0]+'/*.wav'))
                dA2_tracks = sorted(glob.glob(duos[1]+'/*.wav'))
                
                duos_tracks = [[t1,t2] for t1,t2 in zip(dA1_tracks,dA2_tracks)]

                generate_examples(duos_tracks,model,k,args.with_coupling,track_duration,chunk_duration,
                                segmentation_strategy,args.crossfade_time,args.max_duration,save_dir,smaller=args.smaller)
                
                
                #get all tracks in each trio
                dA1_tracks = sorted(glob.glob(trios[0]+'/*.wav'))
                dA2_tracks = sorted(glob.glob(trios[1]+'/*.wav'))
                dA3_tracks = sorted(glob.glob(trios[2]+'/*.wav'))   
                
                trios_tracks = [[t1,t2,t3] for t1,t2,t3 in zip(dA1_tracks,dA2_tracks,dA3_tracks)]
                generate_examples(trios_tracks,model,k,args.with_coupling,track_duration,chunk_duration,
                                segmentation_strategy,args.crossfade_time,args.max_duration,save_dir,smaller=args.smaller)                  
                    
        if 'model' in args.task:
            prGreen("Evaluating model...")
            #evaliate code related metrics        
            #load dataset 
            if args.data == 'canonne':            
                eval_roots = [duos,trios]
            
            elif args.data == 'moises':
                eval_roots = moises_tracks
            
            eval_fetcher = build_coupling_ds(eval_roots,4,
                                            track_duration,chunk_duration,
                                            segmentation_strategy=segmentation_strategy,
                                            pre_segmentation='uniform',
                                            SAMPLING_RATE=16000,
                                            direction="stem",distributed=False)
            eval_fetcher.device = device

            #compute metrics
            output = evaluate_model(model,eval_fetcher,k)
            
            #save output to file
            save_to_file({'k':k},eval_file)
            save_to_file(output,eval_file)
        
        #if no folders are given use the ones form generation
        if 'quality' in args.task :
            prGreen("Evaluating audio quality...")
            #get the tgt folder if not given
            tgt_folder = args.quality_tgt_folder
            if tgt_folder == None:
                tgt_folder = os.path.join(save_dir,"response")
            
            if not os.path.exists(tgt_folder):
                raise RuntimeError("Wrong or no corresponding folder for audio quality measure. Please generate audio or use '--generate'")
            
                
            ref_folder = f"/data3/anasynth_nonbp/bujard/data/{dataset}/eval/audio_quality/{args.split}"
            
            #compute audio quality
            score = evaluate_audio_quality(ref_folder,tgt_folder)
            print("audio quality =",score)
            #save to file
            save_to_file({'audio_quality':score},eval_file)
            
        if 'apa' in args.task :
            prGreen("Evaluating APA...")
            tgt_folder = args.apa_tgt_folder
            if tgt_folder == None:
                #get mix folder 
                tgt_folder = os.path.join(save_dir,"mix")
            
            if not os.path.exists(tgt_folder):
                raise RuntimeError("Wrong or no corresponding folder for APA measure. Please generate audio with '--generate'")
            
            APA_root =  f"/data3/anasynth_nonbp/bujard/data/{dataset}/eval/APA/{args.split}"
            bg_folder = APA_root + '/background'
            fake_bg_folder = APA_root+'/misaligned'
            
            #compute APA
            score,fads = evaluate_APA(bg_folder,fake_bg_folder,tgt_folder)
            
            print("APA =", score,"\nFADs :",fads)
            #save to file
            save_to_file({'APA':score,"FADs":fads},eval_file)
        
        if 'similarity' in args.task :
            prGreen("Evaluating Music Similarity...")
            #we need to iterate over the response folder and the info.txt to get the corresponding gt
            tgt_folder = args.similarity_tgt_folder
            if tgt_folder == None :
                tgt_folder = os.path.join(save_dir,"response")
            
            if not os.path.exists(tgt_folder):
                raise RuntimeError("Wrong or no corresponding folder for similarity measure. Please generate audio with '--generate'")    
            
            targets = sorted(glob.glob(tgt_folder+'/*.wav'))
            
            gt_folder = args.similarity_gt_folder
            if gt_folder==None:
                gt_folder = os.path.join(save_dir,"memory") #we use the memory folder to compute similarity
            
            if not os.path.exists(gt_folder):
                raise RuntimeError("Wrong or no corresponding ground truth folder for similarity measure.")    
            
            gts = sorted(glob.glob(gt_folder+'/*.wav'))

            #compute similarity
            sims = []
            for tgt,gt in zip(targets,gts):
                sim = evaluate_similarity(tgt,gt)
                sims.append(sim)
            
            mean_sim = np.mean(sims)
            std_sim = np.std(sims)
            median_sim = np.median(sims)
            print(mean_sim,std_sim,median_sim)
            #save to file
            save_to_file({"music_similarity (mean, std, median)" : [round(mean_sim,2), round(std_sim,2), round(median_sim,2)]},eval_file)
            
            
    
            
    
    
        
    

if __name__=='__main__':
    main()