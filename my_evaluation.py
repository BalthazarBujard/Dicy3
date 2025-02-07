#%%
from utils.utils import lock_gpu
device = lock_gpu()[0][0]
import os
import torch
from architecture.Seq2Seq import Seq2SeqBase,Seq2SeqCoupling
from architecture.Model import load_model_checkpoint
from MusicDataset.MusicDataset_v2 import Fetcher
from utils.metrics import compute_accuracy,compute_entropy,evaluate_APA,evaluate_audio_quality, evaluate_similarity
from utils.utils import predict_topK_P,prGreen,build_coupling_ds, extract_memory_path,prYellow, find_non_empty
from utils.coupling_ds_generator import extract_all_groups
from top_k_validity import compute_similarity
from tqdm import tqdm
import numpy as np 
from pathlib import Path
from typing import Union, List
from librosa import load
from munch import Munch
import glob
import argparse
from generate import generate_example
import datetime
from itertools import permutations


    

#this function evaluates the Decision module and the perception's quantized encoding
def evaluate_model(model : Seq2SeqBase, eval_fetcher : Fetcher, k: int):
    
    model.eval()
    acc=0
    diversity = 0 #predicted tokens entropy
    cosine_sim = 0
    
    accs = []
    divs = []
    sims = []
    cb_usage=[]
    perplexs=[]
    
    eos_idx = model.special_tokens_idx["eos"].to(device)
    codebook = model.vocab_embedding_table
    
    with torch.no_grad():
        for _ in tqdm(range(len(eval_fetcher))):
            inputs = next(eval_fetcher)
            
            if type(model)==Seq2SeqCoupling:
                src, tgt, src_pad_mask, tgt_pad_mask, _ = inputs.values()
                #compute output
                logits, tgt, tgt_idx, _ = model.forward(src, tgt, src_pad_mask, tgt_pad_mask)
                
            elif type(model)==Seq2SeqBase: #for autocompletion
                src, src_pad_mask, _, _ = inputs.values() 
                #compute output
                logits, tgt, tgt_idx, _ = model.forward(src, src_pad_mask)
            
            #encoded inputs
            encoded_src = model.encoder.forward(src, padding_mask = src_pad_mask[0])[1] #only take cb indexes to compute entropy
                    
            tgt_out = tgt_idx[:,1:] #ground truth (B,T)
            
            #topK search
            preds = predict_topK_P(k,logits,tgt_out) #(B*T,)
            
            #compute perplexity of predicted sequence
            probs = logits.softmax(-1)
            preds_probs = probs.reshape(-1,logits.size(-1)).gather(1,preds.unsqueeze(1)).squeeze(1) #retrieve logits of predicted tokens (B*T,)
            avg_nll = -sum(torch.log(preds_probs))/len(preds_probs)
            ppl = avg_nll.exp()
            perplexs.append(ppl.item())
            
            #accuracy with topK
            accs.append(compute_accuracy(preds,tgt_out.reshape(-1),pad_idx=model.special_tokens_idx["pad"]))
            
            #diversity of pred
            divs.append(compute_entropy(preds,min_length=model.vocab_size).item())
            
            #cb usage -> % of vocab size
            cb_usage.append(2**compute_entropy(encoded_src.reshape(-1),min_length=model.codebook_size).item()/model.codebook_size)
            
            #topK validity
            sims.append(compute_similarity(tgt, tgt_idx, logits,
                                             codebook, eos_idx, k).numpy(force=True))
            
            #del src, tgt, src_pad_mask, tgt_pad_mask, src_mask_indices, logits, tgt_idx, tgt_out, preds
    
    acc = np.mean(accs)
    acc_std = np.std(accs)
    
    diversity = np.mean(divs)
    diversity_std = np.std(divs)
    
    perplexity = np.mean(perplexs)
    perplexity_std = np.std(perplexs)
    
    codebook_usage = np.mean(cb_usage)
    codebook_usage_std = np.std(cb_usage)
    
    sims = np.array(sims) #convert to numpy
    cosine_sim = np.mean(sims[:,0])
    cosine_sim_std = np.mean(sims[:,1]) #mean std over top-K values (dont do another std otherwise its the std of the std)
    
    #torch.cuda.empty_cache()
    #del src, tgt, src_pad_mask, tgt_pad_mask, src_mask_indices, logits, tgt_idx, tgt_out, preds
    
    return Munch(accuracy={"mean":acc,"std":acc_std},
                 diversity={"mean":diversity,"std":diversity_std},
                 perpleity={"mean":perplexity,"std":perplexity_std},
                 codebook_usage = {"mean":codebook_usage,"std":codebook_usage_std},
                 topK_sim = {"mean":cosine_sim,"std":cosine_sim_std})

def generate_eval_examples(tracks_list : List[List], 
                           model : Seq2SeqCoupling, 
                           k : int, with_coupling : bool, decoding_type : str, temperature : float, force_coupling : bool, 
                           track_duration : float, chunk_duration : float, 
                           segmentation : str, pre_segmentation : str, 
                           crossfade_time : float, max_duration : float, 
                           save_dir : Path, 
                           smaller : bool = False, random_subset : bool = True, remove : bool = False, batch_size : int = 8):
    
    #tracks list is like [[t11,t12,...],...,[tm1,tm2,..,tmn]]
    bar = tqdm(range(len(tracks_list)))
    
    for tracks in tracks_list: 
        
        #TODO : FOR BETTER EVALUATION PICK EVERY DUO OF MEMORY/GUIDE IN TRACK LIST
        
        
        
        #pick a source and memory
        # memory = np.random.choice(tracks)
        # srcs = [t for t in tracks if t!=memory]
        # print(srcs)
        # if random_subset:
        #     src = np.random.choice(srcs, size = np.random.randint(1,len(srcs)),replace=False).tolist() if len(srcs)>1 else srcs #pick random subset of mix 
        # else : src = srcs
        # print(src)
        
        duos = list(permutations(tracks,2))
        
        for memory, src in duos:
        
            if smaller: #find small chunk in track
                y,sr = load(src[np.random.randint(0,len(src))],sr=None)
                t0,t1 = find_non_empty(y,max_duration,sr,return_time=True)
                timestamps = [[t0/sr,t1/sr],[t0/sr,t1/sr]] #in seconds
            else : timestamps=[None,None]
            
            #generate
            generate_example(
                model,
                memory, src,
                track_duration, chunk_duration, segmentation, pre_segmentation,
                with_coupling, remove, k, decoding_type, temperature, force_coupling,
                crossfade_time, save_dir, smaller, batch_size, True,
                max_duration, device=device, tgt_sampling_rates={'solo':48000,'mix':48000}, mix_channels=1
            )
        
        
        bar.update(1)
        
    
def save_to_file(data : dict, file : Path):
    with open(file,'a') as f:
        for key, value in data.items():
            f.write(f'{key}:{value}\n')

def parse_args():
        
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['all','model','quality','apa','similarity','none'],nargs="*")
    parser.add_argument('--model_ckp',nargs='*')
    parser.add_argument("--model_ckps_folder",type=str,default=None)
    parser.add_argument('--data',choices=['canonne','moises']) #canonne/moises
    parser.add_argument("--split", choices = ['val','val_subset','test'])
    parser.add_argument('--k',type=float, default=5)
    #if already generated audio samples
    parser.add_argument("--generate",action='store_true')
    parser.add_argument("--with_coupling",action='store_true')
    parser.add_argument("--sliding",action='store_true')
    parser.add_argument("--decoding_type",type=str,choices=['greedy','beam'])
    parser.add_argument("--force_coupling",action='store_true')
    parser.add_argument("--temperature",type=float, default=1.)
    parser.add_argument("--max_duration",type=float,default=60)
    parser.add_argument("--smaller",action='store_true')
    parser.add_argument("--crossfade_time",type=float,default=0.05)
    parser.add_argument("--apa_emb",type=str,choices=["CLAP","L-CLAP"], default = "CLAP")
    parser.add_argument("--fad_inf",action="store_true")
    #here you can specify "memory" or "original" for baseline evaluation
    parser.add_argument("--quality_tgt_folder",default=None, help="target folder ofr generated samples to evaluate audio quality of the model") 
    parser.add_argument("--apa_tgt_folder",default=None, help="target folder to generated to evaluate apa")
    parser.add_argument("--similarity_tgt_folder",default=None, help="target folder to evaluate music similarity")
    parser.add_argument("--similarity_gt_folder",default=None, help="ground truth folder to evaluate music similarity")
    args = parser.parse_args()
    
    del parser 
    
    return args

           
def main():
    
    args = parse_args()
    if args.model_ckp == None:
        assert args.model_ckps_folder != None, "If no model ckp given, specify folder containing all the models to evaluate."
        
        #find all ckp (*.pt) in the folder recursively
        model_ckps = sorted(glob.glob(f"{args.model_ckps_folder}/**/*.pt",recursive=True))
        
    else :
        model_ckps=args.model_ckp #with nargs=* --> always as list


    if args.k>=1: 
        k = int(args.k)
    else : k = args.k #total probability
    
    pre_segmentation = "uniform" if not args.sliding else "sliding"
    
    for model_ckp in model_ckps:
        prYellow(os.path.basename(model_ckp))        
        
        #extract data
        
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
            model,params,_ = load_model_checkpoint(model_ckp)
            model.has_masking=False #during evaluation model doesnt mask time indices
            model.to(device)
            
            #extract segmentation params
            track_duration = params['tracks_size']
            chunk_duration = params['chunk_size']
            segmentation_strategy = params['segmentation']
        
        
        path=os.path.abspath(__file__)
        dir = os.path.dirname(path)
        
        k_fname = f"k_{int(args.k)}" if args.k>=1 else f"p_{int(args.k*100)}%"
        
        save_dir = os.path.join(dir,f'evaluation/{os.path.basename(model_ckp).split(".pt")[0]}/{args.split}/{args.data}/{k_fname}')
        
        # new_save_dir =  save_dir
        # idx=1
        # while os.path.exists(new_save_dir):
        #     new_save_dir=save_dir + f" {idx}"
        #     idx+=1
        # save_dir = new_save_dir
        
        os.makedirs(save_dir,exist_ok=True)
        eval_file = os.path.join(save_dir,"eval.txt")
        
        #depends on dataset and split
        dataset = 'moisesdb_v2' if args.data == 'moises' else 'BasesDeDonnees'
        
        #save arguments/metadata in file
        save_to_file({"":"-"*50,'date':datetime.datetime.now()},eval_file)
        
        #generate audio from corresponding data folder
        if args.generate:
            generation_metadata={'task':'generate','k':k,"decoding type":args.decoding_type,"force_coupling":args.force_coupling,"temperature":args.temperature, "sliding":args.sliding}
            save_to_file(generation_metadata,eval_file)
            
            prGreen("Generating audio...")
            args.crossfade_time = min(args.crossfade_time,chunk_duration/2) #if fade_t too big for single chunks

            if args.data=='moises':
                generate_eval_examples(moises_tracks,model,k,args.with_coupling,
                                  args.decoding_type,args.temperature,args.force_coupling,
                                  track_duration,chunk_duration,
                                segmentation_strategy,pre_segmentation,args.crossfade_time,args.max_duration,save_dir,smaller=args.smaller)
                
            elif args.data == 'canonne':
                #get all tracks in each duo
                dA1_tracks = sorted(glob.glob(duos[0]+'/*.wav'))
                dA2_tracks = sorted(glob.glob(duos[1]+'/*.wav'))
                
                duos_tracks = [[t1,t2] for t1,t2 in zip(dA1_tracks,dA2_tracks)]

                generate_eval_examples(
                                        duos_tracks,model,k,args.with_coupling,
                                        args.decoding_type,args.temperature,args.force_coupling,
                                        track_duration,chunk_duration,segmentation_strategy,pre_segmentation,
                                        args.crossfade_time,args.max_duration,save_dir,smaller=args.smaller,
                                        random_subset=False
                                        )
                
                
                #get all tracks in each trio
                dA1_tracks = sorted(glob.glob(trios[0]+'/*.wav'))
                dA2_tracks = sorted(glob.glob(trios[1]+'/*.wav'))
                dA3_tracks = sorted(glob.glob(trios[2]+'/*.wav'))   
                
                trios_tracks = [[t1,t2,t3] for t1,t2,t3 in zip(dA1_tracks,dA2_tracks,dA3_tracks)]
                
                generate_eval_examples(
                                        trios_tracks,model,k,args.with_coupling,
                                        args.decoding_type,args.temperature,args.force_coupling,
                                        track_duration,chunk_duration,segmentation_strategy,pre_segmentation,
                                        args.crossfade_time,args.max_duration,save_dir,smaller=args.smaller,
                                        random_subset=False
                                        )
                    
        if 'model' in args.task:
            prGreen("Evaluating model...")
            
            model_eval_metadata = {'task':'model',"k":k, "decoding":"greedy GT selection"}
            save_to_file(model_eval_metadata,eval_file)
            
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
            #save_to_file({'k':k},eval_file)
            save_to_file(output,eval_file)
        
        #if no folders are given use the ones form generation
        if 'quality' in args.task :
            prGreen("Evaluating audio quality...")
            
            #get the tgt folder if not given
            tgt_folder = args.quality_tgt_folder
            if tgt_folder == None:
                tgt_folder = os.path.join(save_dir,"response")
            
            #for badeline
            if tgt_folder == "memory":
                tgt_folder=os.path.join(save_dir,"memory")
            
            if not os.path.exists(tgt_folder):
                raise RuntimeError("Wrong or no corresponding folder for audio quality measure. Please generate audio or use '--generate'")
            
            
            quality_metadata = {"task":"audio quality", "fad_inf":args.fad_inf, "tgt folder" : tgt_folder}
            save_to_file(quality_metadata,eval_file)
            
            
                
            ref_folder = f"/data3/anasynth_nonbp/bujard/data/{dataset}/eval/audio_quality/{args.split}"
            
            #compute audio quality
            score = evaluate_audio_quality(ref_folder,tgt_folder,args.fad_inf,device=device)
            print("audio quality =",score)
            #save to file
            save_to_file({'audio_quality':score},eval_file)
            
        if 'apa' in args.task :
            prGreen("Evaluating APA...")
            
            tgt_folder = args.apa_tgt_folder
            if tgt_folder == None:
                #get mix folder 
                tgt_folder = os.path.join(save_dir,"mix")
            
            #for baseline 
            elif tgt_folder == "original":
                tgt_folder == os.path.join(save_dir,"original")
            
            if not os.path.exists(tgt_folder):
                raise RuntimeError("Wrong or no corresponding folder for APA measure. Please generate audio with '--generate'")
            
            
            apa_metadata = {"task":"APA","embedding":args.apa_emb,"fad_inf":args.fad_inf,"tgt folder":tgt_folder}
            save_to_file(apa_metadata,eval_file)
            
            
            APA_root =  f"/data3/anasynth_nonbp/bujard/data/{dataset}/eval/APA/{args.split}"
            bg_folder = APA_root + '/background'
            fake_bg_folder = APA_root+'/misaligned'
            
            #compute APA
            score,fads = evaluate_APA(bg_folder,fake_bg_folder,tgt_folder,args.apa_emb,args.fad_inf,device=device)
            
            print("APA =", score,"\nFADs :",fads)
            #save to file
            save_to_file({'APA':score,"FADs":fads},eval_file)
        
        if 'similarity' in args.task :
            prGreen("Evaluating Music Similarity...")
            
            tgt_folder = args.similarity_tgt_folder
            
            #we need to iterate over the response folder
            if tgt_folder == None :
                tgt_folder = os.path.join(save_dir,"response")
            
            #for baseline
            if tgt_folder == "memory":
                tgt_folder=os.path.join(save_dir,"memory")
            
            if not os.path.exists(tgt_folder):
                raise RuntimeError("Wrong or no corresponding folder for similarity measure. Please generate audio with '--generate'")    
            
            
            sim_metadata={"task":"music similarity","tgt folder":tgt_folder}
            save_to_file(sim_metadata,eval_file)
            
            
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
                target, tgt_sr= load(tgt,sr=None, mono=True)
                gt, gt_sr = load(gt,sr=None,mono=True)
                
                sim = evaluate_similarity(target, tgt_sr,gt, gt_sr)
                sims.append(sim)
            
            mean_sim = np.mean(sims)
            std_sim = np.std(sims)
            median_sim = np.median(sims)
            #percentile90 = np.percentile(np.abs(sims),90)
            print(mean_sim,std_sim,median_sim)#,percentile90)
            print(sims)
            #save to file
            save_to_file({"music_similarity (mean, std, median, 90th percentile)" : [round(mean_sim,2), round(std_sim,2), round(median_sim,2)]},eval_file)
            
            
    
            
    
    
        
    

if __name__=='__main__':
    main()