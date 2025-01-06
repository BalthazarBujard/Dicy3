from utils.utils import lock_gpu, find_non_empty
DEVICES,_=lock_gpu()
DEVICE = DEVICES[0]
from utils.dicy2_generator import generate
from utils.coupling_ds_generator import extract_group
from architecture.Model import load_model_checkpoint
import os
from IPython.display import Audio
import numpy as np
import argparse
from librosa import load


def generate_examples(model_ckp, with_coupling, remove, k, decoding_type, temperature, force_coupling, fade_time, num_examples, data, save_dir, batch_size, from_subset=False):
    model, params, _ = load_model_checkpoint(model_ckp)
    #model.freeze()
    model.eval()
    _=model.to(DEVICE) 

    MAX_CHUNK_DURATION=params['chunk_size']
    MAX_TRACK_DURATION=params['tracks_size']
    SEGMENTATION_STRATEGY = model.segmentation
    PRE_SEGMENTATION_STARTEGY="uniform"
    
    if k<1 :
        k=int(k*params['vocab_size'])
    else : k=int(k)
    
    val_folder = "val_subset" if from_subset else "val"
    
    if 'canonne' in data:
        #clement cannone
        canonne_t = f"../data/BasesDeDonnees/ClementCannone_Trios/4analysis_Exports_Impros_Coupees_Niveau/{val_folder}"
        canonne_d = f"../data/BasesDeDonnees/ClementCannone_Duos/separate_and_csv/separate tracks/{val_folder}"
        
        canonne = [canonne_t,canonne_d]
        if data=='canonne':
            canonne = canonne[np.random.randint(0,2)]
        elif data=='canonne_duos':
            canonne = canonne_d
        elif data=='canonne_trios':
            canonne=canonne_t
        else :
            raise ValueError(f"{data} not good")
        
        A1,A2,A3 = os.path.join(canonne,"A1"),os.path.join(canonne,"A2"),os.path.join(canonne,"A3")
        num_files = len(os.listdir(A1))
        idxs = np.random.choice(range(num_files),size=num_examples,replace=True)

        
        for track_idx in idxs:

            a1 = sorted(os.listdir(A1))[track_idx]
            a2 = sorted(os.listdir(A2))[track_idx]
            
            a1 = os.path.join(A1,a1)
            a2 = os.path.join(A2,a2)
            
            if data=="canonne_trios":
                a3 = sorted(os.listdir(A3))[track_idx]
                a3 = os.path.join(A3,a3)
                tracks = [a1,a2,a3]
            else : tracks = [a1,a2]

            #choose memory and source
            id = np.random.randint(0,len(tracks))
            memory = tracks[id] 
            src = [tracks[i] for i in range(len(tracks)) if i != id ]

            output = generate(memory,src,model,k,with_coupling,decoding_type, temperature, force_coupling,
                            MAX_TRACK_DURATION,MAX_CHUNK_DURATION,
                            track_segmentation=PRE_SEGMENTATION_STARTEGY,
                            chunk_segmentation=SEGMENTATION_STRATEGY,
                            batch_size=batch_size,
                            concat_fade_time=fade_time,
                            remove=remove,
                            save_dir=save_dir,
                            device=DEVICE)
        

    elif data == 'moises':
        root= f"../data/moisesdb_v2/{val_folder}"
        track_folders = [os.path.join(root,track) for track in os.listdir(root)]
        idxs = np.random.choice(range(len(track_folders)),size=num_examples,replace=False)
        for idx in idxs:
            track_folder = track_folders[idx]
            tracks = extract_group(track_folder,instruments_to_ignore=["other","drums","percussion"])
            #print("\n".join(tracks))


            #choose memory and source
            id= np.random.randint(0,len(tracks))
            memory = tracks[id] 
            src = [tracks[i] for i in range(len(tracks)) if i != id ]

            output = generate(memory,src,model,k,with_coupling, decoding_type, temperature, force_coupling,
                            MAX_TRACK_DURATION,MAX_CHUNK_DURATION,track_segmentation=PRE_SEGMENTATION_STARTEGY,
                            chunk_segmentation=SEGMENTATION_STRATEGY,
                            batch_size=batch_size,
                            concat_fade_time=fade_time,
                            remove=remove,
                            save_dir=save_dir,
                            device=DEVICE)
            
def generate_example(model_ckp,memory,src,with_coupling,remove,k, decoding_type, temperature, force_coupling, fade_time,save_dir,smaller, batch_size,max_duration=60.):
    model, params, _ = load_model_checkpoint(model_ckp)
    #model.freeze()
    model.eval()
    _=model.to(DEVICE) 

    MAX_CHUNK_DURATION=params['chunk_size']
    MAX_TRACK_DURATION=params['tracks_size']
    SEGMENTATION_STRATEGY = model.segmentation
    PRE_SEGMENTATION_STARTEGY="uniform"
    
    if k<1 :
        k=int(k*params['vocab_size'])
    else : k=int(k)
    
    if smaller: #find small chunk in track
        y,sr = load(src[np.random.randint(0,len(src))],sr=None)
        t0,t1 = find_non_empty(y,max_duration,sr,return_time=True)
        timestamps = [t0/sr,t1/sr] #in seconds
    else : timestamps=None
    
    output = generate(memory,src,model,k,with_coupling,decoding_type, temperature, force_coupling,
                      MAX_TRACK_DURATION,MAX_CHUNK_DURATION,track_segmentation=PRE_SEGMENTATION_STARTEGY,
                            chunk_segmentation=SEGMENTATION_STRATEGY,
                            batch_size=batch_size,
                            concat_fade_time=fade_time,
                            remove=remove, timestamps=timestamps,
                            save_dir=save_dir,
                            device=DEVICE)



if __name__=='__main__':
    
    #ckp = "runs/coupling/All_res0.5s_len30.0s_mix2stem_8.pt"
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_ckp",type=str)
    parser.add_argument("--batch_size",type=int,default=1)
    parser.add_argument("--ckp_file",type=str,help="Path to file containing the list of model checkpoints to generate with")
    """ parser.add_argument('-cd',"--chunk_duration",type=float)
    parser.add_argument('-td',"--track_duration",type=float) """
    parser.add_argument("--with_coupling",action='store_true')
    parser.add_argument("--remove",action='store_true')
    parser.add_argument("--decoding_type", type = str, choices=['greedy','beam'])
    parser.add_argument("--temperature", type = float, default=1.)
    parser.add_argument("--k",type=float,default=5)
    parser.add_argument("--force_coupling", action = 'store_true')
    parser.add_argument('--fade_time',type=float,default=0.04)
    parser.add_argument("--num_examples",type=int, default=1)
    parser.add_argument("--smaller",action='store_true')
    parser.add_argument("--data")
    parser.add_argument("--from_subset", action = 'store_true')
    parser.add_argument('--memory')
    parser.add_argument('--source',nargs='*')
    parser.add_argument("--save_dir")
    args = parser.parse_args()
    
    with_coupling = args.with_coupling
    save_dir = "output" if args.save_dir==None else args.save_dir
    save_dir = os.path.join(save_dir,os.path.basename(args.model_ckp).split(".pt")[0]) 
    os.makedirs(save_dir,exist_ok=True)
    
    # If a file with checkpoint paths is provided, read it and add to model_ckp
    if args.ckp_file:
        with open(args.ckp_file, 'r') as f:
            file_ckps = [line.strip() for line in f.readlines()]
    
    elif args.model_ckp:
        file_ckps=[args.model_ckp]
    
    else : raise ValueError("Either give a checkpoint file or a model checkpoint")
    
    for model_ckp in file_ckps:
        print("Generating with model :",os.path.basename(model_ckp))
        if args.data!=None:
            generate_examples(model_ckp,with_coupling,args.remove,
                            args.k, args.decoding_type, args.temperature, args.force_coupling,
                            fade_time=args.fade_time,
                            num_examples=args.num_examples,data=args.data,save_dir=save_dir, 
                            batch_size=args.batch_size,
                            from_subset=args.from_subset)
        
        elif args.memory!=None and args.source!=None:
            generate_example(model_ckp,args.memory,args.source,
                            args.with_coupling,args.remove,args.k,args.decoding_type, args.temperature, args.force_coupling,
                            args.fade_time,
                            save_dir,
                            args.smaller,
                            args.batch_size)
        else : raise ValueError("Either specify 'data' or give a source and memory path")