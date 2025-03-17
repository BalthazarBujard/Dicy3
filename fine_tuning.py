import os
import torch
from architecture.Model import load_model_checkpoint
from utils.utils import build_fine_tuning_ds
from trainer import Seq2SeqTrainer
from argparse import ArgumentParser
from pathlib import Path
import json

#global var
SAMPLING_RATE = 16000 #sampling rate for wav2vec2 bb model. shouldn't be changed !

IGNORE = ["drums", "percussion", "other"]

def argparser():
    parser = ArgumentParser()
    
    parser.add_argument("--ckp", type = Path, help="Path to trained checkpoint from which fine tune")
    parser.add_argument("--guide", type = str, help = "Path to the input (guide) audio file") #str type because Dataset class still needs update to enable Path type...
    parser.add_argument("--target", type = str, help = "Path to the output (target) audio file")
    parser.add_argument("--batch_size", type = int, default=None, help = "batch size : default None. If not specified batch size is computed as to have a batch = the whole track")
    parser.add_argument("--pre_segmentation", type=str, default = "sliding", choices = ["sliding", "uniform"])
    parser.add_argument('-lr','--learning_rate',type=float,default=1e-5)
    parser.add_argument('-lr_bb','--learning_rate_backbone',type=float,default=-1)
    parser.add_argument('--epochs',type=int,default=60)
    parser.add_argument("--scheduled_sampling", action = 'store_true')
    parser.add_argument("--scheduler_alpha", type=float, default = 4)
    
    return parser

def main(args, device):

    seq2seq, params, _ = load_model_checkpoint(args.ckp)
    try:
        run_id = params['run_id']
    except KeyError:
        run_id = os.path.basename(args.ckp).split(".pt")[0]
    
    
    
    guide_path = args.guide
    target_path = args.target
    
    batch_size=args.batch_size
    
    chunk_duration = params["chunk_size"]
    track_duration = params["tracks_size"]
    segmentation= params["segmentation"]
    pre_segmentation= args.pre_segmentation #how to segment the tracks in sub-tracks (sliding or uniform expected)

    
    train_fetcher = build_fine_tuning_ds(guide_path,target_path,track_duration,chunk_duration,SAMPLING_RATE,segmentation,pre_segmentation,batch_size,device)
    train_fetcher.device = device 
    
    val_fetcher = None #TODO : peut etre split la track et garder un peu pour la validation
    
    
    lr = args.learning_rate
    lr_bb = lr if args.learning_rate_backbone == -1 else args.learning_rate_backbone
    weight_decay= args.weight_decay #ADD WEIGHT DECAY AND INCREASE LAYER NORM EPS IN DECISION IF INSTABILITY
    betas=(0.9, 0.999) #default betas for Adam
    reg_alpha = 0
    
    k = args.k
    if k>=1:
        k=int(k)
    
    criterion = torch.nn.functional.cross_entropy #torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    
    model = seq2seq.to(device)
    
    bb_params = model.encoder.encoder.backbone.parameters()
    rest_params = []
    for name,p in model.named_parameters():
        if 'backbone' not in name:
            rest_params.append(p) 
    
    optimizer_bb = torch.optim.Adam(bb_params,lr=lr_bb,betas=betas,weight_decay=weight_decay)
    optimizer_rest = torch.optim.Adam(rest_params,lr=lr_bb,betas=betas,weight_decay=weight_decay)
    optimizer = [optimizer_bb,optimizer_rest]
    
    
    rank=0
    trainer = Seq2SeqTrainer(model,rank, criterion, optimizer, run_id,
                            segmentation=segmentation,
                            k = k,
                            chunk_size=chunk_duration,
                            track_size=track_duration,
                            scheduled_sampling = args.scheduled_sampling,
                            scheduler_alpha=args.scheduler_alpha)
    
    epochs = args.epochs
    trainer.train(train_fetcher,val_fetcher,epochs,reg_alpha=reg_alpha, evaluate=val_fetcher!=None)




if __name__=='__main__':
    from utils.utils import lock_gpu
    
    device = lock_gpu()[0][0]
    
    args = argparser().parse_args()
    
    run_id=args.run_id
    if run_id=='None' :
        # run_id = f"{args.data}_{args.chunk_duration}s_{args.track_duration}s_A{args.vocab_size}_{args.pre_post_chunking}_D{args.dim}"
        # run_id += f"_masking_{args.mask_prob}" if args.has_masking else ""
        # run_id+= "_learn_cb" if args.learnable_cb else ""
        # run_id+= "_restart_cb" if args.restart_codebook else ""
        run_id = f"{args.data}_{args.chunk_duration}s_A{args.vocab_size}"
        run_id += "_SchedSamp" if args.scheduled_sampling else ""
        run_id += "_SpecVQ" if args.special_vq else ""
        run_id += "_RelPE" if args.relative_pe else ""
        
    
    new_run_id=run_id
    i=1
    while os.path.exists(f"runs/coupling/{new_run_id}.pt") and args.resume_ckp=='': #find new name if not resume training
        new_run_id=run_id+f'_{i}'
        i+=1
    args.run_id=new_run_id
    
    #save config_file
    with open(f'runs/coupling/train_args_{args.run_id}.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    main(args, device)
