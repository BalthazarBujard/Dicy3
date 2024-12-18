
import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from architecture.Model import SimpleSeq2SeqModel, load_model_checkpoint,myDDP
from utils.utils import build_coupling_ds,prGreen,prRed
from utils.coupling_ds_generator import extract_all_groups
from trainer import Seq2SeqTrainer
from MusicDataset.MusicDataset_v2 import MIN_RESOLUTION
import math

#global var
SAMPLING_RATE = 16000 #sampling rate for wav2vec2 bb model. shouldn't be changed !

IGNORE = ["drums", "percussion", "other"]



def build_model(args):
    
    #BACKBONE
    pretrained_bb_checkpoint = "../w2v_music_checkpoint.pt"
    bb_type="w2v"
    dim=args.dim
    pre_post_chunking = args.pre_post_chunking
    freeze_backbone=args.freeze_backbone 

    vocab_size = args.vocab_size

    #VQ
    #dim=768  #quantizer output dimension. if different than backbone dim must be learnable codebook (becomes an nn.Embedding layer to be learned)
    learnable_codebook=args.learnable_cb#args.learnable_cb #if the codebooks should get closer to the unquantized inputs
    restart_codebook=args.restart_codebook #update dead codevectors
    if restart_codebook and not learnable_codebook: prRed("restart codebook without learnable codebook") 
    
    MAX_CHUNK_DURATION = args.chunk_duration#[sec] #equivalent to resolution for decision input
    MAX_TRACK_DURATION = args.track_duration#(30/0.5)*MAX_CHUNK_DURATION #[sec] comme ca on a tojours des sequences de 60 tokens dans decision #[sec]
    SEGMENTATION_STRATEGY= args.segmentation
    

    #POS ENCODING
    div_term = MAX_CHUNK_DURATION if SEGMENTATION_STRATEGY in ['uniform','sliding'] else MIN_RESOLUTION
    max_len = max(2000,int(math.ceil(MAX_TRACK_DURATION/div_term)+1 + 3)) #gives the max number of chunks per (sub-)track +1 cuz of padding of last track chunk +3 for special tokens

    encoder_head=args.encoder_head #COLLAPSE method
    condense_type=args.condense_type if encoder_head!='mean' else None

    use_special_tokens=True #always used otherwise its weird...

    task = args.task

    #DECISION
    transformer_layers = args.transformer_layers
    decoder_only=True #args.decoder_only 
    inner_dim=args.inner_dim
    heads=args.heads
    dropout = args.dropout

    seq2seq=SimpleSeq2SeqModel(pretrained_bb_checkpoint,
                                    bb_type,
                                    dim,
                                    vocab_size,
                                    max_len,
                                    encoder_head,
                                    chunking=pre_post_chunking,
                                    use_special_tokens=use_special_tokens,
                                    task=task,
                                    condense_type=condense_type,
                                    freeze_backbone=freeze_backbone,
                                    learnable_codebook=learnable_codebook,
                                    restart_codebook=restart_codebook,
                                    transformer_layers=transformer_layers,
                                    dropout=dropout,
                                    decoder_only=decoder_only,
                                    inner_dim=inner_dim,
                                    heads=heads,
                                    )
    return seq2seq

def build_ds(args):
    
    train_set = "train" if not args.train_subset else "train_subset"

    D_A1=f"/data3/anasynth_nonbp/bujard/data/BasesDeDonnees/ClementCannone_Duos/separate_and_csv/separate tracks/{train_set}/A1"
    D_A2=f"/data3/anasynth_nonbp/bujard/data/BasesDeDonnees/ClementCannone_Duos/separate_and_csv/separate tracks/{train_set}/A2"
    T_A1 = f"/data3/anasynth_nonbp/bujard/data/BasesDeDonnees/ClementCannone_Trios/4analysis_Exports_Impros_Coupees_Niveau/{train_set}/A1"
    T_A2 = f"/data3/anasynth_nonbp/bujard/data/BasesDeDonnees/ClementCannone_Trios/4analysis_Exports_Impros_Coupees_Niveau/{train_set}/A2"
    T_A3 = f"/data3/anasynth_nonbp/bujard/data/BasesDeDonnees/ClementCannone_Trios/4analysis_Exports_Impros_Coupees_Niveau/{train_set}/A3"

    moisesdb_train = f"../data/moisesdb_v2/{train_set}"
    moises_tracks = extract_all_groups(moisesdb_train,instruments_to_ignore=["drums", "percussions", "other"])
    
    
    
    if args.data=='all':
        train_roots=[[D_A1,D_A2],[T_A1,T_A2,T_A3]]+moises_tracks
    if args.data=='canonne':
        train_roots=[[D_A1,D_A2],[T_A1,T_A2,T_A3]]
    if args.data=='moises':
        train_roots=moises_tracks

    val_set = "val" if not args.train_subset else "val_subset"
    
    D_A1=f"/data3/anasynth_nonbp/bujard/data/BasesDeDonnees/ClementCannone_Duos/separate_and_csv/separate tracks/{val_set}/A1"
    D_A2=f"/data3/anasynth_nonbp/bujard/data/BasesDeDonnees/ClementCannone_Duos/separate_and_csv/separate tracks/{val_set}/A2"
    T_A1 = f"/data3/anasynth_nonbp/bujard/data/BasesDeDonnees/ClementCannone_Trios/4analysis_Exports_Impros_Coupees_Niveau/{val_set}/A1"
    T_A2 = f"/data3/anasynth_nonbp/bujard/data/BasesDeDonnees/ClementCannone_Trios/4analysis_Exports_Impros_Coupees_Niveau/{val_set}/A2"
    T_A3 = f"/data3/anasynth_nonbp/bujard/data/BasesDeDonnees/ClementCannone_Trios/4analysis_Exports_Impros_Coupees_Niveau/{val_set}/A3"

    moisesdb_val = f"../data/moisesdb_v2/{val_set}"
    moises_tracks = extract_all_groups(moisesdb_val,instruments_to_ignore=["drums", "percussions", "other"])


    if args.data=='all':
        val_roots=[[D_A1,D_A2],[T_A1,T_A2,T_A3]]+moises_tracks
    elif args.data=='canonne':
        val_roots=[[D_A1,D_A2],[T_A1,T_A2,T_A3]]
    elif args.data=='moises':
        val_roots=moises_tracks
    
    return train_roots,val_roots

# WHEN LAUNCING MULTIPLE DDP MANUALLY MODIFY mastr_port
def setup(rank, world_size,mastr_port=12358):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = f'{mastr_port}'

    # initialize the process group
    try :
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    except:
        setup(rank,world_size,mastr_port+1)

def cleanup():
    dist.destroy_process_group()


def main(rank, world_size, args):
    prGreen(f"Running DDP train on rank {rank}.")
    setup(rank, world_size)
    
    train_roots,val_roots=build_ds(args)
    
    batch_size=args.batch_size
    
    MAX_CHUNK_DURATION = args.chunk_duration#[sec] #equivalent to resolution for decision input
    MAX_TRACK_DURATION = args.track_duration#(30/0.5)*MAX_CHUNK_DURATION #[sec] comme ca on a tojours des sequences de 60 tokens dans decision #[sec]
    SEGMENTATION_STRATEGY= args.segmentation
    PRE_SEGMENTATION= args.pre_segmentation #how to segment the tracks in sub-tracks (sliding or uniform expected)
    DIRECTION= args.direction
    
    train_fetcher = build_coupling_ds(train_roots,batch_size,
                                        MAX_TRACK_DURATION,MAX_CHUNK_DURATION,
                                        segmentation_strategy=SEGMENTATION_STRATEGY,
                                        pre_segmentation=PRE_SEGMENTATION,
                                        SAMPLING_RATE=SAMPLING_RATE,
                                        direction=DIRECTION,
                                        mask_prob=args.mask_prob,
                                        mask_len=args.mask_len,
                                        distributed=True)
    train_fetcher.device = rank
    
    #QUESTION : IF MASKING, APPLY MASK TO VAL DATASET ?
    val_fetcher = build_coupling_ds(val_roots,batch_size,
                                        MAX_TRACK_DURATION,MAX_CHUNK_DURATION,
                                        segmentation_strategy=SEGMENTATION_STRATEGY,
                                        pre_segmentation=PRE_SEGMENTATION,
                                        SAMPLING_RATE=SAMPLING_RATE,
                                        direction=DIRECTION,distributed=True)
    val_fetcher.device = rank

    if args.resume_ckp!='':
        seq2seq, params, optim_state_dict = load_model_checkpoint(args.resume_ckp)
    
    else : seq2seq=build_model(args)
    
    
    
    lr = args.learning_rate
    lr_bb = lr if args.learning_rate_backbone == -1 else args.learning_rate_backbone
    weight_decay= args.weight_decay #ADD WEIGHT DECAY AND INCREASE LAYER NORM EPS IN DECISION IF INSTABILITY
    betas=(0.9, 0.999) #default betas for Adam
    grad_accum_steps=args.grad_accum #effective batch size = batch_size * grad_accum
    reg_alpha = args.reg_alpha
    codebook_loss_weight = args.codebook_loss_weight
    
    k = args.k
    if k<1:
        k = int(k*args.vocab_size)
    else : k = int(k)
    
    PAD_IDX = seq2seq.special_tokens_idx["pad"] if seq2seq.use_special_tokens else -100 #pad index ignored for loss
    criterion = torch.nn.functional.cross_entropy #torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    
    model = seq2seq.to(rank)
    ddp_model = myDDP(model, device_ids=[rank],
                      find_unused_parameters=
                      args.freeze_backbone==False or args.learnable_cb) #find unused if backbone train or learn codebook
    
    bb_params = ddp_model.module.encoder.encoder.backbone.parameters()
    rest_params = []
    for name,p in ddp_model.named_parameters():
        if 'backbone' not in name:
            rest_params.append(p) 
    
    optimizer_bb = torch.optim.Adam(bb_params,lr=lr_bb,betas=betas,weight_decay=weight_decay)
    optimizer_rest = torch.optim.Adam(rest_params,lr=lr_bb,betas=betas,weight_decay=weight_decay)
    optimizer = [optimizer_bb,optimizer_rest]
    
    
    if args.resume_ckp!='':
        if type(optim_state_dict) == list :
            for i,optim in enumerate(optim_state_dict):
                optimizer[i].load_state_dict(optim)
        
        else : optimizer.load_state_dict(optim_state_dict)
    
    run_id = args.run_id
    
    #weighed crossentropy
    weighed = args.weighed_crossentropy
    
    
    trainer = Seq2SeqTrainer(ddp_model,rank, criterion, optimizer, run_id,
                            segmentation=SEGMENTATION_STRATEGY,
                            k = k,
                            grad_accum_steps=grad_accum_steps,
                            codebook_loss_weight=codebook_loss_weight,
                            chunk_size=args.chunk_duration,
                            track_size=args.track_duration,
                            resume_epoch=args.resume_epoch,
                            weighed_crossentropy=weighed)
    
    epochs = args.epochs
    trainer.train(train_fetcher,val_fetcher,epochs,reg_alpha=reg_alpha)
    
    cleanup() #destroy process




if __name__=='__main__':
    import json
    from launch_ddp import train_parser
    
    args = train_parser().parse_args()
    
    world_size = torch.cuda.device_count()
    
    os.environ["CUDA_VISIBLE_DEVICES"]=args.device_ids
    
    run_id=args.run_id
    if run_id=='None' :
        run_id = f"{args.data}_{args.chunk_duration}s_{args.track_duration}s_A{args.vocab_size}_{args.pre_post_chunking}_D{args.dim}"
        #run_id+= "learn_bb" if not args.freeze_backbone else ""
    
    
    new_run_id=run_id
    i=1
    while os.path.exists(f"runs/coupling/{new_run_id}.pt") and args.resume_ckp=='': #find new name if not resume training
        new_run_id=run_id+f'_{i}'
        i+=1
    args.run_id=new_run_id
    
    #save config_file
    with open(f'runs/coupling/train_args_{args.run_id}.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    #launch distributed training
    mp.spawn(main,args=(world_size,args),nprocs=world_size)
