#%%
from utils.utils import lock_gpu
DEVICES,ids=lock_gpu(num_devices=1)
WORLD_SIZE=len(DEVICES)
DEVICE=DEVICES[0] #for one device
import torch
from architecture.Model import SimpleSeq2SeqModel
from utils.utils import *
from utils.coupling_ds_generator import extract_all_groups
from utils.coupling_ds_generator import generate_couples, generate_couples_from_root
import math
import os
from trainer import Seq2SeqTrainer
from MusicDataset.MusicDataset_v2 import MIN_RESOLUTION
import argparse

def main():
    parser = argparse.ArgumentParser(description="Coupling training script")

    parser.add_argument('-cd','--chunk_duration', type=float, default = 0.5)
    parser.add_argument('-td','--track_duration', type = float, default=30.)
    parser.add_argument('-seg',"--segmentation", type=str, default="uniform")
    parser.add_argument('-p_seg',"--pre_segmentation", type=str, default="sliding")
    parser.add_argument('-dir','--direction',type=str,choices=["stem","mix"],default="stem")
    parser.add_argument('--ignore',type=list,nargs='+',default=["drums", "percussions", "other"])
    parser.add_argument('-vocab','--vocab_size',type=int,choices=[16,32,64,128,256,512,1024],required=True)
    parser.add_argument('--learnable_cb',action='store_true')
    parser.add_argument('--codebook_loss_weight',type=float,default=1.)
    parser.add_argument('-head','--encoder_head',type=str,choices=['mean','attention'],required=True)
    parser.add_argument('-condense','--condense_type',type=str,choices=['mask','weighed'],default=None)
    parser.add_argument('-layers','--transformer_layers',type=int,default=8)
    parser.add_argument('-decoder','--decoder_only',action='store_true')
    parser.add_argument('--inner_dim',type=int,default=2048)
    parser.add_argument('--heads',type=int,default=12)
    parser.add_argument('--dropout',type=float,default=0.5)
    parser.add_argument('--task',type=str,choices=['completion','coupling'],default='coupling')
    parser.add_argument('-lr','--learning_rate',type=float,default=1e-4)
    parser.add_argument('-epochs',type=int,default=20)
    parser.add_argument('-batch_size',type=int,default=8)
    parser.add_argument('-decay','--weight_decay',type=float,default=5e-2)
    parser.add_argument('-reg_alpha',type=float,default=0.2)
    parser.add_argument('-grad_accum',type=int,default=1)
    parser.add_argument('-k',type=int,default=5)
    parser.add_argument("--run_id",required=True)

    args=parser.parse_args()

    #%%
    # global var
    SAMPLING_RATE = 16000 #sampling rate for wav2vec2 bb model. shouldn't be changed !
    MAX_CHUNK_DURATION = args.chunk_duration #[sec] #equivalent to resolution for decision input
    MAX_TRACK_DURATION = args.track_duration#(30/0.5)*MAX_CHUNK_DURATION #[sec] comme ca on a tojours des sequences de 60 tokens dans decision #[sec]
    SEGMENTATION_STRATEGY=args.segmentation
    PRE_SEGMENTATION=args.pre_segmentation #how to segment the tracks in sub-tracks (sliding or uniform expected)
    DIRECTION=args.direction
    IGNORE = args.ignore#["drums", "percussions", "other"]

    vocab_size = args.vocab_size
    lr = args.learning_rate

    BATCH_SIZE = args.batch_size
    

    dir = "mix2stem" if DIRECTION == "stem" else "stem2mix"
    run_id = args.run_id#f"All_res{MAX_CHUNK_DURATION}s_len{MAX_TRACK_DURATION}s_{dir}_12" 

    if os.path.exists(f"runs/coupling/{run_id}.pt"):
        raise RuntimeError("'run_id' already exists")


    #%%

    #dataset, loader and fetcher

    D_A1="/data3/anasynth_nonbp/bujard/data/BasesDeDonnees/ClementCannone_Duos/separate_and_csv/separate tracks/train/A1"
    D_A2="/data3/anasynth_nonbp/bujard/data/BasesDeDonnees/ClementCannone_Duos/separate_and_csv/separate tracks/train/A2"
    T_A1 = "/data3/anasynth_nonbp/bujard/data/BasesDeDonnees/ClementCannone_Trios/4analysis_Exports_Impros_Coupees_Niveau/train/A1"
    T_A2 = "/data3/anasynth_nonbp/bujard/data/BasesDeDonnees/ClementCannone_Trios/4analysis_Exports_Impros_Coupees_Niveau/train/A2"
    T_A3 = "/data3/anasynth_nonbp/bujard/data/BasesDeDonnees/ClementCannone_Trios/4analysis_Exports_Impros_Coupees_Niveau/train/A3"

    moisesdb_train = "../data/moisesdb_v2/train"
    moises_tracks = extract_all_groups(moisesdb_train,IGNORE)
    train_roots=[[D_A1,D_A2],[T_A1,T_A2,T_A3]]+moises_tracks

    D_A1="/data3/anasynth_nonbp/bujard/data/BasesDeDonnees/ClementCannone_Duos/separate_and_csv/separate tracks/val/A1"
    D_A2="/data3/anasynth_nonbp/bujard/data/BasesDeDonnees/ClementCannone_Duos/separate_and_csv/separate tracks/val/A2"
    T_A1 = "/data3/anasynth_nonbp/bujard/data/BasesDeDonnees/ClementCannone_Trios/4analysis_Exports_Impros_Coupees_Niveau/val/A1"
    T_A2 = "/data3/anasynth_nonbp/bujard/data/BasesDeDonnees/ClementCannone_Trios/4analysis_Exports_Impros_Coupees_Niveau/val/A2"
    T_A3 = "/data3/anasynth_nonbp/bujard/data/BasesDeDonnees/ClementCannone_Trios/4analysis_Exports_Impros_Coupees_Niveau/val/A3"

    moisesdb_val = "../data/moisesdb_v2/val"
    moises_tracks = extract_all_groups(moisesdb_val,IGNORE)

    val_roots=[[D_A1,D_A2],[T_A1,T_A2,T_A3]]+moises_tracks

    #track_folder = "../data/moisesdb_v0.1/737356b2-ce9c-448b-877b-e42b3ed94563"
    #roots = generate_couples(track_folder)

    train_fetcher = build_coupling_ds(train_roots,BATCH_SIZE,
                                        MAX_TRACK_DURATION,MAX_CHUNK_DURATION,
                                        segmentation_strategy=SEGMENTATION_STRATEGY,
                                        pre_segmentation=PRE_SEGMENTATION,
                                        SAMPLING_RATE=SAMPLING_RATE,
                                        direction=DIRECTION,distributed=WORLD_SIZE>1)
    train_fetcher.device = DEVICE

    val_fetcher = build_coupling_ds(val_roots,BATCH_SIZE,
                                        MAX_TRACK_DURATION,MAX_CHUNK_DURATION,
                                        segmentation_strategy=SEGMENTATION_STRATEGY,
                                        pre_segmentation=PRE_SEGMENTATION,
                                        SAMPLING_RATE=SAMPLING_RATE,
                                        direction=DIRECTION,distributed=WORLD_SIZE>1)
    val_fetcher.device = DEVICE


    #%%

    #model building

    #BACKBONE
    pretrained_bb_checkpoint = "../w2v_music_checkpoint.pt"
    bb_type="w2v"

    #VQ
    dim=768  #quantizer output dimension. if different than backbone dim must be learnable codebook (becomes an nn.Embedding layer to be learned)
    learnable_codebook=args.learnable_cb #if the codebooks should get closer to the unquantized inputs
    codebook_loss_weight = args.codebook_loss_weight
    if learnable_codebook : assert codebook_loss_weight>0

    #POS ENCODING
    div_term = MAX_CHUNK_DURATION if SEGMENTATION_STRATEGY in ['uniform','sliding'] else MIN_RESOLUTION
    max_len = max(2000,int(math.ceil(MAX_TRACK_DURATION/div_term)+1 + 3)) #gives the max number of chunks per (sub-)track +1 cuz of padding of last track chunk +3 for special tokens

    encoder_head=args.encoder_head #COLLAPSE method
    condense_type=args.condense_type

    use_special_tokens=True #always used otherwise its weird...

    # -------- old params for other vq classes ----------
    commit_weight=0.
    diversity_weight=0.1
    kmeans_init=True #codebook initialization with kmeans
    threshold_ema_dead_code = 5 #threshold for reinitialization. if cluster size of a code is less than ... restart codebook
    #----------------------------------------------------

    task = args.task

    #DECISION
    transformer_layers = args.transformer_layers
    decoder_only=args.decoder_only 
    inner_dim=args.inner_dim
    heads=args.heads
    dropout = args.dropout

    seq2seq=SimpleSeq2SeqModel(pretrained_bb_checkpoint,
                            bb_type,
                            dim,
                            vocab_size,
                            max_len,
                            encoder_head,
                            use_special_tokens=use_special_tokens,
                            task=task,
                            condense_type=condense_type,
                            freeze_backbone=True,
                            learnable_codebook=learnable_codebook,
                            transformer_layers=transformer_layers,
                            dropout=dropout,
                            decoder_only=decoder_only,
                            inner_dim=inner_dim,
                            heads=heads,
                            #kmeans_init=kmeans_init,
                            #threshold_ema_dead_code=threshold_ema_dead_code,
                            commit_weight=commit_weight,
                            #diversity_weight=diversity_weight
                            )
    _=seq2seq.to(DEVICE)

    # %% 
    # build trainer
    PAD_IDX = seq2seq.special_tokens_idx["pad"] if seq2seq.use_special_tokens else -100 #pad index ignored for loss
    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    weight_decay= args.weight_decay #ADD WEIGHT DECAY AND INCREASE LAYER NORM EPS IN DECISION IF INSTABILITY
    betas=(0.9, 0.999) #default betas for Adam
    grad_accum_steps=args.grad_accum #effective batch size = batch_size * grad_accum

    #----------old params for vq---------------
    init_temp = 1. #use temperature to favor codebook diversity
    min_temp=0.5
    with_decay=False #if temp annealing
    #------------------------------------------

    optimizer = [torch.optim.Adam(seq2seq.parameters(),lr=lr,betas=betas,weight_decay=weight_decay)]


    #----------top-K parameter---------------
    k = args.k

    trainer = Seq2SeqTrainer(seq2seq, 0, criterion, optimizer, run_id,
                            segmentation=SEGMENTATION_STRATEGY,
                            k = k,
                            grad_accum_steps=grad_accum_steps,
                            codebook_loss_weight=codebook_loss_weight,
                            init_sample_temperature=init_temp,
                            min_temperature=min_temp,
                            with_decay=with_decay)

    # %% 
    # launch training
    #train_fetcher=fetcher
    #val_fetcher=test_fetcher 
    eval=val_fetcher!=None
    epochs = args.epochs

    prGreen(f"Training with \n len {MAX_TRACK_DURATION}s, \n resolution {MAX_CHUNK_DURATION}s, \n direction {dir}, \n vocab {vocab_size}")

    reg_alpha=args.reg_alpha
    trainer.train(train_fetcher,val_fetcher,epochs,eval,reg_alpha=reg_alpha) 

if __name__=='__main__':
    main()
