#script to compute clusters from latent vectors

#%%
from utils.utils import lock_gpu,prYellow,prGreen

prYellow(f"Locking GPU...")
DEVICE = lock_gpu()[0][0]

from architecture.Model import build_backbone
from wav2vec2.wav2vec2_utils import DataCollatorForWav2Vec2 # type: ignore
from MusicDataset.MusicDataset_v2 import MusicContainer,Fetcher,MusicDataCollator
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
#from torch_kmeans import KMeans
from torch.utils.data import DataLoader
from transformers import Wav2Vec2FeatureExtractor
from tqdm import tqdm
import numpy as np
import torch
from typing import List

#%%
def generate_codebook(codebook_sizes : List[int], dim : int, normalize : bool = False):
    
    #codebook_size=16

    assert dim in [256,768], "the dimension for the codebook should be either 768 (for hidden state output) or 256 (final projection layer output)"
    output_final_proj = dim==256

    #%%
    #model
    prYellow("Loading model from checkpoint...")
    checkpoint="../w2v_music_checkpoint.pt"
    model = build_backbone(checkpoint,type="w2v",mean=False,pooling=False,output_final_proj=output_final_proj)
    model.to(DEVICE)
    model.freeze()
    model.eval()

    #%%

    #load data
    folders = [
        "BasesDeDonnees/ClementCannone_Duos/separate_and_csv/separate tracks/train/A1",
        "BasesDeDonnees/ClementCannone_Duos/separate_and_csv/separate tracks/train/A2",
        "BasesDeDonnees/ClementCannone_Trios/4analysis_Exports_Impros_Coupees_Niveau/train/A1",
        "BasesDeDonnees/ClementCannone_Trios/4analysis_Exports_Impros_Coupees_Niveau/train/A2",
        "BasesDeDonnees/ClementCannone_Trios/4analysis_Exports_Impros_Coupees_Niveau/train/A3",
        "moisesdb_v2/train"
        ] 

    roots = [f"/data3/anasynth_nonbp/bujard/data/{root}" for root in folders]

    max_duration=5 #the longer the sequence the better
    sr=16000
    segmentation_strategy="sliding" #normally this doesnt affect the kmeans centers since we use the codes from the finest resolution (output of w2v)

    s="\n ".join(folders)
    prYellow(f"Creating dataset from folders :\n {s}")
    ds = MusicContainer(roots,max_duration,sr,segmentation_strategy,ignore_instrument=["other","drums","percussion"])
    # %%
    # dataloader
    batch_size=64
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    #TODO : CHANGE COLLATOR TO OUR MUSICDATACOLLATOR. -> IT DOESNT WORK WITH MUSICCONTAINER AS IS
    collate_fn=DataCollatorForWav2Vec2(model.backbone,feature_extractor,split="test")
    
    loader = DataLoader(ds,batch_size,shuffle=True,collate_fn=collate_fn)
    fetcher=Fetcher(loader)
    fetcher.device=DEVICE

    # %%
    # kmeans
    k_means_batch_size=int(batch_size*max_duration*sr/400) #corresponds to the number of samples per batch. 400 is approx the subsample coeficient (1 sample is 0.025s)

    scaler = StandardScaler()
    
    #iterate over codebook sizes
    for codebook_size in codebook_sizes:
        torch.cuda.empty_cache() #free cached ememory ?
        
        prYellow(f"instanciating kmeans with vocab size = {codebook_size} (aka numbr of centers) and dim {dim}")
        k_means=MiniBatchKMeans(codebook_size,batch_size=k_means_batch_size,random_state=42)

        #%%
        #fit kmeans to data
        epochs=1
        bar=tqdm(range(epochs*len(fetcher)))
        old_centers = np.zeros((codebook_size,model.dim))
        for epoch in range(epochs):
            prYellow(f"Epoch {epoch+1}/{epochs}...")
            for iter in range(len(fetcher)):
                inputs = next(fetcher)
                x = inputs.x #get batched data
                
                #pass through model
                z = model(x)
                
                #reshape as N,latent_dim and to numpy for sklearn compatibility
                z = z.reshape(-1,z.size(-1)).numpy(force=True) #(B,dim)
                
                #TODO : JE CROIS IL FAUT FAIRE UN TRANSPOSE POUR FAIRE LA NORMALISATION DECRITE
                if normalize:
                    z = scaler.fit_transform(z)
                
                #partial fit kmeans
                k_means.partial_fit(z)
                #z=z.reshape(1,-1,z.size(-1))
                #results=kmeans(z)
                
                #TODO : ADD CODNITION TO END EPOCH IF CENTERS DONT CHANGE ENOUGH
                
                #dist = np.linalg.norm(old_centers-k_means.cluster_centers_,axis=1)
                #print(dist)
                #print(k_means.cluster_centers_[:5])
                old_centers=k_means.cluster_centers_
                
                bar.update(1)        
            
            #save centers for later use as VQ at end of epoch
            centers=k_means.cluster_centers_
            prYellow(f"Saving kmeans centers...")
            np.save(f"clustering/kmeans_centers_{codebook_size}_{dim}{"_normalized" if normalize else ""}.npy",centers,allow_pickle=True)
            




# %%


if __name__=="__main__":   
    
    codebook_sizes = [2**n for n in range(4,11)]
    dim=256
    
    generate_codebook(codebook_sizes,dim)

    # for codebook_size in codebook_sizes:
    #     prGreen(f"Generating kmeans VQ with {codebook_size} centers of dim {dim}")
    #     generate_codebook(codebook_size, dim)

    

