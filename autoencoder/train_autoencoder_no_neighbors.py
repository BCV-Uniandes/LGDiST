import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
from Transformer_encoder_decoder import *
from Transformer_simple import Transformer
from utils import *
import wandb
from pytorch_lightning.loggers import WandbLogger
from datetime import datetime
import torch.nn.functional as F
import copy
from transformer_dataloader import CombinedDataset
from tqdm import tqdm

def main():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    current_device = torch.cuda.current_device()

    #argparse
    parser = get_main_parser()
    args = parser.parse_args()
    args_dict = vars(args) #Not uses, maybe later usage

    seed_everything(args.seed) 
    feedforward_dim = args.embedding_dim*2

    # Configurar el logger de wandb
    wandb.login()
    exp_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    wandb.init(project="autoencoder_project", entity="spared_v2", name=exp_name, dir="/home/dvegaa/LDiST/autoencoder")
    wandb_logger = WandbLogger(log_model="best")
    wandb.log({"args": vars(args),
               "feedforward_dim": feedforward_dim})

    adata_128 = ad.read_h5ad(f"/media/SSD0/pcardenasg2/c_dif_layers/datasets/original/{args.dataset}.h5ad")
    adata = ad.read_h5ad(f"/media/SSD0/pcardenasg2/c_dif_layers/datasets/1024/{args.dataset}_1024.h5ad")
    #adata = ad.read_h5ad(f"/media/SSD4/dlmanrique/Spatial_Transcriptomics/datasets/adatas_1024/{args.dataset}_1024.h5ad")

    # Sort adatas 
    adata, adata_128 = sort_adatas(adata=adata, adata_128=adata_128)

    # Get genes to evaluate
    genes_evaluate = []
    genes_128 = adata_128.var["gene_ids"].unique().tolist()
    genes_1024 = adata.var["gene_ids"].unique().tolist()

    for gene in genes_1024:
        if gene in genes_128:
            genes_evaluate.append(1)
        else:
            genes_evaluate.append(0)

    gene_weights = torch.tensor(genes_evaluate, dtype=torch.float32)
    model_autoencoder = None

    # Get neighbor data
    list_nn, max_min_enc = get_neigbors_dataset(adata, args.prediction_layer, args.num_hops, model_autoencoder, args)
    list_no_nn, _ = get_no_neighbors_dataset(adata, args.prediction_layer)

    data = copy.deepcopy(list_no_nn)

    splits = adata.obs["split"].unique().tolist()
    train_data = data["train"]
    val_data = data["val"]

    # Get mask for extreme completion
    ## TEST
    if "test" in splits:
        test_data = data["test"]
        test_tensor = torch.stack([torch.tensor(arr) for arr in test_data])

        mask_extreme = np.zeros((test_tensor.shape[0], 1024, 1))
        #mask 1024
        mask_extreme_completion_test = get_mask_extreme_completion(adata[adata.obs["split"]=="test"], mask_extreme, genes_evaluate)
        mask_extreme_completion_test = torch.tensor(mask_extreme_completion_test).permute(0,2,1)

        test_dataset = CombinedDataset(test_tensor, mask_extreme_completion_test, mask_extreme_completion_test)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    train_tensor = torch.stack([torch.tensor(arr) for arr in train_data])  
    mask_extreme = np.zeros((train_tensor.shape[0], 1024, 1))

    ## TRAIN
    mask_extreme_completion_train = get_mask_extreme_completion(adata[adata.obs["split"]=="train"], mask_extreme, genes_evaluate)
    mask_extreme_completion_train = torch.tensor(mask_extreme_completion_train).permute(0,2,1)

    val_tensor = torch.stack([torch.tensor(arr) for arr in val_data])
    mask_extreme = np.zeros((val_tensor.shape[0], 1024, 1))

    ## VALID
    mask_extreme_completion_val = get_mask_extreme_completion(adata[adata.obs["split"]=="val"], mask_extreme, genes_evaluate)
    mask_extreme_completion_val = torch.tensor(mask_extreme_completion_val).permute(0,2,1)

    train_dataset = CombinedDataset(train_tensor, mask_extreme_completion_train, mask_extreme_completion_train)
    val_dataset = CombinedDataset(val_tensor, mask_extreme_completion_val, mask_extreme_completion_val)

    # Define Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Initiate model
    ############################################################################################
    # Define model
    model = Transformer(input_dim=args.input_dim, 
                        latent_dim=args.latent_dim, 
                        output_dim=args.output_dim,
                        embedding_dim=args.embedding_dim,
                        feedforward_dim=feedforward_dim,
                        num_layers=args.num_layers,
                        num_heads=args.num_heads,
                        lr=args.lr,
                        gene_weights=gene_weights)

    # Initialize the Trainer
    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        logger = wandb_logger,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        enable_checkpointing=False
    )

    # Run the training loop
    trainer.fit(model, train_loader, val_loader)

    # Test the model
    if "test" not in splits:
        test_split = "val"
        test_loader = val_loader
        adata_test = adata[adata.obs["split"] == "val"]
        max_test = max_min_enc["val"][0].item()
        min_test = max_min_enc["val"][1].item()
        mask_extreme_completion_test = mask_extreme_completion_val
    else:
        test_split = "test"
        adata_test = adata[adata.obs["split"] == "test"]
        max_test = max_min_enc["test"][0].item()
        min_test = max_min_enc["test"][1].item()


    trainer.test(model, test_loader)

    # Test model
    auto_pred = []

    # Get model predictions
    with torch.no_grad():
        for data in tqdm(test_loader):
            # Move data to the specified device
            inputs = data[0].to(device)
            inputs = inputs.float()
            model = model.to(device)
            # Make predictions
            outputs = model(inputs, inputs.shape[0])
            # Move outputs to CPU and convert to NumPy if needed
            auto_pred.append(outputs.cpu().numpy())
            
    auto_pred = np.concatenate(auto_pred, axis=0)

    # Denormalized data
    auto_data = []
    for spot in range(0, auto_pred.shape[0]):
        spot_data = auto_pred[spot]
        if args.normalize_encoder == "1-1":
            spot_data = denormalize_from_minus_one_to_one(spot_data, max_test, min_test)
        auto_data.append(spot_data)

    # Get ground truth, predictions and boolean mask
    auto_data_array = np.stack(auto_data, axis=0)
    gt = torch.tensor(adata_test.layers[args.prediction_layer])
    pred = torch.tensor(auto_data_array)
    mask_boolean = mask_extreme_completion_test.bool()
    mask_boolean = mask_boolean[:,0,:]

    # Get MSE metric
    mse = F.mse_loss(gt[mask_boolean], pred[mask_boolean])
    wandb.log({"mse":mse})

    # Save model
    if not os.path.exists(os.path.join("/media/SSD4/dvegaa/autoencoder/no_neighbors/c_d_deltas", f"{args.dataset}", exp_name)):
        os.makedirs(os.path.join("/media/SSD4/dvegaa/autoencoder/no_neighbors/c_d_deltas", f"{args.dataset}", exp_name))

    trainer.save_checkpoint(os.path.join("/media/SSD4/dvegaa/autoencoder/no_neighbors/c_d_deltas", f"{args.dataset}", exp_name, "autoencoder_model.ckpt"))

if __name__ == "__main__":
    main()