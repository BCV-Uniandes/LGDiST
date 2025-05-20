from Transformer_simple import Transformer
from model.model_2D import DiT_stDiff
from dataset import SpaREDData
from train import train_stDiff
from datetime import datetime
from utils import *
import torch
import glob
import os

parser = get_main_parser()
args = parser.parse_args()

# seed everything
seed = args.seed
os.environ['GLOBAL_SEED'] = str(seed)
seed_everything(seed)

# Set the inference step to which the current run/predictions correspond
if "c_d" in args.pred_layer:
    STEP = "StepA"
else:
    STEP = "StepB"

def main():
    # Load trained autoencoder
    print("Using a Transformer-MLP autoencoder for gene preprocessing")
    autoencoder = Transformer(
        input_dim=1024, 
        latent_dim=128, 
        output_dim=1024,
        embedding_dim=args.ae_embedding_dim,
        num_layers=args.ae_num_layers,
        num_heads=args.ae_num_heads
        )
    
    checkpoints = torch.load(args.autoencoder_ckpts_path)
    autoencoder.load_state_dict(checkpoints['state_dict'])
    autoencoder = autoencoder.to(device)

    # Get data with both 1024-gene set and the original set
    spared_data = SpaREDData(args, autoencoder)  

    # Define the diffusion model
    model = DiT_stDiff(
      input_size=[128, args.num_neighs+1],  # 128 because it is the gene-autoencoder's latent dimension
      hidden_size=args.dit_hidden_size, 
      depth=args.dit_depth,
      num_heads=args.num_heads,
      classes=6, 
      args=args,
      mlp_ratio=4.0).to(device)

    best_model_path = args.dit_ckpts_path
    
    # Load the best model after training
    model.load_state_dict(torch.load(best_model_path))

    # Test in test split if available or in val split otherwise
    test_metrics, test_imputation_data, _, _, _ = inference_function(
        data=spared_data,
        model=model,
        diffusion_steps=args.sample_diffusion_steps,
        device=device,
        args=args,
        model_autoencoder=autoencoder,
        process="test" if spared_data.test_data_available else "val",
    )
    print(f"Test metrics: {test_metrics}")

    if args.full_inference:
        full_metrics, full_imputation_data, eval_mask, full_imputation_data_cut, eval_mask_cut = inference_function(
            data=spared_data,
            model=model,
            diffusion_steps=args.sample_diffusion_steps,
            device=device,
            args=args,
            model_autoencoder=autoencoder,
            process="all",
        )
  
    breakpoint()
    # Check that adata has imputed matrices
    pred_log1p_exists = True if f'c_{STEP}_log1p' in spared_data.full_adata.layers else False
    pred_deltas_exists = True if f'c_{STEP}_deltas' in spared_data.full_adata.layers else False

    # Get mask from adata so that even the median-completed data in the context genes is updated with the new predictions for stageB-training
    real_mask = torch.tensor(spared_data.full_adata.layers['mask'])

    if not pred_log1p_exists: # Complete missing values in the 1024-genes adata

        original_exp = torch.tensor(spared_data.full_adata.layers['c_t_log1p'])

        # Imput predicted gene expression only in missing data
        imputed_exp = torch.where(real_mask, original_exp, full_imputation_data)
        
        # Add imputed data to adata
        spared_data.full_adata.layers[f'c_{STEP}_log1p'] = imputed_exp.cpu().numpy()
        
        if not pred_deltas_exists:
            spared_data.full_adata = get_deltas(spared_data.full_adata, f'c_{STEP}_log1p', f'c_{STEP}_deltas')
        else:
            print('Deltas already exist in adata')
        
        # Replace adata file in spared > processed_data
        preprocessed_dataset_path = spared_data.adata_path
        # Remove "general_mask" from the adata before rewriting it
        if "general_mask" in spared_data.full_adata.layers:    
            del spared_data.full_adata.layers["general_mask"]
        
        print('Replacing adata file ...')
        spared_data.full_adata.write(os.path.join(preprocessed_dataset_path))
        print(f'Dataset: {args.dataset}')
        print(f"LDiST completion into layers c_{STEP}_log1p and c_{STEP}_deltas in adata from {preprocessed_dataset_path} done.")
        print('-----------------------------------------------------------------------------------------------------------------')

        # Get predictions for original adatas (128 or 32 genes)
        original_genes = eval_mask.sum(dim=0)!=0
        imputed_exp = imputed_exp[:, original_genes]
        # Add imputed data to adata
        spared_data.original_full_adata.layers[f'c_{STEP}_log1p'] = imputed_exp.cpu().numpy()
        # Add deltas layer
        if not pred_deltas_exists:
            spared_data.original_full_adata = get_deltas(spared_data.original_full_adata, f'c_{STEP}_log1p', f'c_{STEP}_deltas')
        else:
            print('Deltas already exist in adata')

        # Remove unnecessary layers
        del spared_data.original_full_adata.layers["masked_expression_matrix"]
        del spared_data.original_full_adata.layers["random_mask"]

        # Replace adata file in spared > processed_data
        preprocessed_dataset_path = spared_data.original_adata_path
        print('Replacing adata file ...')
        spared_data.original_full_adata.write(os.path.join(preprocessed_dataset_path))
        print(f'Dataset: {args.dataset}')
        print(f"LDiST completion into layers c_{STEP}_log1p and c_{STEP}_deltas in adata from {preprocessed_dataset_path} done.")
        print('-----------------------------------------------------------------------------------------------------------------')

    else:
        print(f'LDiST completion already done for layers c_{STEP}_log1p and c_{STEP}_deltas.')
        print('-----------------------------------------------------------------------------------------------------------------')

if __name__ == "__main__":

    exp_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print(args)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
 
    main()