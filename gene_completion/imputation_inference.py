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

def main():

  # Define and create path to save results
  save_path = os.path.join("results", args.dataset, exp_name)
  os.makedirs(save_path, exist_ok=True)

  # Load trained autoencoder
  print("Using a Transformer-MLP autoencoder for gene preprocessing")
  autoencoder = Transformer(
    input_dim=1024, 
    latent_dim=128, 
    output_dim=1024,
    embedding_dim=256,
    num_layers=2,
    num_heads=2,
    lr=args.lr
    )
    
  checkpoints = torch.load(args.autoencoder_ckpts_path)
  autoencoder.load_state_dict(checkpoints['state_dict'])
  autoencoder = autoencoder.to(device)

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

  if args.train:
    best_model_path = train_stDiff(
      model,
      data=spared_data,
      args=args,
      model_autoencoder=autoencoder,
      save_path=save_path,
      device=device,
      exp_name=exp_name,
      )
    
  else: 
    best_model_path = args.dit_ckpts_path
    
  # Load the best model after training
  model.load_state_dict(torch.load(best_model_path))

  # Test in all available splits
  if args.test:
    train_metrics, test_imputation_data, _ = inference_function(
      data=spared_data,
      model=model,
      diffusion_steps=args.sample_diffusion_steps,
      device=device,
      args=args,
      model_autoencoder=autoencoder,
      process="train",
      )

    val_metrics, test_imputation_data, _ = inference_function(
      data=spared_data,
      model=model,
      diffusion_steps=args.sample_diffusion_steps,
      device=device,
      args=args,
      model_autoencoder=autoencoder,
      process="val",
      )

    if spared_data.test_data_available:
      test_metrics, test_imputation_data, _ = inference_function(
        data=spared_data,
        model=model,
        diffusion_steps=args.sample_diffusion_steps,
        device=device,
        args=args,
        model_autoencoder=autoencoder,
        process="test",
        )

  if args.full_inference:
      full_metrics, full_imputation_data, eval_mask = inference_function(
        data=spared_data,
        model=model,
        diffusion_steps=args.sample_diffusion_steps,
        device=device,
        args=args,
        model_autoencoder=autoencoder,
        process="all",
        )
  
  # Check that adata has imputed matrices
  c_dif_log1p_exists = True if 'c_dif_log1p' in spared_data.original_full_adata.layers else False
  c_dif_deltas_exists = True if 'c_dif_deltas' in spared_data.original_full_adata.layers else False

  if not c_dif_log1p_exists:
    
    original_genes = eval_mask.sum(dim=0)!=0
    original_exp = torch.tensor(spared_data.original_full_adata.layers['c_t_log1p'])
    predicted_exp = full_imputation_data[:,original_genes]
    original_genes_mask = eval_mask[:,original_genes]
    
    assert torch.allclose(original_genes_mask, torch.tensor(spared_data.original_full_adata.layers['mask']))
    
    # Imput predicted gene expression only in missing data
    imputed_exp = torch.where(eval_mask[:,original_genes], original_exp, predicted_exp)
    
    # Add imputed data to adata
    spared_data.original_full_adata.layers['c_dif_log1p'] = imputed_exp.cpu().numpy()
    
    if not c_dif_deltas_exists:
      spared_data.original_full_adata = get_deltas(spared_data.original_full_adata, 'c_dif_log1p', 'c_dif_deltas')
    else:
      print('Deltas already exist in adata')
    
    # Replace adata file in spared > processed_data
    author = args.dataset.split('_')[0]
    if '10x' in author:
        author = 'Visium'

    elif 'fan' in author:
        author = 'fan_mouse_brain'

    preprocessed_dataset_path = os.path.join('/home', 'daruizl', 'SpaRED', 'spared', 'spared', 'processed_data', f'{author}_data', args.dataset, '**', 'adata.h5ad')

    if 'parigi' in author:
        preprocessed_dataset_path = os.path.join('/home', 'daruizl', 'SpaRED', 'spared', 'spared', 'processed_data', f'{author}_data', '**', 'adata.h5ad')

    preprocessed_dataset_path = glob.glob(preprocessed_dataset_path)[0]
    print('Replacing adata file ...')
    spared_data.original_full_adata.write(os.path.join(preprocessed_dataset_path))
    print(f'Dataset: {args.dataset}')
    print("LDiST Imputation for layers c_t_log1p and c_t_deltas into c_dif_log1p and c_dif_deltas done.")
    print('-----------------------------------------------------------------------------------------------------------------')

  else:
    print('LDiST Imputation already done')
    print('-----------------------------------------------------------------------------------------------------------------')

if __name__ == "__main__":

  exp_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
  print(args)
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
 
  main()