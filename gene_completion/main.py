from Transformer_simple import Transformer
from model.model_2D import DiT_stDiff
from dataset import SpaREDData
from train import train_model
from datetime import datetime
from utils import *
import torch
import wandb
import os

parser = get_main_parser()
args = parser.parse_args()

# seed everything
seed = args.seed
os.environ['GLOBAL_SEED'] = str(seed)
seed_everything(seed)

def main():

  # Adjust entity to personal WandB account
  wandb.init(
    project='LGDiST', 
    #entity = '',
    config=vars(args), 
    name=exp_name
    )

  # Define and create path to save results
  save_path = os.path.join("results", args.pred_layer, args.dataset, exp_name)
  if args.train:
    os.makedirs(save_path, exist_ok=True)

  # Load trained autoencoder
  print("Using a Transformer-MLP autoencoder for gene preprocessing")
  autoencoder = Transformer(
    input_dim=args.ae_input_dim, 
    latent_dim=args.ae_latent_dim, 
    output_dim=args.ae_output_dim,
    embedding_dim=args.ae_embedding_dim,
    num_layers=args.ae_num_layers,
    num_heads=args.ae_num_heads
    )

  # Load checkpoints of autoencoder for neighborhood gene expression 
  checkpoints = torch.load(args.autoencoder_ckpts_path)
  autoencoder.load_state_dict(checkpoints['state_dict'])
  autoencoder = autoencoder.to(device)

  # Create dataset object
  spared_data = SpaREDData(args, autoencoder)  

  # Define the diffusion model
  model = DiT_stDiff(
      input_size=[args.ae_latent_dim, args.num_neighs+1],  
      hidden_size=args.dit_hidden_size, 
      depth=args.dit_depth,
      num_heads=args.num_heads,
      classes=6, 
      args=args,
      mlp_ratio=4.0).to(device)

  if args.train:
    best_model_path = train_model(
      model,
      data=spared_data,
      wandb_logger=wandb,
      args=args,
      model_autoencoder=autoencoder,
      save_path=save_path,
      device=device,
      exp_name=exp_name
      )
    
  else: 
    best_model_path = args.dit_ckpts_path
    
  # Load the best model after training
  model.load_state_dict(torch.load(best_model_path))

  # Test in all available splits
  if args.test:
    train_metrics, test_imputation_data, _, _, _ = inference_function(
      data=spared_data,
      model=model,
      diffusion_steps=args.sample_diffusion_steps,
      device=device,
      args=args,
      model_autoencoder=autoencoder,
      wandb_logger=wandb,
      process="train"
      )
    wandb.log({"test_MSE_Train": train_metrics["MSE"], "test_PCC_Train": train_metrics["PCC-Gene"]})

    val_metrics, test_imputation_data, _, _, _ = inference_function(
      data=spared_data,
      model=model,
      diffusion_steps=args.sample_diffusion_steps,
      device=device,
      args=args,
      model_autoencoder=autoencoder,
      wandb_logger=wandb,
      process="val"
      )
    wandb.log({"test_MSE_Val": val_metrics["MSE"], "test_PCC_Val": val_metrics["PCC-Gene"]})

    if spared_data.test_data_available:
      test_metrics, test_imputation_data, _, _, _ = inference_function(
        data=spared_data,
        model=model,
        diffusion_steps=args.sample_diffusion_steps,
        device=device,
        args=args,
        model_autoencoder=autoencoder,
        wandb_logger=wandb,
        process="test"
        )
      wandb.log({"test_MSE_Test": test_metrics["MSE"], "test_PCC_Test": test_metrics["PCC-Gene"]})

  if args.full_inference:
      full_metrics, full_imputation_data, eval_mask, _, _ = inference_function(
        data=spared_data,
        model=model,
        diffusion_steps=args.sample_diffusion_steps,
        device=device,
        args=args,
        model_autoencoder=autoencoder,
        wandb_logger=wandb,
        process="all"
        )
      wandb.log({"test_MSE_all": full_metrics["MSE"], "test_PCC_all": full_metrics["PCC-Gene"]})

if __name__ == "__main__":

  # Set unique name for current run
  exp_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
  print(args)
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
 
  main()