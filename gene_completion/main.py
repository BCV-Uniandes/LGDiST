from Transformer_simple import Transformer
from model.model_2D import DiT_stDiff
from dataset import SpaREDData
from train import train_stDiff
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

  wandb.init(
    project='stLDM', 
    entity = 'spared_v2',
    config=vars(args), 
    name=exp_name
    )

  # Define and create path to save results
  save_path = os.path.join("results", args.dataset, exp_name)
  os.makedirs(save_path, exist_ok=True)

  # Load trained autoencoder
  if args.decode_as_matrix:
    print("Using a Transformer-Transformer autoencoder for gene preprocessing")
    autoencoder = AutoEncoder2D_Transformer(
        input_shape=[7, 1024]
        )
  else: 
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
      wandb_logger=wandb,
      args=args,
      model_autoencoder=autoencoder,
      save_path=save_path,
      device=device,
      exp_name=exp_name
      )
    
  else: 
    best_model_path = args.dit_ckpts_path
    
  # best_model_path = "/home/pcardenasg/ST_Diffusion/stDiff_Spared/Experiments/2025-01-28-20-01-12/villacampa_lung_organoid_12_1024_0.0001_noise.pt"
  # Load the best model after training
  model.load_state_dict(torch.load(best_model_path))

  if args.test:
    test_metrics, test_imputation_data = inference_function(
      data=spared_data,
      model=model,
      diffusion_steps=args.sample_diffusion_steps,
      device=device,
      args=args,
      model_autoencoder=autoencoder,
      wandb_logger=wandb,
      process="test"
      )
    
    wandb.log({"test_MSE": test_metrics["MSE"], "test_PCC": test_metrics["PCC-Gene"]})


if __name__ == "__main__":

  exp_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
  print(args)
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
 
  main()