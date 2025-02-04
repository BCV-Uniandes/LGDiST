from model.scheduler import NoiseScheduler
from ray.air import session
import torch.nn as nn
from tqdm import tqdm
from utils import *
import numpy as np
import torch
import os

parser = get_main_parser()
args = parser.parse_args()

#Seed
seed = args.seed
seed_everything(seed)

def train_stDiff(model,
                data,
                args,
                model_autoencoder,
                save_path,
                wandb_logger,
                device=torch.device('cuda'),
                is_tqdm: bool = True,
                is_tune: bool = False,
                exp_name = ""):
    
    """
    Args:
        model (_type_): DiT model
        args (argparse): parser with the values necessary for custom training and test.
        model_autoencoder (_type_): frozen autoencoder for results decoding.
        save_path (str): path where checkpoints will be saved.
        wandb_logger (_type_): wandb to track results.
        device (_type_, optional): Defaults to torch.device('cuda:1').
        is_tqdm (bool, optional): tqdm. Defaults to True.
        is_tune (bool, optional):  ray tune. Defaults to False.

    Raises:
        NotImplementedError: _description_
    """
    #breakpoint()
    noise_scheduler = NoiseScheduler(
        num_timesteps=args.train_diffusion_steps,
        beta_schedule='cosine'
    )

    #Define Loss function
    criterion = nn.MSELoss()
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    
    if is_tqdm:
        t_epoch = tqdm(range(args.num_epochs), ncols=100)
    else:
        t_epoch = range(args.num_epochs)

    model.train()
    best_mse = np.inf
    best_pcc = -np.inf
    best_model_path = ""
    best_epoch = -1

    # TODO: try running based on steps intead of epochs
    #breakpoint()
    for epoch in t_epoch:
        epoch_loss = 0.
        for i, batch in enumerate(data.train_dataloader()): 
            # The mask is a binary array, where 1's are the hidden data (0 in the values we want to predict) <- SET UP ACTUAL
            x, x_cond, mask = batch["encoded_exp_matrix"].float().permute(0,2,1).to(device), batch["condition_matrix"].float().permute(0,2,1).to(device), batch["condition_mask"].permute(0,2,1).to(device)
            
            # Declare random noise
            noise = torch.randn(x.shape).to(device)         # torch.Size([BS, 128, 7])
            # Set timesteps
            timesteps = torch.randint(1, args.train_diffusion_steps, (x.shape[0],)).long()
            x_t = noise_scheduler.add_noise(x,
                                            noise,
                                            timesteps=timesteps.cpu())
            
            mask = torch.tensor(mask).to(device)
            
            # TODO: check the suitability of this flag for the condition.
            if args.adjust_loss: 
                # keep x_t only for values of interest
                x_noisy = x_t * (1 - mask) + x * mask # keep x_t on "missing"/"masked"/"hidden" data and keep xin the remaining data
            else:
                x_noisy = x_t
            
            # torch.Size([128(BS), 128, 7])
            cond = [x_cond, mask]
            
            pred = model(x_noisy, t=timesteps.to(device), y=cond) # torch.Size([128, 128, 7])
            mask_boolean = (1-mask).bool()                        # 1 = False y 0 = True 
            mask_boolean = mask_boolean[:,:,0] 
            pred = pred[:,:,0] 
                
            noise = noise[:,:,0]
            if args.adjust_loss:
                # Compute loss only on values of interest (main spot embedding)
                loss = criterion(noise[mask_boolean], pred[mask_boolean])
            else:
                # Compute loss on complete data
                loss = criterion(noise, pred)
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            
        # Update the learning rate
        if args.scheduler:
            scheduler.step()  
        
        epoch_loss = epoch_loss / (i + 1)  # type: ignore ¿?
        wandb_logger.log({"Loss": epoch_loss})

        if is_tqdm:
            # Get current learning rate
            current_lr = scheduler.get_last_lr()[0]
            t_epoch.set_postfix_str(f'loss:{epoch_loss:.5f} lr:{current_lr:.6f}')  # type: ignore
        
        if is_tune:
            session.report({'loss': epoch_loss})
        
        # Run validation and save best model
        if epoch % (args.num_epochs//10) == 0 and epoch != 0:
        #if epoch % 10 == 0 and epoch != 0:
            metrics_dict, imputation_data, _ = inference_function(
                data=data,
                model=model,
                #diffusion_steps=args.sample_diffusion_steps,
                diffusion_steps=args.train_diffusion_steps,
                device=device,
                args=args,
                model_autoencoder=model_autoencoder,
                wandb_logger=wandb_logger,
                process="val"
                )

            # Compare MSE metrics and save best model
            if metrics_dict["MSE"] < best_mse:
                best_mse = metrics_dict["MSE"]
                best_pcc = metrics_dict["PCC-Gene"]
                best_epoch = epoch
                best_model_path = os.path.join(save_path, f"best_model.pt")
                torch.save(model.state_dict(), best_model_path)
            
            wandb_logger.log({"MSE": metrics_dict["MSE"], "PCC": metrics_dict["PCC-Gene"]})

    # Save the best MSE and best PCC on the validation set
    wandb_logger.log({"best_MSE":best_mse, "best_PCC": best_pcc, "best_epoch": best_epoch})

    return best_model_path


