from utils import seed_everything, get_main_parser
import torch
from tqdm import tqdm
import numpy as np

parser = get_main_parser()
args = parser.parse_args()

#Seed
seed = args.seed
seed_everything(seed)


def model_sample_stDiff(model, device, dataloader, time, condi_flag, step_noise):
    noise = []
    i = 0
    for batch in dataloader: 
        x, x_cond, mask = batch["encoded_exp_matrix"].float().permute(0,2,1).to(device), batch["condition_matrix"].float().permute(0,2,1).to(device), batch["condition_mask"].permute(0,2,1).to(device)
        
        x_cond = x_cond.float().to(device) 
        mask = mask.to(device)
        t = torch.from_numpy(np.repeat(time, x_cond.shape[0])).long().to(device)

        cond = [x_cond, mask]
        n = model(step_noise[i:i+len(x_cond)], t, cond, condi_flag=condi_flag) 
        noise.append(n)
        i = i+len(x_cond)
    noise = torch.cat(noise, dim=0)
    return noise

def sample_stDiff(model,
                dataloader,
                noise_scheduler,
                args,
                device=torch.device('cuda:1'),
                num_step=1000,
                ):
    #mask = None
    """_summary_

    Args:
        model (_type_): denoising model
        dataloader (_type_): _description_
        noise_scheduler (_type_): _description_
        args (argparse): parser with the values necessary for custom training and test.
        device (_type_, optional): _device_. Defaults to torch.device('cuda:1').
        num_step (int, optional): _timestep_. Defaults to 1000.
        
    Returns:
        _type_: recon_x
    """
    # Iterate entire dataloader to get complete tensor of masks and x_conds
    full_mask = []
    full_x_cond = []
    get_metrics_mask = []
    for batch in dataloader:
        full_mask.append(batch["condition_mask"].permute(0,2,1))
        full_x_cond.append(batch["condition_matrix"].float().permute(0,2,1))
        get_metrics_mask.append(batch["exp_mask"].permute(0,2,1))

    full_mask = torch.cat(full_mask, dim=0).to(device)
    full_x_cond = torch.cat(full_x_cond, dim=0).to(device)
    get_metrics_mask = torch.cat(get_metrics_mask, dim=0)

    # Initialize random x_t
    x_t = torch.randn(full_mask.shape).to(device) #torch.Size([439(full split), 128, 7])
    x_t =  x_t * (1 - full_mask) + full_x_cond * full_mask

    model.eval()
    timesteps = list(range(num_step))[::-1] 
    
    ts = tqdm(timesteps)
    for t_idx, time in enumerate(ts):
        ts.set_description_str(desc=f'time: {time}')
        with torch.no_grad():
            #model_output (spots, genes): retorna un tensor con las predicciones de los genes maskeados y cero en el resto (en lo no maskeado)
            model_output = model_sample_stDiff(model,
                                        device=device,
                                        dataloader=dataloader,
                                        time=time,  # t
                                        condi_flag=True,
                                        step_noise=x_t,  # x_t
                                        )
        
        x_t, _ = noise_scheduler.step(model_output,  # noise
                                      torch.from_numpy(np.array(time)).long().to(device),
                                      x_t)
        
        x_t =  x_t * (1 - full_mask) + full_x_cond * full_mask

    recon_x = x_t.detach().cpu().numpy()
    return recon_x, get_metrics_mask

if __name__ == "__main__":
    from utils import get_main_parser

    # Get parser and parse arguments
    parser = get_main_parser()
    args = parser.parse_args()
    args_dict = vars(args)

    #Seed
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
