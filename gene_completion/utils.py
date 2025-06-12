from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
import matplotlib.pyplot as plt
from metrics import get_metrics
import anndata as ad
import squidpy as sq
import numpy as np
import matplotlib
import argparse
import torch

# Auxiliary function to use booleans in parser
str2bool = lambda x: (str(x).lower() == 'true')

def get_main_parser():
    parser = argparse.ArgumentParser(description='Code for expression prediction using contrastive learning implementation.')
    # Dataset parameters #####################################################################################################################################################################
    parser.add_argument('--dataset',                        type=str,               default='villacampa_lung_organoid',      help='Dataset to use.')
    parser.add_argument('--pred_layer',                     type=str,               default='c_d_deltas',                    help='SpaRED prediction layer to use.')
    parser.add_argument('--num_neighs',                     type=int,               default=6,                               help='Amount of neighbors considered to build spot neighborhoods. Must be the same as the ones used to train the autoencoder.')
    parser.add_argument('--normalize_input',                type=str2bool,          default=True,                            help='Whether or not to normalize the DiT input data (encoded matrix) between -1 and 1 when preparing dataloader.')
    parser.add_argument('--noise_fraction',                 type=float,             default=1,                               help='Fraction of missing values/noise within the gene-expression data. If 1, it is an extreme imputation context.')
    parser.add_argument('--autoencoder_ckpts_path',         type=str,               default='/home/pcardenasg/autoencoders/ST/results/villacampa_lung_organoid/2025-01-21-02-52-56/epoch=498-step=6986.ckpt',            help='Path to trained checkpoints of AE corresponding to the dataset used.')
    parser.add_argument('--decode_as_matrix',               type=str2bool,          default=False,                           help='Whether or not the decoder receives 2D inputs.')
    parser.add_argument('--full_inference',                 type=str2bool,          default=False,                           help='Whether or not to prepare a dataloader with all three data splits to perform inference in complete dataset.')
    parser.add_argument('--partial',                        type=str2bool,          default=False,                           help='Set to True to perform partial completion during inference.')
    parser.add_argument('--masking_method',                 type=str,               default='mask_prob',                     help='Available options: mask_prob, prob_median, scale_factor.')
    parser.add_argument('--mask_prob',                      type=float,             default=0.3,                             help='')
    parser.add_argument('--scale_factor',                   type=float,             default=0.8,                             help='')
    # Model parameters #######################################################################################################################################################################
    parser.add_argument('--dit_hidden_size',                type=int,               default=1024,                            help='')
    parser.add_argument('--dit_depth',                      type=int,               default=12,                              help='')
    parser.add_argument('--num_heads',                      type=int,               default=16,                              help='')
    parser.add_argument("--concat_dim",                     type=int,               default=0,                               help='Which dimension used to concat the condition.')
    parser.add_argument('--dit_ckpts_path',                 type=str,               default='',                              help='Path to trained checkpoints of DiT corresponding to the dataset used. Optional.')
    # Autoencoder parameters #################################################################################################################################################################
    parser.add_argument('--num_layers_autoencoder',             type=int,         default=4,                                    help='Number of layers in the autoencoder')
    parser.add_argument('--num_heads_autoencoder',              type=int,         default=2,                                    help='Number of heads in the transformer encoder')
    parser.add_argument("--embedding_dim",          type=int,         default=512,                                  help='Embedding dimensions in the eutoencoder')
    parser.add_argument('--input_dim',              type=int,         default=1024,                                 help='Input dimension of the autoencoder')
    parser.add_argument("--latent_dim",             type=int,         default=128,                                  help='Latent dimension in the autoencoder')
    parser.add_argument("--output_dim",             type=int,         default=1024,                                 help='Ouput dimension of the autoencoder')
    # Train parameters #######################################################################################################################################################################
    parser.add_argument('--seed',                           type=int,               default=1202,                            help='Seed to control initialization')
    parser.add_argument('--train',                          type=str2bool,          default=True,                            help='Train model.')
    parser.add_argument('--test',                           type=str2bool,          default=True,                            help='Test model.')
    parser.add_argument('--normalized_data',                type=str2bool,          default=False,                           help='Whether or not to work with normalized expression matrix.')
    parser.add_argument('--lr',                             type=float,             default=0.0001,                          help='lr to train DiT.')
    parser.add_argument('--batch_size',                     type=int,               default=128,                             help='Batch size used to train the diffusion model.')
    parser.add_argument('--num_epochs',                     type=int,               default=1500,                            help='Number of training epochs.')
    parser.add_argument('--train_diffusion_steps',          type=int,               default=1500,                            help='Number of diffusion steps for training process.')
    parser.add_argument('--sample_diffusion_steps',         type=int,               default=50,                            help='Number of diffusion steps for val or test process.')
    parser.add_argument('--step_size',                      type=float,             default=600,                             help='Step size to use in learning rate scheduler')
    parser.add_argument("--adjust_loss",                    type=str2bool,          default=True,                            help='If True the loss is obtained only on masked data. If False the loss takes into account the entire set of genes and spots.')
    parser.add_argument("--scheduler",                      type=str2bool,          default=True,                            help='Whether to use LR scheduler or not.')
    ##########################################################################################################################################################################################

    return parser

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True    

def data_normalization(data: torch.tensor, data_min, data_max):
    """ 
    This function receives a gene expression matrix and normalizes its content so that it has a range of [-1, 1].
    """
    norm_data = 2 * (data - data_min) / (data_max - data_min) - 1
    
    return norm_data

'''def data_denormalization(norm_data: torch.tensor, data_min: torch.tensor, data_max: torch.tensor):
    """
    This function receives a normalized gene expression matrix and denormalizes its content 
    back to the original range using the provided data_min and data_max.
    """
    denorm_data = (norm_data + 1) / 2 * (data_max - data_min) + data_min
    return denorm_data'''

def denormalize_from_minus_one_to_one(X_norm, X_max, X_min):
    # Apply the denormalization formula 
    X_denorm = ((X_norm + 1) / 2) * (X_max - X_min) + X_min
    return X_denorm

def decode(imputation, model_decoder, decode_as_matrix=False):
    # imputation shape torch.Size([439, 128, 7])
    if not decode_as_matrix:
        imputation = torch.tensor(imputation, dtype=torch.float32)[:,:,0] # shape torch.Size([439, 128])
    else: 
        imputation = imputation.permute(0,2,1)

    dataset = TensorDataset(imputation)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
    
    model_decoder.to("cuda")
    model_decoder.eval()  
    decoded_samples = []
    with torch.no_grad():
        for batch in dataloader: 
            batch = batch[0].to("cuda") 
            decoded_batch = model_decoder.decoder(batch) 
            decoded_samples.append(decoded_batch)

    decoded_samples = torch.cat(decoded_samples, dim=0)

    return decoded_samples

def inference_function(data, model, diffusion_steps, device, args, model_autoencoder, wandb_logger=None, process = "val"):
    # To avoid circular imports
    from model.scheduler import NoiseScheduler
    from model.sample import sample_stDiff
    """
    Function designed to do inference for validation and test steps.
    Params:
        - data (SpaREDData): class with all SpaRED data preprocessed
        - model (diffusion model): diffusion model to do inference
        - diffusion_steps (int): number of steps needed for denoising during sampling (set in argparse)
        - device (str): device cpu or cuda
        - args (argparse): parser with the values necessary for custom training and test
        - model_autoencoder (autoencoder): autoencoder with a "decoder()" attribute
        - wandb_logger (_type_): wandb to track results.
        - process (str): either "val" or "test" to determine the data split that needs to be used

    Returns:
        - metrics_dict (dict): dictionary with all the evaluation metrics
    """
    # Define noise scheduler
    noise_scheduler = NoiseScheduler(
        num_timesteps=diffusion_steps,
        beta_schedule='cosine'
    )
    
    if process == "train":
        dataloader = data.train_dataloader()
        min_norm, max_norm = data.train_data.min_val, data.train_data.max_val
        c_t_log1p_data = torch.tensor(data.spared_train.layers["c_t_log1p"])
    elif process == "val":
        dataloader = data.val_dataloader()
        min_norm, max_norm = data.val_data.min_val, data.val_data.max_val
        c_t_log1p_data = torch.tensor(data.spared_val.layers["c_t_log1p"])
    elif process == "test":
        dataloader = data.test_dataloader()
        min_norm, max_norm = data.test_data.min_val, data.test_data.max_val
        c_t_log1p_data = torch.tensor(data.spared_test.layers["c_t_log1p"])
    else: # predict on all data
        dataloader = data.all_dataloader()
        min_norm, max_norm = data.all_data.min_val, data.all_data.max_val
        c_t_log1p_data = torch.tensor(data.full_adata.layers["c_t_log1p"])

    # Get all ground truths
    ground_truth = []
    for batch in dataloader:
        ground_truth.append(batch['encoded_exp_matrix'].permute(0,2,1))
    ground_truth = torch.cat(ground_truth, dim=0)
    
    # inference using test split
    imputation, test_mask = sample_stDiff(model,
                        dataloader=dataloader,
                        noise_scheduler=noise_scheduler,
                        args=args,
                        device=device,
                        num_step=diffusion_steps
                        ) 
    
    # Compute MSE between predictions and ground truth before decoding
    ground_truth = denormalize_from_minus_one_to_one(ground_truth, max_norm, min_norm)
    imputation = denormalize_from_minus_one_to_one(imputation, max_norm, min_norm)
    
    print(f"Latent Input values - Mean: {ground_truth.mean().item()} - Std dev: {ground_truth.std().item()}")
    print(f"Latent Output values - Mean: {imputation.mean().item()} - Std dev: {imputation.std().item()}")
    
    # Check how much do minor perturbations in the model's prediction affect the output of the decoder
    imputation = torch.tensor(imputation) 
    perturbation = torch.randn_like(imputation) * 0.01
    
    decoded_imputation = decode(imputation=imputation, model_decoder=model_autoencoder)
    decoded_perturbation = decode(imputation=perturbation, model_decoder=model_autoencoder)
    
    mse_pre = F.mse_loss(imputation, perturbation)
    print("MSE pred vs perturbed-pred before decoding: ", mse_pre.item())
    mse_post = F.mse_loss(decoded_imputation, decoded_perturbation)
    print("MSE pred vs perturbed-pred after decoding: ", mse_post.item())

    # Compute MSE for entire encoded prediction
    dit_imputation = torch.tensor(imputation, dtype=torch.float32)[:,:,0]
    dit_gt = torch.tensor(ground_truth, dtype=torch.float32)[:,:,0]
    dit_mse = F.mse_loss(dit_gt, dit_imputation)
    print("mse del dit (encoded pred vs encoded gt): ", dit_mse.item()) # MSE de predicción vs gt pero antes de decodear 
    if wandb_logger:
        wandb_logger.log({"encoded_pred_MSE": dit_mse})
    
    #Decoded imputation data
    if args.decode_as_matrix:
        imputation = decode(imputation=imputation, model_decoder=model_autoencoder, decode_as_matrix=True)
        imputation = imputation[:,0,:]
    else:
        imputation = decode(imputation=imputation, model_decoder=model_autoencoder)
    
    imputation = imputation.detach().cpu() 
    
    #Evaluate only on main spot of each sample
    mask_boolean = test_mask[:,:,0]

    # If working with deltas, add average expression values to get the prediction equivalent to the "c_t_log1p" layer
    if "deltas" in args.pred_layer:
        # Add the gene average values to the prediction to obtain the final log1p preds
        imputation_tensor = imputation + data.average_vals # this is type torch.Tensor
        #imputation_tensor = np.array(imputation_tensor.cpu()) 

    #imputation_tensor = torch.tensor(imputation_tensor, dtype=torch.float32)
    
    #matrix input
    mse_final = F.mse_loss(imputation_tensor[mask_boolean], c_t_log1p_data[mask_boolean])
    print("MSE final sobre log1p - extreme completion: ", mse_final.item())
    metrics_dict = get_metrics(c_t_log1p_data, imputation_tensor, mask_boolean) 
    
    # Perform partial-completion test if needed
    if args.partial:
        original_adata = data.original_full_adata[data.original_full_adata.obs["split"]==process]
        # Get the columns that correspond to the original SpaRED genes
        original_genes = mask_boolean.sum(dim=0)!=0 
        assert torch.allclose(data.gene_weights, original_genes)
        # Keep only the gt and preds for SpaRED genes
        original_exp = c_t_log1p_data[:,original_genes]
        imputation_tensor = imputation_tensor[:,original_genes]
        # Retrieve the random mask from original adata
        mask_boolean = torch.tensor(original_adata.layers["random_mask"]) 
        metrics_dict = get_metrics(original_exp, imputation_tensor, mask_boolean)

    return metrics_dict, imputation_tensor, mask_boolean

def get_deltas(adata: ad.AnnData, from_layer: str, to_layer: str) -> ad.AnnData:
    """
    Compute the deviations from the mean expression of each gene in adata.layers[from_layer] and save them
    in adata.layers[to_layer]. Also add the mean expression of each gene to adata.var[f'{from_layer}_avg_exp'].

    Args:
        adata (ad.AnnData): The AnnData object to update. Must have expression values in adata.layers[from_layer].
        from_layer (str): The layer to take the data from.
        to_layer (str): The layer to store the results of the transformation.

    Returns:
        ad.AnnData: The updated AnnData object with the deltas and mean expression.
    """

    # Get the expression matrix of both train and global data
    glob_expression = adata.to_df(layer=from_layer)
    train_expression = adata[adata.obs['split'] == 'train'].to_df(layer=from_layer)

    # Define scaler
    scaler = StandardScaler(with_mean=True, with_std=False)

    # Fit the scaler to the train data
    scaler = scaler.fit(train_expression)
    
    # Get the centered expression matrix of the global data
    centered_expression = scaler.transform(glob_expression)

    # Add the deltas to adata.layers[to_layer]	
    adata.layers[to_layer] = centered_expression

    # Add the mean expression to adata.var[f'{from_layer}_avg_exp']
    adata.var[f'{from_layer}_avg_exp'] = scaler.mean_

    # Return the updated AnnData object
    return adata

def get_mask_prob_tensor(masking_method, adata, mask_prob=0.3, scale_factor=0.8):
    """
    This function calculates the probability of masking each gene present in the expression matrix. 
    Within this function, there are three different methods for calculating the masking probability, 
    which are differentiated by the 'masking_method' parameter. 
    The return value is a vector of length equal to the number of genes, where each position represents
    the masking probability of that gene.
    
    Args:
        masking_method (str): parameter used to differenciate the method for calculating the probabilities.
        adata (ad.adata): the adata containing the full dataset.
        mask_prob (float): masking probability for all the genes. Only used when 'masking_method = mask_prob' 
        scale_factor (float): maximum probability of masking a gene if masking_method == 'scale_factor'
    Return:
        prob_tensor (torch.Tensor): vector with the masking probability of each gene for testing. Shape: n_genes  
    """

    # Convert glob_exp_frac to tensor
    glob_exp_frac = torch.tensor(adata.var.glob_exp_frac.values, dtype=torch.float32)
    # Calculate the probability of median imputation
    prob_median = 1 - glob_exp_frac

    if masking_method == "prob_median":
        # Calculate masking probability depending on the prob median
        # (the higher the probability of being replaced with the median, the higher the probability of being masked).
        prob_tensor = prob_median/(1-prob_median)

    elif masking_method == "mask_prob":
        # Calculate masking probability according to mask_prob parameter
        # (Mask everything with the same probability)
        prob_tensor = mask_prob/(1-prob_median)

    elif masking_method == "scale_factor":
        # Calculate masking probability depending on the prob median scaled by a factor
        # (Multiply by a factor the probability of being replaced with median to decrease the masking probability).
        prob_tensor = prob_median/(1-prob_median)
        prob_tensor = prob_tensor*scale_factor
        
    # If probability is more than 1, set it to 1
    prob_tensor[prob_tensor>1] = 1

    return prob_tensor

def mask_exp_matrix(adata: ad.AnnData, pred_layer: str, mask_prob_tensor: torch.Tensor, device):
    """
    This function recieves an adata and masks random values of the pred_layer based on the masking probability of each gene, then saves the masked matrix in the corresponding layer. 
    It also saves the final random_mask for metrics computation. True means the values that are real in the dataset and have been masked for the imputation model development.
    
    Args:
        adata (ad.AnnData): adata of the data split that will be masked and imputed.
        pred_layer (str): indicates the adata.layer with the gene expressions that should be masked and later reconstructed. Shape: spots_in_adata, n_genes
        mask_prob_tensor (torch.Tensor):  tensor with the masking probability of each gene for testing. Shape: n_genes
    
    Return:
        adata (ad.AnnData): adata of the data split with the gene expression matrix already masked and the corresponding random_mask in adata.layers.
    """

    # Extract the expression matrix
    expression_mtx = torch.tensor(adata.layers[pred_layer])
    # Calculate the mask based on probability tensor
    random_mask = torch.rand(expression_mtx.shape).to(device) < mask_prob_tensor.to(device)
    median_imp_mask = torch.tensor(adata.layers['mask']).to(device)
    # False depicts the values that were not measured in real life and are artificially completed in adata.
    # Combine random mask with the median imputation mask
    random_mask = random_mask.to(device) & median_imp_mask
    # Mask chosen values - In random_mask, True represents the values that are synthetically masked and that will be predicted
    expression_mtx[random_mask] = 0
    # Save masked expression matrix in the data_split annData
    adata.layers['masked_expression_matrix'] = np.asarray(expression_mtx.cpu())
    #Save final mask for metric computation
    adata.layers['random_mask'] = np.asarray(random_mask.cpu())

    return adata