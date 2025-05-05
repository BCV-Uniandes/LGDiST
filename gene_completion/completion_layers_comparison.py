import matplotlib.pyplot as plt
import squidpy as sq
import anndata as ad
import numpy as np
import matplotlib
import os

def log_genes_for_slide(dataset_name, genes, slide_adata, data_set, set_name, layers):
    """
    This function receives a slide adata and logs the visualizations for the top and bottom genes

    Args:
        dataset_name (str): dataset name
        genes (list): genes to visualize
        slide_adata (AnnData): slide AnnData
        data_set (str, optional): data set name (train, val, test)
        set_name (str, optional): set name. Defaults to ''. It can be 'Top_Genes' or 'Bottom_Genes'
        layers (list): list of layers to visualize
    """

    # Define order of rows in dict
    order_dict = {}
    for i, gene in enumerate(genes):
        order_dict[gene] = i
        
    num_cols = len(layers)
    fig, ax = plt.subplots(nrows=len(genes), ncols=num_cols, layout='constrained')
    fig.set_size_inches(num_cols * 5, len(genes) * 4)
    
    # Iterate over the genes
    for g in genes: 

        # Get current row
        row = order_dict[g]
        # List for save min and max values of the gene in the slide layers
        norms = []
        for layer in layers:
            # Get min and max of the selected gene in the slide and layer        
            layer_min = np.nanmin(slide_adata[:, g].layers[layer]) 
            layer_max = np.nanmax(slide_adata[:, g].layers[layer])
            norms.append([layer_min, layer_max])
        general_norm = [min([norm[0] for norm in norms]), max([norm[1] for norm in norms])]
        general_norm = matplotlib.colors.Normalize(vmin=general_norm[0], vmax=general_norm[1])

        # Plot layers
        for i,layer in enumerate(layers):
            # Plot spatial scatter
            if layer == "mask":
                norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
                slide_adata.layers["mask_int"] = slide_adata.layers["mask"].astype(int).copy()
                assert np.unique(slide_adata.layers["mask_int"]).shape[0] == 2, "Mask layer should have only 2 unique values (0 and 1)"
                sq.pl.spatial_scatter(slide_adata, color=[str(g)], layer="mask_int", fig=fig, ax=ax[row,i], cmap='gray', norm=norm, colorbar=True, title="")
            else:
                norm = matplotlib.colors.Normalize(vmin=norms[i][0], vmax=norms[i][1])
                sq.pl.spatial_scatter(slide_adata, color=[str(g)], layer=layer, fig=fig, ax=ax[row,i], cmap='jet', norm=norm, colorbar=True, title="")
            # Set y labels
            ax[row,i].set_ylabel('')    
            # Set x labels
            ax[row,i].set_xlabel('')
        
        slide_name = list(slide_adata.obs.slide_id.unique())[0]
        exp_frac = float(slide_adata[:, g].var['exp_frac'].iloc[0])
        ax[row,0].set_ylabel(f'{slide_adata.var.loc[g]["gene_ids"]}:\n{slide_name}\nexp fraction:{round(exp_frac,3)}', fontsize='large')
        ax[row,0].set_xticks([])
        ax[row,0].set_yticks([])
    
    # Format figure
    for i, axis in enumerate(ax.flatten()):
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
        if ((i+1)%num_cols) != 0: 
            axis.spines['bottom'].set_visible(False)
            axis.spines['left'].set_visible(False)

    # Set titles
    for i in range(len(layers)):
        ax[0, i].set_title(layers[i], fontsize='xx-large')

    fig_path = os.path.join('completion_layers_comparison_results', dataset_name, data_set)
    os.makedirs(fig_path, exist_ok=True)
    fig.savefig(os.path.join(fig_path, f'{set_name}.png'))
    plt.close('all')

def plot_completion_layers(dataset_name, data_set, adata, n_genes, slide, layers):
    """
    This function receives the predictions of sota and difussion model, as well as the gt and mask for visualizing the predictions comparison.

    Args:
        dataset_name (str): dataset name
        data_set (str): data set name (train, val, test)
        adata (AnnData): adata for the corresponding set
        n_genes (int, optional): number of genes to plot (top and bottom genes).
        slide (str, optional): slide to plot. If none is given it plots the first slide of the adata.
    """

    # Get the selected slides. NOTE: Only first slide is always selected in case slides is not specified by parameter.
    if slide == "":
        slide = list(adata.obs.slide_id.unique())[0]
    
    # Get adata for slide
    slide_adata = adata[adata.obs['slide_id'] == slide].copy()
    # Modify the uns dictionary to include only the information of the slide
    slide_adata.uns['spatial'] = {slide: adata.uns['spatial'][slide]}

    # Get selected genes based on the expression fraction
    gene_len = len(adata.var['exp_frac'])
    n_top = adata.var.nlargest(gene_len, columns='exp_frac').index.to_list()
    n_bottom = adata.var.nsmallest(gene_len, columns='exp_frac').index.to_list()

    # Takes top and worst genes according to n_genes
    top_genes = n_top[:n_genes]
    bottom_genes = n_bottom[:n_genes]
    selected_genes = []
    selected_genes.append(top_genes)
    selected_genes.append(bottom_genes)
    top_bottom = ["Top_Genes", "Bottom_Genes"]

    print('Creating visualization plots ...')
    for i, gene in enumerate(selected_genes):
        log_genes_for_slide(
            dataset_name=dataset_name,
            genes=gene, 
            slide_adata=slide_adata, 
            data_set=data_set,
            set_name=top_bottom[i],
            layers=layers
        )

if __name__ == "__main__":

    # List of datasets in spared benchmark
    all_datasets = ['10xgenomic_human_brain', '10xgenomic_human_breast_cancer', '10xgenomic_mouse_brain_coronal', 
                    '10xgenomic_mouse_brain_sagittal_anterior', '10xgenomic_mouse_brain_sagittal_posterior', 
                    'abalo_human_squamous_cell_carcinoma', 'erickson_human_prostate_cancer_p1', 'erickson_human_prostate_cancer_p2', 
                    'fan_mouse_brain_coronal', 'fan_mouse_olfatory_bulb', 'mirzazadeh_human_colon_p1', 'mirzazadeh_human_colon_p2', 
                    'mirzazadeh_human_pediatric_brain_tumor_p1', 'mirzazadeh_human_pediatric_brain_tumor_p2', 
                    'mirzazadeh_human_prostate_cancer', 'mirzazadeh_human_small_intestine', 'mirzazadeh_mouse_bone', 
                    'mirzazadeh_mouse_brain_p1', 'mirzazadeh_mouse_brain_p2', 'mirzazadeh_mouse_brain', 'parigi_mouse_intestine', 
                    'vicari_human_striatium', 'vicari_mouse_brain', 'villacampa_kidney_organoid', 'villacampa_lung_organoid', 'villacampa_mouse_brain']
    
    # Define list of layers to compare
    layers = ["mask","c_d_log1p","c_t_log1p","c_dif_log1p"]

    for dataset_name in all_datasets:
        print(f"Processing {dataset_name} ...")
        # Load adata
        adata = ad.read_h5ad(f"../datasets/original/{dataset_name}.h5ad")
        # Check if adata has spatial information
        try:
            adata.uns['spatial']
        except:
            print(f"{dataset_name} does not have spatial information")
            continue

        for data_set in ["train", "val", "test"]:
            if data_set == "test" and "test" not in adata.obs["split"].unique():
                continue
            # Define adata for the data set
            adata_set = adata[adata.obs["split"]==data_set].copy()
            # Plot layers
            plot_completion_layers(
                dataset_name = dataset_name,
                data_set = data_set,
                adata = adata_set,
                n_genes = 3, 
                slide = "",
                layers = layers
            )
