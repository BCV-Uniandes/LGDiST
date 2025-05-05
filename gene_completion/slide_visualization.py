import matplotlib.colors as mcolors
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
import squidpy as sq
import anndata as ad
import numpy as np
import matplotlib
import os

def plot_completion_layers(dataset_name, data_set, adata, gene_name, slide, layer, masked, hist_image):
    """
    This function receives the predictions of sota and difussion model, as well as the gt and mask for visualizing the predictions comparison.

    Args:
        dataset_name (str): dataset name
        data_set (str): data set name (train, val, test)
        adata (AnnData): adata for the corresponding set
        n_genes (int, optional): number of genes to plot (top and bottom genes).
        slide (str, optional): slide to plot. If none is given it plots the first slide of the adata.
        layer (str): layer to visualize
        masked (bool): if True, the layer is masked with the mask layer
        hist_image (bool): if True, the histological image is plotted
    """

    # Get the selected slides. NOTE: Only first slide is always selected in case slides is not specified by parameter.
    if slide == "":
        slide = list(adata.obs.slide_id.unique())[0]
    
    # Get adata for slide
    slide_adata = adata[adata.obs['slide_id'] == slide].copy()
    # Modify the uns dictionary to include only the information of the slide
    slide_adata.uns['spatial'] = {slide: adata.uns['spatial'][slide]}
    # Get slide name
    slide_name = list(slide_adata.obs.slide_id.unique())[0]
    
    dpi = 600
    fig, ax = plt.subplots(figsize=(8000/dpi, 6000/dpi), dpi=dpi)
    
    # Get gene id from gene name
    if isinstance(gene_name, int):
        g = gene_name
    elif isinstance(gene_name, str):
        g = slide_adata.var[slide_adata.var['gene_ids'] == gene_name].index[0]

    # Get min and max of the selected gene in the slide and layer        
    layer_min = np.nanmin(slide_adata[:, g].layers[layer]) 
    layer_max = np.nanmax(slide_adata[:, g].layers[layer])
    norm = matplotlib.colors.Normalize(vmin=layer_min, vmax=layer_max)

    slide_adata.layers["mask_int"] = slide_adata.layers["mask"].astype(int).copy()
    assert np.unique(slide_adata.layers["mask_int"]).shape[0] == 2, "Mask layer should have only 2 unique values (0 and 1)"
    
    # Plot spatial scatter
    if layer == "mask":
        sq.pl.spatial_scatter(slide_adata, color=[str(g)], layer="mask_int", fig=fig, ax=ax, cmap='gray', norm=norm, colorbar=False, title="")
        fig_name = f'{dataset_name}_{data_set}_{slide_name}_{gene_name}_{layer}.png'
    else:
        if hist_image:
            # Plot only the histological image
            sq.pl.spatial_scatter(slide_adata, color=None, library_id=slide, fig=fig, ax=ax, colorbar=False, title="")
            fig_name = f'{dataset_name}_{data_set}_{slide_name}_hist_image.png'
        else:
            if masked:
                cmap_original = get_cmap('jet')
                new_colors = cmap_original(np.linspace(0, 1, 256))  # Obtener los colores originales
                new_colors[0] = [0, 0, 0, 1]  # Reemplazar el primer color con negro (RGBA)
                new_cmap = mcolors.ListedColormap(new_colors) # Crear un nuevo colormap con los colores modificados
                slide_adata.layers["layer_masked"] = np.where(slide_adata.layers["mask_int"] == 1, slide_adata.layers[layer], 0)
                sq.pl.spatial_scatter(slide_adata, color=[str(g)], layer="layer_masked", fig=fig, ax=ax, cmap=new_cmap, norm=norm, colorbar=False, title="")
                fig_name = f'{dataset_name}_{data_set}_{slide_name}_{gene_name}_{layer}_masked.png'
            else:
                sq.pl.spatial_scatter(slide_adata, color=[str(g)], layer=layer, fig=fig, ax=ax, cmap='jet', norm=norm, colorbar=False, title="")
                fig_name = f'{dataset_name}_{data_set}_{slide_name}_{gene_name}_{layer}.png'
    
    fig_path = os.path.join('slides_visualizations')
    os.makedirs(fig_path, exist_ok=True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    plt.axis('off')
    plt.show()
    fig.savefig(os.path.join(fig_path, fig_name))
    plt.close('all')

if __name__ == "__main__":
    
    datasets= [
        #"abalo_human_squamous_cell_carcinoma",
        #"10xgenomic_human_breast_cancer",
        "10xgenomic_human_brain", 
        "erickson_human_prostate_cancer_p1", 
        "fan_mouse_brain_coronal", 
        "mirzazadeh_human_colon_p1", 
        "mirzazadeh_mouse_bone", 
        "mirzazadeh_mouse_brain_p2", 
        "10xgenomic_mouse_brain_sagittal_posterior", 
        #"villacampa_mouse_brain"
    ]
    data_set = 'train'
    slide = ''
    layer = "c_t_log1p"
    gene_name = 20
    masked = False
    hist_image = True

    for dataset_name in datasets:
        print(f"Plotting {dataset_name} ...")
        # Load adata
        adata = ad.read_h5ad(f"../datasets/original/{dataset_name}.h5ad")

        # Check if adata has spatial information
        try:
            adata.uns['spatial']
        except:
            print(f"{dataset_name} does not have spatial information")
            
        # Define adata for the data set
        if data_set == "test" and "test" not in adata.obs["split"].unique():
                print(f"{dataset_name} does not have a test set")
        else:
            adata_set = adata[adata.obs["split"]==data_set].copy()
            # Plot layers
            plot_completion_layers(
                dataset_name = dataset_name,
                data_set = data_set,
                adata = adata_set,
                gene_name = gene_name, 
                slide = slide,
                layer = layer,
                masked = masked, 
                hist_image = hist_image
            )