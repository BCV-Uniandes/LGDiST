from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from utils import get_main_parser, seed_everything
import torch.nn as nn
import numpy as np
import einops
import torch
import math

parser = get_main_parser()
args = parser.parse_args()

#Seed
seed = args.seed
seed_everything(seed)

class Attention2(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Cambio de lineal a conv1d
        # Conv1d layers for q, k, v
        #self.qkv = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias)
        self.q_conv = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias) #kernel cambiado de 1 a 7
        self.k_conv = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias) #kernel cambiado de 1 a 7
        self.v_conv = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias) #kernel cambiado de 1 a 7


        #self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # Opcion convolucional
        # x shape: (batch_size, seq_length, feature_dim)
        B, G, C = x.shape

        # Transpose for Conv1d: (batch_size, feature_dim, seq_length)
        # Se permuta para que pueda entrar en la capa convolucional
        x = x.permute(0, 2, 1)

        # Compute q, k, v using Conv1d
        q = self.q_conv(x)
        k = self.k_conv(x)
        v = self.v_conv(x)
        # B, num_head, G, head_dim--> q, v ,k
        # Reshape for multi-head attention: (batch_size, num_heads, seq_length, head_dim)
        q = q.reshape(B, self.num_heads, self.head_dim, G).permute(0, 1, 3, 2)  # (B, num_heads, seq_length, head_dim)
        k = k.reshape(B, self.num_heads, self.head_dim, G).permute(0, 1, 3, 2)  # (B, num_heads, seq_length, head_dim)
        v = v.reshape(B, self.num_heads, self.head_dim, G).permute(0, 1, 3, 2)  # (B, num_heads, seq_length, head_dim)

        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, num_heads, seq_length, seq_length)
        attn = attn.softmax(dim=-1)  # Apply softmax to get probabilities
        attn = self.attn_drop(attn)  # Dropout for regularization

        # Apply attention scores to the value vectors
        x = torch.matmul(attn, v)  # (B, num_heads, seq_length, head_dim)

        # Combine heads: (batch_size, seq_length, feature_dim)
        x = x.permute(0, 2, 1, 3).reshape(B, G, C)

        # Project back to original dimension
        x = x.permute(0, 2, 1)  # Back to (batch_size, feature_dim, seq_length) for Conv1d
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.permute(0, 2, 1)  # Return to (batch_size, seq_length, feature_dim)
        #print("attention Conv1D")
        
        # Opcion lineal con entrada matricial
        """
        # x shape: (batch_size, seq_length, feature_dim)
        B, G, C = x.shape
        qkv = self.qkv(x).reshape(B, G, 3, self.num_heads, self.head_dim)
        qkv = einops.rearrange(qkv, 'b g n h d -> n b h g d')
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple) (h c fph)
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # Dot product and scaling
        attn = attn.softmax(dim=-1)  # Apply softmax to get probabilities
        attn = self.attn_drop(attn)  # Dropout for regularization

        # Apply attention scores to the value vectors
        x = attn @ v  # Weighted sum of values
        x = einops.rearrange(x, 'b h g d -> b g (h d)')  # Combine heads

        # Project back to original dimension
        x = self.proj(x)
        x = self.proj_drop(x)
        """
        return x
    
def modulate(x, shift, scale):
    res = x * (1 + scale) + shift
    return res

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    time emb to frequency_embedding_size dim, then to hidden_size
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
   
class DiTblock(nn.Module):
    # adaBN -> attn -> mlp
    def __init__(self,
                 feature_dim=2000,
                 mlp_ratio=4.0,
                 num_heads=10,
                 **kwargs) -> None:
        super().__init__()
        
        self.norm1 = nn.LayerNorm(feature_dim, elementwise_affine=False, eps=1e-6)
        # TODO: what was the difference between Attention() and Attention2()? 
        self.attn = Attention2(feature_dim, num_heads=num_heads, qkv_bias=True, **kwargs) 
        
        self.norm2 = nn.LayerNorm(feature_dim, elementwise_affine=False, eps=1e-6)
        approx_gelu = lambda: nn.GELU()
        
        mlp_hidden_dim = int(feature_dim * mlp_ratio)
        self.mlp = Mlp(in_features=feature_dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        
        #self.adaLN_modulation = nn.Sequential(
        #    nn.SiLU(),
        #    nn.Conv1d(in_channels=feature_dim, out_channels=6 * feature_dim, kernel_size=1, bias=True)
        #)
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(feature_dim, 6 * feature_dim, bias=True)
        )
        
        #self.adaLN_modulation_v2 = nn.Sequential(nn.SiLU(), nn.Linear(feature_dim, 6 * feature_dim, bias=True))
    
    def forward(self, x, time):
        """
        Forward pass for DiTblock
        :param x: Input tensor of shape (batch_size, seq_length, feature_dim)
        :param c: Condition tensor of shape (batch_size, seq_length, feature_dim)
        :return: Output tensor of shape (batch_size, seq_length, feature_dim)
        """
        # TODO: debug process
        # Project condition to 6 * hiddensize and then slice it into 6 parts along the column.
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(time).chunk(6, dim=2) #dim cambiado de 1 a 2
        
        """
        # Permute all outputs back to the original shape
        shift_msa = shift_msa.permute(0, 2, 1)  # [256, 512, 7]
        scale_msa = scale_msa.permute(0, 2, 1)  # [256, 512, 7]
        gate_msa = gate_msa.permute(0, 2, 1)    # [256, 512, 7]
        shift_mlp = shift_mlp.permute(0, 2, 1)  # [256, 512, 7]
        scale_mlp = scale_mlp.permute(0, 2, 1)  # [256, 512, 7]
        gate_mlp = gate_mlp.permute(0, 2, 1)    # [256, 512, 7]
        """
        # attention blk
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        # mlp
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class FinalLayer(nn.Module):
    """
    The final layer of DiT. adaLN -> linear
    """
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # Linear for projection into output shape
        self.linear = nn.Linear(hidden_size, out_size, bias=True)
        # AdaLN modulation    
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        
        #self.adaLN_modulation_v2 = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        # Shift scale
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=2) #dim cambiado de 1 a 2 para recibir matrices
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        x = x.permute(0, 2, 1)
        return x      

BaseBlock = {"dit":DiTblock}

def get_positional_encoding(seq_len, d_model):
    positional_encoding = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        operational_pos = pos + 1
        for i in range(0, d_model, 2):
            positional_encoding[pos, i] = np.sin(operational_pos / (10000 ** (i / d_model)))
            if i + 1 < d_model:
                positional_encoding[pos, i + 1] = np.cos(operational_pos / (10000 ** ((i) / d_model)))
    return positional_encoding

class DiT_stDiff(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 depth,
                 num_heads,
                 classes,
                 args,
                 mlp_ratio=4.0,
                 dit_type="dit",
                 **kwargs) -> None:
        
        """ Denoising model

        Args:
            input_size (_type_): input dim
            hidden_size (_type_): scale input to hidden dim
            depth (_type_): dit block num
            num_heads (_type_): transformer head num
            classes (_type_): class num
            args (__type__): model args
            mlp_ratio (float, optional): _description_. Defaults to 4.0.
            dit_type (_type_): which type of block to use. Only "dit" available in this script.
        """        
        super().__init__()
        self.args = args
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.classes = classes
        self.mlp_ratio = mlp_ratio
        self.dit_type = dit_type
        self.in_channels = self.input_size[0]
        self.out_size = self.input_size[0]

        if args.concat_dim == 1:
            # concat by feature dim
            self.project_size = hidden_size*3 
        else:
            self.project_size = hidden_size
        
        # Set time embedder
        self.time_emb = TimestepEmbedder(hidden_size=self.hidden_size)
        
        # 1D Convolutional layer for condition
        self.cond_layer = nn.Sequential(
            nn.Conv1d(in_channels=self.in_channels, out_channels=hidden_size, kernel_size=3, stride=1, padding=1) #kernel_size=7, stride=1, padding=3
        )
    
        # 1D Convolutional in-layer for x tensor
        self.in_layer = nn.Sequential(
            nn.Conv1d(in_channels=self.in_channels, out_channels=hidden_size, kernel_size=3, stride=1, padding=1) #kernel_size=7, stride=1, padding=3
        )
        
        # conditional embedding
        self.condi_emb = nn.Embedding(classes, hidden_size)

        # Declare set of DiT blocks
        self.blks = nn.ModuleList([
            BaseBlock[dit_type](self.project_size, mlp_ratio=self.mlp_ratio, num_heads=self.num_heads) for _ in
            range(self.depth)
        ])

        # out layer
        self.out_layer = FinalLayer(self.project_size, self.out_size)
        self.out_layer_dim2 = nn.Linear(self.input_size[1]*3, self.input_size[1], bias=True)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                # xavier 
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    # bias  0
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        # Initialize label embedding table:
        nn.init.normal_(self.condi_emb.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.time_emb.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_emb.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        if self.dit_type == 'dit':
            for block in self.blks:
                # adaLN_modulation , adaLN  0 initial
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.out_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.out_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.out_layer.linear.weight, 0)
        nn.init.constant_(self.out_layer.linear.bias, 0)

    def forward(self, x, t, y,**kwargs): 
        x = x.float()
        num_neighs = x.shape[2]

        x = self.in_layer(x)                # torch.Size([128, 1024, 7])
        t = self.time_emb(t)                # torch.Size([128, 1024])
        
        cond = self.cond_layer(y[0])        # torch.Size([128, 1024, 7])
        mask = self.cond_layer(y[1])        # torch.Size([128, 1024, 7])
        
        t = t.unsqueeze(2).repeat(1, 1, num_neighs)
        c = t + cond # torch.Size([128, 1024, 7])
        
        # Permute so that the data can pass through ViT
        # From (batch_size, feature_dim, seq_length) to  (batch_size, seq_length, feature_dim)
        # This is done so that the self.adaLN_modulation(c) layer, which consist of a linear proyection, is able to process this input
        c = c.permute(0, 2, 1)             # torch.Size([128, 7, 1024])
        x = x.permute(0, 2, 1)             # torch.Size([128, 7, 1024])

        # torch.Size([256, 7, 512])
        for blk in self.blks: 
            x = blk(x, c)
        x = self.out_layer(x, c)

        return x                           # torch.Size([128, 128, 7])
        

if __name__ == "__main__":
    # Get parser and parse arguments
    parser = get_main_parser()
    args = parser.parse_args()
    args_dict = vars(args)

    #Seed
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)