import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

def build_rope_cache(
    seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000,
) -> torch.Tensor:
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, dtype=dtype, device=device) / n_elem))
    seq_idx = torch.arange(seq_len, dtype=dtype, device=device)
    idx_theta = torch.outer(seq_idx, theta)

    dtypes_requiring_casting = [torch.float16, torch.bfloat16, torch.int8]
    working_dtype = torch.float32 if dtype in dtypes_requiring_casting else dtype
    complex_dtype = torch.complex32 if dtype in dtypes_requiring_casting else torch.complex64

    cache = torch.polar(
        torch.ones_like(idx_theta).to(working_dtype),
        idx_theta.to(working_dtype),
    ).to(complex_dtype)  
    return cache

def apply_rope(x: torch.Tensor, rope_cache: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
    x = x.transpose(1, 2)
    T = x.size(1)
    rope_cache = rope_cache[start_pos : start_pos + T]  

    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))  
    rope_cache = rope_cache.view(1, T, 1, xc.size(3))                    
    x_out = torch.view_as_real(xc * rope_cache).flatten(3)               
    return x_out.transpose(1, 2).type_as(x)                              



class HOIEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, number_per_frames=3, emb_first=0, learnable=True):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.number_per_frames = number_per_frames
        self.emb_first = emb_first
        self.learnable = learnable

        if self.learnable:
            self.learnable_pe = nn.ParameterList([
                nn.Parameter(torch.zeros(1, 1, d_model)) for _ in range(self.number_per_frames)
            ])
            for p in self.learnable_pe:
                nn.init.normal_(p, std=0.02)
        else:
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)
            self.interval = pe.size(1)

    def forward(self, x):
        N = self.number_per_frames
        for i in range(N):
            if self.learnable:
                x[:, i+self.emb_first::N, :] = x[:, i+self.emb_first::N, :] + self.learnable_pe[i]
            else:
                x[:, i+self.emb_first::N, :] = x[:, i+self.emb_first::N, :] + self.pe[:, self.interval*i//N:self.interval*i//N+1, :]
        return self.dropout(x)

class AdaLN(nn.Module):
    def __init__(self, latent_dim, embed_dim=None):
        super().__init__()
        if embed_dim is None:
            embed_dim = latent_dim
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            zero_module(nn.Linear(embed_dim, 2 * latent_dim, bias=True)),
        )
        self.norm = nn.LayerNorm(latent_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, h, emb):
        emb_out = self.emb_layers(emb)
        scale, shift = torch.chunk(emb_out, 2, dim=-1)
        if scale.dim() == 2 and h.dim() == 3:
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
        return self.norm(h) * (1 + scale) + shift


class MHAWithRoPE(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0, num_modalities: int = 1):
        super().__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        assert self.head_dim % 2 == 0

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = dropout
        self.rope_cache = None
        self.rope_max_time = -1
        self.rope_orig_dtype = None
        self.num_modalities = num_modalities

    def _maybe_init_rope(self, max_time_steps: int, x: torch.Tensor, base: int = 10000):
        if (self.rope_cache is None or 
            self.rope_max_time < max_time_steps or 
            self.rope_cache.device != x.device or 
            self.rope_orig_dtype != x.dtype):
            
            base_cache = build_rope_cache(
                seq_len=max_time_steps, n_elem=self.head_dim,
                dtype=x.dtype, device=x.device, base=base,
            )
            if self.num_modalities > 1:
                self.rope_cache = torch.repeat_interleave(base_cache, repeats=self.num_modalities, dim=0)
            else:
                self.rope_cache = base_cache
                
            self.rope_max_time = max_time_steps
            self.rope_orig_dtype = x.dtype

    def forward(
        self, x_q, x_kv, *, is_causal, attn_mask=None,
        use_rope=True, rope_base=10000, rope_max_time=None, q_start_pos=0, k_start_pos=0
    ):
        B, Tq, D = x_q.shape
        _, Tk, _ = x_kv.shape

        q = self.q_proj(x_q).view(B, Tq, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_kv).view(B, Tk, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_kv).view(B, Tk, self.n_head, self.head_dim).transpose(1, 2)

        if use_rope:
            actual_time_steps = (Tq + q_start_pos) // self.num_modalities
            max_time = rope_max_time if rope_max_time is not None else actual_time_steps
            self._maybe_init_rope(max_time_steps=max_time, x=x_q, base=rope_base)
            q = apply_rope(q, self.rope_cache, start_pos=q_start_pos)
            k = apply_rope(k, self.rope_cache, start_pos=k_start_pos)

        # additive_mask = None
        # if key_padding_mask is not None:
        #     # key_padding_mask 中 True 表示需要被 Mask 的位置
        #     pad = key_padding_mask[:, None, None, :].to(torch.bool)
        #     pad_mask = torch.zeros((B, 1, 1, Tk), device=x_q.device, dtype=x_q.dtype)
        #     pad_mask = pad_mask.masked_fill(pad, float("-inf"))
        #     additive_mask = pad_mask if additive_mask is None else (additive_mask + pad_mask)
        # print(q.shape,k.shape,v.shape)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, # additive_mask,
            dropout_p=self.dropout if self.training else 0.0, is_causal=False, 
        )
        return self.o_proj(y.transpose(1, 2).contiguous().view(B, Tq, D))

class TransformerDecoderLayerRoPE(nn.Module):
    def __init__(
        self, d_model: int, n_head: int, dim_ff: int = 2048, dropout: float = 0.1, 
        rope_on_cross_attn: bool = False, num_modalities: int = 3
    ):
        super().__init__()
        self.rope_on_cross_attn = rope_on_cross_attn
        self.self_attn = MHAWithRoPE(d_model, n_head, dropout=dropout, num_modalities=num_modalities)
        self.cross_attn = MHAWithRoPE(d_model, n_head, dropout=dropout, num_modalities=1)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model, bias=False),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, tgt_modulated, tgt_base, memory, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # 加入了 tgt_key_padding_mask
        attn_out = self.self_attn(
            x_q=tgt_modulated, x_kv=tgt_modulated, is_causal=False, 
            attn_mask=tgt_key_padding_mask, use_rope=True
        )
        x = tgt_base + self.drop(attn_out)

        # 加入了 memory_key_padding_mask
        cross_out = self.cross_attn(
            x_q=self.norm2(x), x_kv=memory, is_causal=False, 
            attn_mask=memory_key_padding_mask, use_rope=self.rope_on_cross_attn
        )
        x = x + self.drop(cross_out)

        ffn_out = self.ffn(self.norm3(x))
        x = x + self.drop(ffn_out)

        return x


def padmask_to_attn_mask(memory_frame_mask: torch.Tensor):
    """
    memory_frame_mask: [B, Tm], 1=valid, 0=pad (for K/V)
    return additive attn_mask: [B, 1, 1, Tm] (broadcastable to [B,H,T,Tm])
    """
    B, Tm = memory_frame_mask.shape
    device = memory_frame_mask.device
    key_pad = ~memory_frame_mask        # [B, Tm], True=pad
    attn_mask = torch.zeros((B, 1, 1, Tm), device=device, dtype=torch.float32)
    attn_mask = attn_mask.masked_fill(key_pad[:, None, None, :], float("-inf"))
    return attn_mask

def sa_to_attn_mask(frame_mask: torch.Tensor):
    """
    memory_frame_mask: [B, Tm], 1=valid, 0=pad (for K/V)
    return additive attn_mask: [B, 1, 1, Tm] (broadcastable to [B,H,T,Tm])
    """
    attn_mask = (frame_mask.bool())[:, None, None, :]
    # B, Tm = memory_frame_mask.shape
    # device = memory_frame_mask.device
    # key_pad = ~memory_frame_mask        # [B, Tm], True=pad
    # attn_mask = torch.zeros((B, 1, 1, Tm), device=device, dtype=dtype)
    # attn_mask = attn_mask.masked_fill(key_pad[:, None, None, :], float("-inf"))
    return attn_mask

class MultimodalDiffusionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, cond_dim: int, ff_size:int,dropout: float, num_modalities: int = 3):
        super().__init__()
        self.num_modalities = num_modalities
        self.adalns = nn.ModuleList([AdaLN(d_model, embed_dim=cond_dim) for _ in range(num_modalities)])
        # self.adaln_body = AdaLN(d_model, embed_dim=cond_dim)
        # self.adaln_hand = AdaLN(d_model, embed_dim=cond_dim)
        # self.adaln_obj  = AdaLN(d_model, embed_dim=cond_dim)
        
        self.transformer_block = TransformerDecoderLayerRoPE(
            d_model=d_model, n_head=n_head, num_modalities=num_modalities,dim_ff=ff_size,dropout=dropout
        )
    def encode_adaln(self,x,emb_list):
        B, L, D = x.shape
        if self.num_modalities ==1:
            return self.adalns[0](x,emb_list[0])
        else:
            # print(self.adalns[0].device,)
            return torch.stack([self.adalns[i](x[:,i::self.num_modalities],emb_list[i]) for i in range(self.num_modalities)], dim=2).view(B,L,D)
    def forward(self, x_base, text_memory, emb_list,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        B, L, D = x_base.shape
        
        # h_body = self.adaln_body(x_base[:, 0::3, :], emb_body)
        # h_hand = self.adaln_hand(x_base[:, 1::3, :], emb_hand)
        # h_obj  = self.adaln_obj(x_base[:, 2::3, :], emb_obj)
        
        h_modulated = self.encode_adaln(x_base,emb_list)
        
        # h_concat = torch.stack([h_body, h_hand, h_obj], dim=2)
        # h_modulated = h_concat.view(B, L, D)
        
        out = self.transformer_block(
            tgt_modulated=h_modulated,  
            tgt_base=x_base,            
            memory=text_memory,
            tgt_key_padding_mask=tgt_key_padding_mask,       # 透传 tgt mask
            memory_key_padding_mask=memory_key_padding_mask  # 透传 memory mask
        )
        return out


class FullMultimodalDecoder(nn.Module):
    def __init__(self, num_layers=6, d_model=512, n_head=8, cond_dim=256, num_modalities=3,dropout=0.1,ff_size=1024):
        super().__init__()
        self.num_modalities = num_modalities
        # self.hoi_pe = HOIEncoding(d_model, number_per_frames=num_modalities, learnable=True)
        
        self.blocks = nn.ModuleList([
            MultimodalDiffusionBlock(d_model=d_model, n_head=n_head, cond_dim=cond_dim, num_modalities=num_modalities,ff_size = ff_size,dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # self.final_norm = nn.LayerNorm(d_model)

    def merge_te_w_condition(self,te,obj_emb):
        if self.num_modalities ==1:
            return [te+obj_emb[0].unsqueeze(1)+obj_emb[1].unsqueeze(1)]
        elif self.num_modalities == 2:
            return [te[:,0::2]+obj_emb[1].unsqueeze(1),te[:,1::2]+obj_emb[0].unsqueeze(1)]
        else:
            return [te[:,0::3]+obj_emb[1].unsqueeze(1),te[:,1::3]+obj_emb[0].unsqueeze(1),te[:,2::3]+obj_emb[1].unsqueeze(1)]
        
            
            
    def forward(self, x, emb_list, text_memory, 
                frames_mask=None, text_mask=None):
        # B, T, D = x_body.shape
        
        # x_concat = torch.stack([x_body, x_hand, x_obj], dim=2)
        # x = x_concat.view(B, T * self.num_modalities, D)
        # x = self.hoi_pe(x) 
        
       
        frames_mask_used = sa_to_attn_mask(~frames_mask)
        # print(frames_mask_used[0,0,0])
        text_mask_used = sa_to_attn_mask(~text_mask)
        # print(text_mask_used[0,0,0])
        # print(x.shape,self.num_modalities)
        for block in self.blocks:
            x = block(
                x, emb_list, text_memory,
                frames_mask_used, 
                text_mask_used # 
            )
            
        # x = self.final_norm(x)
        
        # out_body = x[:, 0::3, :]
        # out_hand = x[:, 1::3, :]
        # out_obj  = x[:, 2::3, :]
        
        return x # out_body, out_hand, out_obj



if __name__ == "__main__":
    BATCH = 2
    TIME = 16
    D_MODEL = 512
    N_HEAD = 8
    COND_DIM = 256
    TEXT_LEN = 24
    NUM_LAYERS = 2
    
    model = FullMultimodalDiT(num_layers=NUM_LAYERS, d_model=D_MODEL, n_head=N_HEAD, cond_dim=COND_DIM)
    model.eval() 
    
    x_body = torch.randn(BATCH, TIME, D_MODEL)
    x_hand = torch.randn(BATCH, TIME, D_MODEL)
    x_obj  = torch.randn(BATCH, TIME, D_MODEL)
    
    emb_body = torch.randn(BATCH, COND_DIM)
    emb_hand = torch.randn(BATCH, COND_DIM)
    emb_obj  = torch.randn(BATCH, COND_DIM)
    text_memory = torch.randn(BATCH, TEXT_LEN, D_MODEL)
  
    frames_mask = torch.zeros(BATCH, TIME, dtype=torch.bool)
    frames_mask[0, 10:] = True 
    
    # 假设文本 1 长度 20，文本 2 长度 24
    text_mask = torch.zeros(BATCH, TEXT_LEN, dtype=torch.bool)
    text_mask[0, 20:] = True
    
    with torch.no_grad():
        out_b, out_h, out_o = model(
            x_body, x_hand, x_obj, 
            emb_body, emb_hand, emb_obj, 
            text_memory,
            frames_mask=frames_mask,  # 
            text_mask=text_mask       # 
        )
        
   