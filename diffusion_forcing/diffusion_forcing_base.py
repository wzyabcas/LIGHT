"""
This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research 
template [repo](https://github.com/buoyancy99/research-template). 
By its MIT license, you must keep the above sentence in `README.md` 
and the `LICENSE` file to credit the author.
"""
import math
from torch.autograd import Variable
from pytorch3d.transforms import rotation_6d_to_matrix
from utils.eval_t2m_utils import *
from utils.loss_util import masked_l2,diff_l1

from typing import Optional
from tqdm import tqdm
from torch import nn
import random

from omegaconf import DictConfig
import numpy as np
import torch
import torch.nn.functional as F
from typing import Any
from einops import rearrange

# from lightning.pytorch.utilities.types import STEP_OUTPUT


from .diffusion import Diffusion

    
from abc import ABC, abstractmethod
import warnings
from typing import Any, Union, Sequence, Optional

# from lightning.pytorch.utilities.types import STEP_OUTPUT
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
import numpy as np
from PIL import Image
import wandb
import einops

# BasePytorchAlgo

class MomentumBuffer:
    def __init__(self, momentum: float):
        self.momentum = momentum
        self.running_average = 0
    def update(self, update_value: torch.Tensor):
        new_average = self.momentum * self.running_average
        self.running_average = update_value - new_average
class DiffusionForcingBase(nn.Module):
    def __init__(self, cfg, causal=False):
        super().__init__()
        self.cfg = cfg
       
        self.guidance_scale = cfg.guidance_scale
      
        self.causal = causal
        self.h2o = cfg.h2o
        self.hand_split = cfg.hand_split

        self.uncertainty_scale = cfg.uncertainty_scale
        self.timesteps = cfg.timesteps
        self.sampling_timesteps = cfg.sampling_timesteps
        self.clip_noise = cfg.clip_noise


        self.validation_step_outputs = []
        self.bias = cfg.bias
        # self.device = torch.device('cuda:0')
        self.diffusion_model = Diffusion(
            
            is_causal=self.causal,
            cfg=self.cfg,
        )
        self.infer_noise = cfg.infer_noise
        self.chunk_noise = cfg.chunk_noise
        self.t_noise = cfg.t_noise
        self.u_noise = cfg.u_noise
        self.rand = cfg.rand
        self.snr_gamma = cfg.gamma
        self.split_ho = cfg.split_ho
        self.df_delta = cfg.df_delta
        self.df_weight = cfg.df_weight
        self.df_divider = cfg.df_divider
        self.df_upstop = cfg.df_upstop
        self.df_decay = cfg.df_decay
        self.df_begin = cfg.df_begin
        self.df_cfg = cfg.df_cfg
        self.df_mode = cfg.df_mode
        self.df_mom = cfg.df_mom
        self.df_r = cfg.df_r
        self.df_full_mode = cfg.df_full_mode
        self.df_add = cfg.df_add
        self.df_star = cfg.df_star
        self.df_tweight = cfg.df_tweight
        self.df_prob = cfg.df_prob
        self.df_same = cfg.df_same
        self.df_rescale = cfg.df_rescale
        self.df_gw = cfg.df_gw

        self.debug = 0
        if not self.split_ho:
            self.pyramid = self._generate_pyramid_scheduling_matrix_new(300,self.timesteps-1,1)
            self.multi = 1
        elif not self.hand_split:
            self.multi =2 
        else:
            self.multi = 3
        

    def configure_optimizers(self):
        params = tuple(self.diffusion_model.parameters())
        optimizer_dynamics = torch.optim.AdamW(
            params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay, betas=self.cfg.optimizer_beta
        )
        return optimizer_dynamics

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # update params
        optimizer.step(closure=optimizer_closure)

        # manually warm up lr without a scheduler
        if self.trainer.global_step < self.cfg.warmup_steps:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.cfg.warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.cfg.lr

    def training_step(self, model, batch1,batch2, cond,batch3=None,debug=0) :
        
        
        if batch1.shape == batch2.shape:
            noise_levels1,modes = self._generate_noise_levels(batch1,masks = cond['mask'][...,:].squeeze(1).squeeze(1).to(batch1.device))
            noise_levels2 = noise_levels1
            noise_levels3 = noise_levels2
        elif batch3 is None:
            if not self.split_ho:
                MM = cond['mask'][...,:].squeeze(1).squeeze(1).to(batch1.device)
                noise_levels1,modes1= self._generate_noise_levels(batch1,masks = MM)

                noise_levels2,modes2=self._generate_noise_levels(batch2,masks = MM)
                noise_levels3 = noise_levels2
            else:
                noise_levels_all,modes = self._generate_noise_levels(batch1,masks = self.expand_mask(cond['mask'][...,:].squeeze(1).squeeze(1).to(batch1.device)))
                noise_levels_all=noise_levels_all.reshape(batch1.shape[0],-1,2)
                noise_levels1 = noise_levels_all[:,:,0]
                noise_levels2 = noise_levels_all[:,:,1]
                noise_levels3 = noise_levels2
            
        else:
            if not self.split_ho:
                MM = cond['mask'][...,:].squeeze(1).squeeze(1).to(batch1.device)
                noise_levels1= self._generate_noise_levels(batch1,masks = MM)

                noise_levels2=self._generate_noise_levels(batch2,masks = MM)
                noise_levels3=self._generate_noise_levels(batch3,masks = MM)
            else:
                
                noise_levels_all,modes = self._generate_noise_levels(batch1,masks = self.expand_mask(cond['mask'][...,:].squeeze(1).squeeze(1).to(batch1.device)))
                noise_levels_all=noise_levels_all.reshape(batch1.shape[0],-1,3)
                noise_levels1 = noise_levels_all[:,:,0]
                noise_levels2 = noise_levels_all[:,:,1]
                noise_levels3 = noise_levels_all[:,:,2]
                
                
        if debug:
            batch_size = noise_levels1.shape[0]
            num_frames = noise_levels1.shape[1]
            random_batch_vals = torch.randint(1,
                                                self.timesteps,
                                                (batch_size, 1),
                                                device=batch1.device)
            noise_full_diffusion = random_batch_vals.expand(-1, num_frames)  # shape (B, F)
            noise_levels1 = noise_full_diffusion
            noise_levels2 =noise_levels1
            noise_levels3 =noise_levels1
            
        xs_pred1, target1 ,xs_pred2, target2,xs_pred3, target3, loss_weight1,loss_weight2,loss_weight3,reg1,reg2,reg3 = self.diffusion_model(model,batch1,batch2 ,batch3, cond,noise_levels1,noise_levels2,noise_levels3,modes = modes)
        return xs_pred1, target1 ,xs_pred2, target2,xs_pred3, target3, loss_weight1,loss_weight2,loss_weight3,reg1,reg2,reg3
        
    def expand_mask(self,mask):
        B, T = mask.shape
        valid_lens = mask.sum(dim=1)

        idx = torch.arange(self.multi * T).unsqueeze(0).expand(B, -1).to(mask.device)
        return idx < (valid_lens.unsqueeze(1) * self.multi)
    @torch.no_grad()
    def sample_step(self,model, batch1,batch2,batch3, batch_idx,conditions, scheduling_mode='pyramid',namespace="validation",guidance_fn=None,guidance_param=0) :
        if self.df_cfg:
            model.cfg = True
        else:
            model.cfg = False
        
        self.cfg.scheduling_matrix = scheduling_mode
        horizon = batch1.shape[-1]
        batch_size = batch1.shape[0]
        
        if scheduling_mode!='h2o':
            xs_pred = torch.clamp(batch1, -self.clip_noise, self.clip_noise).to(batch1.device)
            zs_pred = torch.clamp(batch3, -self.clip_noise, self.clip_noise).to(batch1.device)
        else:
            xs_pred = batch1
            zs_pred = batch3
        if scheduling_mode!='o2h':
            
            ys_pred = torch.clamp(batch2, -self.clip_noise, self.clip_noise).to(batch2.device)
        else:
            ys_pred = batch2
        xs_pred_orig = xs_pred.clone()
        ys_pred_orig = ys_pred.clone()
        zs_pred_orig = zs_pred.clone()
        scheduling_matrix = self._generate_scheduling_matrix(3*horizon) #B,T
        start_frame = 0
        for m in tqdm(range(scheduling_matrix.shape[0] - 1)):
         
            if self.df_cfg:
                model.cfg=True
            else:
                model.cfg=False
           
            from_noise_levels = scheduling_matrix[m]
            to_noise_levels = scheduling_matrix[m + 1]
            to_noise_levels = to_noise_levels[None,:].repeat(batch_size, axis=0)
            from_noise_levels = from_noise_levels[None,:].repeat(batch_size, axis=0)
            
            xs_pred,ys_pred,zs_pred = self.diffusion_model.sample_step(
                    model,
                    xs_pred[:],
                    ys_pred,
                    zs_pred,
                    conditions,
                    # conditions[start_frame : ],
                    from_noise_levels[start_frame:],
                    to_noise_levels[start_frame:],
                    guidance_fn=guidance_fn
                )
            if scheduling_mode=='h2o':
                zs_pred = zs_pred_orig
                xs_pred = xs_pred_orig
            elif scheduling_mode=='o2h':
                ys_pred = ys_pred_orig
           
        joints = torch.cat([xs_pred[:,:22*3],zs_pred[:,:30*3]],1)
        if xs_pred.shape[1]> 22*3 + 14:
            xs_pred = torch.cat([joints,xs_pred[:,22*3:22*3+3+22*6],zs_pred[:,30*3:],xs_pred[:,22*3+3+22*6:],ys_pred],1)
        elif zs_pred.shape[1]> 30*3:
            xs_pred = torch.cat([joints,xs_pred[:,22*3:],zs_pred[:,30*3:],ys_pred],1)
        else:
            xs_pred = torch.cat([joints,xs_pred[:,22*3:],ys_pred],1)
            
        return xs_pred
    
    
    
    
    
    
    @torch.no_grad()
    def sample_step_new_cfg(self,model, batch1,batch2,batch3, batch_idx,conditions, scheduling_mode='pyramid',namespace="validation",guidance_fn=None,guidance_param=1.5) :
     
        if self.df_mom:
            self.diffusion_model.momentum_buffers=[MomentumBuffer(self.df_mom) for _ in range(3)]
        self.cfg.scheduling_matrix = scheduling_mode
        horizon = batch1.shape[-1]
        batch_size = batch1.shape[0]
        
                
       
            
           
        if scheduling_mode!='h2o':
            xs_pred = torch.clamp(batch1, -self.clip_noise, self.clip_noise).to(batch1.device)
            zs_pred = torch.clamp(batch3, -self.clip_noise, self.clip_noise).to(batch1.device)
        else:
            xs_pred = batch1
            zs_pred = batch3
        if scheduling_mode!='o2h':
            
            ys_pred = torch.clamp(batch2, -self.clip_noise, self.clip_noise).to(batch2.device)
        else:
            ys_pred = batch2
        xs_pred_orig = xs_pred.clone()
        ys_pred_orig = ys_pred.clone()
        zs_pred_orig = zs_pred.clone()
        
        scheduling_matrix = self._generate_scheduling_matrix(3*horizon) #B,T
        start_frame = 0
        endstep = self.df_delta
        
        real_steps = torch.linspace(-1, self.timesteps - 1, steps=self.sampling_timesteps + 1, device=batch1.device).long()
        xs_pred_orig_bad = xs_pred.clone()
        ys_pred_orig_bad = ys_pred.clone()
        zs_pred_orig_bad = zs_pred.clone()
    
        noise_level_bad = 0
        old_lists = []
        # model.cfg = False
        if self.df_cfg:
            model.cfg = True
        else:
            model.cfg=False
        for m in tqdm(range(scheduling_matrix.shape[0] - 1)):
            old_lists.append([xs_pred.clone(),ys_pred.clone(),zs_pred.clone(),xs_pred_orig,ys_pred_orig,zs_pred_orig])
            from_noise_levels = scheduling_matrix[m]
            to_noise_levels = scheduling_matrix[m + 1]
            to_noise_levels = to_noise_levels[None,:].repeat(batch_size, axis=0)
            from_noise_levels = from_noise_levels[None,:].repeat(batch_size, axis=0)
            cfg_weight = 0
            curr_noise_level_cfg_xy = None
            raw_noise = None
            
            

                
            xs_pred,ys_pred,zs_pred = self.diffusion_model.sample_step(
                    model,
                    xs_pred[:],
                    ys_pred,
                    zs_pred,
                    conditions,
                    # conditions[start_frame : ],
                    from_noise_levels[start_frame:],
                    to_noise_levels[start_frame:],
                    guidance_fn=guidance_fn,
                    cfg_weight = cfg_weight,
                    curr_noise_level_cfg = curr_noise_level_cfg_xy,
                    raw_noise = raw_noise,
                    
                    
                )
            if scheduling_mode=='h2o':
                xs_pred = xs_pred_orig
                zs_pred = zs_pred_orig
            elif scheduling_mode=='o2h':
                ys_pred = ys_pred_orig
                
        
        
        
        
    
        old_lists.append([xs_pred.clone(),ys_pred.clone(),zs_pred.clone(),xs_pred_orig,ys_pred_orig,zs_pred_orig])
        
        
        xs_pred2_good = xs_pred_orig.clone()
        ys_pred2_good = ys_pred_orig.clone()
        zs_pred2_good = zs_pred_orig.clone()
        xs_pred2 = xs_pred_orig.clone()
        ys_pred2 = ys_pred_orig.clone()
        zs_pred2 = zs_pred_orig.clone()
        
        scheduling_matrix_for_good = scheduling_matrix[:]
        
        
        for m in tqdm(range(scheduling_matrix.shape[0] - 1)):
            
            #     return xs_pred
            from_noise_levels = scheduling_matrix_for_good[m]
            to_noise_levels = scheduling_matrix_for_good[m + 1]
            to_noise_levels = to_noise_levels[None,:].repeat(batch_size, axis=0)
            from_noise_levels = from_noise_levels[None,:].repeat(batch_size, axis=0)
            
           
            stepper = m
       
           
            if m+self.df_delta<=scheduling_matrix.shape[0] - 3 and stepper<= scheduling_matrix.shape[0] - self.df_upstop and m>=self.df_begin and m%self.df_divider==0: # a
                
                if self.df_decay:
                    cfg_weight = self.df_weight
                    
                    curr_noise_level_cfg_xy = scheduling_matrix[scheduling_matrix.shape[0] - 3]

                    raw_noise = old_lists[scheduling_matrix.shape[0] - 3]
                    model.cfg= False
                    cfg_text = self.df_cfg
                else:
                    if self.df_gw == 1:
                        cfg_weight = 2*self.df_weight*(1-(m-self.df_begin)/(scheduling_matrix.shape[0] - self.df_upstop-self.df_begin-self.df_delta))
                    elif self.df_gw == 2:
                        cfg_weight = 2*self.df_weight*((m-self.df_begin)/(scheduling_matrix.shape[0] - self.df_upstop-self.df_begin-self.df_delta))
                    elif self.df_gw == 0:
                        cfg_weight = self.df_weight
                    index_used = min(m+self.df_delta,scheduling_matrix.shape[0] - 3)
                    curr_noise_level_cfg_xy = scheduling_matrix[index_used]

                    raw_noise = old_lists[index_used]
                  
                    model.cfg= False
                    cfg_text = self.df_cfg
            else:
                cfg_weight = None
                curr_noise_level_cfg_xy = None
                raw_noise = None
                model.cfg= self.df_cfg
                cfg_text = False
            
            xs_pred2_good,ys_pred2_good,zs_pred2_good = self.diffusion_model.sample_step(
                model,
                xs_pred2_good[:],
                ys_pred2_good,
                zs_pred2_good,
                conditions,
                # conditions[start_frame : ],
                from_noise_levels[start_frame:],
                to_noise_levels[start_frame:],
                guidance_fn=guidance_fn,
                cfg_weight = cfg_weight,
                curr_noise_level_cfg = curr_noise_level_cfg_xy,
                raw_noise = raw_noise,
                cfg_text = cfg_text,
                mode = self.df_mode,
                star = self.df_star,
                add = self.df_add,
                tweight = self.df_tweight,
                rescale = self.df_rescale
                    
            )
            if scheduling_mode=='h2o':
                xs_pred2_good = xs_pred_orig
                zs_pred2_good = zs_pred_orig
            elif scheduling_mode=='o2h':
                ys_pred2_good = ys_pred_orig
            continue
                
                
                

        joints = torch.cat([xs_pred2_good[:,:22*3],zs_pred2_good[:,:30*3]],1)
        
        
        if xs_pred.shape[1]> 22*3 + 14:
            xs_pred = torch.cat([joints,xs_pred2_good[:,22*3:22*3+3+22*6],zs_pred2_good[:,30*3:],xs_pred2_good[:,22*3+3+22*6:],ys_pred2_good],1)
        elif zs_pred.shape[1]> 30*3:
            xs_pred = torch.cat([joints,xs_pred2_good[:,22*3:],zs_pred2_good[:,30*3:],ys_pred2_good],1)
        else:
            xs_pred = torch.cat([joints,xs_pred2_good[:,22*3:],ys_pred2_good],1)
        return xs_pred
    
   
    
    def test_step(self, *args: Any, **kwargs: Any) :
        return self.validation_step(*args, **kwargs, namespace="test")

    
    
    def test_epoch_end(self) -> None:
        self.on_validation_epoch_end(namespace="test")

    def _generate_noise_levels(self, xs: torch.Tensor, masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate noise levels for training.
        """
        modes = None
        rand_float = random.random()
 
        if self.rand:
            batch_size, _, _ , num_frames = xs.shape
            num_frames = num_frames * self.multi
    
            
           
            noise_full_random = torch.randint(
                1, self.timesteps, 
                (batch_size, num_frames), 
                device=xs.device
            )

            rand_t = torch.randint(1, self.timesteps, (batch_size, 1), device=xs.device)
            noise_full_diffusion = rand_t.expand(-1, num_frames)

           
            chunk = num_frames // 3
            noise_chunk = torch.randint(1, self.timesteps, (batch_size, chunk), device=xs.device)
            # shape → (batch, chunk, 1) → repeat out to 3 channels → flatten back to frames
            noise_repeated_t = (
                noise_chunk.unsqueeze(2)
                        .repeat(1, 1, 3)
                        .reshape(batch_size, -1)
            )
            self.df_same = 0
            
            # 4) Per-object constant: each of the 3 objects gets its own single t, same over time
            # if not self.df_same:
            rand1 = torch.randint(1, self.timesteps, (batch_size, 1), device=xs.device)
            rand2 = torch.randint(1, self.timesteps, (batch_size, 1), device=xs.device)
            rand3 = torch.randint(1, self.timesteps, (batch_size, 1), device=xs.device)
            noise_mod1 = rand1.repeat(1, chunk).unsqueeze(2)
            noise_mod2 = rand2.repeat(1, chunk).unsqueeze(2)
            noise_mod3 = rand3.repeat(1, chunk).unsqueeze(2)
            noise_per_object = torch.cat([noise_mod1, noise_mod2, noise_mod3], dim=-1).reshape(batch_size,-1)
            
                
               ## Here: What we use: Asychronous denosing for each modality
            noise_levels = noise_per_object
                    
            
                
                    
                    

                
                
                

            
        elif rand_float>self.infer_noise:
            batch_size, _, _ , num_frames = xs.shape
            num_frames = num_frames * self.multi
            O = self.pyramid
            indices = np.random.choice(O.shape[0], size=batch_size, replace=False)
            noise_levels = torch.tensor(O[indices]).long().to(xs.device)
            # a=1
        elif self.chunk_noise == 1:
            batch_size, _, _ , num_frames = xs.shape
            num_frames = num_frames * self.multi
            # match self.cfg.noise_level:
            #     case "random_all":  # entirely random noise levels
            noise_levels = torch.randint(0, self.timesteps, (batch_size, num_frames), device=xs.device)
        else:
            batch_size, _, _ , num_frames = xs.shape
            num_frames = num_frames * self.multi
            H = self.chunk_noise
            num_chunks = math.ceil(num_frames / H)
            random_chunks = torch.randint(0, self.timesteps, (batch_size, num_chunks), device=xs.device)
            noise_levels = random_chunks.repeat_interleave(H, dim=1)[:, :num_frames]
        if self.t_noise:
            batch_size, _, _ , num_frames = xs.shape
            num_frames = num_frames * self.multi
            noise_levels = noise_levels[:1].repeat(batch_size, 1)

        
        if masks is not None:
            # mask B,T: True non-pad
            # for frames that are not available, treat as full noise
            # discard = torch.all(~rearrange(masks.bool(), "(t fs) b -> t b fs", fs=self.frame_stack), -1)
            noise_levels = torch.where((~masks.bool()), torch.full_like(noise_levels, self.timesteps - 1).to(noise_levels.device), noise_levels)

        return noise_levels,modes
    def _generate_pyramid_scheduling_matrix_new(self, horizon, sampling_timesteps,uncertainty_scale):
        height = sampling_timesteps + int((horizon - 1) * uncertainty_scale) + 1
        
        scheduling_matrix = np.zeros((height, horizon), dtype=np.int64)
        for m in range(height):
            for t in range(horizon):
                scheduling_matrix[m, t] = sampling_timesteps + int(t * uncertainty_scale) - m

        return np.clip(scheduling_matrix, 1, sampling_timesteps)

    def _generate_scheduling_matrix(self, horizon,mask=None):
        # if self.cfg.scheduling_matrix == "pyramid":
        #     matrix =  self._generate_pyramid_scheduling_matrix(horizon, self.uncertainty_scale)
        if "pyramid" in self.cfg.scheduling_matrix:
            matrix =  self._generate_pyramid_scheduling_matrix(horizon//self.multi, self.uncertainty_scale)
            matrix = np.repeat(np.expand_dims(matrix, axis=-1),repeats=self.multi, axis=2).reshape(matrix.shape[0],-1)
        ## hasn't done 
        elif self.cfg.scheduling_matrix == "of":
            first_matrix =  np.arange(self.sampling_timesteps, -1, -1)[:, None].repeat(horizon//self.multi, axis=1)
            matrix = np.zeros((2*self.sampling_timesteps+1,horizon)).astype(np.int64)
            if self.multi == 2:
                matrix[:first_matrix.shape[0],1::2] = first_matrix
                matrix[:first_matrix.shape[0],0::2] = self.sampling_timesteps
                matrix[first_matrix.shape[0]:,0::2] = first_matrix[1:]
            elif self.multi == 3:
                matrix[:first_matrix.shape[0],1::3] = first_matrix
                matrix[:first_matrix.shape[0],0::3] = self.sampling_timesteps
                matrix[:first_matrix.shape[0],2::3] = self.sampling_timesteps
                matrix[first_matrix.shape[0]:,0::3] = first_matrix[1:]
                matrix[first_matrix.shape[0]:,2::3] = first_matrix[1:]
                

        elif self.cfg.scheduling_matrix == "hf":
            first_matrix =  np.arange(self.sampling_timesteps, -1, -1)[:, None].repeat(horizon//self.multi, axis=1)
            matrix = np.zeros((2*self.sampling_timesteps+1,horizon)).astype(np.int64)
            if self.multi == 2:
                matrix[:first_matrix.shape[0],0::2] = first_matrix
                matrix[:first_matrix.shape[0],1::2] = self.sampling_timesteps
                matrix[first_matrix.shape[0]:,1::2] = first_matrix[1:]
            elif self.multi == 3:
                matrix[:first_matrix.shape[0],0::3] = first_matrix
                matrix[:first_matrix.shape[0],2::3] = first_matrix
                matrix[:first_matrix.shape[0],1::3] = self.sampling_timesteps
                # matrix[:first_matrix.shape[0],2::3] = self.sampling_timesteps
                # matrix[first_matrix.shape[0]:,0::3] = first_matrix[1:]
                matrix[first_matrix.shape[0]:,1::3] = first_matrix[1:]
            
            
        elif self.cfg.scheduling_matrix == "ho_double":
        
            matrix =  self._generate_pyramid_scheduling_matrix(horizon//2, self.uncertainty_scale)
            matrix = np.repeat(np.expand_dims(matrix, axis=-1),repeats=2, axis=2).reshape(matrix.shape[0],-1)
        
        elif self.cfg.scheduling_matrix == "full_sequence":
       
            matrix =  np.arange(self.sampling_timesteps, -1, -1)[:, None].repeat(horizon, axis=1)
        elif self.cfg.scheduling_matrix == "autoregressive":
            matrix =  self._generate_pyramid_scheduling_matrix(horizon, self.sampling_timesteps)
        elif self.cfg.scheduling_matrix == "trapezoid":
            matrix =  self._generate_trapezoid_scheduling_matrix(horizon, self.uncertainty_scale)
        elif self.cfg.scheduling_matrix == 'h2o':
           
            # np.arange(self.sampling_timesteps, 0, -1)
            if self.multi == 2:
                matrix = 2*np.ones((self.sampling_timesteps+1, horizon), dtype=int)
                matrix = matrix.astype(np.long)
                values = np.arange(self.sampling_timesteps , -1, -1)[:, None]

                
                matrix[:, 1::2] = values.repeat(horizon//2, axis=1)
            elif self.multi == 3:
                matrix = np.ones((self.sampling_timesteps+1, horizon), dtype=int)*2
                matrix = matrix.astype(np.long)
                values = np.arange(self.sampling_timesteps , -1, -1)[:, None]

                
                matrix[:, 1::3] = values.repeat(horizon//3, axis=1)
            # o2h = np.zeros((self.timesteps - 1, 600), dtype=int)
            # o2h[:, 0::2] = values.repeat(300, axis=1)
            # insert
            
        elif self.cfg.scheduling_matrix == 'o2h': # stupid
            if self.multi == 2:
                matrix = np.ones((self.sampling_timesteps+1, horizon), dtype=int)*1
                matrix = matrix.astype(np.int32)
                values = np.arange(self.sampling_timesteps , -1, -1)[:, None]

                
                matrix[:, 0::2] = values.repeat(horizon//2, axis=1)
            elif self.multi == 3:
                matrix = np.ones((self.sampling_timesteps+1, horizon), dtype=int)*2
                matrix = matrix.astype(np.int32)
               
                values = np.arange(self.sampling_timesteps , -1, -1)[:, None]

                
                matrix[:, 0::3] = values.repeat(horizon//3, axis=1)
                matrix[:, 2::3] = values.repeat(horizon//3, axis=1)
                
        
        return matrix # T,L
    

    def _generate_pyramid_scheduling_matrix(self, horizon: int, uncertainty_scale: float):
        height = self.sampling_timesteps + int((horizon - 1) * uncertainty_scale) + 1
        scheduling_matrix = np.zeros((height, horizon), dtype=np.int64)
        for m in range(height):
            for t in range(horizon):
                scheduling_matrix[m, t] = self.sampling_timesteps + int(t * uncertainty_scale) - m

        return np.clip(scheduling_matrix, 0, self.sampling_timesteps)

    def _generate_trapezoid_scheduling_matrix(self, horizon: int, uncertainty_scale: float):
        height = self.sampling_timesteps + int((horizon + 1) // 2 * uncertainty_scale)
        scheduling_matrix = np.zeros((height, horizon), dtype=np.int64)
        for m in range(height):
            for t in range((horizon + 1) // 2):
                scheduling_matrix[m, t] = self.sampling_timesteps + int(t * uncertainty_scale) - m
                scheduling_matrix[m, -t] = self.sampling_timesteps + int(t * uncertainty_scale) - m

        return np.clip(scheduling_matrix, 0, self.sampling_timesteps)

    def reweight_loss(self, loss, weight=None):
        # Note there is another part of loss reweighting (fused_snr) inside the Diffusion class!
        loss = rearrange(loss, "t b (fs c) ... -> t b fs c ...", fs=self.frame_stack)
        if weight is not None:
            expand_dim = len(loss.shape) - len(weight.shape) - 1
            weight = rearrange(
                weight,
                "(t fs) b ... -> t b fs ..." + " 1" * expand_dim,
                fs=self.frame_stack,
            )
            loss = loss * weight

        return loss.mean()

    def _preprocess_batch(self, batch):
        xs = batch[0]
        batch_size, n_frames = xs.shape[:2]

        if n_frames % self.frame_stack != 0:
            raise ValueError("Number of frames must be divisible by frame stack size")
        if self.context_frames % self.frame_stack != 0:
            raise ValueError("Number of context frames must be divisible by frame stack size")

        masks = torch.ones(n_frames, batch_size).to(xs.device)
        n_frames = n_frames // self.frame_stack

        if self.external_cond_dim:
            conditions = batch[1]
            conditions = torch.cat([torch.zeros_like(conditions[:, :1]), conditions[:, 1:]], 1)
            conditions = rearrange(conditions, "b (t fs) d -> t b (fs d)", fs=self.frame_stack).contiguous()
        else:
            conditions = [None for _ in range(n_frames)]

        xs = self._normalize_x(xs)
        xs = rearrange(xs, "b (t fs) c ... -> t b (fs c) ...", fs=self.frame_stack).contiguous()

        return xs, conditions, masks

    def _normalize_x(self, xs):
        shape = [1] * (xs.ndim - self.data_mean.ndim) + list(self.data_mean.shape)
        mean = self.data_mean.reshape(shape)
        std = self.data_std.reshape(shape)
        return (xs - mean) / std

    def _unnormalize_x(self, xs):
        shape = [1] * (xs.ndim - self.data_mean.ndim) + list(self.data_mean.shape)
        mean = self.data_mean.reshape(shape)
        std = self.data_std.reshape(shape)
        return xs * std + mean

    def _unstack_and_unnormalize(self, xs):
        xs = rearrange(xs, "t b (fs c) ... -> (t fs) b c ...", fs=self.frame_stack)
        return self._unnormalize_x(xs)
    



class BasePytorchAlgo(ABC):
    """
    A base class for Pytorch algorithms using Pytorch Lightning.
    See https://lightning.ai/docs/pytorch/stable/starter/introduction.html for more details.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.debug = self.cfg.debug
        self._build_model()

    @abstractmethod
    def _build_model(self):
        """
        Create all pytorch nn.Modules here.
        """
        raise NotImplementedError

    @abstractmethod
    def training_step(self, *args: Any, **kwargs: Any) :
        r"""Here you compute and return the training loss and some additional metrics for e.g. the progress bar or
        logger.

        Args:
            batch: The output of your data iterable, normally a :class:`~torch.utils.data.DataLoader`.
            batch_idx: The index of this batch.
            dataloader_idx: (only if multiple dataloaders used) The index of the dataloader that produced this batch.

        Return:
            Any of these options:
            - :class:`~torch.Tensor` - The loss tensor
            - ``dict`` - A dictionary. Can include any keys, but must include the key ``'loss'``.
            - ``None`` - Skip to the next batch. This is only supported for automatic optimization.
                This is not supported for multi-GPU, TPU, IPU, or DeepSpeed.

        In this step you'd normally do the forward pass and calculate the loss for a batch.
        You can also do fancier things like multiple forward passes or something model specific.

        Example::

            def training_step(self, batch, batch_idx):
                x, y, z = batch
                out = self.encoder(x)
                loss = self.loss(out, x)
                return loss

        To use multiple optimizers, you can switch to 'manual optimization' and control their stepping:

        .. code-block:: python

            def __init__(self):
                super().__init__()
                self.automatic_optimization = False


            # Multiple optimizers (e.g.: GANs)
            def training_step(self, batch, batch_idx):
                opt1, opt2 = self.optimizers()

                # do training_step with encoder
                ...
                opt1.step()
                # do training_step with decoder
                ...
                opt2.step()

        Note:
            When ``accumulate_grad_batches`` > 1, the loss returned here will be automatically
            normalized by ``accumulate_grad_batches`` internally.

        """
        return super().training_step(*args, **kwargs)

    def configure_optimizers(self):
        """
        Return an optimizer. If you need to use more than one optimizer, refer to pytorch lightning documentation:
        https://lightning.ai/docs/pytorch/stable/common/optimization.html
        """
        parameters = self.parameters()
        return torch.optim.Adam(parameters, lr=self.cfg.lr)

    def log_video(
        self,
        key: str,
        video: Union[np.ndarray, torch.Tensor],
        mean: Union[np.ndarray, torch.Tensor, Sequence, float] = None,
        std: Union[np.ndarray, torch.Tensor, Sequence, float] = None,
        fps: int = 12,
        format: str = "mp4",
    ):
        """
        Log video to wandb. WandbLogger in pytorch lightning does not support video logging yet, so we call wandb directly.

        Args:
            video: a numpy array or tensor, either in form (time, channel, height, width) or in the form
                (batch, time, channel, height, width). The content must be be in 0-255 if under dtype uint8
                or [0, 1] otherwise.
            mean: optional, the mean to unnormalize video tensor, assuming unnormalized data is in [0, 1].
            std: optional, the std to unnormalize video tensor, assuming unnormalized data is in [0, 1].
            key: the name of the video.
            fps: the frame rate of the video.
            format: the format of the video. Can be either "mp4" or "gif".
        """

        if isinstance(video, torch.Tensor):
            video = video.detach().cpu().numpy()

        expand_shape = [1] * (len(video.shape) - 2) + [3, 1, 1]
        if std is not None:
            if isinstance(std, (float, int)):
                std = [std] * 3
            if isinstance(std, torch.Tensor):
                std = std.detach().cpu().numpy()
            std = np.array(std).reshape(*expand_shape)
            video = video * std
        if mean is not None:
            if isinstance(mean, (float, int)):
                mean = [mean] * 3
            if isinstance(mean, torch.Tensor):
                mean = mean.detach().cpu().numpy()
            mean = np.array(mean).reshape(*expand_shape)
            video = video + mean

        if video.dtype != np.uint8:
            video = np.clip(video, a_min=0, a_max=1) * 255
            video = video.astype(np.uint8)

        self.logger.experiment.log(
            {
                key: wandb.Video(video, fps=fps, format=format),
            },
            step=self.global_step,
        )

    def log_image(
        self,
        key: str,
        image: Union[np.ndarray, torch.Tensor, Image.Image, Sequence[Image.Image]],
        mean: Union[np.ndarray, torch.Tensor, Sequence, float] = None,
        std: Union[np.ndarray, torch.Tensor, Sequence, float] = None,
        **kwargs: Any,
    ):
        """
        Log image(s) using WandbLogger.
        Args:
            key: the name of the video.
            image: a single image or a batch of images. If a batch of images, the shape should be (batch, channel, height, width).
            mean: optional, the mean to unnormalize image tensor, assuming unnormalized data is in [0, 1].
            std: optional, the std to unnormalize tensor, assuming unnormalized data is in [0, 1].
            kwargs: optional, WandbLogger log_image kwargs, such as captions=xxx.
        """
        if isinstance(image, Image.Image):
            image = [image]
        elif len(image) and not isinstance(image[0], Image.Image):
            if isinstance(image, torch.Tensor):
                image = image.detach().cpu().numpy()

            if len(image.shape) == 3:
                image = image[None]

            if image.shape[1] == 3:
                if image.shape[-1] == 3:
                    warnings.warn(f"Two channels in shape {image.shape} have size 3, assuming channel first.")
                image = einops.rearrange(image, "b c h w -> b h w c")

            if std is not None:
                if isinstance(std, (float, int)):
                    std = [std] * 3
                if isinstance(std, torch.Tensor):
                    std = std.detach().cpu().numpy()
                std = np.array(std)[None, None, None]
                image = image * std
            if mean is not None:
                if isinstance(mean, (float, int)):
                    mean = [mean] * 3
                if isinstance(mean, torch.Tensor):
                    mean = mean.detach().cpu().numpy()
                mean = np.array(mean)[None, None, None]
                image = image + mean

            if image.dtype != np.uint8:
                image = np.clip(image, a_min=0.0, a_max=1.0) * 255
                image = image.astype(np.uint8)
                image = [img for img in image]

        self.logger.log_image(key=key, images=image, **kwargs)

    def log_gradient_stats(self):
        """Log gradient statistics such as the mean or std of norm."""

        with torch.no_grad():
            grad_norms = []
            gpr = []  # gradient-to-parameter ratio
            for param in self.parameters():
                if param.grad is not None:
                    grad_norms.append(torch.norm(param.grad).item())
                    gpr.append(torch.norm(param.grad) / torch.norm(param))
            if len(grad_norms) == 0:
                return
            grad_norms = torch.tensor(grad_norms)
            gpr = torch.tensor(gpr)
            self.log_dict(
                {
                    "train/grad_norm/min": grad_norms.min(),
                    "train/grad_norm/max": grad_norms.max(),
                    "train/grad_norm/std": grad_norms.std(),
                    "train/grad_norm/mean": grad_norms.mean(),
                    "train/grad_norm/median": torch.median(grad_norms),
                    "train/gpr/min": gpr.min(),
                    "train/gpr/max": gpr.max(),
                    "train/gpr/std": gpr.std(),
                    "train/gpr/mean": gpr.mean(),
                    "train/gpr/median": torch.median(gpr),
                }
            )

    def register_data_mean_std(
        self, mean: Union[str, float, Sequence], std: Union[str, float, Sequence], namespace: str = "data"
    ):
        """
        Register mean and std of data as tensor buffer.

        Args:
            mean: the mean of data.
            std: the std of data.
            namespace: the namespace of the registered buffer.
        """
        for k, v in [("mean", mean), ("std", std)]:
            if isinstance(v, str):
                if v.endswith(".npy"):
                    v = torch.from_numpy(np.load(v))
                elif v.endswith(".pt"):
                    v = torch.load(v)
                else:
                    raise ValueError(f"Unsupported file type {v.split('.')[-1]}.")
            else:
                v = torch.tensor(v)
            self.register_buffer(f"{namespace}_{k}", v.float().to(self.device))

class BaseAlgo(ABC):
    """
    A base class for generic algorithms.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.debug = self.cfg.debug

    @abstractmethod
    def run(*args: Any, **kwargs: Any) -> Any:
        """
        Run the algorithm.
        """
        raise NotImplementedError
