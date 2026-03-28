from typing import Optional, Callable
from collections import namedtuple
from omegaconf import DictConfig
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
import numpy as np
from copy import deepcopy

# from .unet3d import Unet3D
# from .transformer import Transformer
from .utils import linear_beta_schedule, cosine_beta_schedule, sigmoid_beta_schedule, extract, EinopsWrapper

ModelPrediction = namedtuple("ModelPrediction", ["pred_noise1", "pred_x_start1", "model_out1","pred_noise2", "pred_x_start2", "model_out2","pred_noise3", "pred_x_start3", "model_out3"])


class Diffusion(nn.Module):
    # Special thanks to lucidrains for the implementation of the base Diffusion model
    # https://github.com/lucidrains/denoising-diffusion-pytorch

    def __init__(
        self,
        is_causal: bool,
        cfg: DictConfig,
    ):
        super().__init__()
        self.cfg = cfg
        # x_shape: torch.Size,
        # external_cond_dim: int,
        # # self.x_shape = x_shape
        # self.external_cond_dim = external_cond_dim
        self.timesteps = cfg.timesteps
        self.sampling_timesteps = cfg.sampling_timesteps
        self.beta_schedule = cfg.beta_schedule
        self.schedule_fn_kwargs = cfg.schedule_fn_kwargs
        self.objective = cfg.objective
        self.use_fused_snr = cfg.use_fused_snr
        self.rand = cfg.rand
        self.momentum_buffers = None
        # self.snr_gamma = cfg.snr_gamma
        self.snr_clip = cfg.snr_clip
        self.cum_snr_decay = cfg.cum_snr_decay
        self.ddim_sampling_eta = cfg.ddim_sampling_eta
        self.clip_noise = cfg.clip_noise
        # self.arch = cfg.architecture
        self.stabilization_level = cfg.stabilization_level
        self.is_causal = is_causal
        self.df_divider = cfg.df_divider
        self.df_upstop = cfg.df_upstop
        self.df_weight = cfg.df_weight
        self.df_r = cfg.df_r
        self.df_mom = cfg.df_mom

        # self._build_model()
        self._build_buffer()

    

    def _build_buffer(self):
        if self.beta_schedule == "linear":
            beta_schedule_fn = linear_beta_schedule
        elif self.beta_schedule == "cosine":
            beta_schedule_fn = cosine_beta_schedule
        elif self.beta_schedule == "sigmoid":
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f"unknown beta schedule {self.beta_schedule}")

        betas = beta_schedule_fn(self.timesteps, **self.schedule_fn_kwargs)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # sampling related parameters
        assert self.sampling_timesteps <= self.timesteps
        self.is_ddim_sampling = True
        # self.sampling_timesteps < self.timesteps

        # self.is_ddim_sampling = self.sampling_timesteps < self.timesteps

        # helper function to register buffer from float64 to float32
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer("posterior_variance", posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # calculate p2 reweighting

        # register_buffer(
        #     "p2_loss_weight",
        #     (self.p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod))
        #     ** -self.p2_loss_weight_gamma,
        # )

        # derive loss weight
        # https://arxiv.org/abs/2303.09556
        # snr: signal noise ratio
        snr = alphas_cumprod / (1 - alphas_cumprod+1e-6)
        clipped_snr = snr.clone()
        clipped_snr.clamp_(max=self.snr_clip)

        register_buffer("clipped_snr", clipped_snr)
        register_buffer("snr", snr)

    def add_shape_channels(self, x):
        return x.unsqueeze(1).unsqueeze(1)
        # return rearrange(x, f"... -> ...{' 1' * len(self.x_shape)}")

    def model_predictions(self, model,x,y,z, t, external_cond=None,modes=None):
        
        model_output = model(x,  t, external_cond, is_causal=self.is_causal,x2=y,x3=z,modes=modes)
        X_shape = x.shape[1]
        if self.objective == "pred_noise":
            if y is not None and z is None:  ## Two None
                model_output1 = model_output[:,:X_shape]
                pred_noise1 = model_output1
                pred_noise1 = torch.clamp(pred_noise1, -self.clip_noise, self.clip_noise)
                
                x_start1 = self.predict_start_from_noise(x, t[...,0], pred_noise1)
                
                model_output2 = model_output[:,X_shape:]
                pred_noise2 = model_output2
                pred_noise2 = torch.clamp(pred_noise2, -self.clip_noise, self.clip_noise)
                
                x_start2 = self.predict_start_from_noise(y, t[...,1], pred_noise2)
                x_start3 = x_start2
                model_output3 = model_output2
                pred_noise3 = pred_noise2
                
            elif y is None: ## One
                pred_noise1 = model_output
                pred_noise1 = torch.clamp(model_output, -self.clip_noise, self.clip_noise)
                
                x_start1 = self.predict_start_from_noise(x, t, pred_noise1)
                x_start2 = x_start1
                pred_noise2 = pred_noise1
                model_output1 = model_output
                model_output2 = model_output
                
                x_start3 = x_start2
                model_output3 = model_output2
                pred_noise3 = pred_noise2
            else: ## One
                z_shape = z.shape[1]
                model_output1 = model_output[:,:X_shape]
                pred_noise1 = model_output1
                pred_noise1 = torch.clamp(pred_noise1, -self.clip_noise, self.clip_noise)
                
                x_start1 = self.predict_start_from_noise(x, t[...,0], pred_noise1)
                
                model_output2 = model_output[:,X_shape:-z_shape]
                pred_noise2 = model_output2
                pred_noise2 = torch.clamp(pred_noise2, -self.clip_noise, self.clip_noise)
                
                x_start2 = self.predict_start_from_noise(y, t[...,1], pred_noise2)
                
                model_output3 = model_output[:,-z_shape:]
                pred_noise3 = model_output3
                pred_noise3 = torch.clamp(pred_noise3, -self.clip_noise, self.clip_noise)
                
                x_start3 = self.predict_start_from_noise(z, t[...,2], pred_noise3)
        elif self.objective == "pred_x0":
            if y is not None and z is None:
                model_output1 = model_output[:,:X_shape]
                model_output2 = model_output[:,X_shape:]

                x_start1 = model_output1
                pred_noise1 = self.predict_noise_from_start(x, t[...,0], x_start1)
                
                x_start2 = model_output2
                pred_noise2 = self.predict_noise_from_start(y, t[...,1], x_start2)
                
                x_start3 = x_start2
                model_output3 = model_output2
                pred_noise3 = pred_noise2
            elif y is None :
                
                x_start1 = model_output
                pred_noise1 = self.predict_noise_from_start(x, t, x_start1)
                x_start2 = x_start1
                pred_noise2 = pred_noise1
                
                model_output1 = model_output
                model_output2 = model_output
                
                x_start3 = x_start2
                model_output3 = model_output2
                pred_noise3 = pred_noise2
            else:
                z_shape = z.shape[1]
                model_output1 = model_output[:,:X_shape]
                model_output2 = model_output[:,X_shape:-z_shape]
                model_output3 = model_output[:,-z_shape:]

                x_start1 = model_output1
                pred_noise1 = self.predict_noise_from_start(x, t[...,0], x_start1)
                
                x_start2 = model_output2
                pred_noise2 = self.predict_noise_from_start(y, t[...,1], x_start2)
                
                x_start3 = model_output3
                pred_noise3 = self.predict_noise_from_start(z, t[...,2], x_start3)
                
                

        elif self.objective == "pred_v":
            if y is not None and z is None:
                model_output1 = model_output[:,:X_shape]
                model_output2 = model_output[:,X_shape:]
                
                v = model_output1
                x_start1 = self.predict_start_from_v(x, t[...,0], v)
                pred_noise1 = self.predict_noise_from_start(x, t[...,0], x_start1)
                
                v2 = model_output2
                x_start2 = self.predict_start_from_v(y, t[...,1], v2)
                pred_noise2 = self.predict_noise_from_start(y, t[...,1], x_start2)
                
                x_start3 = x_start2
                model_output3 = model_output2
                pred_noise3 = pred_noise2
                
            
            elif y is None:    
                v = model_output
                x_start1 = self.predict_start_from_v(x, t, v)
                pred_noise1 = self.predict_noise_from_start(x, t, x_start1)
                
                x_start2 = x_start1
                pred_noise2 = pred_noise1
                
                model_output1 = model_output
                model_output2 = model_output
                
                x_start3 = x_start2
                model_output3 = model_output2
                pred_noise3 = pred_noise2
            else:
                z_shape = z.shape[1]
                model_output1 = model_output[:,:X_shape]
                model_output2 = model_output[:,X_shape:-z_shape]
                model_output3 = model_output[:,-z_shape:]
                
                v = model_output1
                x_start1 = self.predict_start_from_v(x, t[...,0], v)
                pred_noise1 = self.predict_noise_from_start(x, t[...,0], x_start1)
                
                v2 = model_output2
                x_start2 = self.predict_start_from_v(y, t[...,1], v2)
                pred_noise2 = self.predict_noise_from_start(y, t[...,1], x_start2)
                
                v3 = model_output3
                x_start3 = self.predict_start_from_v(z, t[...,2], v3)
                pred_noise3 = self.predict_noise_from_start(z, t[...,2], x_start3)
                

        return ModelPrediction(pred_noise1, x_start1, model_output1,pred_noise2, x_start2, model_output2,pred_noise3, x_start3, model_output3)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / extract(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
            noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)
       
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_mean_variance(self, model,x, t, external_cond=None):
        model_pred = self.model_predictions(model=model,x=x, t=t, external_cond=external_cond)
        x_start = model_pred.pred_x_start
        return self.q_posterior(x_start=x_start, x_t=x, t=t)

    def compute_loss_weights(self, noise_levels: torch.Tensor):

        snr = self.snr[noise_levels]
        clipped_snr = self.clipped_snr[noise_levels]
        normalized_clipped_snr = clipped_snr / self.snr_clip
        normalized_snr = snr / self.snr_clip

        if not self.use_fused_snr:
            if self.objective == "pred_noise":
                return clipped_snr / snr
            elif self.objective == "pred_x0":
                # weights = clipped_snr / (snr+1e-6)
                # weights = clipped_snr / (snr+1e-6)
                # weights[snr == 0 ] =1
                return clipped_snr , noise_levels # 1- noise_levels/self.timesteps
                # weights[snr ==0] = 1
                # return weights # update
            elif self.objective == "pred_v":
                return clipped_snr / (snr + 1)
            else:
                raise ValueError("Unknown objective")

            # min SNR reweighting
            # match self.objective:
            #     case "pred_noise":
            #         return clipped_snr / snr
            #     case "pred_x0":
            #         return clipped_snr
            #     case "pred_v":
            #         return clipped_snr / (snr + 1)

        cum_snr = torch.zeros_like(normalized_snr)
        for t in range(0, noise_levels.shape[0]):
            if t == 0:
                cum_snr[t] = normalized_clipped_snr[t]
            else:
                cum_snr[t] = self.cum_snr_decay * cum_snr[t - 1] + (1 - self.cum_snr_decay) * normalized_clipped_snr[t]

        cum_snr = F.pad(cum_snr[:-1], (0, 0, 1, 0), value=0.0)
        clipped_fused_snr = 1 - (1 - cum_snr * self.cum_snr_decay) * (1 - normalized_clipped_snr)
        fused_snr = 1 - (1 - cum_snr * self.cum_snr_decay) * (1 - normalized_snr)

        
        if self.objective == "pred_noise":
            return clipped_fused_snr / fused_snr
        elif self.objective == "pred_x0":
            return clipped_fused_snr * self.snr_clip
        elif self.objective == "pred_v":
            return clipped_fused_snr * self.snr_clip / (fused_snr * self.snr_clip + 1)
        # else:
        #     raise ValueError(f"unknown objective {self.objective}")

        # match self.objective:
        #     case "pred_noise":
        #         return clipped_fused_snr / fused_snr
        #     case "pred_x0":
        #         return clipped_fused_snr * self.snr_clip
        #     case "pred_v":
        #         return clipped_fused_snr * self.snr_clip / (fused_snr * self.snr_clip + 1)
        #     case _:
        #         raise ValueError(f"unknown objective {self.objective}")

    def forward(
        self,
        model,
        x,
        y,
        z,
        external_cond,
        noise_levels_x,
        noise_levels_y,
        noise_levels_z,
        modes = None
    ):
        if x.shape == y.shape: 
            noise = torch.randn_like(x)
            noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)

            noised_x = self.q_sample(x_start=x, t=noise_levels_x, noise=noise)
            model_pred = self.model_predictions(model=model,x=noised_x,y=None,z=None, t=noise_levels_x, external_cond=external_cond,modes = modes)

            pred = model_pred.model_out1
            x_pred = model_pred.pred_x_start1

            if self.objective == "pred_noise":
                target = noise
            elif self.objective == "pred_x0":
                target = x
            elif self.objective == "pred_v":
                target = self.predict_v(x, noise_levels_x, noise)
            else:
                raise ValueError(f"unknown objective {self.objective}")

            # loss = F.mse_loss(pred, target.detach(), reduction="none")
            loss_weight , reg_schedule = self.compute_loss_weights(noise_levels_x)
            
            return pred,target.detach(),None,None,None,None,loss_weight,None,None,reg_schedule,None,None
        elif z is None:
            noise = torch.randn_like(x)
            noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)
            noised_x = self.q_sample(x_start=x, t=noise_levels_x, noise=noise)
            # model_pred = self.model_predictions(model=model,x=noised_x,y=noised_x t=noise_levels_x, external_cond=external_cond)
            
            noise2 = torch.randn_like(y)
            
            noise2 = torch.clamp(noise2, -self.clip_noise, self.clip_noise)
            noised_y = self.q_sample(x_start=y, t=noise_levels_y, noise=noise2)
            
            noise_levels_all = torch.cat([noise_levels_x.unsqueeze(2),noise_levels_y.unsqueeze(2)],-1)
            model_pred = self.model_predictions(model=model,x=noised_x, y=noised_y,z=None, t=noise_levels_all, external_cond=external_cond,modes=modes)

            
            pred1 = model_pred.model_out1
            x_pred1 = model_pred.pred_x_start1

            if self.objective == "pred_noise":
                target1 = noise
            elif self.objective == "pred_x0":
                target1 = x
            elif self.objective == "pred_v":
                target1 = self.predict_v(x, noise_levels_x, noise)
            else:
                raise ValueError(f"unknown objective {self.objective}")

            # loss = F.mse_loss(pred, target.detach(), reduction="none")
            loss_weight1,reg_schedule1 = self.compute_loss_weights(noise_levels_x)
            
            pred2 = model_pred.model_out2
            x_pred2 = model_pred.pred_x_start2

            if self.objective == "pred_noise":
                target2 = noise2
            elif self.objective == "pred_x0":
                target2 = y
            elif self.objective == "pred_v":
                target2 = self.predict_v(y, noise_levels_y, noise2)
            else:
                raise ValueError(f"unknown objective {self.objective}")

            # loss = F.mse_loss(pred, target.detach(), reduction="none")
            loss_weight2,reg_schedule2 = self.compute_loss_weights(noise_levels_y)
            
            return pred1,target1.detach(),pred2,target2.detach(),None,None,loss_weight1,loss_weight2,None,reg_schedule1,reg_schedule2,None
        else:
            noise = torch.randn_like(x)
            noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)
            noised_x = self.q_sample(x_start=x, t=noise_levels_x, noise=noise)
            # model_pred = self.model_predictions(model=model,x=noised_x,y=noised_x t=noise_levels_x, external_cond=external_cond)
            
            noise2 = torch.randn_like(y)
            
            noise2 = torch.clamp(noise2, -self.clip_noise, self.clip_noise)
            noised_y = self.q_sample(x_start=y, t=noise_levels_y, noise=noise2)
            
            noise3 = torch.randn_like(z)
            
            noise3 = torch.clamp(noise3, -self.clip_noise, self.clip_noise)
            noised_z = self.q_sample(x_start=z, t=noise_levels_z, noise=noise3)
            
            noise_levels_all = torch.cat([noise_levels_x.unsqueeze(2),noise_levels_y.unsqueeze(2),noise_levels_z.unsqueeze(2)],-1)
            model_pred = self.model_predictions(model=model,x=noised_x, y=noised_y,z=noised_z, t=noise_levels_all, external_cond=external_cond,modes=modes)

            
            pred1 = model_pred.model_out1
            x_pred1 = model_pred.pred_x_start1

            if self.objective == "pred_noise":
                target1 = noise
            elif self.objective == "pred_x0":
                target1 = x
            elif self.objective == "pred_v":
                target1 = self.predict_v(x, noise_levels_x, noise)
            else:
                raise ValueError(f"unknown objective {self.objective}")

            # loss = F.mse_loss(pred, target.detach(), reduction="none")
            loss_weight1,reg_schedule1 = self.compute_loss_weights(noise_levels_x)
            
            pred2 = model_pred.model_out2
            x_pred2 = model_pred.pred_x_start2

            if self.objective == "pred_noise":
                target2 = noise2
            elif self.objective == "pred_x0":
                target2 = y
            elif self.objective == "pred_v":
                target2 = self.predict_v(y, noise_levels_y, noise2)
            else:
                raise ValueError(f"unknown objective {self.objective}")

            # loss = F.mse_loss(pred, target.detach(), reduction="none")
            loss_weight2,reg_schedule2 = self.compute_loss_weights(noise_levels_y)
            
            
            pred3 = model_pred.model_out3
            x_pred3 = model_pred.pred_x_start3

            if self.objective == "pred_noise":
                target3 = noise3
            elif self.objective == "pred_x0":
                target3 = z
            elif self.objective == "pred_v":
                target3 = self.predict_v(z, noise_levels_z, noise3)
            else:
                raise ValueError(f"unknown objective {self.objective}")

            # loss = F.mse_loss(pred, target.detach(), reduction="none")
            loss_weight3,reg_schedule3 = self.compute_loss_weights(noise_levels_z)
            return pred1,target1.detach(),pred2,target2.detach(),pred3,target3.detach(),loss_weight1,loss_weight2,loss_weight3,reg_schedule1,reg_schedule2,reg_schedule3
        # loss_weight = loss_weight.view(*loss_weight.shape, *((1,) * (loss.ndim - 2)))
        # loss = loss * loss_weight

        # return x_pred, loss

    def sample_step(
        self,
        model,
        x,
        y,
        z,
        external_cond ,
        curr_noise_level,
        next_noise_level,
        guidance_fn,
        curr_noise_level_cfg = None,
        raw_noise = None,
        cfg_weight = 0,
        cfg_text = None,
        mode = 0,
        star=0,
        add=0,
        tweight=0,
        rescale =0
        
    ):
        real_steps = torch.linspace(-1, self.timesteps - 1, steps=self.sampling_timesteps + 1, device=x.device).long()
    
        real_steps2 = torch.linspace(-1, self.timesteps - 1, steps=self.sampling_timesteps + 1+1, device=x.device).long()
       
        curr_noise_level_orig = curr_noise_level.copy()
        
        B,L= curr_noise_level.shape
        curr_noise_level = real_steps[curr_noise_level.reshape(-1)].reshape(B,L)
        
        next_noise_level = real_steps[next_noise_level.reshape(-1)].reshape(B,L)

      
        assert self.is_ddim_sampling # Only implement the ddim sampling in this project
        
        if self.is_ddim_sampling:
          
            
            if curr_noise_level_cfg is not None:
                
                curr_noise_level_cfg_xyz = real_steps[curr_noise_level_cfg.reshape(-1)].reshape(-1)
                pure_noise_level = np.ones_like(curr_noise_level_cfg.reshape(-1)) * self.sampling_timesteps
                pure_noise_level = real_steps[pure_noise_level].reshape(-1)
            else:
                curr_noise_level_cfg_xyz = None
                pure_noise_level = None
                
            return self.ddim_sample_step_triple(
                model = model,
                x=x,
                y = y,
                z = z,
                external_cond=external_cond,
                curr_noise_level=curr_noise_level,
                next_noise_level=next_noise_level,
                guidance_fn=guidance_fn,
                curr_noise_level_cfg_xyz = curr_noise_level_cfg_xyz,
                pure_noise_level = pure_noise_level,
                cfg_weight = cfg_weight,
                raw_noise = raw_noise,
                cfg_text = cfg_text,
                mode =mode,
                star = star,
                add = add,
                tweight = tweight,
                rescale = rescale
                
            )
                


        # FIXME: temporary code for checking ddpm sampling
        assert torch.all(
            (curr_noise_level - 1 == next_noise_level) | ((curr_noise_level == -1) & (next_noise_level == -1))
        ), "Wrong noise level given for ddpm sampling."

        assert (
            self.sampling_timesteps == self.timesteps
        ), "sampling_timesteps should be equal to timesteps for ddpm sampling."

        return self.ddpm_sample_step(
            model=model,
            x=x,
            external_cond=external_cond,
            curr_noise_level=curr_noise_level,
            guidance_fn=guidance_fn,
        )

    def ddpm_sample_step(
        self,
        model,
        x,
        external_cond,
        curr_noise_level,
        guidance_fn,
        
    ):
        clipped_curr_noise_level = torch.where(
            curr_noise_level < 0,
            torch.full_like(curr_noise_level, self.stabilization_level - 1, dtype=torch.long),
            curr_noise_level,
        )

        # treating as stabilization would require us to scale with sqrt of alpha_cum
        orig_x = x.clone().detach()
        scaled_context = self.q_sample(
            x,
            clipped_curr_noise_level
        )
        x = torch.where(self.add_shape_channels(curr_noise_level < 0), scaled_context, orig_x)

        if guidance_fn is not None:
            raise NotImplementedError("Guidance function is not implemented for ddpm sampling yet.")

        else:
            model_mean, _, model_log_variance = self.p_mean_variance(
                model = model,
                x=x,
                t=clipped_curr_noise_level,
                external_cond=external_cond,
            )

        noise = torch.where(
            self.add_shape_channels(clipped_curr_noise_level > 0),
            torch.randn_like(x),
            0,
        )
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)
        x_pred = model_mean + torch.exp(0.5 * model_log_variance) * noise

        # only update frames where the noise level decreases
        return torch.where(self.add_shape_channels(curr_noise_level == -1), orig_x, x_pred)

    def ddim_sample_step(
        self,
        model,
        x,
        external_cond,
        curr_noise_level,
        next_noise_level,
        guidance_fn,
        curr_noise_level_cfg = None
    ):
        # convert noise level -1 to self.stabilization_level - 1
        clipped_curr_noise_level = torch.where(
            curr_noise_level < 0,
            torch.full_like(curr_noise_level, self.stabilization_level - 1, dtype=torch.long).to(x.device),
            curr_noise_level,
        )

        # treating as stabilization would require us to scale with sqrt of alpha_cum
        orig_x = x.clone().detach()
        scaled_context = self.q_sample(
            x,
            clipped_curr_noise_level,
            noise=torch.zeros_like(x).to(x.device),
        )
        x = torch.where(self.add_shape_channels(curr_noise_level < 0), scaled_context, orig_x)

        alpha = self.alphas_cumprod[clipped_curr_noise_level]
        alpha_next = torch.where(
            next_noise_level < 0,
            torch.ones_like(next_noise_level).to(x.device),
            self.alphas_cumprod[next_noise_level],
        )
        sigma = torch.where(
            next_noise_level < 0,
            torch.zeros_like(next_noise_level).to(x.device),
            self.ddim_sampling_eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt(),
        )
        c = (1 - alpha_next - sigma**2).sqrt()

        alpha_next = self.add_shape_channels(alpha_next)
        c = self.add_shape_channels(c)
        sigma = self.add_shape_channels(sigma)

        if guidance_fn is not None:
            with torch.enable_grad():
                x = x.detach().requires_grad_()

                model_pred = self.model_predictions(model=model,
                    x=x,
                    t=clipped_curr_noise_level,
                    external_cond=external_cond,
                )

                guidance_loss = guidance_fn(model_pred.pred_x_start)
                grad = -torch.autograd.grad(
                    guidance_loss,
                    x,
                )[0]

                pred_noise = model_pred.pred_noise + (1 - alpha_next).sqrt() * grad
                x_start = self.predict_start_from_noise(x, clipped_curr_noise_level, pred_noise)

        else:
            with torch.no_grad():
                model_pred = self.model_predictions(model=model,
                    x=x,y=None,z=None,
                    t=clipped_curr_noise_level,
                    external_cond=external_cond,
                )
                x_start = model_pred.pred_x_start1
                pred_noise = model_pred.pred_noise1

        noise = torch.randn_like(x)
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)

        x_pred = x_start * alpha_next.sqrt() + pred_noise * c + sigma * noise

        # only update frames where the noise level decreases
        mask = curr_noise_level == next_noise_level
        x_pred = torch.where(
            self.add_shape_channels(mask),
            orig_x,
            x_pred,
        )

        return x_pred,None
    
    
    def find_tprime_batch(
        self,
        t,                # [B, T], long indices in [0, Tmax-1]
        r        # scalar or broadcastable to [B, T]
              # tie-breaking: pick higher/lower index when equal distance
    ) :
    
        alphas_cumprod = self.alphas_cumprod
        Tmax = alphas_cumprod.shape[0]
        device = alphas_cumprod.device
        dtype  = alphas_cumprod.dtype

        # Ensure index tensor is long and on the right device
        # t = t.to(device=device, dtype=torch.long)
        # t = t.clamp_(0, Tmax - 1)

        # alpha_t = \bar{alpha}_t  (2-D)  — your requested indexing form
        alpha_t = alphas_cumprod[t].to(dtype)

        # target = r * \bar{alpha}_t  (broadcast OK)
        # if not torch.is_tensor(r):
        #     r = torch.tensor(r, device=device, dtype=dtype)
        target = (r * alpha_t)

        # searchsorted needs ascending input: flip to ascending
        arr_rev = torch.flip(alphas_cumprod, dims=[0])  # [Tmax], ascending

        # Find insertion positions for each target: shape [B, T], values in [0..Tmax]
        idx = torch.searchsorted(arr_rev, target)

        # Nearest neighbors in the ascending array
        hi = idx.clamp(max=Tmax - 1)  
        t_prime  = (Tmax - 1 - hi).to(t.dtype)# right neighbor (or boundary)
        return t_prime
    
    def ddim_sample_step_triple(
        self,
        model,
        x,
        y,
        z,
        external_cond,
        curr_noise_level,
        next_noise_level,
        guidance_fn,
        curr_noise_level_cfg_xyz = None, # B,3*t
        pure_noise_level = None, # b,3*t
        cfg_weight = 0,
        raw_noise = None,
        cfg_text = 0,
        mode = 0,
        star = 0, 
        add = 0,
        tweight = 0,
        rescale = 0
    ):
        
        clipped_curr_noise_level = torch.where(
            curr_noise_level < 0,
            torch.full_like(curr_noise_level, self.stabilization_level - 1, dtype=torch.long).to(x.device),
            curr_noise_level,
        )
        clipped_curr_noise_level = clipped_curr_noise_level.reshape(clipped_curr_noise_level.shape[0],-1,3)
        clipped_curr_noise_level_x = clipped_curr_noise_level[:,:,0]
        clipped_curr_noise_level_y = clipped_curr_noise_level[:,:,1]
        clipped_curr_noise_level_z = clipped_curr_noise_level[:,:,2]
        
        curr_noise_level = curr_noise_level.reshape(curr_noise_level.shape[0],-1,3)
        curr_noise_level_x = curr_noise_level[:,:,0]
        curr_noise_level_y = curr_noise_level[:,:,1]
        curr_noise_level_z = curr_noise_level[:,:,2]
        
        next_noise_level = next_noise_level.reshape(next_noise_level.shape[0],-1,3)
        next_noise_level_x = next_noise_level[:,:,0]
        next_noise_level_y = next_noise_level[:,:,1]
        next_noise_level_z = next_noise_level[:,:,2]
        # treating as stabilization would require us to scale with sqrt of alpha_cum
        orig_x = x.clone().detach()
        scaled_context_x = self.q_sample(
            x,
            clipped_curr_noise_level_x,
            noise=torch.zeros_like(x).to(x.device),
        )
        x = torch.where(self.add_shape_channels(curr_noise_level_x < 0), scaled_context_x, orig_x)

        alpha_x = self.alphas_cumprod[clipped_curr_noise_level_x]
        alpha_next_x = torch.where(
            next_noise_level_x < 0,
            torch.ones_like(next_noise_level_x).to(x.device),
            self.alphas_cumprod[next_noise_level_x],
        )
        sigma_x = torch.where(
            next_noise_level_x < 0,
            torch.zeros_like(next_noise_level_x).to(x.device),
            self.ddim_sampling_eta * ((1 - alpha_x / alpha_next_x) * (1 - alpha_next_x) / (1 - alpha_x)).sqrt(),
        )
        c_x = (1 - alpha_next_x - sigma_x**2).sqrt()

        alpha_next_x = self.add_shape_channels(alpha_next_x)
        c_x = self.add_shape_channels(c_x)
        sigma_x = self.add_shape_channels(sigma_x)
        

        
        
        orig_y = y.clone().detach()
        scaled_context_y = self.q_sample(
            y,
            clipped_curr_noise_level_y,
            noise=torch.zeros_like(y).to(x.device),
        )
        y = torch.where(self.add_shape_channels(curr_noise_level_y < 0), scaled_context_y, orig_y)
        
        alpha_y = self.alphas_cumprod[clipped_curr_noise_level_y]
        alpha_next_y = torch.where(
            next_noise_level_y < 0,
            torch.ones_like(next_noise_level_y).to(x.device),
            self.alphas_cumprod[next_noise_level_y],
        )
        sigma_y = torch.where(
            next_noise_level_y < 0,
            torch.zeros_like(next_noise_level_y).to(x.device),
            self.ddim_sampling_eta * ((1 - alpha_y / alpha_next_y) * (1 - alpha_next_y) / (1 - alpha_y)).sqrt(),
        )
        c_y = (1 - alpha_next_y - sigma_y**2).sqrt()

        alpha_next_y = self.add_shape_channels(alpha_next_y)
        c_y = self.add_shape_channels(c_y)
        sigma_y = self.add_shape_channels(sigma_y)
        
        
        orig_z = z.clone().detach()
        scaled_context_z = self.q_sample(
            z,
            clipped_curr_noise_level_z,
            noise=torch.zeros_like(z).to(z.device),
        )
        z = torch.where(self.add_shape_channels(curr_noise_level_z < 0), scaled_context_z, orig_z)

        alpha_z = self.alphas_cumprod[clipped_curr_noise_level_z]
        alpha_next_z = torch.where(
            next_noise_level_z < 0,
            torch.ones_like(next_noise_level_z).to(z.device),
            self.alphas_cumprod[next_noise_level_z],
        )
        sigma_z = torch.where(
            next_noise_level_z < 0,
            torch.zeros_like(next_noise_level_z).to(z.device),
            self.ddim_sampling_eta * ((1 - alpha_z / alpha_next_z) * (1 - alpha_next_z) / (1 - alpha_z)).sqrt(),
        )
        c_z = (1 - alpha_next_z - sigma_z**2).sqrt()

        alpha_next_z = self.add_shape_channels(alpha_next_z)
        c_z = self.add_shape_channels(c_z)
        sigma_z = self.add_shape_channels(sigma_z)
        
        if curr_noise_level_cfg_xyz is not None:

            x_full, y_full,z_full,noise_x_init,noise_y_init,noise_z_init = raw_noise
            
            
            x_full,clipped_curr_noise_level_cfg_x,curr_noise_level_cfg_x = self.clip_noise_and_input(curr_noise_level_cfg_xyz[0::3].unsqueeze(0).repeat(x_full.shape[0],1),x_full)
            y_full,clipped_curr_noise_level_cfg_y,curr_noise_level_cfg_y = self.clip_noise_and_input(curr_noise_level_cfg_xyz[1::3].unsqueeze(0).repeat(x_full.shape[0],1),y_full)
            z_full,clipped_curr_noise_level_cfg_z,curr_noise_level_cfg_z = self.clip_noise_and_input(curr_noise_level_cfg_xyz[2::3].unsqueeze(0).repeat(x_full.shape[0],1),z_full)

            if add ==1:
                alpha_x_noiser = self.alphas_cumprod[clipped_curr_noise_level_cfg_x]
                weight_add = (alpha_x_noiser/alpha_x).unsqueeze(1).unsqueeze(1)
                x_full = torch.sqrt(weight_add)*x + noise_x_init*torch.sqrt(1-weight_add)
                
                alpha_y_noiser = self.alphas_cumprod[clipped_curr_noise_level_cfg_y]
                weight_add = (alpha_y_noiser/alpha_y).unsqueeze(1).unsqueeze(1)
                y_full = torch.sqrt(weight_add)*y + noise_y_init*torch.sqrt(1-weight_add)
                
                alpha_z_noiser = self.alphas_cumprod[clipped_curr_noise_level_cfg_z]
                weight_add = (alpha_z_noiser/alpha_z).unsqueeze(1).unsqueeze(1)
                z_full = torch.sqrt(weight_add)*z + noise_z_init*torch.sqrt(1-weight_add)
                
            elif add>0:
                 
                clipped_curr_noise_level_cfg_x = self.find_tprime_batch(clipped_curr_noise_level_x ,add)
                weight_add = (self.alphas_cumprod[clipped_curr_noise_level_cfg_x]/alpha_x).unsqueeze(1).unsqueeze(1)
                x_full = torch.sqrt(weight_add)*x + noise_x_init*torch.sqrt(1-weight_add)
                
                clipped_curr_noise_level_cfg_y = self.find_tprime_batch(clipped_curr_noise_level_y ,add)
                weight_add = (self.alphas_cumprod[clipped_curr_noise_level_cfg_y]/alpha_y).unsqueeze(1).unsqueeze(1)
                y_full = torch.sqrt(weight_add)*y + noise_y_init*torch.sqrt(1-weight_add)
                
                clipped_curr_noise_level_cfg_z = self.find_tprime_batch(clipped_curr_noise_level_z ,add)
                weight_add = (self.alphas_cumprod[clipped_curr_noise_level_cfg_z]/alpha_z).unsqueeze(1).unsqueeze(1)
                z_full = torch.sqrt(weight_add)*z + noise_z_init*torch.sqrt(1-weight_add)
                
                           
                
            
            
            
            
            
            
            
            

        if guidance_fn is not None: # Not implemented here
            with torch.enable_grad():
                x = x.detach().requires_grad_()
                y = y.detach().requires_grad_()
                z = z.detach().requires_grad_()

                model_pred = self.model_predictions(model=model,
                    x=x,y=y,z=z,
                    t=clipped_curr_noise_level,
                    external_cond=external_cond,
                )
                Time = clipped_curr_noise_level[0,0,0]
                if (self.df_divider==1 and Time<self.df_upstop) or (self.df_divider!=1 and Time%self.df_divider ==0 and Time<self.df_upstop) or Time<self.df_upstop:
        
                    INPUT =torch.cat([model_pred.pred_x_start1,model_pred.pred_x_start3,model_pred.pred_x_start2],1)
                    
                    
                    posterior_variance = extract(self.posterior_variance, clipped_curr_noise_level.reshape(clipped_curr_noise_level.shape[0],-1), INPUT.shape)
                    posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, clipped_curr_noise_level.reshape(clipped_curr_noise_level.shape[0],-1), INPUT.shape)
                    
                    guidance_loss = guidance_fn(INPUT,external_cond)
                    grad = -torch.autograd.grad(
                        guidance_loss,
                        INPUT,
                    )[0]
                
                    x_start = model_pred.pred_x_start1 + posterior_variance[...,0::3]*1*grad[:,:model_pred.pred_x_start1.shape[1]]
                    y_start = model_pred.pred_x_start2 + posterior_variance[...,0::3]*posterior_variance[...,1::3]*self.df_weight*grad[:,model_pred.pred_x_start1.shape[1]+model_pred.pred_x_start3.shape[1]:]
                    z_start = model_pred.pred_x_start3 + posterior_variance[...,2::3]*1*grad[:,model_pred.pred_x_start1.shape[1]:model_pred.pred_x_start1.shape[1]+model_pred.pred_x_start3.shape[1]]
                    
                    # pred_noise = model_pred.pred_noise + (1 - alpha_next).sqrt() * grad
                    pred_noise_x = self.predict_noise_from_start(x, clipped_curr_noise_level[...,0], x_start)
                    pred_noise_y = self.predict_noise_from_start(y, clipped_curr_noise_level[...,1], y_start)
                    pred_noise_z = self.predict_noise_from_start(z, clipped_curr_noise_level[...,2], z_start)
                else:
                    INPUT =torch.cat([model_pred.pred_x_start1,model_pred.pred_x_start3,model_pred.pred_x_start2],1)
                    
                    
                    posterior_variance = extract(self.posterior_variance, clipped_curr_noise_level.reshape(clipped_curr_noise_level.shape[0],-1), INPUT.shape)
                    posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, clipped_curr_noise_level.reshape(clipped_curr_noise_level.shape[0],-1), INPUT.shape)
                    
                    guidance_loss = guidance_fn.gradients_vel(INPUT,external_cond)
                    grad = -torch.autograd.grad(
                        guidance_loss,
                        INPUT,allow_unused=True
                    )[0]
                
                    x_start = model_pred.pred_x_start1 + posterior_variance[...,0::3]*1*grad[:,:model_pred.pred_x_start1.shape[1]]
                    y_start = model_pred.pred_x_start2 + posterior_variance[...,0::3]*posterior_variance[...,1::3]*self.df_weight*grad[:,model_pred.pred_x_start1.shape[1]+model_pred.pred_x_start3.shape[1]:]
                    z_start = model_pred.pred_x_start3 + posterior_variance[...,2::3]*1*grad[:,model_pred.pred_x_start1.shape[1]:model_pred.pred_x_start1.shape[1]+model_pred.pred_x_start3.shape[1]]
                    
                    # pred_noise = model_pred.pred_noise + (1 - alpha_next).sqrt() * grad
                    pred_noise_x = self.predict_noise_from_start(x, clipped_curr_noise_level[...,0], x_start)
                    pred_noise_y = self.predict_noise_from_start(y, clipped_curr_noise_level[...,1], y_start)
                    pred_noise_z = self.predict_noise_from_start(z, clipped_curr_noise_level[...,2], z_start)
                # x_start = self.predict_start_from_noise(x, clipped_curr_noise_level, pred_noise)

        else:
            with torch.no_grad():
                model_pred = self.model_predictions(model=model,
                    x=x,y=y,z=z,
                    t=clipped_curr_noise_level,
                    external_cond=external_cond,
                )
                
                x_start = model_pred.pred_x_start1
                pred_noise_x = model_pred.pred_noise1
                
                y_start = model_pred.pred_x_start2
                pred_noise_y = model_pred.pred_noise2
                
                z_start = model_pred.pred_x_start3
                pred_noise_z = model_pred.pred_noise3
                
                if curr_noise_level_cfg_xyz is not None: # step
                    
                    x_start,y_start,z_start,pred_noise_x,pred_noise_y,pred_noise_z =self.cfg_step(x_start,y_start,z_start,pred_noise_x,pred_noise_y,pred_noise_z,model,external_cond,[x,y,z],[x_full,y_full,z_full],clipped_curr_noise_level,
                                  [clipped_curr_noise_level_cfg_x,clipped_curr_noise_level_cfg_y,clipped_curr_noise_level_cfg_z],mode,cfg_weight,cfg_text,star,tweight,rescale,model_pred)
       
                     
                    

        noise_x = torch.randn_like(x).to(x.device)
        noise_x = torch.clamp(noise_x, -self.clip_noise, self.clip_noise)
        x_pred = x_start * alpha_next_x.sqrt() + pred_noise_x * c_x + sigma_x * noise_x
      
        mask = curr_noise_level_x == next_noise_level_x
        x_pred = torch.where(
            self.add_shape_channels(mask),
            orig_x,
            x_pred,
        )
        noise_y = torch.randn_like(y).to(x.device)
        noise_y = torch.clamp(noise_y, -self.clip_noise, self.clip_noise)

        y_pred = y_start * alpha_next_y.sqrt() + pred_noise_y * c_y + sigma_y * noise_y

        # only update frames where the noise level decreases
        mask = curr_noise_level_y == next_noise_level_y
        y_pred = torch.where(
            self.add_shape_channels(mask),
            orig_y,
            y_pred,
        )
        
        noise_z = torch.randn_like(z).to(x.device)
        noise_z = torch.clamp(noise_z, -self.clip_noise, self.clip_noise)
        z_pred = z_start * alpha_next_z.sqrt() + pred_noise_z * c_z + sigma_z * noise_z
        # only update frames where the noise level decreases
        mask = curr_noise_level_z == next_noise_level_z
        z_pred = torch.where(
            self.add_shape_channels(mask),
            orig_z,
            z_pred,
        )

        return x_pred,y_pred,z_pred
    def clip_noise_and_input(self,curr_noise_level,x):
        clipped_curr_noise_level_x = torch.where(
            curr_noise_level < 0,
            torch.full_like(curr_noise_level, self.stabilization_level - 1, dtype=torch.long).to(x.device),
            curr_noise_level,
        )
        # clipped_curr_noise_level = clipped_curr_noise_level.reshape(clipped_curr_noise_level.shape[0],-1,3)
        # clipped_curr_noise_level_x = clipped_curr_noise_level[:,:,0]
        # clipped_curr_noise_level_y = clipped_curr_noise_level[:,:,1]
        # clipped_curr_noise_level_z = clipped_curr_noise_level[:,:,2]
        
        # curr_noise_level = curr_noise_level.reshape(curr_noise_level.shape[0],-1,3)
        # curr_noise_level_x = curr_noise_level[:,:,0]
        # curr_noise_level_y = curr_noise_level[:,:,1]
        # curr_noise_level_z = curr_noise_level[:,:,2]
        
        # next_noise_level = next_noise_level.reshape(next_noise_level.shape[0],-1,3)
        # next_noise_level_x = next_noise_level[:,:,0]
        # next_noise_level_y = next_noise_level[:,:,1]
        # next_noise_level_z = next_noise_level[:,:,2]
        # treating as stabilization would require us to scale with sqrt of alpha_cum
        orig_x = x.clone().detach()
        scaled_context_x = self.q_sample(
            x,
            clipped_curr_noise_level_x,
            noise=torch.zeros_like(x).to(x.device),
        )
        x = torch.where(self.add_shape_channels(curr_noise_level < 0), scaled_context_x, orig_x)
        return x,clipped_curr_noise_level_x,curr_noise_level
    def calculate_multiplier(self,x1,x2):
        N1 = torch.norm(x1, p=2, dim=1, keepdim=True)+1e-6
        dot_product = (x1*x2).sum(dim=1,keepdim=True)
        return dot_product/(N1**2)
    def masked_std(self,x, mask, dim=None, keepdim=False, correction=1):
    
        if dim is None:
            dim = tuple(range(1, x.ndim))
        w = mask.to(dtype=x.dtype)
        cnt = w.sum(dim=dim, keepdim=keepdim)
        cnt_safe = cnt.clamp(min=1.0)
        mean = (x * w).sum(dim=dim, keepdim=keepdim) / cnt_safe
        var_num = ((x - mean)**2 * w).sum(dim=dim, keepdim=keepdim)
        denom = (cnt - correction).clamp(min=1.0)
        std = (var_num / denom).sqrt()
        std = torch.where(cnt > correction, std, torch.zeros_like(std))
        return std
    def _normalize_l2(self,x: torch.Tensor, dim, eps: float = 1e-12) -> torch.Tensor:
    
        if isinstance(dim, int):
            dim = (dim,)
        denom = torch.linalg.vector_norm(x, ord=2, dim=dim, keepdim=True).clamp_min(eps)
        return x / denom

    def project_bt1t(self,
        v0,  # [B, D, 1, T]
        v1,  # [B, D, 1, T]
        mode=0,
        eps=1e-12,
        mask =None
    ):
        dtype = v0.dtype
        if mask is not None:
            v0d = v0*mask.to(dtype)
            v1d = v1*mask.to(dtype)
        v0d, v1d = v0d.double(), v1d.double()

        if mode == 0:
            v1n = self._normalize_l2(v1d, dim=-3, eps=eps)
            coeff = (v0d * v1n).sum(dim=-3, keepdim=True)  
            v0_parallel = coeff * v1n                                  # [B,D,1,T]
        else:
            v1n = self._normalize_l2(v1d, dim=(-3, -2, -1), eps=eps)       # (D,1,T)
            coeff = (v0d * v1n).sum(dim=(-3, -2, -1), keepdim=True)   # [B,1,1,1]
            v0_parallel = coeff * v1n                                  # [B,D,1,T]

        v0_orth = v0d - v0_parallel
        return v0_parallel.to(dtype), v0_orth.to(dtype)
    
    def cfg_substep_one(self,star,mask,rescale,cfg_text,cfg_weight,clipped_curr_noise_level,i,y,y_start,pred_noise_y,y_start_other_alluncond,pred_noise_y_other_alluncond,y_start_other_textuncond,pred_noise_y_other_textuncond):
        timer1= 1
        timer2=1
        
        if star ==0:
            if rescale>0:
                # 
                pred_noise_y2 = pred_noise_y_other_alluncond - cfg_text*(pred_noise_y_other_textuncond-pred_noise_y_other_alluncond*timer1)+cfg_weight*(pred_noise_y-pred_noise_y_other_textuncond)
                
                std_pos = self.masked_std(pred_noise_y,mask=mask, keepdim=True)
                std_cfg = self.masked_std(pred_noise_y2,mask=mask, keepdim=True)
                factor = std_pos / std_cfg
                factor = rescale * factor + (1 - rescale)
                
                pred_noise_y2 = pred_noise_y2* factor
                y_start2 = self.predict_start_from_noise(y,clipped_curr_noise_level[...,i],pred_noise_y2)
            
            else:
                
                # diff =pred_noise_y_other_textuncond-pred_noise_y_other_alluncond*timer1    
                # diff =  -y_start_other_textuncond+y_start_other_alluncond # Good
                diff =  -y_start+y_start_other_alluncond
                has_nan1 = torch.isnan(y_start).any()
                has_nan2 = torch.isnan(y_start_other_alluncond).any()
                
                diff_text = y_start - y_start_other_textuncond
                
                if self.df_r > 0:
                    
                    ones = torch.ones_like(diff)
                    diff_norm = diff.norm(p=2, dim=[-1, -2, -3], keepdim=True)
                    scale_factor = torch.minimum(ones, self.df_r / diff_norm)
                    diff = diff * scale_factor
                y_start2 = y_start + (cfg_weight - 1) * diff + (cfg_text-1)*diff_text
                pred_noise_y2 = self.predict_noise_from_start(y, clipped_curr_noise_level[...,i], y_start2)
               

            
            
            # pred_noise_y2 = pred_noise_y_other_alluncond + cfg_text*(pred_noise_y_other_textuncond-pred_noise_y_other_alluncond*timer1)+cfg_weight*(pred_noise_y-pred_noise_y_other_textuncond)
        elif star in [6,7]: ## ALL_STAR
            diff = -y_start_other_textuncond +y_start_other_alluncond

            # diff =pred_noise_y_other_textuncond-pred_noise_y_other_alluncond*timer1    
            if self.momentum_buffers is not None:
                self.momentum_buffers[i].update(diff)
                diff = self.momentum_buffers[i].running_average
            if self.df_r > 0:
                ones = torch.ones_like(diff)
                diff_norm = diff.norm(p=2, dim=[-1, -2, -3], keepdim=True)
                scale_factor = torch.minimum(ones, self.df_r / diff_norm)
                diff = diff * scale_factor
            if star ==6:
                diff_parallel, diff_orthogonal = self.project_bt1t(diff, y_start,mode=0,mask=mask)
            else:
                diff_parallel, diff_orthogonal = self.project_bt1t(diff, y_start,mode=1,mask=mask)

            eta = 0
            normalized_update = diff_orthogonal + eta * diff_parallel
            
            
            y_start2 = y_start + (cfg_text - 1) * normalized_update
            pred_noise_y2 = self.predict_noise_from_start(y, clipped_curr_noise_level[...,i], y_start2)
           
        
        elif star ==3:
            ## cfg++
            y_start2 = y_start_other_alluncond - cfg_text*(y_start_other_textuncond -y_start_other_alluncond)+cfg_weight*(y_start-y_start_other_textuncond)  
            # sb woshisb
            pred_noise_y2 = self.predict_noise_from_start(y, clipped_curr_noise_level[...,i], y_start2)
            y_start2 = y_start_other_alluncond
        elif star ==4:
            ##cfg++
            pred_noise_y2 = pred_noise_y_other_alluncond - cfg_text*(pred_noise_y_other_textuncond-pred_noise_y_other_alluncond*timer1)+cfg_weight*(pred_noise_y-pred_noise_y_other_textuncond)
           
            # y_start2 = y_start_other_alluncond + cfg_text*(y_start_other_textuncond -y_start_other_alluncond)+cfg_weight*(y_start-y_start_other_textuncond)  
            y_start2 = self.predict_start_from_noise(y,clipped_curr_noise_level[...,i],pred_noise_y2)
            pred_noise_y2 = pred_noise_y_other_alluncond
            # sb woshisb
            # pred_noise_y2 = self.predict_noise_from_start(y, clipped_curr_noise_level[...,i], y_start2)
            
        
        elif star ==2:
            # cfg-zero*
            timer1 = self.calculate_multiplier(pred_noise_y_other_alluncond,pred_noise_y_other_textuncond)
            timer2 = self.calculate_multiplier(pred_noise_y_other_textuncond,pred_noise_y)
            pred_noise_y2 = pred_noise_y_other_alluncond*timer2*timer1 - cfg_text*(pred_noise_y_other_textuncond*timer2-pred_noise_y_other_alluncond*timer1*timer2)+cfg_weight*(pred_noise_y-pred_noise_y_other_textuncond*timer2)
            y_start2 = self.predict_start_from_noise(y,clipped_curr_noise_level[...,i],pred_noise_y2)
        elif star==1:
             # cfg-zero*
            v_y_other_alluncond = self.predict_v( y_start, clipped_curr_noise_level[...,i], pred_noise_y_other_alluncond)
            v_y_other_textuncond = self.predict_v( y_start, clipped_curr_noise_level[...,i], pred_noise_y_other_textuncond)
            v_y = self.predict_v( y_start, clipped_curr_noise_level[...,i], pred_noise_y)
            timer1 = self.calculate_multiplier(v_y_other_alluncond,v_y_other_textuncond)
            timer2 = self.calculate_multiplier(v_y_other_textuncond,v_y)
            
            
            v_y = v_y_other_alluncond*timer2*timer1 - cfg_text*(v_y_other_textuncond*timer2-v_y_other_alluncond*timer1*timer2)+cfg_weight*(v_y-v_y_other_textuncond*timer2)
            y_start2 = self.predict_start_from_v( y, clipped_curr_noise_level[...,i], v_y)
            pred_noise_y2 = self.predict_noise_from_start(y, clipped_curr_noise_level[...,i], y_start2) ## previously without 2
        return y_start2,pred_noise_y2
    
    def cfg_substep_one_1(self,star,mask,rescale,cfg_text,cfg_weight,clipped_curr_noise_level,i,y,y_start,pred_noise_y,y_start_other_alluncond,pred_noise_y_other_alluncond,y_start_other_textuncond,pred_noise_y_other_textuncond):
        timer1= 1
        timer2=1
        
        if star ==0:
            if rescale>0:
                # 
                pred_noise_y2 = pred_noise_y_other_alluncond - cfg_text*(pred_noise_y_other_textuncond-pred_noise_y_other_alluncond*timer1)+cfg_weight*(pred_noise_y-pred_noise_y_other_textuncond)
                # std_pos = pred_noise_y.std(dim=list(range(1, pred_noise_y.ndim)), keepdim=True)
                # std_cfg = pred_noise_y2.std(dim=list(range(1, pred_noise_y2.ndim)), keepdim=True)
                std_pos = self.masked_std(pred_noise_y,mask=mask, keepdim=True)
                std_cfg = self.masked_std(pred_noise_y2,mask=mask, keepdim=True)
                factor = std_pos / std_cfg
                factor = rescale * factor + (1 - rescale)
                
                pred_noise_y2 = pred_noise_y2* factor
                y_start2 = self.predict_start_from_noise(y,clipped_curr_noise_level[...,i],pred_noise_y2)
            
            else:
                
                # diff =pred_noise_y_other_textuncond-pred_noise_y_other_alluncond*timer1    
                # diff =  -y_start_other_textuncond+y_start_other_alluncond # Good
                diff =  y_start-y_start_other_alluncond
                has_nan1 = torch.isnan(y_start).any()
                has_nan2 = torch.isnan(y_start_other_alluncond).any()
                
                diff_text = y_start - y_start_other_textuncond
                
                if self.df_r > 0:
                    ones = torch.ones_like(diff)
                    diff_norm = diff.norm(p=2, dim=[-1, -2, -3], keepdim=True)
                    scale_factor = torch.minimum(ones, self.df_r / diff_norm)
                    diff = diff * scale_factor
                y_start2 = y_start + (cfg_weight - 1) * diff + (cfg_text-1)*diff_text
                pred_noise_y2 = self.predict_noise_from_start(y, clipped_curr_noise_level[...,i], y_start2)
                

            
            
            # pred_noise_y2 = pred_noise_y_other_alluncond + cfg_text*(pred_noise_y_other_textuncond-pred_noise_y_other_alluncond*timer1)+cfg_weight*(pred_noise_y-pred_noise_y_other_textuncond)
        elif star in [6,7]: ## ALL_STAR
            diff = -y_start_other_textuncond +y_start_other_alluncond

            # diff =pred_noise_y_other_textuncond-pred_noise_y_other_alluncond*timer1    
            if self.momentum_buffers is not None:
                self.momentum_buffers[i].update(diff)
                diff = self.momentum_buffers[i].running_average
            if self.df_r > 0:
                ones = torch.ones_like(diff)
                diff_norm = diff.norm(p=2, dim=[-1, -2, -3], keepdim=True)
                scale_factor = torch.minimum(ones, self.df_r / diff_norm)
                diff = diff * scale_factor
            if star ==6:
                diff_parallel, diff_orthogonal = self.project_bt1t(diff, y_start,mode=0,mask=mask)
            else:
                diff_parallel, diff_orthogonal = self.project_bt1t(diff, y_start,mode=1,mask=mask)

            eta = 0
            normalized_update = diff_orthogonal + eta * diff_parallel
            
            
            y_start2 = y_start + (cfg_text - 1) * normalized_update
            pred_noise_y2 = self.predict_noise_from_start(y, clipped_curr_noise_level[...,i], y_start2)
           
        
        elif star ==3:
            ## cfg++
            y_start2 = y_start_other_alluncond - cfg_text*(y_start_other_textuncond -y_start_other_alluncond)+cfg_weight*(y_start-y_start_other_textuncond)  
            pred_noise_y2 = self.predict_noise_from_start(y, clipped_curr_noise_level[...,i], y_start2)
            y_start2 = y_start_other_alluncond
        elif star ==4:
            ##cfg++
            pred_noise_y2 = pred_noise_y_other_alluncond - cfg_text*(pred_noise_y_other_textuncond-pred_noise_y_other_alluncond*timer1)+cfg_weight*(pred_noise_y-pred_noise_y_other_textuncond)
           

            # y_start2 = y_start_other_alluncond + cfg_text*(y_start_other_textuncond -y_start_other_alluncond)+cfg_weight*(y_start-y_start_other_textuncond)  
            y_start2 = self.predict_start_from_noise(y,clipped_curr_noise_level[...,i],pred_noise_y2)
            pred_noise_y2 = pred_noise_y_other_alluncond
          
        
        elif star ==2:
            # cfg-zero*
            timer1 = self.calculate_multiplier(pred_noise_y_other_alluncond,pred_noise_y_other_textuncond)
            timer2 = self.calculate_multiplier(pred_noise_y_other_textuncond,pred_noise_y)
            pred_noise_y2 = pred_noise_y_other_alluncond*timer2*timer1 - cfg_text*(pred_noise_y_other_textuncond*timer2-pred_noise_y_other_alluncond*timer1*timer2)+cfg_weight*(pred_noise_y-pred_noise_y_other_textuncond*timer2)
            y_start2 = self.predict_start_from_noise(y,clipped_curr_noise_level[...,i],pred_noise_y2)
        elif star==1:
             # cfg-zero*
            v_y_other_alluncond = self.predict_v( y_start, clipped_curr_noise_level[...,i], pred_noise_y_other_alluncond)
            v_y_other_textuncond = self.predict_v( y_start, clipped_curr_noise_level[...,i], pred_noise_y_other_textuncond)
            v_y = self.predict_v( y_start, clipped_curr_noise_level[...,i], pred_noise_y)
            timer1 = self.calculate_multiplier(v_y_other_alluncond,v_y_other_textuncond)
            timer2 = self.calculate_multiplier(v_y_other_textuncond,v_y)
            
            
            v_y = v_y_other_alluncond*timer2*timer1 - cfg_text*(v_y_other_textuncond*timer2-v_y_other_alluncond*timer1*timer2)+cfg_weight*(v_y-v_y_other_textuncond*timer2)
            y_start2 = self.predict_start_from_v( y, clipped_curr_noise_level[...,i], v_y)
            pred_noise_y2 = self.predict_noise_from_start(y, clipped_curr_noise_level[...,i], y_start2) ## previously without 2
        return y_start2,pred_noise_y2
    
    
    
    def cfg_substep_one_3(self,star,mask,rescale,cfg_text,cfg_weight,clipped_curr_noise_level,i,y,y_start,pred_noise_y,y_start_other_alluncond,pred_noise_y_other_alluncond,y_start_other_textuncond,pred_noise_y_other_textuncond):
        timer1= 1
        timer2=1
       
        if star ==0:
            if rescale>0:
                # 
                pred_noise_y2 = pred_noise_y_other_alluncond - cfg_text*(pred_noise_y_other_textuncond-pred_noise_y_other_alluncond*timer1)+cfg_weight*(pred_noise_y-pred_noise_y_other_textuncond)
                # std_pos = pred_noise_y.std(dim=list(range(1, pred_noise_y.ndim)), keepdim=True)
                # std_cfg = pred_noise_y2.std(dim=list(range(1, pred_noise_y2.ndim)), keepdim=True)
                std_pos = self.masked_std(pred_noise_y,mask=mask, keepdim=True)
                std_cfg = self.masked_std(pred_noise_y2,mask=mask, keepdim=True)
                factor = std_pos / std_cfg
                factor = rescale * factor + (1 - rescale)
                
                pred_noise_y2 = pred_noise_y2* factor
                y_start2 = self.predict_start_from_noise(y,clipped_curr_noise_level[...,i],pred_noise_y2)
            
            else:
                
                # diff =pred_noise_y_other_textuncond-pred_noise_y_other_alluncond*timer1    
                # diff =  -y_start_other_textuncond+y_start_other_alluncond # Good
                diff =  -y_start_other_textuncond+y_start_other_alluncond # THIRD ONE
                has_nan1 = torch.isnan(y_start).any()
                has_nan2 = torch.isnan(y_start_other_alluncond).any()
                
                diff_text = y_start - y_start_other_textuncond
                
                if self.df_r > 0:
                    ones = torch.ones_like(diff)
                    diff_norm = diff.norm(p=2, dim=[-1, -2, -3], keepdim=True)
                    scale_factor = torch.minimum(ones, self.df_r / diff_norm)
                    diff = diff * scale_factor
                y_start2 = y_start + (cfg_weight - 1) * diff + 0*(cfg_text-1)*diff_text
                pred_noise_y2 = self.predict_noise_from_start(y, clipped_curr_noise_level[...,i], y_start2)
               

            
            
        elif star in [6,7]: ## ALL_STAR
            diff = -y_start_other_textuncond +y_start_other_alluncond

            # diff =pred_noise_y_other_textuncond-pred_noise_y_other_alluncond*timer1    
            if self.momentum_buffers is not None:
                self.momentum_buffers[i].update(diff)
                diff = self.momentum_buffers[i].running_average
            if self.df_r > 0:
                ones = torch.ones_like(diff)
                diff_norm = diff.norm(p=2, dim=[-1, -2, -3], keepdim=True)
                scale_factor = torch.minimum(ones, self.df_r / diff_norm)
                diff = diff * scale_factor
            if star ==6:
                diff_parallel, diff_orthogonal = self.project_bt1t(diff, y_start,mode=0,mask=mask)
            else:
                diff_parallel, diff_orthogonal = self.project_bt1t(diff, y_start,mode=1,mask=mask)

            eta = 0
            normalized_update = diff_orthogonal + eta * diff_parallel
            
            
            y_start2 = y_start + (cfg_text - 1) * normalized_update
            pred_noise_y2 = self.predict_noise_from_start(y, clipped_curr_noise_level[...,i], y_start2)
            # pred_noise_y2 = pred_noise_y + (cfg_text - 1) * normalized_update
            # # pred_noise_y2 = pred_noise_y_other_alluncond + cfg_text*(diff)+cfg_weight*(pred_noise_y-pred_noise_y_other_textuncond)
            # y_start2 =self.predict_start_from_noise(y,clipped_curr_noise_level[...,i],pred_noise_y2)
        
        elif star ==3:
            ## cfg++
            y_start2 = y_start_other_alluncond - cfg_text*(y_start_other_textuncond -y_start_other_alluncond)+cfg_weight*(y_start-y_start_other_textuncond)  
            # sb woshisb
            pred_noise_y2 = self.predict_noise_from_start(y, clipped_curr_noise_level[...,i], y_start2)
            y_start2 = y_start_other_alluncond
        elif star ==4:
            ##cfg++
            pred_noise_y2 = pred_noise_y_other_alluncond - cfg_text*(pred_noise_y_other_textuncond-pred_noise_y_other_alluncond*timer1)+cfg_weight*(pred_noise_y-pred_noise_y_other_textuncond)
            # woshisb
            

            # y_start2 = y_start_other_alluncond + cfg_text*(y_start_other_textuncond -y_start_other_alluncond)+cfg_weight*(y_start-y_start_other_textuncond)  
            y_start2 = self.predict_start_from_noise(y,clipped_curr_noise_level[...,i],pred_noise_y2)
            pred_noise_y2 = pred_noise_y_other_alluncond
            # sb woshisb
            # pred_noise_y2 = self.predict_noise_from_start(y, clipped_curr_noise_level[...,i], y_start2)
            
        
        elif star ==2:
            # cfg-zero*
            timer1 = self.calculate_multiplier(pred_noise_y_other_alluncond,pred_noise_y_other_textuncond)
            timer2 = self.calculate_multiplier(pred_noise_y_other_textuncond,pred_noise_y)
            pred_noise_y2 = pred_noise_y_other_alluncond*timer2*timer1 - cfg_text*(pred_noise_y_other_textuncond*timer2-pred_noise_y_other_alluncond*timer1*timer2)+cfg_weight*(pred_noise_y-pred_noise_y_other_textuncond*timer2)
            y_start2 = self.predict_start_from_noise(y,clipped_curr_noise_level[...,i],pred_noise_y2)
        elif star==1:
             # cfg-zero*
            v_y_other_alluncond = self.predict_v( y_start, clipped_curr_noise_level[...,i], pred_noise_y_other_alluncond)
            v_y_other_textuncond = self.predict_v( y_start, clipped_curr_noise_level[...,i], pred_noise_y_other_textuncond)
            v_y = self.predict_v( y_start, clipped_curr_noise_level[...,i], pred_noise_y)
            timer1 = self.calculate_multiplier(v_y_other_alluncond,v_y_other_textuncond)
            timer2 = self.calculate_multiplier(v_y_other_textuncond,v_y)
            
            
            v_y = v_y_other_alluncond*timer2*timer1 - cfg_text*(v_y_other_textuncond*timer2-v_y_other_alluncond*timer1*timer2)+cfg_weight*(v_y-v_y_other_textuncond*timer2)
            y_start2 = self.predict_start_from_v( y, clipped_curr_noise_level[...,i], v_y)
            pred_noise_y2 = self.predict_noise_from_start(y, clipped_curr_noise_level[...,i], y_start2) ## previously without 2
        return y_start2,pred_noise_y2
    
  
    
    def cfg_step(self,x_start,y_start,z_start,pred_noise_x,pred_noise_y,pred_noise_z,model,external_cond,inputs,inputs_full,clipped_curr_noise_level,
                                  clipp_noise_level_full,mode,cfg_weight,cfg_text,star,tweight,rescale,model_pred):
        if tweight ==0:
            cfg_text = cfg_weight # ALL COND, ORIGINAL 1.5
        else:
            cfg_text = tweight
        x,y,z = inputs
        x_full,y_full,z_full = inputs_full ## noiser
        clipped_curr_noise_level_cfg_x,clipped_curr_noise_level_cfg_y,clipped_curr_noise_level_cfg_z = clipp_noise_level_full
        
        
        
        clipped_curr_noise_level_full= clipped_curr_noise_level.clone()
        # clipped_curr_noise_level_yfull = clipped_curr_noise_level.clone()
        # clipped_curr_noise_level_xzuncond = clipped_curr_noise_level.clone() pure_noise_level
        # x_full,clipped_curr_noise_level_cfg_x,curr_noise_level_cfg_x = self.clip_noise_and_input(curr_noise_level_cfg_xyz[0::3].unsqueeze(0).repeat(x_full.shape[0],1),x_full)
        # y_full,clipped_curr_noise_level_cfg_y,curr_noise_level_cfg_y = self.clip_noise_and_input(curr_noise_level_cfg_xyz[1::3].unsqueeze(0).repeat(x_full.shape[0],1),y_full)
        # z_full,clipped_curr_noise_level_cfg_z,curr_noise_level_cfg_z = self.clip_noise_and_input(curr_noise_level_cfg_xyz[2::3].unsqueeze(0).repeat(x_full.shape[0],1),z_full)
        if mode ==0:
            clipped_curr_noise_level_full[...,1] = clipped_curr_noise_level_cfg_y
            clipped_curr_noise_level_full[...,2] = clipped_curr_noise_level_cfg_z
        elif mode ==1:
            clipped_curr_noise_level_full[...,0] = clipped_curr_noise_level_cfg_x
            clipped_curr_noise_level_full[...,2] = clipped_curr_noise_level_cfg_z
        elif mode ==2:
            clipped_curr_noise_level_full[...,0] = clipped_curr_noise_level_cfg_x
            clipped_curr_noise_level_full[...,1] = clipped_curr_noise_level_cfg_y
        elif mode ==3:
            clipped_curr_noise_level_full[...,0] = clipped_curr_noise_level_cfg_x
            
        elif mode ==4:
            clipped_curr_noise_level_full[...,1] = clipped_curr_noise_level_cfg_y
        elif mode ==5:
            clipped_curr_noise_level_full[...,2] = clipped_curr_noise_level_cfg_z
        elif mode ==6:
            clipped_curr_noise_level_full_x= clipped_curr_noise_level.clone()
            clipped_curr_noise_level_full_x[...,1] = clipped_curr_noise_level_cfg_y
            clipped_curr_noise_level_full_x[...,2] = clipped_curr_noise_level_cfg_z
            
            clipped_curr_noise_level_full_y= clipped_curr_noise_level.clone()
            clipped_curr_noise_level_full_y[...,0] = clipped_curr_noise_level_cfg_x
            clipped_curr_noise_level_full_y[...,2] = clipped_curr_noise_level_cfg_z
            
            clipped_curr_noise_level_full_z= clipped_curr_noise_level.clone()
            clipped_curr_noise_level_full_z[...,1] = clipped_curr_noise_level_cfg_y
            clipped_curr_noise_level_full_z[...,0] = clipped_curr_noise_level_cfg_x
        
            
            
            
            
        
        external_uncond =deepcopy(external_cond)
        if tweight>0:
        #     external_uncond['uncond'] = True
        # else:
            external_uncond['uncond'] = True
            
        ## ALL COND
        if mode == 0:
            
            model_pred_xzuncond_textuncond = self.model_predictions(model=model,
            x=x,y=y_full,z=z_full,
            t=clipped_curr_noise_level_full,
            external_cond=external_cond,
            )
        elif mode ==1:
            model_pred_xzuncond_textuncond = self.model_predictions(model=model,
            x=x_full,y=y,z=z_full,
            t=clipped_curr_noise_level_full,
            external_cond=external_cond,
            )
            # model_pred_xzuncond_textuncond = self.model_predictions(model=model,
            # x=x,y=y,z=z,
            # t=clipped_curr_noise_level_full,
            # external_cond=external_cond,
            # )
        elif mode ==2:
            model_pred_xzuncond_textuncond = self.model_predictions(model=model,
            x=x_full,y=y_full,z=z,
            t=clipped_curr_noise_level_full,
            external_cond=external_cond,
            )
        elif mode ==3:
            model_pred_xzuncond_textuncond = self.model_predictions(model=model,
            x=x_full,y=y,z=z,
            t=clipped_curr_noise_level_full,
            external_cond=external_cond,
            )
        elif mode ==4:
            model_pred_xzuncond_textuncond = self.model_predictions(model=model,
            x=x,y=y_full,z=z,
            t=clipped_curr_noise_level_full,
            external_cond=external_cond,
            )
        elif mode ==5:
            model_pred_xzuncond_textuncond = self.model_predictions(model=model,
            x=x,y=y,z=z_full,
            t=clipped_curr_noise_level_full,
            external_cond=external_cond,
            )
        elif mode ==6:
            model_pred_uncond_textuncond_x = self.model_predictions(model=model,
            x=x,y=y_full,z=z_full,
            t=clipped_curr_noise_level_full_x,
            external_cond=external_cond,
            )
            model_pred_uncond_textuncond_y = self.model_predictions(model=model,
            x=x_full,y=y,z=z_full,
            t=clipped_curr_noise_level_full_y,
            external_cond=external_cond,
            )
            model_pred_uncond_textuncond_z = self.model_predictions(model=model,
            x=x_full,y=y_full,z=z,
            t=clipped_curr_noise_level_full_z,
            external_cond=external_cond,
            )
            
        
        # model_pred_textuncond = self.model_predictions(model=model,
        # x=x,y=y,z=z,
        # t=clipped_curr_noise_level_full,
        # external_cond=external_cond,
        # )
        ## Working
        if tweight>0:
            model_pred_textuncond = self.model_predictions(model=model,
            x=x,y=y,z=z,
            t=clipped_curr_noise_level,
            external_cond=external_uncond,
            )
        else:
            model_pred_textuncond = model_pred
        # model_pred_textuncond = self.model_predictions(model=model,
        # x=x,y=y,z=z,
        # t=clipped_curr_noise_level_full,
        # external_cond=external_uncond,
        # )
        if tweight>0:
            x_start_other_textuncond = model_pred_textuncond.pred_x_start1
            diff_textx =  x_start - x_start_other_textuncond
            
            x_start_other_textuncond = model_pred_textuncond.pred_x_start2
            diff_texty =  y_start - x_start_other_textuncond
            
            x_start_other_textuncond = model_pred_textuncond.pred_x_start3
            diff_textz =  z_start - x_start_other_textuncond
        # stupid
        
        # star 0 original
        # star 1 v
        # star 2 noise
        mask = external_uncond['mask']
       
        if mode ==0:
            # if not star:
            x_start_other_alluncond = model_pred_xzuncond_textuncond.pred_x_start1
            pred_noise_x_other_alluncond = model_pred_xzuncond_textuncond.pred_noise1
            x_start_other_textuncond = model_pred_textuncond.pred_x_start1
            pred_noise_x_other_textuncond = model_pred_textuncond.pred_noise1
           
            x_start,pred_noise_x = self.cfg_substep_one(star,mask,rescale,cfg_text,cfg_weight,clipped_curr_noise_level,0,x,x_start,pred_noise_x,x_start_other_alluncond,pred_noise_x_other_alluncond,x_start_other_textuncond,pred_noise_x_other_textuncond)

            
                

                
                
                
                
            
        elif mode ==1:
            
           
            y_start_other_alluncond = model_pred_xzuncond_textuncond.pred_x_start2
            pred_noise_y_other_alluncond = model_pred_xzuncond_textuncond.pred_noise2
            y_start_other_textuncond = model_pred_textuncond.pred_x_start2
            pred_noise_y_other_textuncond = model_pred_textuncond.pred_noise2
            
            y_start,pred_noise_y = self.cfg_substep_one(star,mask,rescale,cfg_text,cfg_weight,clipped_curr_noise_level,1,y,y_start,pred_noise_y,y_start_other_alluncond,pred_noise_y_other_alluncond,y_start_other_textuncond,pred_noise_y_other_textuncond)
    
        elif mode ==2:
        
        
            y_start_other_alluncond = model_pred_xzuncond_textuncond.pred_x_start3
            pred_noise_y_other_alluncond = model_pred_xzuncond_textuncond.pred_noise3
            y_start_other_textuncond = model_pred_textuncond.pred_x_start3
            pred_noise_y_other_textuncond = model_pred_textuncond.pred_noise3
            z_start,pred_noise_z = self.cfg_substep_one(star,mask,rescale,cfg_text,cfg_weight,clipped_curr_noise_level,2,z,z_start,pred_noise_z,y_start_other_alluncond,pred_noise_y_other_alluncond,y_start_other_textuncond,pred_noise_y_other_textuncond)
    
            # z_start = y_start_other_alluncond + cfg_text*(y_start_other_textuncond -y_start_other_alluncond)+cfg_weight*(z_start-y_start_other_textuncond)  
            # pred_noise_z = pred_noise_y_other_alluncond + cfg_text*(pred_noise_y_other_textuncond-pred_noise_y_other_alluncond)+cfg_weight*(pred_noise_z-pred_noise_y_other_textuncond)
        
        elif mode ==3:
            ############ y
            y_start_other_alluncond = model_pred_xzuncond_textuncond.pred_x_start2
            pred_noise_y_other_alluncond = model_pred_xzuncond_textuncond.pred_noise2
            y_start_other_textuncond = model_pred_textuncond.pred_x_start2
            pred_noise_y_other_textuncond = model_pred_textuncond.pred_noise2
            
            y_start,pred_noise_y = self.cfg_substep_one(star,mask,rescale,cfg_text,cfg_weight,clipped_curr_noise_level,1,y,y_start,pred_noise_y,y_start_other_alluncond,pred_noise_y_other_alluncond,y_start_other_textuncond,pred_noise_y_other_textuncond)

            
            # y_start = y_start_other_alluncond + cfg_text*(y_start_other_textuncond -y_start_other_alluncond)+cfg_weight*(y_start-y_start_other_textuncond)  
            # pred_noise_y = pred_noise_y_other_alluncond + cfg_text*(pred_noise_y_other_textuncond-pred_noise_y_other_alluncond)+cfg_weight*(pred_noise_y-pred_noise_y_other_textuncond)
            
            ################ Z
            y_start_other_alluncond = model_pred_xzuncond_textuncond.pred_x_start3
            pred_noise_y_other_alluncond = model_pred_xzuncond_textuncond.pred_noise3
            y_start_other_textuncond = model_pred_textuncond.pred_x_start3
            pred_noise_y_other_textuncond = model_pred_textuncond.pred_noise3
            
            z_start,pred_noise_z = self.cfg_substep_one(star,mask,rescale,cfg_text,cfg_weight,clipped_curr_noise_level,2,z,z_start,pred_noise_z,y_start_other_alluncond,pred_noise_y_other_alluncond,y_start_other_textuncond,pred_noise_y_other_textuncond)

            
            # z_start = y_start_other_alluncond + cfg_text*(y_start_other_textuncond -y_start_other_alluncond)+cfg_weight*(z_start-y_start_other_textuncond)  
            # pred_noise_z = pred_noise_y_other_alluncond + cfg_text*(pred_noise_y_other_textuncond-pred_noise_y_other_alluncond)+cfg_weight*(pred_noise_z-pred_noise_y_other_textuncond)
        elif mode==4:
            ##### X
            y_start_other_alluncond = model_pred_xzuncond_textuncond.pred_x_start1
            pred_noise_y_other_alluncond = model_pred_xzuncond_textuncond.pred_noise1
            y_start_other_textuncond = model_pred_textuncond.pred_x_start1
            pred_noise_y_other_textuncond = model_pred_textuncond.pred_noise1
            
            x_start,pred_noise_x = self.cfg_substep_one(star,mask,rescale,cfg_text,cfg_weight,clipped_curr_noise_level,0,x,x_start,pred_noise_x,y_start_other_alluncond,pred_noise_y_other_alluncond,y_start_other_textuncond,pred_noise_y_other_textuncond)

            
            # x_start = x_start_other_alluncond + cfg_text*(x_start_other_textuncond -x_start_other_alluncond)+cfg_weight*(x_start-x_start_other_textuncond)  
            # pred_noise_x = pred_noise_x_other_alluncond + cfg_text*(pred_noise_x_other_textuncond-pred_noise_x_other_alluncond)+cfg_weight*(pred_noise_x-pred_noise_x_other_textuncond)
            
            
            ################ Z
            y_start_other_alluncond = model_pred_xzuncond_textuncond.pred_x_start3
            pred_noise_y_other_alluncond = model_pred_xzuncond_textuncond.pred_noise3
            y_start_other_textuncond = model_pred_textuncond.pred_x_start3
            pred_noise_y_other_textuncond = model_pred_textuncond.pred_noise3
            
            z_start,pred_noise_z = self.cfg_substep_one(star,mask,rescale,cfg_text,cfg_weight,clipped_curr_noise_level,2,z,z_start,pred_noise_z,y_start_other_alluncond,pred_noise_y_other_alluncond,y_start_other_textuncond,pred_noise_y_other_textuncond)

            
            # z_start = y_start_other_alluncond + cfg_text*(y_start_other_textuncond -y_start_other_alluncond)+cfg_weight*(z_start-y_start_other_textuncond)  
            # pred_noise_z = pred_noise_y_other_alluncond + cfg_text*(pred_noise_y_other_textuncond-pred_noise_y_other_alluncond)+cfg_weight*(pred_noise_z-pred_noise_y_other_textuncond)
        elif mode ==5:
            ##### X
            y_start_other_alluncond = model_pred_xzuncond_textuncond.pred_x_start1
            pred_noise_y_other_alluncond = model_pred_xzuncond_textuncond.pred_noise1
            y_start_other_textuncond = model_pred_textuncond.pred_x_start1
            pred_noise_y_other_textuncond = model_pred_textuncond.pred_noise1
            x_start,pred_noise_x = self.cfg_substep_one(star,mask,rescale,cfg_text,cfg_weight,clipped_curr_noise_level,0,x,x_start,pred_noise_x,y_start_other_alluncond,pred_noise_y_other_alluncond,y_start_other_textuncond,pred_noise_y_other_textuncond)

            # x_start = x_start_other_alluncond + cfg_text*(x_start_other_textuncond -x_start_other_alluncond)+cfg_weight*(x_start-x_start_other_textuncond)  
            # pred_noise_x = pred_noise_x_other_alluncond + cfg_text*(pred_noise_x_other_textuncond-pred_noise_x_other_alluncond)+cfg_weight*(pred_noise_x-pred_noise_x_other_textuncond)
            
             ############ y
            y_start_other_alluncond = model_pred_xzuncond_textuncond.pred_x_start2
            pred_noise_y_other_alluncond = model_pred_xzuncond_textuncond.pred_noise2
            y_start_other_textuncond = model_pred_textuncond.pred_x_start2
            pred_noise_y_other_textuncond = model_pred_textuncond.pred_noise2
            y_start,pred_noise_y = self.cfg_substep_one(star,mask,rescale,cfg_text,cfg_weight,clipped_curr_noise_level,1,y,y_start,pred_noise_y,y_start_other_alluncond,pred_noise_y_other_alluncond,y_start_other_textuncond,pred_noise_y_other_textuncond)
        elif mode == 6:
            y_start_other_alluncond = model_pred_uncond_textuncond_x.pred_x_start1
            pred_noise_y_other_alluncond = model_pred_uncond_textuncond_x.pred_noise1
            y_start_other_textuncond = model_pred_textuncond.pred_x_start1
            pred_noise_y_other_textuncond = model_pred_textuncond.pred_noise1
            x_start,pred_noise_x = self.cfg_substep_one(star,mask,rescale,cfg_text,cfg_weight,clipped_curr_noise_level,0,x,x_start,pred_noise_x,y_start_other_alluncond,pred_noise_y_other_alluncond,y_start_other_textuncond,pred_noise_y_other_textuncond)
            
            
            y_start_other_alluncond = model_pred_uncond_textuncond_y.pred_x_start2
            pred_noise_y_other_alluncond = model_pred_uncond_textuncond_y.pred_noise2
            y_start_other_textuncond = model_pred_textuncond.pred_x_start2
            pred_noise_y_other_textuncond = model_pred_textuncond.pred_noise2
            y_start,pred_noise_y = self.cfg_substep_one(star,mask,rescale,cfg_text,cfg_weight,clipped_curr_noise_level,1,y,y_start,pred_noise_y,y_start_other_alluncond,pred_noise_y_other_alluncond,y_start_other_textuncond,pred_noise_y_other_textuncond)
            
            
            y_start_other_alluncond = model_pred_uncond_textuncond_z.pred_x_start3
            pred_noise_y_other_alluncond = model_pred_uncond_textuncond_z.pred_noise3
            y_start_other_textuncond = model_pred_textuncond.pred_x_start3
            pred_noise_y_other_textuncond = model_pred_textuncond.pred_noise3
            
            z_start,pred_noise_z = self.cfg_substep_one(star,mask,rescale,cfg_text,cfg_weight,clipped_curr_noise_level,2,z,z_start,pred_noise_z,y_start_other_alluncond,pred_noise_y_other_alluncond,y_start_other_textuncond,pred_noise_y_other_textuncond)
            # y_start = y_start_other_alluncond + cfg_text*(y_start_other_textuncond -y_start_other_alluncond)+cfg_weight*(y_start-y_start_other_textuncond)  
            # pred_noise_y = pred_noise_y_other_alluncond + cfg_text*(pred_noise_y_other_textuncond-pred_noise_y_other_alluncond)+cfg_weight*(pred_noise_y-pred_noise_y_other_textuncond)
        
        if tweight>0:
            x_start = x_start+ (tweight-1)*diff_textx
            y_start = y_start+ (tweight-1)*diff_texty
            z_start = z_start+ (tweight-1)*diff_textz
            pred_noise_x = self.predict_noise_from_start(x, clipped_curr_noise_level[...,0], x_start)
            pred_noise_y = self.predict_noise_from_start(y, clipped_curr_noise_level[...,1], y_start)

            pred_noise_z = self.predict_noise_from_start(z, clipped_curr_noise_level[...,2], z_start)
        
        
        return x_start,y_start,z_start,pred_noise_x,pred_noise_y,pred_noise_z
    
    
    
    
    def cfg_step_3(self,x_start,y_start,z_start,pred_noise_x,pred_noise_y,pred_noise_z,model,external_cond,inputs,inputs_full,clipped_curr_noise_level,
                                  clipp_noise_level_full,mode,cfg_weight,cfg_text,star,tweight,rescale,model_pred):
        if tweight ==0:
            cfg_text = cfg_weight # ALL COND, ORIGINAL 1.5
        else:
            cfg_text = tweight
        x,y,z = inputs
        x_full,y_full,z_full = inputs_full ## noiser
        clipped_curr_noise_level_cfg_x,clipped_curr_noise_level_cfg_y,clipped_curr_noise_level_cfg_z = clipp_noise_level_full
        
        
        
        clipped_curr_noise_level_full= clipped_curr_noise_level.clone()
        # clipped_curr_noise_level_yfull = clipped_curr_noise_level.clone()
        # clipped_curr_noise_level_xzuncond = clipped_curr_noise_level.clone() pure_noise_level
        # x_full,clipped_curr_noise_level_cfg_x,curr_noise_level_cfg_x = self.clip_noise_and_input(curr_noise_level_cfg_xyz[0::3].unsqueeze(0).repeat(x_full.shape[0],1),x_full)
        # y_full,clipped_curr_noise_level_cfg_y,curr_noise_level_cfg_y = self.clip_noise_and_input(curr_noise_level_cfg_xyz[1::3].unsqueeze(0).repeat(x_full.shape[0],1),y_full)
        # z_full,clipped_curr_noise_level_cfg_z,curr_noise_level_cfg_z = self.clip_noise_and_input(curr_noise_level_cfg_xyz[2::3].unsqueeze(0).repeat(x_full.shape[0],1),z_full)
        if mode ==0:
            clipped_curr_noise_level_full[...,1] = clipped_curr_noise_level_cfg_y
            clipped_curr_noise_level_full[...,2] = clipped_curr_noise_level_cfg_z
        elif mode ==1:
            clipped_curr_noise_level_full[...,0] = clipped_curr_noise_level_cfg_x
            clipped_curr_noise_level_full[...,2] = clipped_curr_noise_level_cfg_z
        elif mode ==2:
            clipped_curr_noise_level_full[...,0] = clipped_curr_noise_level_cfg_x
            clipped_curr_noise_level_full[...,1] = clipped_curr_noise_level_cfg_y
        elif mode ==3:
            clipped_curr_noise_level_full[...,0] = clipped_curr_noise_level_cfg_x
            
        elif mode ==4:
            clipped_curr_noise_level_full[...,1] = clipped_curr_noise_level_cfg_y
        elif mode ==5:
            clipped_curr_noise_level_full[...,2] = clipped_curr_noise_level_cfg_z
        elif mode ==6:
            clipped_curr_noise_level_full_x= clipped_curr_noise_level.clone()
            clipped_curr_noise_level_full_x[...,1] = clipped_curr_noise_level_cfg_y
            clipped_curr_noise_level_full_x[...,2] = clipped_curr_noise_level_cfg_z
            
            clipped_curr_noise_level_full_y= clipped_curr_noise_level.clone()
            clipped_curr_noise_level_full_y[...,0] = clipped_curr_noise_level_cfg_x
            clipped_curr_noise_level_full_y[...,2] = clipped_curr_noise_level_cfg_z
            
            clipped_curr_noise_level_full_z= clipped_curr_noise_level.clone()
            clipped_curr_noise_level_full_z[...,1] = clipped_curr_noise_level_cfg_y
            clipped_curr_noise_level_full_z[...,0] = clipped_curr_noise_level_cfg_x
        
            
            
            
                # clipped_curr_noise_level_yfull[...,1] = clipped_curr_noise_level_cfg_y
                # clipped_curr_noise_level_xzfull[...,0] = clipped_curr_noise_level_cfg_x
                # clipped_curr_noise_level_xzfull[...,2] = clipped_curr_noise_level_cfg_z            
        
        
        external_uncond =deepcopy(external_cond)
        if tweight>0:
        #     external_uncond['uncond'] = True
        # else:
            external_uncond['uncond'] = True
            
        ## ALL COND
        if mode == 0:
            
            model_pred_xzuncond_textuncond = self.model_predictions(model=model,
            x=x,y=y_full,z=z_full,
            t=clipped_curr_noise_level_full,
            external_cond=external_uncond,
            )
        elif mode ==1:
            model_pred_xzuncond_textuncond = self.model_predictions(model=model,
            x=x_full,y=y,z=z_full,
            t=clipped_curr_noise_level_full,
            external_cond=external_uncond,
            )
            # model_pred_xzuncond_textuncond = self.model_predictions(model=model,
            # x=x,y=y,z=z,
            # t=clipped_curr_noise_level_full,
            # external_cond=external_cond,
            # )
        elif mode ==2:
            model_pred_xzuncond_textuncond = self.model_predictions(model=model,
            x=x_full,y=y_full,z=z,
            t=clipped_curr_noise_level_full,
            external_cond=external_uncond,
            )
        elif mode ==3:
            model_pred_xzuncond_textuncond = self.model_predictions(model=model,
            x=x_full,y=y,z=z,
            t=clipped_curr_noise_level_full,
            external_cond=external_uncond,
            )
        elif mode ==4:
            model_pred_xzuncond_textuncond = self.model_predictions(model=model,
            x=x,y=y_full,z=z,
            t=clipped_curr_noise_level_full,
            external_cond=external_uncond,
            )
        elif mode ==5:
            model_pred_xzuncond_textuncond = self.model_predictions(model=model,
            x=x,y=y,z=z_full,
            t=clipped_curr_noise_level_full,
            external_cond=external_uncond,
            )
        elif mode ==6:
            model_pred_uncond_textuncond_x = self.model_predictions(model=model,
            x=x,y=y_full,z=z_full,
            t=clipped_curr_noise_level_full_x,
            external_cond=external_uncond,
            )
            model_pred_uncond_textuncond_y = self.model_predictions(model=model,
            x=x_full,y=y,z=z_full,
            t=clipped_curr_noise_level_full_y,
            external_cond=external_uncond,
            )
            model_pred_uncond_textuncond_z = self.model_predictions(model=model,
            x=x_full,y=y_full,z=z,
            t=clipped_curr_noise_level_full_z,
            external_cond=external_uncond,
            )
            
        
        # model_pred_textuncond = self.model_predictions(model=model,
        # x=x,y=y,z=z,
        # t=clipped_curr_noise_level_full,
        # external_cond=external_cond,
        # )
        ## Working
        if tweight>0:
            model_pred_textuncond = self.model_predictions(model=model,
            x=x,y=y,z=z,
            t=clipped_curr_noise_level,
            external_cond=external_uncond,
            )
        else:
            model_pred_textuncond = model_pred
        
        
        if tweight>0:
            x_start_other_textuncond = model_pred_textuncond.pred_x_start1
            diff_textx =  x_start - x_start_other_textuncond
            
            x_start_other_textuncond = model_pred_textuncond.pred_x_start2
            diff_texty =  y_start - x_start_other_textuncond
            
            x_start_other_textuncond = model_pred_textuncond.pred_x_start3
            diff_textz =  z_start - x_start_other_textuncond
        # model_pred_textuncond = self.model_predictions(model=model,
        # x=x,y=y,z=z,
        # t=clipped_curr_noise_level_full,
        # external_cond=external_uncond,
        # )
        # stupid
        
        # star 0 original
        # star 1 v
        # star 2 noise
        mask = external_uncond['mask']
        
        if mode ==0:
            # if not star:
            x_start_other_alluncond = model_pred_xzuncond_textuncond.pred_x_start1
            pred_noise_x_other_alluncond = model_pred_xzuncond_textuncond.pred_noise1
            x_start_other_textuncond = model_pred_textuncond.pred_x_start1
            pred_noise_x_other_textuncond = model_pred_textuncond.pred_noise1
            timer1 = 1
            timer2= 1
            x_start,pred_noise_x = self.cfg_substep_one(star,mask,rescale,cfg_text,cfg_weight,clipped_curr_noise_level,0,x,x_start,pred_noise_x,x_start_other_alluncond,pred_noise_x_other_alluncond,x_start_other_textuncond,pred_noise_x_other_textuncond)

            # if star ==0:
            #     x_start = x_start_other_alluncond + cfg_text*(x_start_other_textuncond -x_start_other_alluncond)+cfg_weight*(x_start-x_start_other_textuncond)  
            #     pred_noise_x = pred_noise_x_other_alluncond + cfg_text*(pred_noise_x_other_textuncond-pred_noise_x_other_alluncond*timer1)+cfg_weight*(pred_noise_x-pred_noise_x_other_textuncond)
            # elif star ==2:
            #     timer1 = self.calculate_multiplier(pred_noise_x_other_alluncond,pred_noise_x_other_textuncond)
            #     timer2 = self.calculate_multiplier(pred_noise_x_other_textuncond,pred_noise_x)
            #     pred_noise_x = pred_noise_x_other_alluncond + cfg_text*(pred_noise_x_other_textuncond-pred_noise_x_other_alluncond*timer1)+cfg_weight*(pred_noise_x-pred_noise_x_other_textuncond*timer2)
            #     x_start = self.predict_start_from_noise(x,clipped_curr_noise_level[...,0],pred_noise_x)
            # elif star==1:
                
            #     v_x_other_alluncond = self.predict_v( x_start, clipped_curr_noise_level[...,0], pred_noise_x_other_alluncond)
            #     v_x_other_textuncond = self.predict_v( x_start, clipped_curr_noise_level[...,0], pred_noise_x_other_textuncond)
            #     v_x = self.predict_v( x_start, clipped_curr_noise_level[...,0], pred_noise_x)
            #     timer1 = self.calculate_multiplier(v_x_other_alluncond,v_x_other_textuncond)
            #     timer2 = self.calculate_multiplier(v_x_other_textuncond,v_x)
                
            #     v_x = v_x_other_alluncond + cfg_text*(v_x_other_textuncond-v_x_other_alluncond*timer1)+cfg_weight*(v_x-v_x_other_textuncond*timer2)
            #     x_start = self.predict_start_from_v( x, clipped_curr_noise_level[...,0], v_x)
            #     pred_noise_x = self.predict_noise_from_start(x, clipped_curr_noise_level[...,0], x_start)
                

                
                
                
                
            
        elif mode ==1:
            
            timer1= 1
            timer2=1
            y_start_other_alluncond = model_pred_xzuncond_textuncond.pred_x_start2
            pred_noise_y_other_alluncond = model_pred_xzuncond_textuncond.pred_noise2
            y_start_other_textuncond = model_pred_textuncond.pred_x_start2
            pred_noise_y_other_textuncond = model_pred_textuncond.pred_noise2
            
            y_start,pred_noise_y = self.cfg_substep_one(star,mask,rescale,cfg_text,cfg_weight,clipped_curr_noise_level,1,y,y_start,pred_noise_y,y_start_other_alluncond,pred_noise_y_other_alluncond,y_start_other_textuncond,pred_noise_y_other_textuncond)
            
        elif mode ==2:
        
        
            y_start_other_alluncond = model_pred_xzuncond_textuncond.pred_x_start3
            pred_noise_y_other_alluncond = model_pred_xzuncond_textuncond.pred_noise3
            y_start_other_textuncond = model_pred_textuncond.pred_x_start3
            pred_noise_y_other_textuncond = model_pred_textuncond.pred_noise3
            z_start,pred_noise_z = self.cfg_substep_one(star,mask,rescale,cfg_text,cfg_weight,clipped_curr_noise_level,2,z,z_start,pred_noise_z,y_start_other_alluncond,pred_noise_y_other_alluncond,y_start_other_textuncond,pred_noise_y_other_textuncond)
    
            # z_start = y_start_other_alluncond + cfg_text*(y_start_other_textuncond -y_start_other_alluncond)+cfg_weight*(z_start-y_start_other_textuncond)  
            # pred_noise_z = pred_noise_y_other_alluncond + cfg_text*(pred_noise_y_other_textuncond-pred_noise_y_other_alluncond)+cfg_weight*(pred_noise_z-pred_noise_y_other_textuncond)
        
        elif mode ==3:
            ############ y
            y_start_other_alluncond = model_pred_xzuncond_textuncond.pred_x_start2
            pred_noise_y_other_alluncond = model_pred_xzuncond_textuncond.pred_noise2
            y_start_other_textuncond = model_pred_textuncond.pred_x_start2
            pred_noise_y_other_textuncond = model_pred_textuncond.pred_noise2
            
            y_start,pred_noise_y = self.cfg_substep_one(star,mask,rescale,cfg_text,cfg_weight,clipped_curr_noise_level,1,y,y_start,pred_noise_y,y_start_other_alluncond,pred_noise_y_other_alluncond,y_start_other_textuncond,pred_noise_y_other_textuncond)

            
            # y_start = y_start_other_alluncond + cfg_text*(y_start_other_textuncond -y_start_other_alluncond)+cfg_weight*(y_start-y_start_other_textuncond)  
            # pred_noise_y = pred_noise_y_other_alluncond + cfg_text*(pred_noise_y_other_textuncond-pred_noise_y_other_alluncond)+cfg_weight*(pred_noise_y-pred_noise_y_other_textuncond)
            
            ################ Z
            y_start_other_alluncond = model_pred_xzuncond_textuncond.pred_x_start3
            pred_noise_y_other_alluncond = model_pred_xzuncond_textuncond.pred_noise3
            y_start_other_textuncond = model_pred_textuncond.pred_x_start3
            pred_noise_y_other_textuncond = model_pred_textuncond.pred_noise3
            
            z_start,pred_noise_z = self.cfg_substep_one(star,mask,rescale,cfg_text,cfg_weight,clipped_curr_noise_level,2,z,z_start,pred_noise_z,y_start_other_alluncond,pred_noise_y_other_alluncond,y_start_other_textuncond,pred_noise_y_other_textuncond)

            
            # z_start = y_start_other_alluncond + cfg_text*(y_start_other_textuncond -y_start_other_alluncond)+cfg_weight*(z_start-y_start_other_textuncond)  
            # pred_noise_z = pred_noise_y_other_alluncond + cfg_text*(pred_noise_y_other_textuncond-pred_noise_y_other_alluncond)+cfg_weight*(pred_noise_z-pred_noise_y_other_textuncond)
        elif mode==4:
            ##### X
            y_start_other_alluncond = model_pred_xzuncond_textuncond.pred_x_start1
            pred_noise_y_other_alluncond = model_pred_xzuncond_textuncond.pred_noise1
            y_start_other_textuncond = model_pred_textuncond.pred_x_start1
            pred_noise_y_other_textuncond = model_pred_textuncond.pred_noise1
            
            x_start,pred_noise_x = self.cfg_substep_one(star,mask,rescale,cfg_text,cfg_weight,clipped_curr_noise_level,0,x,x_start,pred_noise_x,y_start_other_alluncond,pred_noise_y_other_alluncond,y_start_other_textuncond,pred_noise_y_other_textuncond)

            
            # x_start = x_start_other_alluncond + cfg_text*(x_start_other_textuncond -x_start_other_alluncond)+cfg_weight*(x_start-x_start_other_textuncond)  
            # pred_noise_x = pred_noise_x_other_alluncond + cfg_text*(pred_noise_x_other_textuncond-pred_noise_x_other_alluncond)+cfg_weight*(pred_noise_x-pred_noise_x_other_textuncond)
            
            
            ################ Z
            y_start_other_alluncond = model_pred_xzuncond_textuncond.pred_x_start3
            pred_noise_y_other_alluncond = model_pred_xzuncond_textuncond.pred_noise3
            y_start_other_textuncond = model_pred_textuncond.pred_x_start3
            pred_noise_y_other_textuncond = model_pred_textuncond.pred_noise3
            
            z_start,pred_noise_z = self.cfg_substep_one(star,mask,rescale,cfg_text,cfg_weight,clipped_curr_noise_level,2,z,z_start,pred_noise_z,y_start_other_alluncond,pred_noise_y_other_alluncond,y_start_other_textuncond,pred_noise_y_other_textuncond)

            
            # z_start = y_start_other_alluncond + cfg_text*(y_start_other_textuncond -y_start_other_alluncond)+cfg_weight*(z_start-y_start_other_textuncond)  
            # pred_noise_z = pred_noise_y_other_alluncond + cfg_text*(pred_noise_y_other_textuncond-pred_noise_y_other_alluncond)+cfg_weight*(pred_noise_z-pred_noise_y_other_textuncond)
        elif mode ==5:
            ##### X
            y_start_other_alluncond = model_pred_xzuncond_textuncond.pred_x_start1
            pred_noise_y_other_alluncond = model_pred_xzuncond_textuncond.pred_noise1
            y_start_other_textuncond = model_pred_textuncond.pred_x_start1
            pred_noise_y_other_textuncond = model_pred_textuncond.pred_noise1
            x_start,pred_noise_x = self.cfg_substep_one(star,mask,rescale,cfg_text,cfg_weight,clipped_curr_noise_level,0,x,x_start,pred_noise_x,y_start_other_alluncond,pred_noise_y_other_alluncond,y_start_other_textuncond,pred_noise_y_other_textuncond)

            # x_start = x_start_other_alluncond + cfg_text*(x_start_other_textuncond -x_start_other_alluncond)+cfg_weight*(x_start-x_start_other_textuncond)  
            # pred_noise_x = pred_noise_x_other_alluncond + cfg_text*(pred_noise_x_other_textuncond-pred_noise_x_other_alluncond)+cfg_weight*(pred_noise_x-pred_noise_x_other_textuncond)
            
             ############ y
            y_start_other_alluncond = model_pred_xzuncond_textuncond.pred_x_start2
            pred_noise_y_other_alluncond = model_pred_xzuncond_textuncond.pred_noise2
            y_start_other_textuncond = model_pred_textuncond.pred_x_start2
            pred_noise_y_other_textuncond = model_pred_textuncond.pred_noise2
            y_start,pred_noise_y = self.cfg_substep_one(star,mask,rescale,cfg_text,cfg_weight,clipped_curr_noise_level,1,y,y_start,pred_noise_y,y_start_other_alluncond,pred_noise_y_other_alluncond,y_start_other_textuncond,pred_noise_y_other_textuncond)
        elif mode == 6:
            
            y_start_other_alluncond = model_pred_uncond_textuncond_x.pred_x_start1
            pred_noise_y_other_alluncond = model_pred_uncond_textuncond_x.pred_noise1
            y_start_other_textuncond = model_pred_textuncond.pred_x_start1
            pred_noise_y_other_textuncond = model_pred_textuncond.pred_noise1
            x_start,pred_noise_x = self.cfg_substep_one(star,mask,rescale,cfg_text,cfg_weight,clipped_curr_noise_level,0,x,x_start,pred_noise_x,y_start_other_alluncond,pred_noise_y_other_alluncond,y_start_other_textuncond,pred_noise_y_other_textuncond)
            
            
            y_start_other_alluncond = model_pred_uncond_textuncond_y.pred_x_start2
            pred_noise_y_other_alluncond = model_pred_uncond_textuncond_y.pred_noise2
            y_start_other_textuncond = model_pred_textuncond.pred_x_start2
            pred_noise_y_other_textuncond = model_pred_textuncond.pred_noise2
            y_start,pred_noise_y = self.cfg_substep_one(star,mask,rescale,cfg_text,cfg_weight,clipped_curr_noise_level,1,y,y_start,pred_noise_y,y_start_other_alluncond,pred_noise_y_other_alluncond,y_start_other_textuncond,pred_noise_y_other_textuncond)
            
            
            y_start_other_alluncond = model_pred_uncond_textuncond_z.pred_x_start3
            pred_noise_y_other_alluncond = model_pred_uncond_textuncond_z.pred_noise3
            y_start_other_textuncond = model_pred_textuncond.pred_x_start3
            pred_noise_y_other_textuncond = model_pred_textuncond.pred_noise3
            
            z_start,pred_noise_z = self.cfg_substep_one(star,mask,rescale,cfg_text,cfg_weight,clipped_curr_noise_level,2,z,z_start,pred_noise_z,y_start_other_alluncond,pred_noise_y_other_alluncond,y_start_other_textuncond,pred_noise_y_other_textuncond)
            # y_start = y_start_other_alluncond + cfg_text*(y_start_other_textuncond -y_start_other_alluncond)+cfg_weight*(y_start-y_start_other_textuncond)  
            # pred_noise_y = pred_noise_y_other_alluncond + cfg_text*(pred_noise_y_other_textuncond-pred_noise_y_other_alluncond)+cfg_weight*(pred_noise_y-pred_noise_y_other_textuncond)
        
        if tweight>0:
            x_start = x_start+ (tweight-1)*diff_textx
            y_start = y_start+ (tweight-1)*diff_texty
            z_start = z_start+ (tweight-1)*diff_textz
            pred_noise_x = self.predict_noise_from_start(x, clipped_curr_noise_level[...,0], x_start)
            pred_noise_y = self.predict_noise_from_start(y, clipped_curr_noise_level[...,1], y_start)

            pred_noise_z = self.predict_noise_from_start(z, clipped_curr_noise_level[...,2], z_start)


            
        
        
        
        return x_start,y_start,z_start,pred_noise_x,pred_noise_y,pred_noise_z