import copy
import os
import sys
sys.path.append('.')
sys.path.append('..')
import torch.nn as nn

import numpy as np
from utils.loss_util import masked_l2,diff_l1,diff_l2,huber_loss,huber_loss2
import re
from os.path import join as pjoin
from typing import Optional

import blobfile as bf
import torch
from torch.optim import AdamW


from diffusion_forcing import logger
from utils import dist_util

from accelerate.utils import tqdm


from utils.model_util import load_model_wo_clip
from pytorch3d.transforms import *
from accelerate import Accelerator,DeepSpeedPlugin
from accelerate.utils import DistributedDataParallelKwargs

import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from chamfer_distance import ChamferDistance as chamfer_dist
import math

INITIAL_LOG_LOSS_SCALE = 20.0
def align_input_dtype(module, inputs):
    # Some modules (Dropout, GELU, custom view layers) have **no parameters**
    params = list(module.parameters(recurse=False))
    if not params:                     # <-- fixes the StopIteration you hit
        return

    tgt_dtype = params[0].dtype        # weight / bias dtype
    x, *rest = inputs
    if isinstance(x, torch.Tensor) and x.dtype != tgt_dtype:
        x = x.to(tgt_dtype)
    return (x, *rest)

def patch_model(model: nn.Module):
    NEED_CAST = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d,
                 nn.LayerNorm, nn.GroupNorm, nn.MultiheadAttention)
    for m in model.modules():
        if isinstance(m, NEED_CAST):
            m.register_forward_pre_hook(align_input_dtype, prepend=True)

class TrainLoop:
    def __init__(self, args, train_platform, model, diffusion, data):
        
        ## new compile:
        
        self.args = args
        self.chd = chamfer_dist()
        if args.ltype =='l2':
            self.loss_type = diff_l2
        elif args.ltype=='l1':
            self.loss_type = diff_l1
        elif args.ltype=='huber':
            self.loss_type = huber_loss
        else:
            self.loss_type = huber_loss2
        if 'schedule' in args.ltype:
            self.loss_schedule = 1
        else:
            self.loss_schedule=0
            
        self.body_index = list(range(22*3))+list(range(52*3,52*3+self.args.foot))
        self.hand_index = list(range(22*3,52*3+30))

        
            
       
        self.dataset = args.dataset
        self.train_platform = train_platform
        
        
       
        self.diffusion = diffusion
        self.cond_mode = model.cond_mode
        self.data = data
        self.batch_size = args.batch_size
        self.microbatch = args.batch_size  # deprecating this option
        self.lr = args.lr
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.resume_checkpoint = args.resume_checkpoint
        
        self.use_fp16 = False  # deprecating this option
        self.fp16_scale_growth = 1e-3  # deprecating this option
        self.weight_decay = args.weight_decay
        self.lr_anneal_steps = args.lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size # * dist.get_world_size()
        self.num_steps = args.num_steps
        self.num_epochs = self.num_steps // len(self.data) + 1

        self.sync_cuda = torch.cuda.is_available()

       
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        

        self.accelerator = Accelerator( #deepspeed_plugin=ds_plugin,
            mixed_precision="bf16",kwargs_handlers=[ddp_kwargs]
        )
        self.save_dir = args.save_dir
        
        self.overwrite = args.overwrite
        
        self.model = model
        self.model_avg = None
        if self.args.text_encoder_type in ['clip','longclip']:
            self.model.clip_model = None
        if self.accelerator.is_main_process:
            if self.args.use_ema:
                self.model_avg = copy.deepcopy(self.model)
        self._load_and_sync_parameters()
        

        if self.args.use_ema:
            
            EPS = 1e-6
            self.opt = AdamW(
            
                self.model.parameters(), #  if self.use_fp16 else self.mp_trainer.master_params)
                lr=self.lr,
                weight_decay=0,
                betas=(0.9, self.args.adam_beta2),
                eps = EPS,
                fused= True
            )
        else:
            self.opt = AdamW(
                self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay,eps = 1e-4
            )
        

        ds_cfg = {
            "train_micro_batch_size_per_gpu": 32,
            
            "fp16": { "enabled": True},               # ← changed
            "zero_optimization": {"stage": 2},
            "attention_plugin": {
                "type": "flash_attn",
                "params": { "flash_attn_fusion": "fa2" }
            }
        }

        ds_plugin = DeepSpeedPlugin(
            zero_stage=2,
            gradient_accumulation_steps=1,
            hf_ds_config=ds_cfg,
        )


        warmup_schedule = LinearLR(
                self.opt, 
                start_factor=0.05, 
                end_factor=1.0, 
                total_iters=self.lr_anneal_steps
            )
        cosine_lr_scheduler = CosineAnnealingLR(self.opt, T_max=350000, eta_min=1e-6)

        self.scheduler = SequentialLR(
            self.opt,
            schedulers=[warmup_schedule, cosine_lr_scheduler],
            milestones=[self.lr_anneal_steps]
        )
        
        if self.resume_step:
            self._load_optimizer_state()
        
        
        
        
        if os.path.isdir(self.resume_checkpoint):
            self.resume_checkpoint = self.find_resume_checkpoint()
        
        

        self.model, self.opt, self.data,self.scheduler,self.diffusion= self.accelerator.prepare(self.model, self.opt, self.data,self.scheduler,self.diffusion)
        
        # dataset = self.args.dataset
        # if dataset == 'interact':
            
        #     datasets = ['behave', 'intercap', 'neuraldome', 'chairs', 'omomo', 'imhd', 'grab']
        #     # self.datasets = ['neuraldome', 'chairs', 'imhd', 'grab']
        # elif dataset == 'interact_high':
        #     datasets = ['neuraldome', 'chairs', 'imhd', 'grab']
            
        # elif dataset == 'interact_wobehave':
        #     datasets = [ 'neuraldome', 'chairs', 'omomo', 'imhd', 'grab']
        # elif dataset == 'interact_wobehave_correct':
        #     datasets = [ 'neuraldome', 'chairs', 'omomo_correct', 'imhd', 'grab']
        # elif dataset == 'interact_correct':
        #     datasets = [ 'neuraldome', 'chairs', 'omomo_correct', 'imhd', 'grab','behave_correct','intercap_correct']
        # elif dataset == 'interact_woomomo_correct':
        #     datasets = [ 'neuraldome', 'chairs', 'imhd', 'grab','behave_correct','intercap_correct']
        # elif dataset == 'interact_behave_omomo':
        #     datasets = [ 'behave_correct', 'omomo_correct','imhd']
        # else:
        #     datasets = [dataset]
        # self.sdf_dct={}
    
                    
                
            
        
        
        self.normalize = self.args.normalize
        if self.normalize:
            self.mean = torch.from_numpy(data.dataset.t2m_dataset.mean.reshape(1,-1,1,1)).to(self.accelerator.device)
            self.std = torch.from_numpy(data.dataset.t2m_dataset.std.reshape(1,-1,1,1)).to(self.accelerator.device)
            upper = self.args.joint_nums*3+self.args.foot + 30            
                
            if self.normalize:
                self.mean_h = self.mean[:,self.body_index].float()
                self.std_h = self.std[:,self.body_index].float()
                self.mean_hand = self.mean[:,self.hand_index].float()
                self.std_hand = self.std[:,self.hand_index].float()
                self.mean_o = self.mean[:,upper:].float()
                self.std_o = self.std[:,upper:].float()
        
        print("Parameter dtype:", next(self.model.parameters()).dtype)
        print("Native AMP enabled:", self.accelerator.native_amp)
        print("Mixed precision mode:", self.accelerator.state.mixed_precision)
        
       
        bones = [
            # lower body
            [0, 2], [2, 5], [5, 8], [8,11],
            [0, 1], [1, 4], [4, 7], [7,10],

            # spine & head
            [0, 3], [3, 6], [6, 9], [9,12], [12,15],

            # right arm
            [9,14], [14,17], [17,19], [19,21],

            # left arm
            [9,13], [13,16], [16,18], [18,20],

            # left hand
            [20,22],[22,23],[23,24],
            [20,25],[25,26],[26,27],
            [20,28],[28,29],[29,30],
            [20,31],[31,32],[32,33],
            [20,34],[34,35],[35,36],

            # right hand
            [21,37],[37,38],[38,39],
            [21,40],[40,41],[41,42],
            [21,43],[43,44],[44,45],
            [21,46],[46,47],[47,48],
            [21,49],[49,50],[50,51]
        ]
        self.bones = torch.tensor(bones,dtype=torch.long).to(self.accelerator.device)
        self.bones_body = [[0, 2], [2, 5], [5, 8], [8,11],
                [0, 1], [1, 4], [4, 7], [7,10],

                # spine & head
                [0, 3], [3, 6], [6, 9], [9,12], [12,15],

                # right arm
                [9,14], [14,17], [17,19], [19,21],

                # left arm
                [9,13], [13,16], [16,18], [18,20]]
        self.bones_hand = [[0, 1], [1, 2], [3, 4], [4, 5], [6, 7], [7, 8], [9, 10], [10, 11], [12, 13], [13, 14], [15, 16], [16, 17], [18, 19], [19, 20], [21, 22], [22, 23], [24, 25], [25, 26], [27, 28], [28, 29]]
        
        self.bones_hand = torch.tensor(self.bones_hand,dtype=torch.long).to(self.accelerator.device)
        self.bones_body = torch.tensor(self.bones_body,dtype=torch.long).to(self.accelerator.device)
        
        if self.accelerator.is_main_process:
            logger.log('SET_PROGRESS_BAR')
            self.progress_bar = tqdm(self.data)
        


        self.device = torch.device("cpu")
        if torch.cuda.is_available() and dist_util.dev() != 'cpu':
            self.device = torch.device(dist_util.dev())

        self.schedule_sampler_type = 'uniform'
        
        self.eval_wrapper, self.eval_data, self.eval_gt_data = None, None, None
        
        self.use_ddp = False
        self.ddp_model = self.model
    
   
    def _load_and_sync_parameters(self):
        resume_checkpoint = self.find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            
            self.step += 1  
            
            
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint) 
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            state_dict = dist_util.load_state_dict(
                resume_checkpoint, map_location=dist_util.dev())

            if 'model_avg' in state_dict:
                print('loading both model and model_avg')
                state_dict, state_dict_avg = state_dict['model'], state_dict[
                    'model_avg']
                load_model_wo_clip(self.model, state_dict)
                if self.accelerator.is_main_process:
                    load_model_wo_clip(self.model_avg, state_dict_avg)
            else:
                load_model_wo_clip(self.model, state_dict)
                if self.args.use_ema and self.accelerator.is_main_process:
                    print('loading model_avg from model')
                    self.model_avg.load_state_dict(self.model.state_dict(), strict=False)

            

    def _load_optimizer_state(self):
        main_checkpoint = self.find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:09}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            if 'scaler' in state_dict.keys():
                self.opt.load_state_dict(state_dict['opt'])
                # for pg in self.opt.param_groups:
                #     pg['lr'] *= 5.0
                self.accelerator.scaler.load_state_dict(state_dict["scaler"])
                self.scheduler.load_state_dict(state_dict["scheduler"])
            else:
                self.opt.load_state_dict(state_dict['opt'])
                self.scheduler.load_state_dict(state_dict["scheduler"])
               
            for group in self.opt.param_groups:
                group['weight_decay'] = 0.0
            self.opt.param_groups[0]['capturable'] = True

    # def cond_modifiers(self, cond, motion):
    #     # All modifiers must be in-place
    #     self.target_cond_modifier(cond, motion)
    
    # def target_cond_modifier(self, cond, motion):
    #     if self.args.multi_target_cond:
    #         batch_size = motion.shape[0]
    #         cond['target_joint_names'], cond['is_heading'] = sample_goal(batch_size, motion.device, self.args.target_joint_names)

    #         cond['target_cond'] = get_target_location(motion, 
    #                                                   self.data.dataset.mean[None, :, None, None], 
    #                                                   self.data.dataset.std[None, :, None, None], 
    #                                                   cond['lengths'], 
    #                                                   self.data.dataset.t2m_dataset.opt.joints_num, self.model.all_goal_joint_names, cond['target_joint_names'], cond['is_heading']).detach()

    def run_loop(self):
        print('train steps:', self.num_steps)
        for epoch in range(self.num_epochs):
            training_losses = []
            for motion, cond in (self.data):
                
                motion = motion.to(self.accelerator.device)
                

                train_loss = self.run_step(motion, cond)
                if self.total_step() % self.log_interval == 0 and self.accelerator.is_main_process:
                    for k,v in logger.get_current().dumpkvs().items():
                        if k == 'loss':
                            print('step[{}]: loss[{:0.5f}]'.format(self.total_step(), v))

                        if k in ['step', 'samples'] or '_q' in k:
                            continue
                        else:
                            self.train_platform.report_scalar(name=k, value=v, iteration=self.total_step(), group_name='Loss')

                if self.total_step() % self.save_interval == 0 and self.accelerator.is_main_process:
                    self.save()
                    self.model.eval()
                    if self.args.use_ema:
                        self.model_avg.eval()
                    self.model.train()
                    if self.args.use_ema:
                        self.model_avg.train()

                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.total_step() > 0:
                        return
        
        if (self.total_step() - 1) % self.save_interval != 0 and self.accelerator.is_main_process:
            self.save()

    


    def run_step(self, batch, cond):
        self.opt.zero_grad()
        with self.accelerator.autocast(): 
            loss = self.forward_backward(batch, cond)
        self.accelerator.backward(loss)

        
        self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=1)

        step_was_run = self.opt.step()
        if not self.opt.step_was_skipped:
            
            self.scheduler.step()
        else:
            print('Overflow Step')
            
        if self.accelerator.is_main_process:
            self.step = self.step+1
            if self.step%100 ==0:
                
                self.progress_bar.set_postfix(step=self.step)
            if self.args.use_ema:
                raw_model = self.accelerator.unwrap_model(self.model)
                for p, avg_p in zip(raw_model.parameters(), self.model_avg.parameters()):
                    avg_p.data.mul_(self.args.avg_model_beta).add_(p.data, alpha=1 - self.args.avg_model_beta)
        self.log_step()
        return loss.detach().unsqueeze(0)

    def update_average_model(self):
        # update the average model using exponential moving average
        if self.args.use_ema:
            params = self.model.parameters(
            ) if self.use_fp16 else self.mp_trainer.master_params
            for param, avg_param in zip(params, self.model_avg.parameters()):
            
                avg_param.data.mul_(self.args.avg_model_beta).add_(
                    param.data, alpha=1 - self.args.avg_model_beta)
    
    def calc_loss_split_hand(self,gt_h,model_output_h,gt_o,model_output_o,gt_hand,model_output_hand,mask,weight_h=1.0,weight_o=1.0,weight_hand=1.0,reg_h_t=1.0,reg_o_t=1.0,reg_hand_t=1.0,micro_cond=None):
        loss_dict = {}
        TIMESTEP = self.args.diffusion_steps
        reg_h = 1-reg_h_t/TIMESTEP
        reg_o = 1-reg_o_t/TIMESTEP
        reg_hand = 1-reg_hand_t/TIMESTEP
        alpha = - math.log(self.args.huber_c) / TIMESTEP
        if self.loss_schedule:
            huber_c_h = torch.exp(-alpha * reg_h_t)
            huber_c_hand = torch.exp(-alpha * reg_hand_t)
            huber_c_o = torch.exp(-alpha * reg_o_t)
        else:
            huber_c_h = self.args.huber_c
            huber_c_hand = self.args.huber_c
            huber_c_o = self.args.huber_c
            
        if self.args.uniform_reg:
            reg_h =1
            reg_o = 1
            reg_hand =1
    
        use_mean = self.args.mean
        self.use_mean = use_mean
       
            
        B = gt_h.shape[0]
        T = gt_h.shape[-1]
        
        loss_mse_h_b = masked_l2(gt_h, model_output_h, mask,entries_norm=use_mean,w=weight_h,loss_fn=self.loss_type,delta=huber_c_h)
        loss_mse_h_h = masked_l2(gt_hand, model_output_hand, mask,entries_norm=use_mean,w=weight_hand,delta=huber_c_hand,loss_fn=self.loss_type)
        loss_dict['loss_mse_h_b'] = loss_mse_h_b
        loss_dict['loss_mse_h_h'] = loss_mse_h_h
        loss_mse_h = (loss_mse_h_b*self.args.body_w + loss_mse_h_h*self.args.hw)
        loss_dict['loss_mse_h'] = loss_mse_h
        loss_mse_h = loss_mse_h
                

        loss_mse_o_rot =  masked_l2(gt_o[:,:6], model_output_o[:,:6], mask,entries_norm=False,w=weight_o,delta=huber_c_o,loss_fn=self.loss_type)
        loss_mse_o_trans =  masked_l2(gt_o[:,6:], model_output_o[:,6:], mask,entries_norm=False,w=weight_o,delta=huber_c_o,loss_fn=self.loss_type)
        loss_dict['loss_mse_o_trans'] = loss_mse_o_trans/(model_output_o.shape[1]-6) # 0.05
        loss_dict['loss_mse_o_rot'] = loss_mse_o_rot/6 # 0.02
        
        loss_mse =loss_mse_h + (loss_mse_o_trans+self.args.obj_trans_w*loss_mse_o_rot)*self.args.obj_w/(model_output_o.shape[1])
     
            
        if self.args.normalize:
           
            gt_h = gt_h*self.std_h+self.mean_h
            gt_o = gt_o*self.std_o+self.mean_o
            gt_hand = gt_hand*self.std_hand+self.mean_hand
            
            model_output_h = model_output_h*self.std_h+self.mean_h
            model_output_o = model_output_o*self.std_o+self.mean_o
            model_output_hand = model_output_hand*self.std_hand+self.mean_hand
            
            
        gt_skel_body = gt_h[:,:22*3].reshape(B,22,3,1,T)
        out_skel_body = model_output_h[:,:22*3].reshape(B,22,3,1,T)
    
        gt_skel_hand = gt_hand[:,:30*3].reshape(B,30,3,1,T)
        out_skel_hand = model_output_hand[:,:30*3].reshape(B,30,3,1,T)
        weight_contact = 0.0
        
        rot_6d = model_output_o[:, :6, 0].permute(0, 2, 1)   
        rot_6d_gt = gt_o[:, :6, 0].permute(0, 2, 1)           
        ROT = rotation_6d_to_matrix(rot_6d.float())              
        ROT_gt = rotation_6d_to_matrix(rot_6d_gt.float())
        trans = model_output_o[:, 6:9, 0].permute(0, 2, 1)      
        trans_gt = gt_o[:, 6:9, 0].permute(0, 2, 1) 
        
        
        if self.args.contact_weight>0:
            
            object_points = micro_cond['y']['obj_points']
            B, N, C = object_points.shape

            
            relative_ids = micro_cond['y']['relative_ids']
            relative_ids_body = relative_ids[...,:22]
            relative_ids_hand = relative_ids[...,22:]

            T = relative_ids.shape[1]
            obj_exp  = object_points.unsqueeze(1).expand(-1, T, -1, -1) 
            
            idx_exp  = relative_ids_body.unsqueeze(-1)            # [B, T, 52, 1]
            idx_exp  = idx_exp.expand(-1, -1, -1, 3)          # [B, T, 52, 3]

            obj_static_52t_body     = torch.gather(obj_exp, dim=2, index=idx_exp)  
            pts_gt_body = torch.matmul(obj_static_52t_body, ROT_gt.transpose(-1, -2)) +trans_gt.unsqueeze(2) # B,T,52,3
            pts_gt_body = pts_gt_body.permute(0,2,3,1)
            delta_gt_body = gt_skel_body.squeeze(3) - pts_gt_body
            idx_exp  = relative_ids_hand.unsqueeze(-1)            # [B, T, 52, 1]
            idx_exp  = idx_exp.expand(-1, -1, -1, 3)          # [B, T, 52, 3]

            obj_static_52t_hand     = torch.gather(obj_exp, dim=2, index=idx_exp)  
            pts_gt_hand = torch.matmul(obj_static_52t_hand, ROT_gt.transpose(-1, -2)) +trans_gt.unsqueeze(2) # B,T,52,3
            pts_gt_hand = pts_gt_hand.permute(0,2,3,1)
            delta_gt_hand = gt_skel_hand.squeeze(3) - pts_gt_hand
            # norm_gt = torch.norm(delta_gt,dim=2,keepdim=True)
            
            
            mask_dis = micro_cond['y']['contact_label'].permute(0,2,1).unsqueeze(2)*mask
            mask_dis_body = mask_dis[:,:22]
            mask_dis_hand = mask_dis[:,22:]
            
            
            dist_loss = masked_l2((out_skel_body.squeeze(3) - pts_gt_body), delta_gt_body, mask_dis_body,
                            entries_norm=use_mean, w=weight_h*reg_h,delta=huber_c_h,loss_fn=self.loss_type)
            dist_loss += masked_l2((out_skel_hand.squeeze(3) - pts_gt_hand), delta_gt_hand, mask_dis_hand,
                            entries_norm=use_mean, w=weight_hand*reg_hand,delta=huber_c_hand,loss_fn=self.loss_type)
            
            weight_contact = self.args.contact_weight
        
            loss_dict['loss_contact'] = dist_loss
            
            loss_mse = loss_mse + weight_contact*dist_loss
            
        if self.args.foot_weight:
            
            foot_contact = micro_cond['y']['foot_contact'].permute(0,2,1).unsqueeze(2) # B,D,S
            findex = [8*3,8*3+1,8*3+2,11*3,11*3+1,11*3+2,7*3,7*3+1,7*3+2,10*3,10*3+1,10*3+2]
            loc_gt = gt_h[:,findex]
            loc_gtv = loc_gt[...,1:]-loc_gt[...,:-1]
            loc_pred = model_output_h[:,findex]
            
            loc_pred_v = loc_pred[...,1:]-loc_pred[...,:-1]
            
            loc_gtv = torch.cat([loc_gtv,loc_gtv[...,-1:].clone().detach()],-1).reshape(loc_gtv.shape[0],4,3,-1)
            loc_pred_v = torch.cat([loc_pred_v,loc_pred_v[...,-1:].clone().detach()],-1).reshape(loc_gtv.shape[0],4,3,-1)
            V = (loc_pred_v)
            loss_foot_skating = masked_l2(V,0,mask.to(V.dtype)*foot_contact.to(V.dtype),entries_norm=False, w=weight_h*reg_h,delta=huber_c_h,loss_fn=self.loss_type)
            loss_dict['loss_foot'] = loss_foot_skating

            loss_mse = loss_mse+loss_foot_skating*self.args.foot_weight/2
                   
        loss_mse = loss_mse/(self.args.hw+self.args.obj_w)
               
                    
            
        return loss_mse,loss_dict
        
      
    
        
        
    def forward_backward(self, batch, cond):
        # self.mp_trainer.zero_grad()
        
        for i in range(0, batch.shape[0], self.microbatch):
            # Eliminates the microbatch feature
            assert i == 0
            assert self.microbatch == self.batch_size
            micro = batch
            micro_cond = cond
            last_batch = (i + self.microbatch) >= batch.shape[0]
            
            upper = self.args.joint_nums*3+self.args.foot + 30
            model_output_h, gt_h,model_output_o, gt_o,model_output_hand, gt_hand, loss_weight_h,loss_weight_o,loss_weight_hand,reg_schedule_h,reg_schedule_o,reg_schedule_hand = self.diffusion.training_step(self.ddp_model,micro[:,self.body_index],micro[:,upper:], micro_cond['y'],micro[:,self.hand_index])
            loss_weight_hand = loss_weight_hand.unsqueeze(1).unsqueeze(1)
           
            loss_weight_h = loss_weight_h.unsqueeze(1).unsqueeze(1)
            loss_weight_o = loss_weight_o.unsqueeze(1).unsqueeze(1)
            reg_schedule_h = reg_schedule_h.unsqueeze(1).unsqueeze(1)
            reg_schedule_o = reg_schedule_o.unsqueeze(1).unsqueeze(1)
            
            reg_schedule_hand = reg_schedule_hand.unsqueeze(1).unsqueeze(1)
         
            if self.args.uniform_weight:
                loss_weight_h = torch.ones_like(loss_weight_h).to(gt_h.device)
                loss_weight_o = torch.ones_like(loss_weight_o).to(gt_h.device)
                loss_weight_hand = torch.ones_like(loss_weight_hand).to(gt_h.device)
            mask = micro_cond['y']['mask'].to(model_output_h.device)         
            with self.accelerator.autocast():
                loss_mse,losses = self.calc_loss_split_hand(gt_h,model_output_h,gt_o,model_output_o,gt_hand,model_output_hand,mask,weight_h=loss_weight_h,weight_o=loss_weight_o,weight_hand=loss_weight_hand,reg_h_t = reg_schedule_h,reg_o_t=reg_schedule_o,reg_hand_t=reg_schedule_hand,micro_cond=micro_cond)
              
            losses["loss"] = loss_mse
            loss = losses["loss"].mean()
            # losses["loss"]=loss.item()
            log_loss_dict_easy({k: v  for k, v in losses.items()})
        return loss
        

    def _anneal_lr(self):
        if not self.lr_anneal_steps or self.total_step() > self.lr_anneal_steps:
            return
        frac_done = self.total_step() / self.lr_anneal_steps
        lr = self.lr * ( frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.total_step())
        logger.logkv("samples", (self.total_step() + 1) * self.global_batch)


    def ckpt_file_name(self):
        return f"model{(self.total_step()):09d}.pt"

         

    
    def find_resume_checkpoint(self) -> Optional[str]:
        '''look for all file in save directory in the pattent of model{number}.pt
            and return the one with the highest step number.

        TODO: Implement this function (alredy existing in MDM), so that find model will call it in case a ckpt exist.
        TODO: Change call for find_resume_checkpoint and send save_dir as arg.
        TODO: This means ignoring the flag of resume_checkpoint in case some other ckpts exists in that dir!
        '''

        matches = {file: re.match(r'model(\d+).pt$', file) for file in os.listdir(self.args.save_dir)}
        models = {int(match.group(1)): file for file, match in matches.items() if match}

        return pjoin(self.args.save_dir, models[max(models)]) if models else None
    
    def total_step(self):
        return self.step + self.resume_step
    
    def save(self):
        def save_checkpoint():
            def del_clip(state_dict):
                # Do not save CLIP weights
                clip_weights = [
                    e for e in state_dict.keys() if e.startswith('clip_model.')
                ]
                for e in clip_weights:
                    del state_dict[e]

           
            state_dict = self.accelerator.unwrap_model(self.model).state_dict()
                
            del_clip(state_dict)

            if self.args.use_ema:
                # save both the model and the average model
                state_dict_avg = self.model_avg.state_dict()
                del_clip(state_dict_avg)
                state_dict = {'model': state_dict, 'model_avg': state_dict_avg}

            logger.log(f"saving model...")
            filename = self.ckpt_file_name()
            with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
                torch.save(state_dict, f)

        save_checkpoint()

        with bf.BlobFile(
            bf.join(self.save_dir, f"opt{(self.total_step()):09d}.pt"),
            "wb",
        ) as f:
            opt_state = self.opt.state_dict()
            
            ## command
            opt_state = {
                    'opt': opt_state,
                    # 'scaler': self.accelerator.scaler.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                }
            
            torch.save(opt_state, f)
        # 
        


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
   
    return logger.get_dir()



def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
def log_loss_dict_easy( losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
       
