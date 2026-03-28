import torch
from model.light import LIGHT 

# from diffusion import gaussian_diffusion as gd
# from diffusion.respace import SpacedDiffusion, space_timesteps
from diffusion_forcing.diffusion_forcing_base import DiffusionForcingBase
from omegaconf import OmegaConf 
from utils import dist_util

from utils.parser_util import get_cond_mode
from data_loaders.humanml_utils import HML_EE_JOINT_NAMES

def load_model_wo_clip(model, state_dict):
 
    
    del state_dict['sequence_pos_encoder.pe']  # no need to load it (fixed), and causes size mismatch for older models
    del state_dict['embed_timestep.sequence_pos_encoder.pe']  # no need to load it (fixed), and causes size mismatch for older models
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    # print(unexpected_keys,'kkk')
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') or 'sequence_pos_encoder' in k for k in missing_keys])


def create_model_and_diffusion(args, data):
    
    model = LIGHT(**get_model_args(args, data))
    diffusion = create_diffusion_forcing(args)
    if args.normalize:
        model.set_mean_std_rt(torch.from_numpy(data.dataset.t2m_dataset.mean_rt).to(dist_util.dev()),torch.from_numpy(data.dataset.t2m_dataset.std_rt).to(dist_util.dev()))
    return model, diffusion




def get_model_args(args, data,extra_keys=[]):

    # default args
   
    cond_mode = get_cond_mode(args)

    data_rep = "hml_vec"
    nfeats = 1 
    njoints = 195 
        
  
    # Compatibility with old models
    if not hasattr(args, 'pred_len'):
        args.pred_len = 0
        args.context_len = 0
    

    model_args_dct =  { 'njoints': njoints, 'nfeats': nfeats, 
            'translation': True, 
            'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
            'dropout': args.dropout, 'activation': "gelu", 'data_rep': data_rep, 'cond_mode': cond_mode,
            'cond_mask_prob': args.cond_mask_prob, 'arch': args.arch,
            'text_encoder_type': args.text_encoder_type,
             'mask_frames': args.mask_frames,
            'foot':args.foot,
            'cond_mask_uniform':args.cond_mask_uniform
            }


    return model_args_dct


def create_diffusion_forcing(args):
    if args.predict_x0:
        objective = "pred_x0"
    else:
        objective ="pred_noise"
    config_dict = {
    "objective": objective,
    "beta_schedule": args.beta_schedule,
    "schedule_fn_kwargs": {},
    "clip_noise": 20.0,
    "use_fused_snr":  args.use_fused_snr,
    "snr_clip": args.snr_gamma,
    "cum_snr_decay": 0.98,
    "timesteps": args.diffusion_steps,
    # sampling configuration
    "sampling_timesteps": 499,  # fixme: number of diffusion steps, should be increased
    "ddim_sampling_eta": args.df_eta,
    "stabilization_level": 1, # 10
    "uncertainty_scale": 1,
    "guidance_scale": 0.0,
# -1 for full trajectory diffusion, number to specify diffusion chunk size
    "scheduling_matrix": "autoregressive",
    "chunk_noise":args.chunk_noise,
    "infer_noise":args.infer_noise,
    't_noise':args.t_noise,
    'u_noise':args.u_noise,
    'split_ho':args.split_ho_emb,
    'h2o':args.h2o,
    'rand':args.rand,
    'gamma':args.snr_gamma,
    'hand_split':args.hand_split,
    'bias':args.bias,
    'df_delta':args.df_delta,
    'df_weight':args.df_weight,
    'df_divider':args.df_divider,
    'df_upstop':args.df_upstop,
    'df_decay':args.df_decay,
    'df_begin':args.df_begin,
    'df_cfg':args.df_cfg,
    'df_mode':args.df_mode,
    'df_star':args.df_star,
    'df_add':args.df_add,
    'df_tweight':args.df_tweight,
    'df_full_mode':args.df_full_mode,
    'df_prob':args.df_prob,
    'df_same':args.df_same,
    'df_rescale':args.df_rescale,
    'df_gw':args.df_gw,
    'df_mom':args.df_mom,
    'df_r':args.df_r

    
    
    }   
    
    
    cfg = OmegaConf.create(config_dict)

    return DiffusionForcingBase(cfg)

def create_gaussian_diffusion(args):
    # default params
    predict_xstart = args.predict_x0  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = args.diffusion_steps
    scale_beta = 1.  # no scaling
    timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]
    
    if hasattr(args, 'lambda_target_loc'):
        lambda_target_loc = args.lambda_target_loc
    else:
        lambda_target_loc = 0.

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_vel=args.lambda_vel,
        lambda_rcxyz=args.lambda_rcxyz,
        lambda_fc=args.lambda_fc,
        lambda_target_loc=lambda_target_loc,
    )

def load_saved_model(model, model_path, use_avg: bool=False):  # use_avg_model
    state_dict = torch.load(model_path, map_location='cpu')
    # Use average model when possible
    if use_avg and 'model_avg' in state_dict.keys():
    # if use_avg_model:
        print('loading avg model')
        state_dict = state_dict['model_avg']
    else:
        if 'model' in state_dict:
            print('loading model without avg')
            state_dict = state_dict['model']
        else:
            print('checkpoint has no avg model, loading as usual.')
    load_model_wo_clip(model, state_dict)
    return model