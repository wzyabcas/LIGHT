from argparse import ArgumentParser
import argparse
import os
import json


def parse_and_load_from_model(parser):
    # args according to the loaded model
    # do not try to specify them from cmd line since they will be overwritten
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    args = parser.parse_args()
    args_to_overwrite = []
    for group_name in ['base','dataset', 'model', 'diffusion']:
        args_to_overwrite += get_args_per_group_name(parser, args, group_name)

    # load args from model
    if args.model_path != '' :  # if not using external results file
        args = load_args_from_model(args, args_to_overwrite)

    if args.cond_mask_prob == 0:
        args.guidance_param = 1
    print(args.diffusion_steps,'DIFFSUION_STEPS')
    
    return apply_rules(args)

def load_args_from_model(args, args_to_overwrite):
    model_path = get_model_path_from_args()
    args_path = os.path.join(os.path.dirname(model_path), 'args.json')
    assert os.path.exists(args_path), 'Arguments json file was not found!'
    with open(args_path, 'r') as fr:
        model_args = json.load(fr)
# group.add_argument("--df_delta", default=30, type=int, help="Use smaller sigma values.")
#     group.add_argument("--df_weight", default=1.5, type=float, help="Use smaller sigma values.")
#     group.add_argument("--df_divider", default=3, type=int, help="Use smaller sigma values.")
#     group.add_argument("--df_upstop", default=30, type=int, help="Use smaller sigma values.")
    # or a=='dataset' or a=='df_delta' or a=='df_weight' or a=='df_divider' or a=='df_upstop' or a=='df_decay' or a=='df_begin' or a=='df_cfg':
    
    DF_PARAMS = {
    'debug', 'df_delta', 'df_weight', 'df_divider', 'df_upstop',
    'df_decay', 'df_begin', 'df_cfg', 'df_add', 'df_star',
    'df_mode', 'df_full_mode', 'df_tweight', 'df_eta', 'df_rescale',
    'df_gw', 'df_mom', 'df_r', 'guidance_param',
    }
    for a in args_to_overwrite:
        if a in DF_PARAMS:
            continue
        if a in model_args.keys():
            setattr(args, a, model_args[a])

        elif 'cond_mode' in model_args: # backward compitability
            unconstrained = (model_args['cond_mode'] == 'no_cond')
            setattr(args, 'unconstrained', unconstrained)

        else:
            print('Warning: was not able to load [{}], using default value [{}] instead.'.format(a, args.__dict__[a]))
    return args

def apply_rules(args):
    # For prefix completion
    if args.pred_len == 0:
        args.pred_len = args.context_len

    # For target conditioning
    if args.lambda_target_loc > 0.:
        args.multi_target_cond = True
    return args


def get_args_per_group_name(parser, args, group_name):
    for group in parser._action_groups:
        if group.title == group_name:
            group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
            return list(argparse.Namespace(**group_dict).__dict__.keys())
    return ValueError('group_name was not found.')

def get_model_path_from_args():
    try:
        dummy_parser = ArgumentParser()
        dummy_parser.add_argument('--model_path')
        dummy_args, _ = dummy_parser.parse_known_args()
        return dummy_args.model_path
    except:
        raise ValueError('model_path argument must be specified.')


def add_base_options(parser):
    group = parser.add_argument_group('base')
    
    group.add_argument("--unet_channels", type=int, default=283)
    group.add_argument("--unet_dim", type=int, default=256)
    group.add_argument("--unet_dim_mults", nargs="+", type=int, default=(2, 4))
    group.add_argument("--unet_resnet_block_groups", type=int, default=8)
    group.add_argument("--unet_norm", type=str, default="group")
    group.add_argument("--unet_kernel", type=int, default=3)
    group.add_argument("--unet_stride", type=int, default=2)
    group.add_argument("--unet_padding", type=int, default=1)
    group.add_argument("--dit_split", type=int, default=0)
    group.add_argument("--norm_first", type=int, default=0)
    group.add_argument("--renorm", type=float, default=0)
    group.add_argument("--obj_w", type=float, default=1)
    group.add_argument("--obj_trans_w", type=float, default=1)
    group.add_argument("--pene", type=float, default=0.0)
    group.add_argument("--adaln_t", type=int, default=2)
    group.add_argument("--ltype", type=str, default='l2')
    group.add_argument("--huber_c", type=float, default=1.0)

    group.add_argument("--self_vel", type=int, default=0)
    group.add_argument("--balance_type", type=int, default=2)
    group.add_argument("--use_bb", type=int, default=0)
    group.add_argument("--repre_bb", type=int, default=0)
    group.add_argument("--only_marker", type=int, default=0)
    group.add_argument("--umarker_mean", type=int, default=0)
    group.add_argument("--chamfer", type=int, default=0)
    group.add_argument("--dynamic_bps", type=int, default=0)
    group.add_argument("--dynamic_marker", type=int, default=0)
    
    
    group.add_argument("--clean", type=int, default=1)
    group.add_argument("--bps_mode", type=int, default=2)
    group.add_argument("--pointnet", type=int, default=0)
    group.add_argument("--bias", type=int, default=2)
    group.add_argument("--marker_mode", type=int, default=2)
    group.add_argument("--cid_num", type=int, default=0)
    group.add_argument("--cat", type=int, default=0)
    group.add_argument("--beta_mode", type=int, default=1)
    group.add_argument("--bone", type=int, default=0)
    group.add_argument("--snr_gamma", type=float, default=2.0)
    group.add_argument("--ome", type=float, default=0)
    group.add_argument("--version", type=int, default=6)
    
    group.add_argument("--h2o", type=int, default=0)
    group.add_argument("--use_bone", type=int, default=0)
    
    group.add_argument("--rand", type=int, default=1)
    group.add_argument("--use_aug", type=int, default=0)

    group.add_argument("--dataset", default='humanml', type=str,
                    help="Dataset name (choose from list).")
    group.add_argument("--process_v", default=0, type=int, help="Use Velocity for training")
    group.add_argument("--mean", default=1, type=int, help="Use mean")
    group.add_argument('--gamma', type=float, default=0.25, help='Learning rate schedule factor')
    group.add_argument('--milestones', default=[90_000,160_000,240_000], nargs="+", type=int,
                            help="learning rate schedule (iterations)")

    group.add_argument("--normalize", type=int, default=0)
    group.add_argument("--load_npz", type=int, default=1)
    group.add_argument("--cid", type=int, default=0)
    group.add_argument("--cw", type=int, default=0)
    group.add_argument("--split_loss", default=0, type=int, help="Split loss for different modality")
    group.add_argument("--split_rt", default=0, type=int, help="Split rotation and translation in training losses")
    group.add_argument("--ht", default=0, type=int, help="")
    group.add_argument("--loss_type", default=0, type=int, help="Training Loss Type")
    group.add_argument("--predict_x0", default=1, type=int, help="Whether Predict X0 for diffusion forcing")
    group.add_argument("--chunk_noise", default=1, type=int, help="Noise chunk for diffusion forcing")
    group.add_argument("--infer_noise", default=0.0, type=float, help="")
    group.add_argument("--t_noise", default=0, type=int, help="")
    group.add_argument("--hw", default=1, type=float, help="hw")
    group.add_argument("--handw", default=0, type=float, help="hand loss weight")
    group.add_argument("--u_noise", default=0, type=int, help="uniform noise")
    group.add_argument("--snr", default=0, type=int, help="Use SNR for diffusion Forcing Training")
    group.add_argument("--split_t", default=0, type=int, help="Split Timestep")
    group.add_argument("--hand_split", default=1, type=int, help="Split Hand modality")
    group.add_argument("--cond_mask_uniform", default=0, type=int, help="Whether only dropout text when modalities have same noise in training")

    group.add_argument("--dropout", default=0.1, type=float, help="")
    group.add_argument("--zero_init", default=1, type=int, help="")
    group.add_argument("--use_mask", default=1, type=int, help="")
    group.add_argument("--use_bps", default=1, type=int, help="")
    group.add_argument("--use_obj_feat", default=0, type=int, help="")
    group.add_argument("--embed_shape", default=1, type=int, help="")
    group.add_argument("--joint_nums", default=52, type=int, help="")
    group.add_argument("--use_fused_snr", default=0, type=int, help="")
    group.add_argument("--ho_pe", default=1, type=int, help="")
    group.add_argument("--save_npz", default=0, type=int, help="")
    group.add_argument("--foot", default=0, type=int, help="")
    group.add_argument("--task", default="", type=str, help="")
    group.add_argument("--uniform_weight", default=0, type=int, help="")
    group.add_argument("--uniform_reg", default=0, type=int, help="")
    group.add_argument("--clean_loss", default=1, type=int, help="")
    group.add_argument("--split_condition", default=0, type=int, help="")

    group.add_argument("--st_att", default=0, type=int, help="")
    group.add_argument("--st_gcn", default=0, type=int, help="")
    group.add_argument("--reg_diff", default=0, type=int, help="")
    group.add_argument("--use_gd", default=0, type=int, help="")
    group.add_argument("--hand_loss", default=0, type=int, help="")
    group.add_argument("--hand_weight", default=0, type=float, help="")
    group.add_argument("--body_w", default=1, type=float, help="")
    group.add_argument("--foot_weight", default=0, type=float, help="")
    group.add_argument("--ground_weight", default=0, type=float, help="")
    group.add_argument("--contact_weight", default=1, type=float, help="")
    group.add_argument("--vel_weight", default=0.01, type=float, help="")
    group.add_argument("--dit", default='dit', type=str, help="")
    group.add_argument("--use_beta", default=1, type=int, help="")
    group.add_argument("--rope", default=1, type=int, help="")
    group.add_argument("--learnable_pe", default=1, type=int, help="")
    group.add_argument("--online", default=1, type=int, help="")

    group.add_argument("--beta_schedule", default='cosine', type=str, help="")
    group.add_argument("--st_depth", default=3, type=int, help="")
    group.add_argument("--st_attn_fuse", default=1, type=int, help="")
    group.add_argument("--st_dim_repre", default=1, type=int, help="")
    
    group.add_argument("--reg_hl", default=0, type=int, help="")

    group.add_argument("--debug", default=0, type=int, help="")
    group.add_argument("--hand_local", default=0, type=float, help="")
    
    group.add_argument("--repre", default=501, type=int, help="")
    group.add_argument("--accel", default=0, type=int, help="")
    group.add_argument("--relative", default=0, type=int, help="")
    group.add_argument("--relative_all", default=0, type=int, help="")
    group.add_argument("--use_marker", default=1, type=int, help="")
    group.add_argument("--use_rot", default=0, type=int, help="")
    group.add_argument("--use_hand_scalar_rot", default=0, type=int, help="")
    group.add_argument("--use_joint", default=1, type=int, help="")
    group.add_argument("--verts_loss", default=1, type=int, help="")
    group.add_argument("--verts_weight", default=0.0, type=float, help="")
    group.add_argument("--verts_vel", default=0, type=int, help="")
    group.add_argument("--perturb", default=0, type=int, help="")
    group.add_argument("--relative_weight", default=0.1, type=float, help="")
    group.add_argument("--original_df", default=0, type=int, help="")

    group.add_argument("--latent_dim_refine", default=64, type=int, help="")
    group.add_argument("--local_window", default=None, type=int, help="")
    group.add_argument("--split_ho_emb", default=1, type=int, help="")
    group.add_argument("--split_pe", default=1, type=int, help="")
    group.add_argument("--weight_obj", default=0, type=int, help="")
    group.add_argument("--split_hand", default=0, type=int, help="")
    group.add_argument("--add_dec", default=0, type=int, help="")
    group.add_argument("--obj_split", default=0, type=int, help="")

    group.add_argument("--sigma", default=0.5, type=float, help="")
    group.add_argument("--use_soft_masking", default=0, type=int, help="")
    group.add_argument("--causal", default=0, type=int, help="")
    group.add_argument("--model_type", default='mdm', type=str, help="")
    group.add_argument("--pad_prefix", default=0, type=int, help="")
    group.add_argument("--predict_prefix", default=0, type=int, help="")
    group.add_argument("--crop_len", default=300, type=int, help="")
    

    group.add_argument("--cuda", default=True, type=bool, help="Use cuda device, otherwise use CPU.")
    group.add_argument("--device", default=0, type=int, help="Device id to use.")
    group.add_argument("--seed", default=10, type=int, help="For fixing random seed.")
    group.add_argument("--batch_size", default=64, type=int, help="Batch size during training.")
    group.add_argument("--train_platform_type", default='NoPlatform', choices=['NoPlatform', 'ClearmlPlatform', 'TensorboardPlatform', 'WandBPlatform'], type=str,
                       help="Choose platform to log results. NoPlatform means no logging.")
    group.add_argument("--external_mode", default=False, type=bool, help="For backward cometability, do not change or delete.")


def add_diffusion_options(parser):
    group = parser.add_argument_group('diffusion')
    group.add_argument("--noise_schedule", default='cosine', choices=['linear', 'cosine'], type=str,
                       help="Noise schedule type")
    group.add_argument("--diffusion_steps", default=1000, type=int,
                       help="Number of diffusion steps (denoted T in the paper)")
    group.add_argument("--sigma_small", default=True, type=bool, help="Use smaller sigma values.")
    group.add_argument("--df_delta", default=50, type=int, help="Use smaller sigma values.")
    group.add_argument("--df_mode", default=0, type=int, help="Use smaller sigma values.")
    group.add_argument("--df_mom", default=0, type=float, help="Use smaller sigma values.")
    group.add_argument("--df_r", default=0, type=float, help="Use smaller sigma values.")
    group.add_argument("--df_gw", default=0, type=float, help="Use smaller sigma values.")
    group.add_argument("--df_eta", default=0, type=float, help="Use smaller sigma values.")
    group.add_argument("--df_full_mode", default=0, type=int, help="Use smaller sigma values.")
    group.add_argument("--df_weight", default=1.5, type=float, help="Use smaller sigma values.")
    group.add_argument("--df_divider", default=3, type=int, help="Use smaller sigma values.")
    group.add_argument("--df_upstop", default=30, type=int, help="Use smaller sigma values.")
    group.add_argument("--df_decay", default=0, type=int, help="Use smaller sigma values.")
    group.add_argument("--df_begin", default=0, type=int, help="Use smaller sigma values.")
    group.add_argument("--df_cfg", default=0, type=int, help="Use smaller sigma values.")
    group.add_argument("--df_add", default=0, type=float, help="Add noise to staged")
    group.add_argument("--df_star", default=0, type=int, help="CFG star.")
    group.add_argument("--df_tweight", default=0, type=float, help="CFG star.")
    group.add_argument("--df_prob", default=1, type=float, help="PROB.")
    group.add_argument("--df_rescale", default=0, type=float, help="Use smaller sigma values.")

    group.add_argument("--df_same", default=0, type=int, help="TRAIN SAME WITH INFERENCE")


def add_model_options(parser):
    group = parser.add_argument_group('model')
    group.add_argument("--arch", default='trans_dec', type=str,
                       help="Architecture types as reported in the paper.")
    group.add_argument("--text_encoder_type", default='clip', type=str, help="Text encoder type.")
    group.add_argument("--emb_trans_dec", action='store_true',
                       help="For trans_dec architecture only, if true, will inject condition as a class token"
                            " (in addition to cross-attention).")
    group.add_argument("--layers", default=8, type=int,
                       help="Number of layers.")
    group.add_argument("--latent_dim", default=512, type=int,
                       help="Transformer/GRU width.")
    group.add_argument("--cond_mask_prob", default=.1, type=float,
                       help="The probability of masking the condition during training."
                            " For classifier-free guidance learning.")
    group.add_argument("--mask_frames", action='store_true', help="If true, will fix Rotem's bug and mask invalid frames.")
    group.add_argument("--lambda_rcxyz", default=0.0, type=float, help="Joint positions loss.")
    group.add_argument("--lambda_vel", default=0.0, type=float, help="Joint velocity loss.")
    group.add_argument("--lambda_fc", default=0.0, type=float, help="Foot contact loss.")
    group.add_argument("--lambda_target_loc", default=0.0, type=float, help="For HumanML only, when . L2 with target location.")
    group.add_argument("--unconstrained", action='store_true',
                       help="Model is trained unconditionally. That is, it is constrained by neither text nor action. "
                            "Currently tested on HumanAct12 only.")
    group.add_argument("--pos_embed_max_len", default=5000, type=int,
                       help="Pose embedding max length.")
    group.add_argument("--use_ema", action='store_true',
                    help="If True, will use EMA model averaging.")
    

    group.add_argument("--multi_target_cond", action='store_true', help="If true, enable multi-target conditioning (aka Sigal's model).")
    group.add_argument("--multi_encoder_type", default='single', choices=['single', 'multi', 'split'], type=str, help="Specifies the encoder type to be used for the multi joint condition.")
    group.add_argument("--target_enc_layers", default=1, type=int, help="Num target encoder layers")


    # Prefix completion model
    group.add_argument("--context_len", default=0, type=int, help="If larger than 0, will do prefix completion.")
    group.add_argument("--pred_len", default=0, type=int, help="If context_len larger than 0, will do prefix completion. If pred_len will not be specified - will use the same length as context_len")
    



def add_data_options(parser):
    group = parser.add_argument_group('dataset')
   
    
    group.add_argument("--data_dir", default="", type=str,
                       help="If empty, will use defaults according to the specified dataset.")


def add_training_options(parser):
    group = parser.add_argument_group('training')
    group.add_argument("--save_dir", required=True, type=str,
                       help="Path to save checkpoints and results.")
    group.add_argument("--overwrite", action='store_true',
                       help="If True, will enable to use an already existing save_dir.")
    group.add_argument("--lr", default=1e-4, type=float, help="Learning rate.")
    group.add_argument("--weight_decay", default=1e-2, type=float, help="Optimizer weight decay.")
    group.add_argument("--lr_anneal_steps", default=0, type=int, help="Number of learning rate anneal steps.")
    group.add_argument("--eval_batch_size", default=32, type=int,
                       help="Batch size during evaluation loop. Do not change this unless you know what you are doing. "
                            "T2m precision calculation is based on fixed batch size 32.")
    group.add_argument("--eval_split", default='test', choices=['val', 'test'], type=str,
                       help="Which split to evaluate on during training.")
    group.add_argument("--eval_during_training", action='store_true',
                       help="If True, will run evaluation during training.")
    group.add_argument("--eval_rep_times", default=3, type=int,
                       help="Number of repetitions for evaluation loop during training.")
    group.add_argument("--eval_num_samples", default=1_000, type=int,
                       help="If -1, will use all samples in the specified split.")
    group.add_argument("--log_interval", default=1_000, type=int,
                       help="Log losses each N steps")
    group.add_argument("--save_interval", default=8_000, type=int,
                       help="Save checkpoints and run evaluation each N steps")
    group.add_argument("--num_steps", default=600_000, type=int,
                       help="Training will stop after the specified number of steps.")
    group.add_argument("--num_frames", default=60, type=int,
                       help="Limit for the maximal number of frames. In HumanML3D and KIT this field is ignored.")
    group.add_argument("--resume_checkpoint", default="", type=str,
                       help="If not empty, will start from the specified checkpoint (path to model###.pt file).")
    group.add_argument("--resume_checkpoint_refine", default="", type=str,
                       help="If not empty, will start from the specified checkpoint (path to model###.pt file).")
    group.add_argument("--gen_during_training", action='store_true',
                       help="If True, will generate motions during training, on each save interval.")
    group.add_argument("--gen_num_samples", default=3, type=int,
                       help="Number of samples to sample while generating")
    group.add_argument("--gen_num_repetitions", default=2, type=int,
                       help="Number of repetitions, per sample (text prompt/action)")
    group.add_argument("--gen_guidance_param", default=1, type=float,
                       help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")
    
    group.add_argument("--avg_model_beta", default=0.9999, type=float, help="Average model beta (for EMA).")
    group.add_argument("--adam_beta2", default=0.999, type=float, help="Adam beta2.")
    
    group.add_argument("--target_joint_names", default='DIMP_FINAL', type=str, help="Force single joint configuration by specifing the joints (coma separated). If None - will use the random mode for all end effectors.")
    group.add_argument("--autoregressive", action='store_true', help="If true, and we use a prefix model will generate motions in an autoregressive loop.")
    group.add_argument("--autoregressive_include_prefix", action='store_true', help="If true, include the init prefix in the output, otherwise, will drop it.")
    group.add_argument("--autoregressive_init", default='data', type=str, choices=['data', 'isaac'], 
                        help="Sets the source of the init frames, either from the dataset or isaac init poses.")


def add_sampling_options(parser):
    group = parser.add_argument_group('sampling')
    group.add_argument("--model_path", required=True, type=str,
                       help="Path to model####.pt file to be sampled.")
    group.add_argument("--output_dir", default='', type=str,
                       help="Path to results dir (auto created by the script). "
                            "If empty, will create dir in parallel to checkpoint.")
    group.add_argument("--num_samples", default=6, type=int,
                       help="Maximal number of prompts to sample, "
                            "if loading dataset from file, this field will be ignored.")
    group.add_argument("--num_repetitions", default=3, type=int,
                       help="Number of repetitions, per sample (text prompt/action)")
    group.add_argument("--guidance_param", default=1, type=float,
                       help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")

    group.add_argument("--autoregressive", action='store_true', help="If true, and we use a prefix model will generate motions in an autoregressive loop.")
    group.add_argument("--autoregressive_include_prefix", action='store_true', help="If true, include the init prefix in the output, otherwise, will drop it.")
    group.add_argument("--autoregressive_init", default='data', type=str, choices=['data', 'isaac'], 
                        help="Sets the source of the init frames, either from the dataset or isaac init poses.")

def add_generate_options(parser):
    group = parser.add_argument_group('generate')
    group.add_argument("--motion_length", default=6.0, type=float,
                       help="The length of the sampled motion [in seconds]. "
                            "Maximum is 9.8 for HumanML3D (text-to-motion), and 2.0 for HumanAct12 (action-to-motion)")
    group.add_argument("--input_text", default='', type=str,
                       help="Path to a text file lists text prompts to be synthesized. If empty, will take text prompts from dataset.")
    group.add_argument("--dynamic_text_path", default='', type=str,
                       help="For the autoregressive mode only! Path to a text file lists text prompts to be synthesized. If empty, will take text prompts from dataset.")
    group.add_argument("--action_file", default='', type=str,
                       help="Path to a text file that lists names of actions to be synthesized. Names must be a subset of dataset/uestc/info/action_classes.txt if sampling from uestc, "
                            "or a subset of [warm_up,walk,run,jump,drink,lift_dumbbell,sit,eat,turn steering wheel,phone,boxing,throw] if sampling from humanact12. "
                            "If no file is specified, will take action names from dataset.")
    group.add_argument("--text_prompt", default='', type=str,
                       help="A text prompt to be generated. If empty, will take text prompts from dataset.")
    group.add_argument("--action_name", default='', type=str,
                       help="An action name to be generated. If empty, will take text prompts from dataset.")
    group.add_argument("--target_joint_names", default='DIMP_FINAL', type=str, help="Force single joint configuration by specifing the joints (coma separated). If None - will use the random mode for all end effectors.")


def add_edit_options(parser):
    group = parser.add_argument_group('edit')
    group.add_argument("--edit_mode", default='in_between', choices=['in_between', 'upper_body'], type=str,
                       help="Defines which parts of the input motion will be edited.\n"
                            "(1) in_between - suffix and prefix motion taken from input motion, "
                            "middle motion is generated.\n"
                            "(2) upper_body - lower body joints taken from input motion, "
                            "upper body is generated.")
    group.add_argument("--text_condition", default='', type=str,
                       help="Editing will be conditioned on this text prompt. "
                            "If empty, will perform unconditioned editing.")
    group.add_argument("--prefix_end", default=0.25, type=float,
                       help="For in_between editing - Defines the end of input prefix (ratio from all frames).")
    group.add_argument("--suffix_start", default=0.75, type=float,
                       help="For in_between editing - Defines the start of input suffix (ratio from all frames).")


def add_evaluation_options(parser):
    group = parser.add_argument_group('eval')
    group.add_argument("--model_path", required=True, type=str,
                       help="Path to model####.pt file to be sampled.")
    group.add_argument("--eval_mode", default='wo_mm', choices=['wo_mm', 'mm_short', 'debug', 'full'], type=str,
                       help="wo_mm (t2m only) - 20 repetitions without multi-modality metric; "
                            "mm_short (t2m only) - 5 repetitions with multi-modality metric; "
                            "debug - short run, less accurate results."
                            "full (a2m only) - 20 repetitions.")
    group.add_argument("--autoregressive", action='store_true', help="If true, and we use a prefix model will generate motions in an autoregressive loop.")
    group.add_argument("--autoregressive_include_prefix", action='store_true', help="If true, include the init prefix in the output, otherwise, will drop it.")
    group.add_argument("--autoregressive_init", default='data', type=str, choices=['data', 'isaac'], 
                        help="Sets the source of the init frames, either from the dataset or isaac init poses.")
    group.add_argument("--guidance_param", default=1, type=float,
                       help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")


def get_cond_mode(args):
    if args.unconstrained:
        cond_mode = 'no_cond'
    # elif args.dataset in ['kit', 'humanml']:
    cond_mode = 'text'
    # else:
    #     cond_mode = 'action'
    return cond_mode


# def train_args():
#     parser = ArgumentParser()
#     add_base_options(parser)
#     add_data_options(parser)
#     add_model_options(parser)
#     add_diffusion_options(parser)
#     add_training_options(parser)
#     return apply_rules(parser.parse_args())

def train_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_training_options(parser)
    args = parser.parse_args()
    
    args_save_path = os.path.join(args.save_dir, 'args.json')
    if args.resume_checkpoint =='' and (not os.path.exists(args_save_path)) :
        return apply_rules(args)
    # if args.resume_checkpoint =='':
    #     return apply_rules(args)
    args_to_overwrite = []
    for group_name in ['base','dataset', 'model', 'diffusion','training']:
        args_to_overwrite += get_args_per_group_name(parser, args, group_name)
    # if args.model_path != '' :  # if not using external results file
    
    model_path = args.resume_checkpoint
    args_path = os.path.join(os.path.dirname(model_path), 'args.json')
    if not os.path.exists(args_path):
        args_path = os.path.join(args.save_dir, 'args.json')
        
    # model_path = args.resume_checkpoint
    # args_path = os.path.join(os.path.dirname(model_path), 'args.json')
    # assert os.path.exists(args_path), 'Arguments json file was not found!'
    with open(args_path, 'r') as fr:
        model_args = json.load(fr)

    for a in args_to_overwrite :
        if a=='debug' or a=='save_dir' or a=='resume_checkpoint'  or a=='obj_w':
            continue
        if a in model_args.keys():
            setattr(args, a, model_args[a])

        elif 'cond_mode' in model_args: # backward compitability
            unconstrained = (model_args['cond_mode'] == 'no_cond')
            setattr(args, 'unconstrained', unconstrained)

        else:
            print('Warning: was not able to load [{}], using default value [{}] instead.'.format(a, args.__dict__[a]))
    
    return args
    # args = load_args_from_model(args, args_to_overwrite)
    # # args = parse_and_load_from_model(parser)
    # return args
    # return apply_rules(parser.parse_args())
# def parse_and_load_from_model(parser):
#     # args according to the loaded model
#     # do not try to specify them from cmd line since they will be overwritten
#     add_data_options(parser)
#     add_model_options(parser)
#     add_diffusion_options(parser)
#     args = parser.parse_args()
#     args_to_overwrite = []
#     for group_name in ['dataset', 'model', 'diffusion']:
#         args_to_overwrite += get_args_per_group_name(parser, args, group_name)

#     # load args from model
#     if args.model_path != '' :  # if not using external results file
#         args = load_args_from_model(args, args_to_overwrite)

#     if args.cond_mask_prob == 0:
#         args.guidance_param = 1
    
#     return apply_rules(args)

def generate_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_sampling_options(parser)
    add_generate_options(parser)
    args = parse_and_load_from_model(parser)
    cond_mode = get_cond_mode(args)

    if (args.input_text or args.text_prompt) and cond_mode != 'text':
        raise Exception('Arguments input_text and text_prompt should not be used for an action condition. Please use action_file or action_name.')
    elif (args.action_file or args.action_name) and cond_mode != 'action':
        raise Exception('Arguments action_file and action_name should not be used for a text condition. Please use input_text or text_prompt.')

    return args


def edit_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_sampling_options(parser)
    add_edit_options(parser)
    return parse_and_load_from_model(parser)


def evaluation_parser():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_evaluation_options(parser)
    return parse_and_load_from_model(parser)