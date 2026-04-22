# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
# from sentence_transformers import SentenceTransformer
from utils.parser_util import evaluation_parser
from datetime import datetime

from utils.fixseed import fixseed
import os
import numpy as np
import torch
import smplx
from accelerate import Accelerator,DeepSpeedPlugin
from accelerate.utils import DistributedDataParallelKwargs

from utils.sampler_util import ClassifierFreeSampleModel, AutoRegressiveSampler
from tqdm import tqdm

import shutil
from trimesh import Trimesh
import trimesh
from scipy.spatial.transform import Rotation

from bps_torch.bps import bps_torch
from bps_torch.tools import sample_sphere_uniform
from bps_torch.tools import sample_uniform_cylinder
from tma.models.architectures.temos.textencoder.distillbert_actor import DistilbertActorAgnosticEncoder
from tma.models.architectures.temos.motionencoder.actor import ActorAgnosticEncoder
from utils.eval_t2m_utils import *
from tqdm import tqdm
from utils.common.quaternion import rotation_6d_to_matrix_np, rotation_6d_to_matrix
from data_loaders.get_data import get_dataset_loader
from utils.sampler_util import ClassifierFreeSampleModel
from collections import OrderedDict
from utils import dist_util
from data_loaders.humanml.scripts.motion_process import *
from data_loaders.humanml.utils.utils import *
from utils.model_util import create_model_and_diffusion, load_saved_model
from pytorch3d.transforms import rotation_6d_to_matrix,matrix_to_axis_angle,matrix_to_rotation_6d

device = torch.device('cuda:0')
######################################## smplh 10 ########################################
# smplh_model_male = smplx.create(MODEL_PATH, model_type='smplh',
#                         gender="male",
#                         use_pca=False,
#                         ext='pkl',flat_hand_mean=True).to(device)

# smplh_model_female = smplx.create(MODEL_PATH, model_type='smplh',
#                         gender="female",
#                         use_pca=False,
#                         ext='pkl',flat_hand_mean=True).to(device)

# smplh_model_neutral = smplx.create(MODEL_PATH, model_type='smplh',
#                         gender="neutral",
#                         use_pca=False,
#                         ext='pkl',flat_hand_mean=True).to(device)

# smplh10 = {'male': smplh_model_male, 'female': smplh_model_female, 'neutral': smplh_model_neutral}


# SMPLH_PATH = MODEL_PATH+'/smplh'
# surface_model_male_fname = os.path.join(SMPLH_PATH,'female', "model.npz")
# surface_model_female_fname = os.path.join(SMPLH_PATH, "male","model.npz")
# surface_model_neutral_fname = os.path.join(SMPLH_PATH, "neutral", "model.npz")
# dmpl_fname = None
# num_dmpls = None 
# num_expressions = None
# num_betas = 16 

# human_faces_single = smplh_model_female.faces.astype(np.int32)

# def forward_human(model_type,poses,trans,betas,gender="male"):
#     if model_type =='smplh10':
#         frame_times = poses.shape[0]
#         smplx_output = smplh10[gender](body_pose=poses[:, 3:66],
#                                         global_orient=poses[:, :3],
#                                         left_hand_pose=poses[:, 66:111],
#                                         right_hand_pose=poses[:, 111:156],
#                                         betas=betas[None, :].repeat(frame_times, 1).float(),
#                                         transl=trans) 
#         verts = smplx_output.vertices
#     else:
#         # if model_type == 'smplh':
#         smpl_model = smplh16[gender]
#         # elif model_type == 'smplx':
#         #     smpl_model = smplx16[gender]
#         smplx_output = smpl_model(pose_body=poses[:, 3:66], 
#                             pose_hand=poses[:, 66:156], 
#                             betas=betas[None, :].repeat(frame_times, 1).float(), 
#                             root_orient=poses[:, :3], 
#                             trans=trans)
#         verts = (smplx_output.v)
        
#     return verts
#     # faces = smpl_model.faces.astype(np.int32)
#     # faces = torch.from_numpy(faces).unsqueeze(0).repeat(frame_times, 1, 1).to(device)

@torch.no_grad()
def eval_t2hoi(args,val_loader, motion_model, motion_diffusion, textencoder,motionencoder,std_enc,mean_enc,repeat_id,obj_v_dict,mean_data,std_data,obj_canrot_dict):
    

    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    
    matching_score_real = 0
    matching_score_pred = 0

    

    nb_sample = 0
    l1_dist = 0
    num_poses = 1
    # for i in range(1):
    ## insert
    print(len(val_loader),'LENGTH')
    for batch_idx,batch in enumerate(val_loader):
        motions, model_kwargs=batch
       
        
        motions_copy = motions.detach().clone()
        bs = len(motions)
        if args.guidance_param != 1:
            model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param
        else:
            model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev())
        model_kwargs['y']['obj_bps'] = model_kwargs['y']['obj_bps'].to(dist_util.dev())
        model_kwargs['y']['obj_points'] = model_kwargs['y']['obj_points'].to(dist_util.dev())
        # B,D,1,T -> b,1,t,d -> B,T,D
        motions = motions.cpu().permute(0, 2, 3, 1).squeeze().float() # b,d,1,t -> b,1,t,d
        if args.normalize:
            motions = motions* std_data.to(motions.device) +mean_data.to(motions.device)
        # full_motion = model_kwargs['y']['full_motion']
        # full_motion = full_motion.cpu().permute(0, 2, 3, 1).squeeze().float()
        text_key = 'text' if 'text' in model_kwargs['y'] else 'action'
        text_list_gt = model_kwargs['y'][text_key]
        # print(text_list_gt)
        lengths = model_kwargs['y']['lengths'].cpu().numpy()
        
        start_index = args.joint_nums*3+30+args.foot
        start_index_small = args.joint_nums*3+args.foot
   
        motions_obj = motions[...,start_index:start_index+9].float()
        obj_points = model_kwargs['y']['obj_points']
        obj_points = obj_points[:,None,...].repeat(1,motions.shape[1],1,1)
        all_obj_name = model_kwargs['y']['seq_name']
        vertices = obj_points.reshape(-1,obj_points.shape[2],3).float().cpu()

        angle, trans = motions_obj[..., :6].reshape(-1,6).float(), motions_obj[..., 6:9].reshape(-1,3).float()

        torch_rot = rotation_6d_to_matrix(angle)
        torch_rot_t =torch_rot.reshape(bs,-1,3,3)
        obj_points_gt = torch.bmm(vertices.float(), torch_rot.transpose(1, 2)) + trans[:, None, :]
        
      
        bps_obj_times = bps_obj.repeat(trans.shape[0], 1, 1).float() + trans[:, None, :]
        bps_time_nonrot = bps_torch.encode(x=obj_points_gt, \
            feature_type=['deltas'], \
            custom_basis=bps_obj_times)['deltas'] # T X N X 3 
        
       
        motions_obj_repre = motions_obj.clone()
        
        
        dataset_names = []
    
        obj_names = []
        for iii in tqdm(range(bs)):
            seq_name = model_kwargs['y']['seq_name'][iii]
            dataset_name, obj_name = str(seq_name).rsplit('_',1)
            dataset_name = dataset_name.split('_')[0]
            dataset_names.append(dataset_name)
            obj_names.append(obj_name)
        
        for iii in tqdm(range(bs)):
            dataset_name = dataset_names[iii]
            obj_name = obj_names[iii]
            canrot = obj_canrot_dict[dataset_name][obj_name]
            rot_can = torch.einsum('tij,jk->tik',torch_rot_t[iii],canrot.permute(1,0))
            can_6d = matrix_to_rotation_6d(rot_can.float())
            motions_obj_repre[iii][...,:6] = can_6d
    
        gt_motions = torch.cat((motions[...,:start_index_small-args.foot].cuda(),motions_obj_repre.cuda(),bps_time_nonrot.reshape(motions.shape[0], motions.shape[1],128*3).cuda()),dim=-1)
            
        motion_shape = (args.batch_size, model.njoints, model.nfeats, n_frames)
        # sample_fn = motion_diffusion.p_sample_loop
        device = dist_util.dev()
        model_kwargs['y'] = {key: val.to(device) if torch.is_tensor(val) else val for key, val in model_kwargs['y'].items()}

       
        obj_start = args.joint_nums*3+args.foot + 30
       
        batch_rand = torch.randn(motion_shape).to(device)
        body_index = list(range(22*3))#+list(range(52*3,52*3+3+22*6))+list(range(52*3+3+52*6,52*3+3+52*6+args.foot))
        hand_index = list(range(22*3,52*3+30))
        
        batch1 = batch_rand[:,body_index]
        batch2 = batch_rand[:,obj_start:]
        batch3 = batch_rand[:,hand_index]
        gbatch1 = motions_copy[:,body_index]
        gbatch2 = motions_copy[:,obj_start:]
        gbatch3 = motions_copy[:,hand_index]
        
       
        # model_kwargs['y']['uncond'] = True
        # sample = motions_copy
        sample = diffusion.sample_step_new_cfg(model, batch1,batch2, batch3,0, model_kwargs['y'],scheduling_mode='full_sequence',guidance_param=0.0001)
        # sample = sample.permute(0, 2, 3, 1).squeeze().float()

        if args.normalize:
            sample = sample*std_data.reshape(1,-1,1,1).to(sample.device)+mean_data.reshape(1,-1,1,1).to(sample.device)
            
        # sample = diffusion.sample_step(model, batch1,batch2, batch3,0, model_kwargs['y'],scheduling_mode='full_sequence')
        
        # sample = motions_copy
        fskat,_ = calculate_skating_ratio(sample[:,:22*3].reshape(sample.shape[0],22,3,-1),model_kwargs['y']['mask'])
        
        # B,T,D
        sample = sample.cpu().permute(0, 2, 3, 1).squeeze().float() # B,D,1,T  b,t,d
        # sample = sample.cpu().permute(0, 2, 3, 1).squeeze().float()
        sample_obj = sample[..., start_index:start_index+9].float()
        angle, trans = sample_obj[..., :6].reshape(-1,6), sample_obj[..., 6:9].reshape(-1,3)
        
        
        torch_rot = rotation_6d_to_matrix(angle)
        
        torch_rot_t = torch_rot.reshape(bs,-1,3,3)
        obj_points_pred = torch.bmm(vertices.float(), torch_rot.transpose(1, 2)) + trans[:, None, :]
        bps_obj_times = bps_obj.repeat(trans.shape[0], 1, 1).float() + trans[:, None, :]
        bps_time = bps_torch.encode(x=obj_points_pred, \
            feature_type=['deltas'], \
            custom_basis=bps_obj_times)['deltas'] # T X N X 3 
       
        sample_obj_repre = sample_obj.clone()
        for iii in tqdm(range(bs)):
            dataset_name = dataset_names[iii]
            obj_name = obj_names[iii]
            canrot = obj_canrot_dict[dataset_name][obj_name]
            rot_can = torch.einsum('tij,jk->tik',torch_rot_t[iii],canrot.permute(1,0))
            can_6d = matrix_to_rotation_6d(rot_can.float())
            sample_obj_repre[iii][...,:6] = can_6d
                
                
   
        pred_motions = torch.cat((sample[...,:start_index_small-args.foot].cuda(),sample_obj_repre.cuda(),bps_time.reshape(sample.shape[0], sample.shape[1],128*3).cuda()),dim=-1)
        
      
        bs = gt_motions.shape[0]
        obj_points_gt_t = obj_points_gt.reshape(gt_motions.shape[0],-1,340,3).to(gt_motions.device)
        obj_points_pred_t = obj_points_pred.reshape(gt_motions.shape[0],-1,340,3).to(gt_motions.device)
        con_percents=[]
        f1s =[]
        recalls=[]
        cprecisions=[]
        
        
        con_percents_h=[]
        recalls_h=[]
        cprecisions_h =[]
        f1s_h =[]
        
        
        
        torch_rot_change = torch_rot.reshape(bs,-1,3,3)
        trans_change = trans.reshape(bs,-1,3)
        pen_dists=[]
        for iii in tqdm(range(bs)):
            dataset_name = dataset_names[iii]
            obj_name = obj_names[iii]
            vertices_mesh = obj_v_dict[dataset_name][obj_name]
            penetration_score = compute_penetration_metric2(args.version,dataset_name,obj_name,sample[iii,:lengths[iii]-4,:52*3].reshape(-1,52,3).cuda(),trans_change[iii,:lengths[iii]-4].cuda(),torch_rot_change[iii,:lengths[iii]-4].cuda())
            pen_dists.append(penetration_score)
            
            obj_points_pred_mesh = torch.matmul(vertices_mesh.float(), torch_rot_change[iii,:lengths[iii]-4].transpose(1,2)) + trans_change[iii,:lengths[iii]-4][:, None, :]

            obj_points_pred_mesh = obj_points_pred_mesh.float().to(device)
            
            contact_metrics = contact_metric_micro(gt_motions[iii,:lengths[iii]-4,:args.joint_nums*3].reshape(-1,args.joint_nums,3),obj_points_gt_t[iii,:lengths[iii]-4],pred_motions[iii,:lengths[iii]-4,:args.joint_nums*3].reshape(-1,args.joint_nums,3),obj_points_pred_t[iii,:lengths[iii]-4],contact_threh = 0.05)

            

            contact_percentage = contact_metrics["mean_pred_contact_percent"]
            f1 = contact_metrics["micro_f1"]
            cprecision = contact_metrics['micro_precision']
            recall = contact_metrics['micro_recall']
            con_percents.append(contact_percentage)
            cprecisions.append(cprecision)
            recalls.append(recall)
            f1s.append(f1)
            
            contact_percentage_h = contact_metrics["mean_pred_contact_percent_hand"]
            f1_h = contact_metrics["micro_f1_hand"]
            cprecision_h = contact_metrics['micro_precision_hand']
            recall_h = contact_metrics['micro_recall_hand']
            con_percents_h.append(contact_percentage_h)
            cprecisions_h.append(cprecision_h)
            recalls_h.append(recall_h)
            f1s_h.append(f1_h)
        pen_dist = np.mean(pen_dists)
            
           
      
        pred_rot6d = sample[:,:,args.joint_nums*3+30:args.joint_nums*3+36]
        pred_trans = sample[:,:,args.joint_nums*3+36:]#.reshape(pred_motions.shape[0],pred_motions.shape[1],-1,3)
        pred_poses = matrix_to_axis_angle(rotation_6d_to_matrix(pred_rot6d.float()).float()).reshape(pred_motions.shape[0],pred_motions.shape[1],-1)
        
        pred_poses = pred_poses.cuda()
        pred_rot6d = pred_rot6d.cuda()
        pred_trans = pred_trans.cuda()
      
        
            
        f1_mean = np.mean(f1s) 
        
        cprecision_mean = np.mean(cprecisions)
        recall_mean = np.mean(recalls)
        contact_percentage_mean = np.mean(con_percents)
        
        f1_mean_h = np.mean(f1s_h) 
        cprecision_mean_h = np.mean(cprecisions_h)
        recall_mean_h = np.mean(recalls_h)
        contact_percentage_mean_h = np.mean(con_percents_h)
        
        pred_motions = (pred_motions - mean_enc)/std_enc
        gt_motions = (gt_motions-mean_enc)/std_enc
        mlength_list =lengths

        em_pred=motionencoder(pred_motions,mlength_list).loc
        em=motionencoder(gt_motions,mlength_list).loc
        et=textencoder(text_list_gt).loc
        et_pred=et
        num_poses+=sum(mlength_list)
        
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)
       

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match
        nb_sample += bs
        
    
    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)
    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
    
    l1_dist = np.zeros(1)


    



    

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 50)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 50)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample
    

    

    msg = "--> \t Eva. Re %d:, FID. %.4f, Diversity Real. %.4f, Diversity. %.4f, R_precision_real. (%.4f, %.4f, %.4f), R_precision. (%.4f, %.4f, %.4f), matching_real. %.4f, matching_pred. %.4f, mae. %.4f" % \
          (repeat_id, fid, diversity_real, diversity, R_precision_real[0], R_precision_real[1], R_precision_real[2],
           R_precision[0], R_precision[1], R_precision[2], matching_score_real, matching_score_pred, l1_dist)
    print(msg)
    return fid, diversity, R_precision, matching_score_pred, l1_dist,fskat,f1_mean,contact_percentage_mean,pen_dist,recall_mean,cprecision_mean,f1_mean_h,contact_percentage_mean_h,recall_mean_h,cprecision_mean_h
def load_dataset(args, max_frames, n_frames):
    data = get_dataset_loader(args,name=args.dataset,
                              batch_size=args.batch_size,
                              num_frames=max_frames,
                              split='test',
                              hml_mode='train',
                              eval_mode = True,
                              fixed_len=args.pred_len + args.context_len, pred_len=args.pred_len, device=dist_util.dev())
    data.fixed_length = n_frames
    return data

if __name__ == '__main__':
    ##
    
    # datasets=['omomo_correct','grab','chairs','imhd','neuraldome','intercap_correct','behave_correct'][:]
    datasets=['omomo','grab','chairs','imhd','neuraldome','intercap','behave'][:]
    # datasets =['neuraldome','intercap_correct','behave_correct']
    # datasets =['behave_correct']
    
    obj_canrot_dict ={}
    for dataset in datasets:
        dataset_used = dataset.split('_')[0]
        obj_canrot_dict[dataset_used] = {}
        for object_name in os.listdir(os.path.join('./InterAct',dataset_used,'objects')):
           
            obj_can_rot = np.load(os.path.join(f'./InterAct/{dataset_used}/objects/{object_name}/',f'{object_name}_can.npy'))
            
            obj_canrot_dict[dataset_used][object_name] = torch.from_numpy(obj_can_rot).float().to(torch.device('cpu'))
        if dataset in ['omomo_correct','behave_correct','intercap_correct']:
            obj_canrot_dict[dataset] = obj_canrot_dict[dataset_used]
    

    args = evaluation_parser()
    print('GT ORIG')
    args.eval_mode = 'debug'
    fixseed(args.seed)
    ## debug
    args.debug = 0
    args.use_joint = 1 
    max_frames = 300
    fps =30
    args.motion_length = 10
    n_frames = min(max_frames, int(args.motion_length*fps))
    args.batch_size = 256# This must be 32! Don't change it! otherwise it will cause a bug in R precision calc!
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    log_name = f'gs_{niter}_mode{args.df_mode}_cfg{args.df_cfg}_star{args.df_star}_tw{args.df_tweight}_ad{args.df_add}_delta{args.df_delta}_w{args.df_weight}_div{args.df_divider}_ust{args.df_upstop}_decay{args.df_decay}_bg{args.df_begin}_eta{args.df_eta}_rescale{args.df_rescale}_gw{args.df_gw}_r{args.df_r}_mom{args.df_mom}'
    
    log_file = os.path.join(os.path.dirname(args.model_path), log_name + f'.txt')
    save_dir = os.path.dirname(log_file)  # has not been tested with WandB

    print(f'Will save to log file [{log_file}]')

  

    print(f'Eval mode [{args.eval_mode}]')
    
    num_samples_limit = 1000  # None means no limit (eval over all dataset)
    run_mm = False
    mm_num_samples = 0
    mm_num_repeats = 0
    mm_num_times = 0
    diversity_times = 5 # about 3 Hrs


    dist_util.setup_dist(args.device)
  
    split = 'test'
    # gt_loader = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=None, split=split, hml_mode='gt')
    # # gen_loader = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=None, split=split, hml_mode='eval')
    # # added new features + support for prefix completion:
    # gen_loader = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=None, split=split, hml_mode='eval',
    #                                 fixed_len=args.context_len+args.pred_len, pred_len=args.pred_len, device=dist_util.dev(),
    #                                 autoregressive=args.autoregressive)

    # num_actions = gen_loader.dataset.num_actions

    # args.use_rot = 1
    data = load_dataset(args, max_frames, n_frames)
    # args.use_rot = 0
    if args.normalize:
        
        MEAN_DATA =torch.from_numpy(data.dataset.t2m_dataset.mean).float()
        STD_DATA =torch.from_numpy(data.dataset.t2m_dataset.std).float()
    else:
        MEAN_DATA=None
        STD_DATA = None
    
    model, diffusion = create_model_and_diffusion(args, data)
    diffusion = diffusion.to(dist_util.dev())
    
    sample_fn = diffusion.sample_step
    if args.autoregressive:
        sample_cls = AutoRegressiveSampler(args, sample_fn, n_frames)
        sample_fn = sample_cls.sample

    load_saved_model(model, args.model_path, use_avg=args.use_ema)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
    
    model.to(dist_util.dev())
    model.eval()  # disable random masking
    
    ###################################################
    ########################### eval model ####################################
    std_path = "./InterAct/stats/std_evaluator.npy"
    std_enc = np.ones_like(np.load(std_path)) ###
    std_enc = torch.from_numpy(std_enc).float().cuda()
    mean_path = "./InterAct/stats/mean_evaluator.npy"
    mean_enc = np.zeros_like(np.load(mean_path)) ##
    mean_enc = torch.from_numpy(mean_enc).float().cuda()
    mean_enc = 0
    std_enc = 1
    eval_model_epoch = "1599"
    # eval_type = "markersbps_3"
    path = './assets/evaluator.ckpt' 
    
    A=torch.load(path)
    STAT_DICT=A['state_dict']
   
    filtered_dict = {k[12:]: v for k, v in STAT_DICT.items() if k.startswith("textencoder")}
    filtered_dict2 = {k[14:]: v for k, v in STAT_DICT.items() if k.startswith("motionencoder")}
    #A['state_dict']=filtered_dict
    modelpath = 'distilbert-base-uncased'

    textencoder = DistilbertActorAgnosticEncoder(modelpath,latent_dim=256, 
    ff_size=1024,num_layers=4).cuda()
    textencoder.load_state_dict(filtered_dict)

    ## nfeats 231: motion dimension, 128*3: bps dimension
    motionencoder=ActorAgnosticEncoder(nfeats=52*3+9+128*3,vae=True,latent_dim=256,ff_size=1024,
                                    num_layers=4).cuda()
    motionencoder.load_state_dict(filtered_dict2)
    motionencoder.eval()
    textencoder.eval()
    
    accelerator = Accelerator( #deepspeed_plugin=ds_plugin,
            mixed_precision="fp16",
        )
    
    if not hasattr(model, "hf_device_map") or model.hf_device_map is None:
        model.hf_device_map = {}

    model, data,diffusion= accelerator.prepare(model,data,diffusion)
    bps_torch = bps_torch()
    bps_obj = np.load("./InterAct/stats/bps_basis_set_128_1.npy")
    bps_obj = torch.from_numpy(bps_obj)
    fid = []
    div = []
    top1 = []
    top2 = []
    top3 = []
    matching = []
    mm = []
    foot_sk = []
    f1s =[]
    contact_percentages =[]
    repeat_time = 2
    pen_dists = []
    recalls=[]
    cprecisions=[]
    
    f1s_h =[]
    recalls_h=[]
    cprecisions_h=[]
    contact_percentages_h =[]
    repeat_time = 2
    pen_dists = [] 
    obj_base_path = './InterAct'
    obj_v_dict={}
    # ['behave_correct','chairs','grab','imhd','intercap_correct','neuraldome','omomo_correct','chairs']:
    for datasets in ['behave','chairs','grab','imhd','intercap','neuraldome','omomo']:
        dataset_obj_path = os.path.join(obj_base_path,datasets,'objects')
        obj_v_dict[datasets]={}
        for obj_name in os.listdir(dataset_obj_path):
            # if args.version<9:
            V=trimesh.load(os.path.join(dataset_obj_path,obj_name,f'{obj_name}.obj')).vertices
            

                
            obj_v_dict[datasets][obj_name] = torch.from_numpy(np.array(V)).float().unsqueeze(0).to(torch.device('cpu'))
   
        
        
    for i in range(repeat_time):
        with torch.no_grad():
            best_fid, best_div, Rprecision, best_matching, best_mm,foot_skating,f1,contact_percentage,pen_dist,recall,cprecision,f1_h,contact_percentage_h,recall_h,cprecision_h = \
                eval_t2hoi(args,data, model, diffusion, textencoder, motionencoder,std_enc,mean_enc,repeat_id=i,obj_v_dict=obj_v_dict,mean_data=MEAN_DATA,std_data=STD_DATA,obj_canrot_dict=obj_canrot_dict)
        fid.append(best_fid)
        div.append(best_div)
        top1.append(Rprecision[0])
        top2.append(Rprecision[1])
        top3.append(Rprecision[2])
        matching.append(best_matching)
        mm.append(best_mm)
        foot_sk.append(foot_skating)
        f1s.append(f1)
        contact_percentages.append(contact_percentage)
        pen_dists.append(pen_dist)
        recalls.append(recall)

        cprecisions.append(cprecision)
        f1s_h.append(f1_h)
        contact_percentages_h.append(contact_percentage_h)
        recalls_h.append(recall_h)
        cprecisions_h.append(cprecision_h)

    fid = np.array(fid)
    div = np.array(div)
    top1 = np.array(top1)
    top2 = np.array(top2)
    top3 = np.array(top3)
    matching = np.array(matching)
    mm = np.array(mm)
    foot_sk = np.array(foot_sk)
    f1s =np.array(f1s)
    contact_percentages =np.array(contact_percentages)
    pen_dists = np.array(pen_dists)
    cprecisions = np.array(cprecisions)
    recalls= np.array(recalls)
    
    f1s_h =np.array(f1s_h)
    contact_percentages_h =np.array(contact_percentages_h)
    
    cprecisions_h = np.array(cprecisions_h)
    recalls_h= np.array(recalls_h)

    f = open(log_file, 'w')
    print('final result:')
    print('final result:', file=f, flush=True)

    msg_final = f"\tFID: {np.mean(fid):.3f}, conf. {np.std(fid) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                f"\tDiversity: {np.mean(div):.3f}, conf. {np.std(div) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                f"\tTOP1: {np.mean(top1):.3f}, conf. {np.std(top1) * 1.96 / np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2) * 1.96 / np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                f"\tMatching: {np.mean(matching):.3f}, conf. {np.std(matching) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                f"\tFSR: {np.mean(foot_sk):.3f}, conf. {np.std(foot_sk) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                f"\tF1: {np.mean(f1s):.3f}, conf. {np.std(f1s) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                f"\tCP: {np.mean(contact_percentages):.3f}, conf. {np.std(contact_percentages) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                f"\tPD: {np.mean(pen_dists):.6f}, conf. {np.std(pen_dists) * 1.96 / np.sqrt(repeat_time):.6f}\n" \
                f"\tCpreicison: {np.mean(cprecisions):.3f}, conf. {np.std(cprecisions) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                f"\tRecalls: {np.mean(recalls):.3f}, conf. {np.std(recalls) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                f"\tF1_H: {np.mean(f1s_h):.3f}, conf. {np.std(f1s_h) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                f"\tCP_H: {np.mean(contact_percentages_h):.3f}, conf. {np.std(contact_percentages_h) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                f"\tCpreicison_H: {np.mean(cprecisions_h):.3f}, conf. {np.std(cprecisions_h) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                f"\tRecalls_H: {np.mean(recalls_h):.3f}, conf. {np.std(recalls_h) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                f"\tMultimodality:{np.mean(mm):.3f}, conf.{np.std(mm) * 1.96 / np.sqrt(repeat_time):.3f}\n\n"
    print(msg_final)
    print(args.df_delta, args.df_weight,args.df_divider,args.df_upstop,args.df_decay,args.df_begin,args.df_mode)
    print(msg_final, file=f, flush=True)
    
    
    

