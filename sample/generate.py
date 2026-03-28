# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.markerset import *
from sample.joints2smpl import SmplhOptmize10_fulljoints
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion, load_saved_model
from utils import dist_util
from utils.sampler_util import ClassifierFreeSampleModel, AutoRegressiveSampler
from data_loaders.get_data import get_dataset_loader
import data_loaders.humanml.utils.paramUtil as paramUtil

from render.mesh_viz import visualize_body_obj
import trimesh

from common.quaternion import rotation_6d_to_matrix_np


import torch

def rigid_align_batch(P: torch.Tensor,
                      V: torch.Tensor,
                      eps: float = 1e-8):
    """
    Args
    ----
    P : (N, 3)           
    V : (T, N, 3)        
    Returns
    -------
    R : (T, 3, 3)        
    t : (T, 3)           
    """

    # 基本维度检查
    assert P.ndim == 2 and P.shape[1] == 3
    assert V.ndim == 3 and V.shape[2] == 3 and V.shape[1] == P.shape[0]

    device = P.device                                # PU
    N         = P.shape[0]
    # P_center = P_mean
    # 1) 去中心化
    P_mean    = P.mean(dim=0)                        # (3,)
    P_center  = P - P_mean                           # (N,3)

    V_mean    = V.mean(dim=1)                        # (T,3)
    V_center  = V - V_mean[:, None, :]               # (T,N,3)

    # 2) 计算批量协方差 H_t = P_c^T V_c
    #    einsum 语义: (ni)*(t,nj) -> (t,ij)
    H = torch.einsum('ni,tnj->tij', P_center, V_center)  # (T,3,3)

    # 3) SVD 分解
    U, S, Vh = torch.linalg.svd(H)                   # SVD; Vh = V^T

    # 4) 处理镜像反射：保证 det(R)=+1
    det_R = torch.det(Vh.transpose(-2, -1) @ U.transpose(-2, -1))  # (T,)
    # 构造对角矩阵 D，最后一维根据 det 调整符号
    D = torch.diag_embed(
            torch.stack([
                torch.ones_like(det_R),
                torch.ones_like(det_R),
                torch.sign(det_R)                    # -1 
            ], dim=-1)
        )                                            # (T,3,3)

    # 5) 得到最终旋转 R = Vh^T D U^T
    R = Vh.transpose(-2, -1) @ D @ U.transpose(-2, -1)  # (T,3,3)

    # 6) 平移 t = v_mean - R * p_mean
    t = V_mean - (R @ P_mean)                        # (T,3)

    return R, t

def theta_to_y_rotation(theta):
    """
    Converts a batch of Y-axis rotation angles (T,) to rotation matrices (T, 3, 3).

    Args:
        theta: (T,) torch tensor of Y-axis rotation angles in radians.

    Returns:
        R: (T, 3, 3) torch tensor of Y-axis rotation matrices.
    """
    T = theta.shape[0]

    # Compute cos and sin values
    cos_t = torch.cos(theta)  # (T,)
    sin_t = torch.sin(theta)  # (T,)

    # Initialize rotation matrix batch (T, 3, 3) with zeros
    R = torch.zeros((T, 3, 3), device=theta.device)

    # Assign values based on Y-axis rotation matrix structure
    R[:, 0, 0] = cos_t
    R[:, 0, 2] = sin_t
    R[:, 1, 1] = 1  # Y-axis remains unchanged
    R[:, 2, 0] = -sin_t
    R[:, 2, 2] = cos_t

    return R

def recover_from_ric2(feature):
    rot_y_theta=np.cumsum(feature[:,0])
    Rotation_y=theta_to_y_rotation(rot_y_theta)
    root_pos=torch.zeros([feature.shape[0],3]).to(feature.device)
    root_pos[:,1]=feature[:,2]
    root_pos[:,[0,2]]=torch.matmul(feature[:,1:4].reshape(-1,1,3).float(),Rotation_y.permute(0,2,1).float())[:,0,[0,2]]
    positions=torch.matmul(feature[:,4:4+77*3].reshape(-1,77,3).float(),Rotation_y.permute(0,2,1).float())+root_pos.reshape(-1,1,3)
    return positions

def main(args=None):
    if args is None:
        # args is None unless this method is called from another function (e.g. during training)
        args = generate_args()
    args.guidance_param = 1.5 
    device = torch.device('cuda:0')
    fixseed(args.seed)
    out_path = args.output_dir
    n_joints = 22 if args.dataset == 'humanml' else 21
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    # max_frames = 196 if args.dataset in ['kit', 'humanml'] else 60
    max_frames = 300
    fps =30
    # fps = 12.5 if args.dataset == 'kit' else 20
    args.motion_length = 10
    n_frames = min(max_frames, int(args.motion_length*fps))
    print(n_frames,args.motion_length,fps,'JKJKJK')
    is_using_data = not any([args.input_text, args.text_prompt, args.action_file, args.action_name])
    if args.context_len > 0:
        is_using_data = True  # For prefix completion, we need to sample a prefix
    dist_util.setup_dist(args.device)
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'samples_{}_{}_{}_seed{}'.format(name, niter,args.guidance_param, args.seed))
        if args.text_prompt != '':
            out_path += '_' + args.text_prompt.replace(' ', '_').replace('.', '')
        elif args.input_text != '':
            out_path += '_' + os.path.basename(args.input_text).replace('.txt', '').replace(' ', '_').replace('.', '')
        elif args.dynamic_text_path != '':
            out_path += '_' + os.path.basename(args.dynamic_text_path).replace('.txt', '').replace(' ', '_').replace('.', '')
    out_path = out_path +'_'+args.dataset+'_mesh_woguidance_fullseq'
    # this block must be called BEFORE the dataset is loaded
    texts = None
    if args.text_prompt != '':
        texts = [args.text_prompt] * args.num_samples
    elif args.input_text != '':
        assert os.path.exists(args.input_text)
        with open(args.input_text, 'r') as fr:
            texts = fr.readlines()
        texts = [s.replace('\n', '') for s in texts]
        args.num_samples = len(texts)
    elif args.dynamic_text_path != '':
        assert os.path.exists(args.dynamic_text_path)
        assert args.autoregressive, "Dynamic text sampling is only supported with autoregressive sampling."
        with open(args.dynamic_text_path, 'r') as fr:
            texts = fr.readlines()
        texts = [s.replace('\n', '') for s in texts]
        n_frames = len(texts) * args.pred_len  # each text prompt is for a single prediction
    elif args.action_name:
        action_text = [args.action_name]
        args.num_samples = 1
    elif args.action_file != '':
        assert os.path.exists(args.action_file)
        with open(args.action_file, 'r') as fr:
            action_text = fr.readlines()
        action_text = [s.replace('\n', '') for s in action_text]
        args.num_samples = len(action_text)

    # args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples
    print(args.batch_size,'BS')
    print('Loading dataset...')
    data = load_dataset(args, max_frames, n_frames)
    total_num_samples = args.num_samples * args.num_repetitions

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)
    diffusion = diffusion.to(dist_util.dev())

    # sample_fn = diffusion.p_sample_loop
    if args.autoregressive:
        sample_cls = AutoRegressiveSampler(args, sample_fn, n_frames)
        sample_fn = sample_cls.sample

    print(f"Loading checkpoints from [{args.model_path}]...")
    load_saved_model(model, args.model_path, use_avg=args.use_ema)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking

    motion_shape = (args.batch_size, model.njoints, model.nfeats, n_frames)
    iterator = iter(data)
    GLOBAL_INDEX = 0
    BATCH_LEN = 0
    for input_motion, model_kwargs in iterator:

        
        input_motion = input_motion.to(dist_util.dev())
        original = model_kwargs['y']['text'] 
        ORIG_LEN =len(original)
        if texts is not None:
 
            if GLOBAL_INDEX+ ORIG_LEN >len(texts):
                break
            
            model_kwargs['y']['text'] = texts[GLOBAL_INDEX:GLOBAL_INDEX+ORIG_LEN]

        model_kwargs['y'] = {key: val.to(dist_util.dev()) if torch.is_tensor(val) else val for key, val in model_kwargs['y'].items()}
        init_image = None    
        
        all_motions = []
        all_lengths = []
        all_text = []
        all_gt =[ ]

        # add CFG scale to batch
        if args.guidance_param != 1:
            model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param

        
        if args.dynamic_text_path != '':
          
            model_kwargs['y']['text'] = [model_kwargs['y']['text']] * args.num_samples
            if args.text_encoder_type == 'bert':
                model_kwargs['y']['text_embed'] = (model_kwargs['y']['text_embed'][0].unsqueeze(0).repeat(args.num_samples, 1, 1, 1), 
                                                model_kwargs['y']['text_embed'][1].unsqueeze(0).repeat(args.num_samples, 1, 1))
            else:
                raise NotImplementedError('DiP model only supports BERT text encoder at the moment. If you implement this, please send a PR!')
       
        human_end = args.joint_nums*3

        obj_start = args.joint_nums*3+args.foot + 30
        NB =args.joint_nums
        body_index = list(range(22*3))+list(range(52*3,52*3+args.foot))
        hand_index = list(range(22*3,52*3+30)) 
        for rep_i in range(args.num_repetitions):
            print(f'### Sampling [repetitions #{rep_i}]')
            batch_rand = torch.randn(motion_shape).to(device)

            if args.split_ho_emb and not args.hand_split:
                batch1 =batch_rand[:,:obj_start]
                batch2 = batch_rand[:,obj_start:]
                batch3 = None
            elif args.hand_split:
                
                batch1 = batch_rand[:,body_index]
                batch2 = batch_rand[:,obj_start:]
                batch3 = batch_rand[:,hand_index]
                print(batch_rand.shape,batch1.shape,batch2.shape,batch3.shape)
                
            else:
                batch1= batch_rand
                batch2 = batch_rand
                batch3 = None
            sample = diffusion.sample_step(model, batch1,batch2, batch3,0, model_kwargs['y'],scheduling_mode='full_sequence') # scheduling_mode 
            # sample = input_motion
            print(sample.shape,'OUTPUT_SAMPLE')
         
            
                
            input_motion = input_motion.squeeze(2).permute(0,2,1).detach().cpu().numpy()
            sample = sample.squeeze(2).permute(0,2,1).detach().cpu().numpy() ## B,L,D
            if args.normalize:
                sample = data.dataset.t2m_dataset.inv_transform(sample)
                input_motion = data.dataset.t2m_dataset.inv_transform(input_motion)
               

            rot2xyz_pose_rep = 'xyz' if model.data_rep in ['xyz', 'hml_vec'] else model.data_rep
            rot2xyz_mask = None if rot2xyz_pose_rep == 'xyz' else model_kwargs['y']['mask'].reshape(args.batch_size, n_frames).bool()
           

            if args.unconstrained:
                all_text += ['unconstrained'] * args.num_samples
            else:
                text_key = 'text' if 'text' in model_kwargs['y'] else 'action_text'
                all_text += model_kwargs['y'][text_key]

            all_motions.append(sample)
            all_gt.append(input_motion)
            _len = model_kwargs['y']['lengths'].cpu().numpy()
            if 'prefix' in model_kwargs['y'].keys():
                _len[:] = sample.shape[1]
            all_lengths.append(_len)

            print(f"created {len(all_motions) * args.batch_size} samples")


        all_motions = np.concatenate(all_motions, axis=0)
        all_gt = np.concatenate(all_gt, axis=0)
        all_motions = all_motions[:total_num_samples]  # [bs, njoints, 6, seqlen]
        all_gt = all_gt[:total_num_samples]
        all_text = all_text[:total_num_samples]
        all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]

       
        os.makedirs(out_path,exist_ok=True)

        npy_path = os.path.join(out_path, f'results_{GLOBAL_INDEX}.npy')
        print(f"saving results file to [{npy_path}]")
        np.save(npy_path,
                {'motion': all_motions,'gt':all_gt, 'text': all_text, 'lengths': all_lengths,
                'num_samples': args.num_samples, 'num_repetitions': args.num_repetitions,'seq_name':model_kwargs['y']['seq_name'][:args.batch_size]})
        
        if args.dynamic_text_path != '':
            text_file_content = '\n'.join(['#'.join(s) for s in all_text])
        else:
            text_file_content = '\n'.join(all_text)
        if texts is not None:
            text_file_content = '\n'.join(texts)
        with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
            fw.write(text_file_content)
        # else:
        #     with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
        #         fw.write(text_file_content)
        with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
            fw.write('\n'.join([str(l) for l in all_lengths]))

        print(f"saving visualizations to [{out_path}]...")
        skeleton = paramUtil.kit_kinematic_chain if args.dataset == 'kit' else paramUtil.t2m_kinematic_chain

        sample_print_template, row_print_template, all_print_template, \
        sample_file_template, row_file_template, all_file_template = construct_template_variables(args.unconstrained)
        max_vis_samples = 64
        num_vis_samples = min(args.num_samples, max_vis_samples)
        animations = np.empty(shape=(args.num_samples, args.num_repetitions), dtype=object)
        max_length = max(all_lengths)

        for sample_i in range(args.num_samples):
            rep_files = []
            for rep_i in range(1):
                caption = all_text[rep_i*args.batch_size + sample_i]
                if args.dynamic_text_path != '':  # caption per frame
                    assert type(caption) == list
                    caption_per_frame = []
                    for c in caption:
                        caption_per_frame += [c] * args.pred_len
                    caption = caption_per_frame

                
                # Trim / freeze motion if needed
                length = all_lengths[rep_i*args.batch_size + sample_i]
                ## B,L,D
                if texts is not None:
                    use_gt = 1
                else:
                    use_gt = 2 
                for i in range(use_gt):
                    # if i==1: 
                    #     continue
                    
                    if i==0:
                        motion = all_motions[rep_i*args.batch_size + sample_i][:length]
                    else:
                        motion = all_gt[rep_i*args.batch_size + sample_i][:length]
                    # if motion.shape[0] > length:
                    #     motion[length:-1] = motion[length-1]
                    
                    # if i==0:
                    #     motion = all_motions[rep_i*args.batch_size + sample_i][:max_length]
                    # else:
                    #     motion = all_gt[rep_i*args.batch_size + sample_i][:max_length]
                    # if motion.shape[0] > length:
                    #     motion[length:-1] = motion[length-1]  # duplicate the last frame to end of motion, so all motions will be in equal length

                    save_file = sample_file_template.format(sample_i, rep_i)
                    animation_save_path = os.path.join(out_path, save_file)
                    gt_frames = np.arange(args.context_len) if args.context_len > 0 and not args.autoregressive else []
                        
                            
                            
                    obj_start = args.joint_nums*3+args.foot + 30
                        
                    NB=args.joint_nums
                        
                    pred_motion = motion
                    motion_obj = pred_motion[:,obj_start:]
                    human_verts = pred_motion[:,:human_end].reshape(-1,NB,3)
                    
                    MODEL2=SmplhOptmize10_fulljoints('male',1,human_verts.shape[0],joint_nums = args.joint_nums)
                        
                    device = torch.device('cuda:0')
                    verts, human_faces,poses,betas,trans= MODEL2.forward(torch.from_numpy(human_verts).float().to(device))
                    human_verts = verts.float().detach().cpu().numpy()
                            # print(human_verts.shape,human_faces.shape,'VERTS_FACE_SHAPE')
                    seq_name = model_kwargs['y']['seq_name'][rep_i*args.batch_size + sample_i]
                    dataset_name1, obj_name = str(seq_name).rsplit('_',1)
                   
                    mesh_path = os.path.join(f"./InterAct/{dataset_name1.split('_')[0]}/objects", obj_name, obj_name + ".obj")
                    
                        
                    obj_mesh = trimesh.load(mesh_path)
                    obj_verts,obj_face = obj_mesh.vertices, obj_mesh.faces
                    
                  
                    obj_angles = motion_obj[...,:6]
                    obj_trans = motion_obj[...,6:]

                    angle_matrix = rotation_6d_to_matrix_np(obj_angles)
                    angle_matrix = rotation_6d_to_matrix_np(obj_angles)
                    
                    obj_verts = (obj_verts)[None, ...]
                    obj_verts = np.matmul(obj_verts, np.transpose(angle_matrix, (0, 2, 1))) + obj_trans[:, None, :]
                    # obj_angles = Rotation.from_matrix(angle_matrix).as_rotvec()
                    # os.path.join(out_path, 'results.npy')
                    INDEX= rep_i*args.batch_size + sample_i
                    if i==0:
                        opath= os.path.join(out_path,f'{GLOBAL_INDEX+INDEX}_{seq_name}_mesh.mp4')
                    else:
                        opath = os.path.join(out_path,f'{GLOBAL_INDEX+INDEX}_{seq_name}_gt_mesh.mp4')
                    # visualize_points_obj(human_verts, obj_verts, obj_face, save_path=opath, show_frame=True, multi_angle=False)
                    if i==0:
                        npz_path= os.path.join(out_path,f'{GLOBAL_INDEX+INDEX}_{seq_name}_mesh.npz')
                    else:
                        npz_path = os.path.join(out_path,f'{GLOBAL_INDEX+INDEX}_{seq_name}_gt_mesh.npz')
                    dct={'poses':poses,'betas':betas,'trans':trans}
                    np.savez(npz_path,**dct)
                    visualize_body_obj(human_verts, human_faces,obj_verts, obj_face, save_path=opath, show_frame=True, multi_angle=False)

                  
        GLOBAL_INDEX = GLOBAL_INDEX + ORIG_LEN

    abs_path = os.path.abspath(out_path)
    print(f'[Done] Results are at [{abs_path}]')

    return out_path


def save_multiple_samples(out_path, file_templates,  animations, fps, max_frames, no_dir=False):
    
    num_samples_in_out_file = 3
    n_samples = animations.shape[0]
    
    for sample_i in range(0,n_samples,num_samples_in_out_file):
        last_sample_i = min(sample_i+num_samples_in_out_file, n_samples)
        all_sample_save_file = file_templates['all'].format(sample_i, last_sample_i-1)
        if no_dir and n_samples <= num_samples_in_out_file:
            all_sample_save_path = out_path
        else:
            all_sample_save_path = os.path.join(out_path, all_sample_save_file)
            print(f'saving {os.path.split(out_path)[1]}/{all_sample_save_file}')

        clips = clips_array(animations[sample_i:last_sample_i])
        clips.duration = max_frames/fps
        
        # import time
        # start = time.time()
        clips.write_videofile(all_sample_save_path, fps=fps, threads=4, logger=None)
        # print(f'duration = {time.time()-start}')
        
        for clip in clips.clips: 
            # close internal clips. Does nothing but better use in case one day it will do something
            clip.close()
        clips.close()  # important
 

def construct_template_variables(unconstrained):
    row_file_template = 'sample{:02d}.mp4'
    all_file_template = 'samples_{:02d}_to_{:02d}.mp4'
    if unconstrained:
        sample_file_template = 'row{:02d}_col{:02d}.mp4'
        sample_print_template = '[{} row #{:02d} column #{:02d} | -> {}]'
        row_file_template = row_file_template.replace('sample', 'row')
        row_print_template = '[{} row #{:02d} | all columns | -> {}]'
        all_file_template = all_file_template.replace('samples', 'rows')
        all_print_template = '[rows {:02d} to {:02d} | -> {}]'
    else:
        sample_file_template = 'sample{:02d}_rep{:02d}.mp4'
        sample_print_template = '["{}" ({:02d}) | Rep #{:02d} | -> {}]'
        row_print_template = '[ "{}" ({:02d}) | all repetitions | -> {}]'
        all_print_template = '[samples {:02d} to {:02d} | all repetitions | -> {}]'

    return sample_print_template, row_print_template, all_print_template, \
           sample_file_template, row_file_template, all_file_template


def load_dataset(args, max_frames, n_frames):
    data = get_dataset_loader(args,name=args.dataset,
                              batch_size=args.batch_size,
                              num_frames=max_frames,
                              split='test',
                              hml_mode='train',
                            #   hml_mode='train' if args.pred_len > 0 else 'text_only',  # We need to sample a prefix from the dataset
                              fixed_len=args.pred_len + args.context_len, pred_len=args.pred_len, device=dist_util.dev())
    data.fixed_length = n_frames
    return data

# data = get_dataset_loader(args,name=args.dataset, 
#                               batch_size=args.batch_size, 
#                               num_frames=args.num_frames, 
#                               fixed_len=args.pred_len + args.context_len, 
#                               pred_len=args.pred_len,
#                               device=dist_util.dev(),)


def is_substr_in_list(substr, list_of_strs):
    return np.char.find(list_of_strs, substr) != -1  # [substr in string for string in list_of_strs]

if __name__ == "__main__":
    main()
