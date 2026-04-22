import numpy as np
from scipy import linalg
import torch
from scipy.ndimage import uniform_filter1d
import json
import pytorch3d.loss
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from pytorch3d.ops import cot_laplacian
from pytorch3d.structures import Meshes
import shutil
# from human_body_prior.tools import tgm_conversion as tgm
import chamfer_distance as chd
def calculate_mpjpe(gt_joints, pred_joints):
    """
    gt_joints: num_poses x num_joints(22) x 3
    pred_joints: num_poses x num_joints(22) x 3
    (obtained from recover_from_ric())
    """
    assert gt_joints.shape == pred_joints.shape, f"GT shape: {gt_joints.shape}, pred shape: {pred_joints.shape}"

    # Align by root (pelvis)
    pelvis = gt_joints[:, [0]].mean(1)
    gt_joints = gt_joints - torch.unsqueeze(pelvis, dim=1)
    pelvis = pred_joints[:, [0]].mean(1)
    pred_joints = pred_joints - torch.unsqueeze(pelvis, dim=1)

    # Compute MPJPE
    mpjpe = torch.linalg.norm(pred_joints - gt_joints, dim=-1) # num_poses x num_joints=22
    mpjpe_seq = mpjpe.mean(-1) # num_poses

    return mpjpe_seq

def compute_metrics( ori_jpos_gt, ori_jpos_pred, \
     gt_obj_com_pos, pred_obj_com_pos, \
    gt_obj_rot_mat, pred_obj_rot_mat):
    # verts_gt: T X Nv X 3 
    # jpos_gt: T X J X 3
    # gt_trans: T X 3
    # gt_rot_mat: T X 22 X 3 X 3 
    # gt_obj_com_pos: T X 3
    # gt_obj_rot_mat: T X 3 X 3
    # human_faces: Nf X 3, array  
    # obj_verts: T X No X 3
    # obj_faces: Nf X 3, array  
    # gt_contact_label: T X 2 (left palm, right palm)
    # pred_contact_label: T X 2
    # actual_len: scale value 

    # ori_verts_gt = ori_verts_gt[:actual_len]
    # ori_verts_pred = ori_verts_pred[:actual_len]
    # ori_jpos_gt = ori_jpos_gt[:actual_len]
    # ori_jpos_pred = ori_jpos_pred[:actual_len]
    # gt_trans = gt_trans[:actual_len]
    # pred_trans = pred_trans[:actual_len]
    # gt_rot_mat = gt_rot_mat[:actual_len]
    # pred_rot_mat = pred_rot_mat[:actual_len]
    # gt_obj_com_pos = gt_obj_com_pos[:actual_len]
    # pred_obj_com_pos = pred_obj_com_pos[:actual_len] 
    # gt_obj_rot_mat = gt_obj_rot_mat[:actual_len]
    # pred_obj_rot_mat = pred_obj_rot_mat[:actual_len] 
    # gt_obj_verts = gt_obj_verts[:actual_len]
    # pred_obj_verts = pred_obj_verts[:actual_len]

    # Calculate global hand joint position error 
    # if use_joints24:
    #     lhand_idx = 22 
    #     rhand_idx = 23 
    # else:
    
    hand_idx = list(range(22,52))
    body_idx = list(range(22,52))
    # hand_index = list(range(22,52))
    lhand_jpos_pred = ori_jpos_pred[:, hand_idx, :].detach().cpu().numpy() 
    # rhand_jpos_pred = ori_jpos_pred[:, rhand_idx, :].detach().cpu().numpy() 
    lhand_jpos_gt = ori_jpos_gt[:, hand_idx, :].detach().cpu().numpy()
    # rhand_jpos_gt = ori_jpos_gt[:, rhand_idx, :].detach().cpu().numpy() 
    lhand_jpe = np.linalg.norm(lhand_jpos_pred - lhand_jpos_gt, axis=-1).mean() * 1000
    # rhand_jpe = np.linalg.norm(rhand_jpos_pred - rhand_jpos_gt, axis=-1).mean() * 1000
    hand_jpe = lhand_jpe

    lhand_jpos_pred = ori_jpos_pred[:, body_idx, :].detach().cpu().numpy() 
    # rhand_jpos_pred = ori_jpos_pred[:, rhand_idx, :].detach().cpu().numpy() 
    lhand_jpos_gt = ori_jpos_gt[:, body_idx, :].detach().cpu().numpy()
    # rhand_jpos_gt = ori_jpos_gt[:, rhand_idx, :].detach().cpu().numpy() 
    body_jpe = np.linalg.norm(lhand_jpos_pred - lhand_jpos_gt, axis=-1).mean() * 1000
    # rhand_jpe = np.linalg.norm(rhand_jpos_pred - rhand_jpos_gt, axis=-1).mean() * 1000
    # hand_jpe = lhand_jpe
    # Calculate MPJPE 
    jpos_pred = ori_jpos_pred - ori_jpos_pred[:, 0:1] # zero out root
    jpos_gt = ori_jpos_gt - ori_jpos_gt[:, 0:1] 
    jpos_pred = jpos_pred.detach().cpu().numpy()
    jpos_gt = jpos_gt.detach().cpu().numpy()
    mpjpe = np.linalg.norm(jpos_pred - jpos_gt, axis=2).mean() * 1000

    # Caculate translation error 
    pred_trans = ori_jpos_pred[:,0]
    gt_trans = ori_jpos_gt[:,0]
    trans_err = np.linalg.norm(pred_trans.detach().cpu().numpy() - gt_trans.detach().cpu().numpy(), axis=-1).mean() * 1000
    
    # Calculate rotation error
    # rot_dist = 0 
    ## foot skating

    # Compute object rotation error.
    obj_rot_mat_pred = pred_obj_rot_mat.detach().cpu().numpy() 
    obj_rot_mat_gt = gt_obj_rot_mat.detach().cpu().numpy()
    obj_rot_dist = get_frobenious_norm_rot_only(obj_rot_mat_pred.reshape(-1, 3, 3), obj_rot_mat_gt.reshape(-1, 3, 3))
    # obj_rot_dist = 0 

    # Compute com error. 
    obj_com_pos_err = np.linalg.norm(pred_obj_com_pos.detach().cpu().numpy() - gt_obj_com_pos.detach().cpu().numpy(), axis=1).mean() * 1000

    

    return hand_jpe, body_jpe, mpjpe, trans_err, obj_rot_dist, obj_com_pos_err 

def get_frobenious_norm_rot_only(x, y):
    # x, y: N X 3 X 3 
    error = 0.0
    for i in range(len(x)):
        x_mat = x[i][:3, :3]
        y_mat_inv = np.linalg.inv(y[i][:3, :3])
        error_mat = np.matmul(x_mat, y_mat_inv)
        ident_mat = np.identity(3)
        error += np.linalg.norm(ident_mat - error_mat, 'fro')
    return error / len(x)

def determine_floor_height_and_contacts(body_joint_seq, fps=30):
    '''
    Input: body_joint_seq N x 22 x 3 numpy array
    Contacts are N x 4 where N is number of frames and each row is left heel/toe, right heel/toe
    '''
    FLOOR_VEL_THRESH = 0.005
    FLOOR_HEIGHT_OFFSET = 0.01

    num_frames = body_joint_seq.shape[0]

    # compute toe velocities
    root_seq = body_joint_seq[:, 0, :]
    left_toe_seq = body_joint_seq[:, 10, :]
    right_toe_seq = body_joint_seq[:, 11, :]
    left_toe_vel = np.linalg.norm(left_toe_seq[1:] - left_toe_seq[:-1], axis=1)
    left_toe_vel = np.append(left_toe_vel, left_toe_vel[-1])
    right_toe_vel = np.linalg.norm(right_toe_seq[1:] - right_toe_seq[:-1], axis=1)
    right_toe_vel = np.append(right_toe_vel, right_toe_vel[-1])

    # now foot heights (z is up)
    left_toe_heights = left_toe_seq[:, 2]
    right_toe_heights = right_toe_seq[:, 2]
    root_heights = root_seq[:, 2]

    # filter out heights when velocity is greater than some threshold (not in contact)
    all_inds = np.arange(left_toe_heights.shape[0])
    left_static_foot_heights = left_toe_heights[left_toe_vel < FLOOR_VEL_THRESH]
    left_static_inds = all_inds[left_toe_vel < FLOOR_VEL_THRESH]
    right_static_foot_heights = right_toe_heights[right_toe_vel < FLOOR_VEL_THRESH]
    right_static_inds = all_inds[right_toe_vel < FLOOR_VEL_THRESH]

    all_static_foot_heights = np.append(left_static_foot_heights, right_static_foot_heights)
    all_static_inds = np.append(left_static_inds, right_static_inds)

    if all_static_foot_heights.shape[0] > 0:
        cluster_heights = []
        cluster_root_heights = []
        cluster_sizes = []
        # cluster foot heights and find one with smallest median
        clustering = DBSCAN(eps=0.005, min_samples=3).fit(all_static_foot_heights.reshape(-1, 1))
        all_labels = np.unique(clustering.labels_)
        # print(all_labels)
       
        min_median = min_root_median = float('inf')
        for cur_label in all_labels:
            cur_clust = all_static_foot_heights[clustering.labels_ == cur_label]
            cur_clust_inds = np.unique(all_static_inds[clustering.labels_ == cur_label]) # inds in the original sequence that correspond to this cluster
           
            # get median foot height and use this as height
            cur_median = np.median(cur_clust)
            cluster_heights.append(cur_median)
            cluster_sizes.append(cur_clust.shape[0])

            # get root information
            cur_root_clust = root_heights[cur_clust_inds]
            cur_root_median = np.median(cur_root_clust)
            cluster_root_heights.append(cur_root_median)
           
            # update min info
            if cur_median < min_median:
                min_median = cur_median
                min_root_median = cur_root_median

        floor_height = min_median 
        offset_floor_height = floor_height - FLOOR_HEIGHT_OFFSET # toe joint is actually inside foot mesh a bit

    else:
        floor_height = offset_floor_height = 0.0
   
    return floor_height

def get_foot_sliding(
    verts,
    up="z",
    threshold = 0.01  # 1 cm/frame
):
    # verts: T X Nv X 3
    vert_velocities = []
    up_coord = 2 if up == "z" else 1
    lowest_vert_idx = np.argmin(verts[:, :, up_coord], axis=1)
    for frame in range(1, verts.shape[0] - 1):
        vert_idx = lowest_vert_idx[frame]
        vel = np.linalg.norm(
            verts[frame + 1, vert_idx, :] - verts[frame - 1, vert_idx, :]
        ) / 2
        vert_velocities.append(vel)
    return np.sum(np.array(vert_velocities) > threshold) / verts.shape[0] * 100

def compute_foot_sliding_for_smpl(pred_global_jpos, floor_height):
    # pred_global_jpos: T X J X 3 
    seq_len = pred_global_jpos.shape[0]

    # Put human mesh to floor z = 0 and compute. 
    pred_global_jpos[:, :, 2] -= floor_height

    lankle_pos = pred_global_jpos[:, 7, :] # T X 3 
    ltoe_pos = pred_global_jpos[:, 10, :] # T X 3 

    rankle_pos = pred_global_jpos[:, 8, :] # T X 3 
    rtoe_pos = pred_global_jpos[:, 11, :] # T X 3 

    H_ankle = 0.08 # meter
    H_toe = 0.04 # meter 

    lankle_disp = np.linalg.norm(lankle_pos[1:, :2] - lankle_pos[:-1, :2], axis = 1) # T 
    ltoe_disp = np.linalg.norm(ltoe_pos[1:, :2] - ltoe_pos[:-1, :2], axis = 1) # T 
    rankle_disp = np.linalg.norm(rankle_pos[1:, :2] - rankle_pos[:-1, :2], axis = 1) # T 
    rtoe_disp = np.linalg.norm(rtoe_pos[1:, :2] - rtoe_pos[:-1, :2], axis = 1) # T 

    lankle_subset = lankle_pos[:-1, -1] < H_ankle
    ltoe_subset = ltoe_pos[:-1, -1] < H_toe
    rankle_subset = rankle_pos[:-1, -1] < H_ankle
    rtoe_subset = rtoe_pos[:-1, -1] < H_toe
   
    lankle_sliding_stats = np.abs(lankle_disp * (2 - 2 ** (lankle_pos[:-1, -1]/H_ankle)))[lankle_subset]
    lankle_sliding = np.sum(lankle_sliding_stats)/seq_len * 1000

    ltoe_sliding_stats = np.abs(ltoe_disp * (2 - 2 ** (ltoe_pos[:-1, -1]/H_toe)))[ltoe_subset]
    ltoe_sliding = np.sum(ltoe_sliding_stats)/seq_len * 1000

    rankle_sliding_stats = np.abs(rankle_disp * (2 - 2 ** (rankle_pos[:-1, -1]/H_ankle)))[rankle_subset]
    rankle_sliding = np.sum(rankle_sliding_stats)/seq_len * 1000

    rtoe_sliding_stats = np.abs(rtoe_disp * (2 - 2 ** (rtoe_pos[:-1, -1]/H_toe)))[rtoe_subset]
    rtoe_sliding = np.sum(rtoe_sliding_stats)/seq_len * 1000

    sliding = (lankle_sliding + ltoe_sliding + rankle_sliding + rtoe_sliding) / 4.

    return sliding 

# (X - X_train)*(X - X_train) = -2X*X_train + X*X + X_train*X_train
def euclidean_distance_matrix(matrix1, matrix2):
    """
        Params:
        -- matrix1: N1 x D
        -- matrix2: N2 x D
        Returns:
        -- dist: N1 x N2
        dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)    # shape (num_test, num_train)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)    # shape (num_test, 1)
    d3 = np.sum(np.square(matrix2), axis=1)     # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting
    return dists

def calculate_top_k(mat, top_k):
    size = mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    bool_mat = (mat == gt_mat)
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
#         print(correct_vec, bool_mat[:, i])
        correct_vec = (correct_vec | bool_mat[:, i])
        # print(correct_vec)
        top_k_list.append(correct_vec[:, None])
    top_k_mat = np.concatenate(top_k_list, axis=1)
    return top_k_mat


def calculate_R_precision(embedding1, embedding2, top_k, sum_all=False):
    dist_mat = euclidean_distance_matrix(embedding1, embedding2)
    argmax = np.argsort(dist_mat, axis=1)
    top_k_mat = calculate_top_k(argmax, top_k)
    if sum_all:
        return top_k_mat.sum(axis=0)
    else:
        return top_k_mat


def calculate_matching_score(embedding1, embedding2, sum_all=False):
    assert len(embedding1.shape) == 2
    assert embedding1.shape[0] == embedding2.shape[0]
    assert embedding1.shape[1] == embedding2.shape[1]

    dist = linalg.norm(embedding1 - embedding2, axis=1)
    if sum_all:
        return dist.sum(axis=0)
    else:
        return dist



def calculate_activation_statistics(activations):
    """
    Params:
    -- activation: num_samples x dim_feat
    Returns:
    -- mu: dim_feat
    -- sigma: dim_feat x dim_feat
    """
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


def calculate_diversity(activation, diversity_times):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist.mean()


def compute_penetration_metric(version,dataset_name, object_name, ori_verts_pred, \
    pred_obj_com_pos, pred_obj_rot_mat, eval_fullbody=False):
    # ori_verts_pred: T X Nv X 3 
    # pred_obj_com_pos: T X 3
    # pred_obj_rot_mat: T X 3 X 3
    ori_verts_pred = ori_verts_pred[None] # 1 X T X Nv X 3 
    pred_obj_com_pos = pred_obj_com_pos[None] # 1 X T X 3 
    pred_obj_rot_mat = pred_obj_rot_mat[None] # 1 X T X 3 X 3 

    # if not eval_fullbody:
    #     hand_verts = ori_verts_pred[:, :, self.hand_vertex_idxs, :] # BS X T X N_hand X 3
    # else:
    hand_verts = ori_verts_pred 

    hand_verts_in_rest_frame = hand_verts - pred_obj_com_pos[:, :, None, :] # BS X T X N_hand X 3 
    hand_verts_in_rest_frame = torch.matmul(pred_obj_rot_mat[:, :, None, :, :].repeat(1, 1, \
                        hand_verts_in_rest_frame.shape[2], 1, 1), \
                        hand_verts_in_rest_frame[:, :, :, :, None]).squeeze(-1) # BS X T X N_hand X 3 

    curr_object_sdf, curr_object_sdf_centroid, curr_object_sdf_extents = \
    load_object_sdf_data(dataset_name,object_name,version=version)

    # Convert hand vertices to align with rest pose object. 
    signed_dists = compute_signed_distances(curr_object_sdf, curr_object_sdf_centroid, \
        curr_object_sdf_extents, hand_verts_in_rest_frame[0]) # we always use bs = 1 now!!!                          
    # signed_dists: T X N_hand (120 X 1535)

    penetration_score = torch.minimum(signed_dists, torch.zeros_like(signed_dists)).abs().mean() # The smaller, the better 
    # penetration_score = torch.minimum(signed_dists, torch.zeros_like(signed_dists)).abs().sum()
    return penetration_score.detach().cpu().numpy()


def compute_penetration_metric2(version,dataset_name, object_name, ori_verts_pred, \
    pred_obj_com_pos, pred_obj_rot_mat, eval_fullbody=False):
    # ori_verts_pred: T X Nv X 3 
    # pred_obj_com_pos: T X 3
    # pred_obj_rot_mat: T X 3 X 3
    ori_verts_pred = ori_verts_pred[None] # 1 X T X Nv X 3 
    pred_obj_com_pos = pred_obj_com_pos[None] # 1 X T X 3 
    pred_obj_rot_mat = pred_obj_rot_mat[None] # 1 X T X 3 X 3 

    # if not eval_fullbody:
    #     hand_verts = ori_verts_pred[:, :, self.hand_vertex_idxs, :] # BS X T X N_hand X 3
    # else:
    hand_verts = ori_verts_pred 

    hand_verts_in_rest_frame = hand_verts - pred_obj_com_pos[:, :, None, :] # BS X T X N_hand X 3 
    
    # torch.Size([1, 96, 52, 3]) torch.Size([1, 96, 3, 3]) SS
    # print(hand_verts_in_rest_frame.shape, pred_obj_rot_mat.shape,'SS',pred_obj_rot_mat[:, :, None, :, :].repeat(1, 1, \
                        # hand_verts_in_rest_frame.shape[2], 1, 1).shape,hand_verts_in_rest_frame[:, :, :, :, None].shape)
    # hand_verts_in_rest_frame = torch.matmul(pred_obj_rot_mat[:, :, None, :, :].repeat(1, 1, \
    #                     hand_verts_in_rest_frame.shape[2], 1, 1), \
    #                     hand_verts_in_rest_frame[:, :, :, :, None]).squeeze(-1) # BS X T X N_hand X 3
    hand_verts_in_rest_frame = torch.einsum("btij,btjk->btik",hand_verts_in_rest_frame, pred_obj_rot_mat) # R^{-1} J
    # print(hand_verts_in_rest_frame.shape,'SSS') 

    curr_object_sdf, curr_object_sdf_centroid, curr_object_sdf_extents = \
    load_object_sdf_data(dataset_name,object_name,version=version)

    # Convert hand vertices to align with rest pose object. 
    signed_dists = compute_signed_distances(curr_object_sdf, curr_object_sdf_centroid, \
        curr_object_sdf_extents, hand_verts_in_rest_frame[0]) # we always use bs = 1 now!!!                          
    # signed_dists: T X N_hand (120 X 1535)

    penetration_score = torch.minimum(signed_dists, torch.zeros_like(signed_dists)).abs().mean() # The smaller, the better 
    # penetration_score = torch.minimum(signed_dists, torch.zeros_like(signed_dists)).abs().sum()
    return penetration_score.detach().cpu().numpy()





def compute_penetration_metric_torch(version,dataset_name, object_name, ori_verts_pred, \
    pred_obj_com_pos, pred_obj_rot_mat, eval_fullbody=False):
    # ori_verts_pred: T X Nv X 3 
    # pred_obj_com_pos: T X 3
    # pred_obj_rot_mat: T X 3 X 3
    ori_verts_pred = ori_verts_pred[None] # 1 X T X Nv X 3 
    pred_obj_com_pos = pred_obj_com_pos[None] # 1 X T X 3 
    pred_obj_rot_mat = pred_obj_rot_mat[None] # 1 X T X 3 X 3 

    # if not eval_fullbody:
    #     hand_verts = ori_verts_pred[:, :, self.hand_vertex_idxs, :] # BS X T X N_hand X 3
    # else:
    hand_verts = ori_verts_pred 

    hand_verts_in_rest_frame = hand_verts - pred_obj_com_pos[:, :, None, :] # BS X T X N_hand X 3 
    hand_verts_in_rest_frame = torch.matmul(pred_obj_rot_mat[:, :, None, :, :].repeat(1, 1, \
                        hand_verts_in_rest_frame.shape[2], 1, 1), \
                        hand_verts_in_rest_frame[:, :, :, :, None]).squeeze(-1) # BS X T X N_hand X 3 

    curr_object_sdf, curr_object_sdf_centroid, curr_object_sdf_extents = \
    load_object_sdf_data(dataset_name,object_name,version=version)

    # Convert hand vertices to align with rest pose object. 
    signed_dists = compute_signed_distances(curr_object_sdf, curr_object_sdf_centroid, \
        curr_object_sdf_extents, hand_verts_in_rest_frame[0]) # we always use bs = 1 now!!!                          
    # signed_dists: T X N_hand (120 X 1535)
    if (signed_dists < 0).any():
        penetration_score = (signed_dists[signed_dists < 0]**2).mean()
    else:
        penetration_score = torch.tensor(0.0, device=signed_dists.device)
    # penetration_score = torch.minimum(signed_dists, torch.zeros_like(signed_dists)).mean() # The smaller, the better 
    # penetration_score = torch.minimum(signed_dists, torch.zeros_like(signed_dists)).abs().sum()
    return penetration_score


def load_object_sdf_data( dataset_name, obj_name,version):
    # if self.test_unseen_objects:
    #     data_folder = os.path.join(self.data_root_folder, "unseen_objects_data/selected_rotated_zeroed_obj_sdf_256_npy_files")
    #     sdf_npy_path = os.path.join(data_folder, object_name+".npy")
    #     sdf_json_path = os.path.join(data_folder, object_name+".json")
    # else:
    object_name = obj_name
    
    if version  ==9:
        sdf_npy_path = os.path.join(data_folder, object_name+"_can.npy")
        sdf_json_path = os.path.join(data_folder, object_name+"_can.json")
    else:
        # sdf_npz_path = os.path.join(data_folder, object_name+".npz")
        dataset_name = dataset_name.split('_')[0]
        sdf_npz_path = f'./InterAct/{dataset_name}/objects/{object_name}/{object_name}'+'_sdf.npz'
        
        # sdf_json_path = os.path.join(data_folder, object_name+".json")
    data = np.load(sdf_npz_path,allow_pickle=True)
    sdf = data['sdf']
    sdf_centroid = data['centroid']
    sdf_extents = data['extents']

    # sdf = np.load(sdf_npy_path) # 256 X 256 X 256 
    # sdf_json_data = json.load(open(sdf_json_path, 'r'))

    # sdf_centroid = np.asarray(sdf_json_data['centroid']) # a list with 3 items -> 3 
    # sdf_extents = np.asarray(sdf_json_data['extents']) # a list with 3 items -> 3 

    sdf = torch.from_numpy(sdf).float()[None].cuda()
    sdf_centroid = torch.from_numpy(sdf_centroid).float()[None].cuda()
    sdf_extents = torch.from_numpy(sdf_extents).float()[None].cuda() 

    return sdf, sdf_centroid, sdf_extents


def compute_signed_distances(
    sdf, sdf_centroid, sdf_extents,
    query_points):
    # sdf: 1 X 256 X 256 X 256 
    # sdf_centroid: 1 X 3, center of the bounding box.  
    # sdf_extents: 1 X 3, width, height, depth of the box.  
    # query_points: T X Nv X 3 

    # query_pts_norm = (query_points - sdf_centroid[None, :, :]) * 2 / sdf_extents[None, :, :] # Convert to range [-1, 1]
    query_pts_norm = (query_points - sdf_centroid[None, :, :]) * 2 / sdf_extents.cpu().detach().numpy().max() # Convert to range [-1, 1]
     
    query_pts_norm = query_pts_norm[...,[2, 1, 0]] # Switch the order to depth, height, width
    
    num_steps, nv, _ = query_pts_norm.shape # T X Nv X 3 

    query_pts_norm = query_pts_norm[None, :, None, :, :] # 1 X T X 1 X Nv X 3 

    signed_dists = F.grid_sample(sdf[:, None, :, :, :], query_pts_norm, \
    padding_mode='border', align_corners=True)
    # F.grid_sample: N X C X D_in X H_in X W_in, N X D_out X H_out X W_out X 3, output: N X C X D_out X H_out X W_out 
    # sdf: 1 X 1 X 256 X 256 X 256, query_pts: 1 X T X 1 X Nv X 3 -> 1 X 1 X T X 1 X Nv  

    signed_dists = signed_dists[0, 0, :, 0, :] * sdf_extents.cpu().detach().numpy().max() / 2. # T X Nv 
    
    return signed_dists


def calculate_skating_ratio(motions,mask):
    thresh_height = 0.05 # 10
    fps = 30.0
    thresh_vel = 0.50 # 20 cm /s 
    avg_window = 5 # frames

    batch_size = motions.shape[0]
    # 10 left, 11 right foot. XZ plane, y up
    # motions [bs, 22, 3, max_len]
    verts_feet = motions[:, [10, 11], :, :].detach().cpu().numpy()  # [bs, 2, 3, max_len]
    verts_feet_plane_vel = np.linalg.norm(verts_feet[:, :, [0, 2], 1:] - verts_feet[:, :, [0, 2], :-1],  axis=2) * fps  # [bs, 2, max_len-1]
    # [bs, 2, max_len-1]
    vel_avg = uniform_filter1d(verts_feet_plane_vel, axis=-1, size=avg_window, mode='constant', origin=0)

    verts_feet_height = verts_feet[:, :, 1, :]  # [bs, 2, max_len]
    # If feet touch ground in agjecent frames
    feet_contact = np.logical_and((verts_feet_height[:, :, :-1] < thresh_height), (verts_feet_height[:, :, 1:] < thresh_height))  # [bs, 2, max_len - 1]
    # skate velocity
    skate_vel = feet_contact * vel_avg

    # it must both skating in the current frame
    skating = np.logical_and(feet_contact, (verts_feet_plane_vel > thresh_vel))
    # and also skate in the windows of frames
    skating = np.logical_and(skating, (vel_avg > thresh_vel))

    # Both feet slide
    skating = np.logical_or(skating[:, 0, :], skating[:, 1, :]) # [bs, max_len -1]
    skating_ratio = np.sum(skating*mask.float().squeeze(1).squeeze(1)[:,:-1].detach().cpu().numpy(), axis=1) / skating.shape[1]
    
    return skating_ratio, skate_vel

def calculate_multimodality(activation, multimodality_times):
    assert len(activation.shape) == 3
    assert activation.shape[1] > multimodality_times
    num_per_sent = activation.shape[1]

    first_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    second_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    dist = linalg.norm(activation[:, first_dices] - activation[:, second_dices], axis=2)
    return dist.mean()


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

    
def point2point_signed(
        x,
        y,
        x_normals=None,
        y_normals=None,
        return_vector=False,
):
    """
    signed distance between two pointclouds
    Args:
        x: FloatTensor of shape (N, P1, D) representing a batch of point clouds
            with P1 points in each batch element, batch size N and feature
            dimension D.
        y: FloatTensor of shape (N, P2, D) representing a batch of point clouds
            with P2 points in each batch element, batch size N and feature
            dimension D.
        x_normals: Optional FloatTensor of shape (N, P1, D).
        y_normals: Optional FloatTensor of shape (N, P2, D).
    Returns:
        - y2x_signed: Torch.Tensor
            the sign distance from y to x
        - y2x_signed: Torch.Tensor
            the sign distance from y to x
        - yidx_near: Torch.tensor
            the indices of x vertices closest to y
    """


    N, P1, D = x.shape
    P2 = y.shape[1]

    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")

    ch_dist = chd.ChamferDistance()

    x_near, y_near, xidx_near, yidx_near = ch_dist(x,y,x_normals=x_normals,y_normals=y_normals)

    xidx_near_expanded = xidx_near.view(N, P1, 1).expand(N, P1, D).to(torch.long)
    x_near = y.gather(1, xidx_near_expanded)

    yidx_near_expanded = yidx_near.view(N, P2, 1).expand(N, P2, D).to(torch.long)
    y_near = x.gather(1, yidx_near_expanded)

    x2y = x - x_near  # y point to x
    y2x = y - y_near  # x point to y

    if x_normals is not None:
        y_nn = x_normals.gather(1, yidx_near_expanded)
        in_out = torch.bmm(y_nn.view(-1, 1, 3), y2x.view(-1, 3, 1)).view(N, -1).sign()
        y2x_signed = y2x.norm(dim=2) * in_out

    else:
        y2x_signed = y2x.norm(dim=2)

    if y_normals is not None:
        x_nn = y_normals.gather(1, xidx_near_expanded)
        in_out_x = torch.bmm(x_nn.view(-1, 1, 3), x2y.view(-1, 3, 1)).view(N, -1).sign()
        x2y_signed = x2y.norm(dim=2) * in_out_x
    else:
        x2y_signed = x2y.norm(dim=2)

    if not return_vector:
        return y2x_signed, x2y_signed, yidx_near, xidx_near
    else:
        return y2x_signed, x2y_signed, yidx_near, xidx_near, y2x, x2y
def contact_metric(gt_joints,gt_obj,pred_joints,pred_obj,contact_threh = 0.05):

        # contact_threh = 0.05
    # print(torch.linalg.norm((gt_joints-pred_joints)))
    # print(torch.linalg.norm((gt_obj-pred_obj)))
    # gt_joints = gt_joints[:,:-30]
    # pred_joints = pred_joints[:,:-30]
    _,gt_dmin,_,_ = point2point_signed(gt_joints,gt_obj)
    

    gt_contact = (gt_dmin < contact_threh)
    # print(gt_contact.shape,gt_dmin.shape)
    # gt_rhand_contact = (gt_rhand2obj_dist_min < contact_threh)
    
    _,pred_dmin,_,_ = point2point_signed(pred_joints,pred_obj)
    
    pred_contact = (pred_dmin < contact_threh)

    

    num_steps = pred_contact.shape[0]
    
    gt_cnt = gt_contact.sum(dim=0).float()  # (J,)
    pred_cnt = pred_contact.sum(dim=0).float()
    
    
    gt_contact_dist_sum   = (gt_dmin  * gt_contact.float()).sum(dim=0)
    pred_contact_dist_sum = (pred_dmin * gt_contact.float()).sum(dim=0)
    # if gt_cnt[j] == 0, we define the average to be 0
    gt_avg_contact_dist   = torch.where(gt_cnt > 0,
                                        gt_contact_dist_sum  / gt_cnt,
                                        torch.zeros_like(gt_cnt))
    pred_avg_contact_dist = torch.where(gt_cnt > 0,
                                        pred_contact_dist_sum / gt_cnt,
                                        torch.zeros_like(gt_cnt))

    TP = ((gt_contact & pred_contact).sum(dim=0)).float()
    FP = ((~gt_contact & pred_contact).sum(dim=0)).float()
    TN = ((~gt_contact & ~pred_contact).sum(dim=0)).float()
    FN = ((gt_contact & ~pred_contact).sum(dim=0)).float()

    # contact percentages
    T =gt_dmin.shape[0]
    gt_contact_percent   = gt_cnt / T
    pred_contact_percent = pred_cnt / T

    # precision, recall, with safe-zero handling
    precision = torch.where(
        (TP + FP) > 0,
        TP / (TP + FP),
        torch.zeros_like(TP)
    )
    recall = torch.where(
        (TP + FN) > 0,
        TP / (TP + FN),
        torch.zeros_like(TP)
    )
    # F1 (also safe)
    f1 = torch.where(
        (precision + recall) > 0,
        2 * (precision * recall) / (precision + recall),
        torch.zeros_like(precision)
    )

    # accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    ############################################################################# STUPID FUCK
    gt_dmin= torch.min(gt_dmin[:,-30:],dim=1,keepdim=True)[0]
    

    gt_contact = (gt_dmin < contact_threh)
    # print(gt_contact.shape,gt_dmin.shape)
    # gt_rhand_contact = (gt_rhand2obj_dist_min < contact_threh)
    
    # _,pred_dmin,_,_ = point2point_signed(pred_joints,pred_obj)
    pred_dmin= torch.min(pred_dmin[:,-30:],dim=1,keepdim=True)[0]
    pred_contact = (pred_dmin < contact_threh)

    

    num_steps = pred_contact.shape[0]
    
    gt_cnt = gt_contact.sum(dim=0).float()  # (J,)
    pred_cnt = pred_contact.sum(dim=0).float()
    
    
    gt_contact_dist_sum   = (gt_dmin  * gt_contact.float()).sum(dim=0)
    pred_contact_dist_sum = (pred_dmin * gt_contact.float()).sum(dim=0)
    # if gt_cnt[j] == 0, we define the average to be 0
    gt_avg_contact_dist_hand   = torch.where(gt_cnt > 0,
                                        gt_contact_dist_sum  / gt_cnt,
                                        torch.zeros_like(gt_cnt))
    pred_avg_contact_dist_hand = torch.where(gt_cnt > 0,
                                        pred_contact_dist_sum / gt_cnt,
                                        torch.zeros_like(gt_cnt))

    TP = ((gt_contact & pred_contact).sum(dim=0)).float()
    FP = ((~gt_contact & pred_contact).sum(dim=0)).float()
    TN = ((~gt_contact & ~pred_contact).sum(dim=0)).float()
    FN = ((gt_contact & ~pred_contact).sum(dim=0)).float()

    # contact percentages
    T =gt_dmin.shape[0]
    gt_contact_percent_hand   = gt_cnt / T
    pred_contact_percent_hand = pred_cnt / T

    # precision, recall, with safe-zero handling
    precision_hand = torch.where(
        (TP + FP) > 0,
        TP / (TP + FP),
        torch.zeros_like(TP)
    )
    recall_hand = torch.where(
        (TP + FN) > 0,
        TP / (TP + FN),
        torch.zeros_like(TP)
    )
    # F1 (also safe)
    f1_hand = torch.where(
        (precision_hand + recall_hand) > 0,
        2 * (precision_hand * recall_hand) / (precision_hand + recall_hand),
        torch.zeros_like(precision_hand)
    )

    # accuracy
    accuracy_hand = (TP + TN) / (TP + TN + FP + FN)
    
    
    
    
    
    

    # 7) Finally: mean across J joints
    metrics = {
        'mean_gt_contact_percent'   : gt_contact_percent.mean().item(),
        'mean_pred_contact_percent' : pred_contact_percent.mean().item(),
        'mean_gt_contact_dist'      : gt_avg_contact_dist.mean().item(),
        'mean_pred_contact_dist'    : pred_avg_contact_dist.mean().item(),
        'mean_precision'            : precision.mean().item(),
        'mean_recall'               : recall.mean().item(),
        'mean_f1'                   : f1.mean().item(),
        'mean_accuracy'             : accuracy.mean().item(),
        'mean_gt_contact_percent_hand'   : gt_contact_percent_hand.mean().item(),
        'mean_pred_contact_percent_hand' : pred_contact_percent_hand.mean().item(),
        'mean_gt_contact_dist_hand'      : gt_avg_contact_dist_hand.mean().item(),
        'mean_pred_contact_dist_hand'    : pred_avg_contact_dist_hand.mean().item(),
        'mean_precision_hand'            : precision_hand.mean().item(),
        'mean_recall_hand'               : recall_hand.mean().item(),
        'mean_f1_hand'                   : f1_hand.mean().item(),
        'mean_accuracy_hand'             : accuracy_hand.mean().item(),
    }

    return metrics 

import torch

def contact_metric_micro(gt_joints, gt_obj, pred_joints, pred_obj, contact_threh = 0.05):
    
    _, gt_dmin, _, _ = point2point_signed(gt_joints, gt_obj)
    gt_contact = (gt_dmin < contact_threh)
    
    _, pred_dmin, _, _ = point2point_signed(pred_joints, pred_obj)
    pred_contact = (pred_dmin < contact_threh)

    num_steps = pred_contact.shape[0]
    T = gt_dmin.shape[0]
    
    gt_cnt = gt_contact.sum(dim=0).float()  # (J,)
    pred_cnt = pred_contact.sum(dim=0).float()
    
    gt_contact_dist_sum   = (gt_dmin  * gt_contact.float()).sum(dim=0)
    pred_contact_dist_sum = (pred_dmin * gt_contact.float()).sum(dim=0)
    
    # if gt_cnt[j] == 0, we define the average to be 0
    gt_avg_contact_dist   = torch.where(gt_cnt > 0,
                                        gt_contact_dist_sum  / gt_cnt,
                                        torch.zeros_like(gt_cnt))
    pred_avg_contact_dist = torch.where(gt_cnt > 0,
                                        pred_contact_dist_sum / gt_cnt,
                                        torch.zeros_like(gt_cnt))

    TP = ((gt_contact & pred_contact).sum(dim=0)).float()
    FP = ((~gt_contact & pred_contact).sum(dim=0)).float()
    TN = ((~gt_contact & ~pred_contact).sum(dim=0)).float()
    FN = ((gt_contact & ~pred_contact).sum(dim=0)).float()

    # contact percentages
    gt_contact_percent   = gt_cnt / T
    pred_contact_percent = pred_cnt / T

   
    total_TP = TP.sum()
    total_FP = FP.sum()
    total_TN = TN.sum()
    total_FN = FN.sum()

    perfect_empty = (total_TP == 0) & (total_FP == 0) & (total_FN == 0)

    micro_precision = torch.where(
        (total_TP + total_FP) > 0,
        total_TP / (total_TP + total_FP),
        torch.where(perfect_empty, torch.tensor(1.0, device=TP.device), torch.tensor(0.0, device=TP.device))
    )

    micro_recall = torch.where(
        (total_TP + total_FN) > 0,
        total_TP / (total_TP + total_FN),
        torch.where(perfect_empty, torch.tensor(1.0, device=TP.device), torch.tensor(0.0, device=TP.device))
    )

    micro_f1 = torch.where(
        (micro_precision + micro_recall) > 0,
        2 * (micro_precision * micro_recall) / (micro_precision + micro_recall),
        torch.where(perfect_empty, torch.tensor(1.0, device=TP.device), torch.tensor(0.0, device=TP.device))
    )

    micro_accuracy = (total_TP + total_TN) / (total_TP + total_TN + total_FP + total_FN)


   
    gt_dmin_hand = torch.min(gt_dmin[:, -30:], dim=1, keepdim=True)[0]
    gt_contact_hand = (gt_dmin_hand < contact_threh)
    
    pred_dmin_hand = torch.min(pred_dmin[:, -30:], dim=1, keepdim=True)[0]
    pred_contact_hand = (pred_dmin_hand < contact_threh)

    gt_cnt_hand = gt_contact_hand.sum(dim=0).float()
    pred_cnt_hand = pred_contact_hand.sum(dim=0).float()

    gt_contact_dist_sum_hand   = (gt_dmin_hand  * gt_contact_hand.float()).sum(dim=0)
    pred_contact_dist_sum_hand = (pred_dmin_hand * gt_contact_hand.float()).sum(dim=0)
    
    gt_avg_contact_dist_hand   = torch.where(gt_cnt_hand > 0,
                                        gt_contact_dist_sum_hand  / gt_cnt_hand,
                                        torch.zeros_like(gt_cnt_hand))
    pred_avg_contact_dist_hand = torch.where(gt_cnt_hand > 0,
                                        pred_contact_dist_sum_hand / gt_cnt_hand,
                                        torch.zeros_like(gt_cnt_hand))

    TP_hand = ((gt_contact_hand & pred_contact_hand).sum(dim=0)).float()
    FP_hand = ((~gt_contact_hand & pred_contact_hand).sum(dim=0)).float()
    TN_hand = ((~gt_contact_hand & ~pred_contact_hand).sum(dim=0)).float()
    FN_hand = ((gt_contact_hand & ~pred_contact_hand).sum(dim=0)).float()

    gt_contact_percent_hand   = gt_cnt_hand / T
    pred_contact_percent_hand = pred_cnt_hand / T

   
    total_TP_hand = TP_hand.sum()
    total_FP_hand = FP_hand.sum()
    total_TN_hand = TN_hand.sum()
    total_FN_hand = FN_hand.sum()

    perfect_empty_hand = (total_TP_hand == 0) & (total_FP_hand == 0) & (total_FN_hand == 0)

    micro_precision_hand = torch.where(
        (total_TP_hand + total_FP_hand) > 0,
        total_TP_hand / (total_TP_hand + total_FP_hand),
        torch.where(perfect_empty_hand, torch.tensor(1.0, device=TP_hand.device), torch.tensor(0.0, device=TP_hand.device))
    )

    micro_recall_hand = torch.where(
        (total_TP_hand + total_FN_hand) > 0,
        total_TP_hand / (total_TP_hand + total_FN_hand),
        torch.where(perfect_empty_hand, torch.tensor(1.0, device=TP_hand.device), torch.tensor(0.0, device=TP_hand.device))
    )

    micro_f1_hand = torch.where(
        (micro_precision_hand + micro_recall_hand) > 0,
        2 * (micro_precision_hand * micro_recall_hand) / (micro_precision_hand + micro_recall_hand),
        torch.where(perfect_empty_hand, torch.tensor(1.0, device=TP_hand.device), torch.tensor(0.0, device=TP_hand.device))
    )

    micro_accuracy_hand = (total_TP_hand + total_TN_hand) / (total_TP_hand + total_TN_hand + total_FP_hand + total_FN_hand)

    metrics = {
        'mean_gt_contact_percent'        : gt_contact_percent.mean().item(),
        'mean_pred_contact_percent'      : pred_contact_percent.mean().item(),
        'mean_gt_contact_dist'           : gt_avg_contact_dist.mean().item(),
        'mean_pred_contact_dist'         : pred_avg_contact_dist.mean().item(),
        
        # 全局 Micro 指标
        'micro_precision'                : micro_precision.item(),
        'micro_recall'                   : micro_recall.item(),
        'micro_f1'                       : micro_f1.item(),
        'micro_accuracy'                 : micro_accuracy.item(),
        
        'mean_gt_contact_percent_hand'   : gt_contact_percent_hand.mean().item(),
        'mean_pred_contact_percent_hand' : pred_contact_percent_hand.mean().item(),
        'mean_gt_contact_dist_hand'      : gt_avg_contact_dist_hand.mean().item(),
        'mean_pred_contact_dist_hand'    : pred_avg_contact_dist_hand.mean().item(),
        
        # 手部 Micro 指标
        'micro_precision_hand'           : micro_precision_hand.item(),
        'micro_recall_hand'              : micro_recall_hand.item(),
        'micro_f1_hand'                  : micro_f1_hand.item(),
        'micro_accuracy_hand'            : micro_accuracy_hand.item(),
    }
    
    return metrics
# def contact_metric(gt_joints,gt_obj,pred_joints,pred_obj,contact_threh = 0.05):

#         # contact_threh = 0.05
#     _,gt_dmin,_,_ = point2point_signed(gt_joints,gt_obj)
    

#     gt_contact = (gt_dmin < contact_threh)
#     # gt_rhand_contact = (gt_rhand2obj_dist_min < contact_threh)
    
#     _,pred_dmin,_,_ = point2point_signed(pred_joints,pred_obj)
    
#     pred_contact = (pred_dmin < contact_threh)

    

#     num_steps = pred_contact.shape[0]
    
#     gt_cnt = gt_contact.sum(dim=0).float()  # (J,)
#     pred_cnt = pred_contact.sum(dim=0).float()
    
    
#     gt_contact_dist_sum   = (gt_dmin  * gt_contact.float()).sum(dim=0)
#     pred_contact_dist_sum = (pred_dmin * gt_contact.float()).sum(dim=0)
#     # if gt_cnt[j] == 0, we define the average to be 0
#     gt_avg_contact_dist   = torch.where(gt_cnt > 0,
#                                         gt_contact_dist_sum  / gt_cnt,
#                                         torch.zeros_like(gt_cnt))
#     pred_avg_contact_dist = torch.where(gt_cnt > 0,
#                                         pred_contact_dist_sum / gt_cnt,
#                                         torch.zeros_like(gt_cnt))

#     TP = ((gt_contact & pred_contact).sum(dim=0)).float()
#     FP = ((~gt_contact & pred_contact).sum(dim=0)).float()
#     TN = ((~gt_contact & ~pred_contact).sum(dim=0)).float()
#     FN = ((gt_contact & ~pred_contact).sum(dim=0)).float()

#     # contact percentages
#     T =gt_dmin.shape[0]
#     gt_contact_percent   = gt_cnt / T
#     pred_contact_percent = pred_cnt / T

#     # precision, recall, with safe-zero handling
#     precision = torch.where(
#         (TP + FP) > 0,
#         TP / (TP + FP),
#         torch.zeros_like(TP)
#     )
#     recall = torch.where(
#         (TP + FN) > 0,
#         TP / (TP + FN),
#         torch.zeros_like(TP)
#     )
#     # F1 (also safe)
#     f1 = torch.where(
#         (precision + recall) > 0,
#         2 * (precision * recall) / (precision + recall),
#         torch.zeros_like(precision)
#     )

#     # accuracy
#     accuracy = (TP + TN) / (TP + TN + FP + FN)

#     # 7) Finally: mean across J joints
#     metrics = {
#         'mean_gt_contact_percent'   : gt_contact_percent.mean().item(),
#         'mean_pred_contact_percent' : pred_contact_percent.mean().item(),
#         'mean_gt_contact_dist'      : gt_avg_contact_dist.mean().item(),
#         'mean_pred_contact_dist'    : pred_avg_contact_dist.mean().item(),
#         'mean_precision'            : precision.mean().item(),
#         'mean_recall'               : recall.mean().item(),
#         'mean_f1'                   : f1.mean().item(),
#         'mean_accuracy'             : accuracy.mean().item(),
#     }

#     return metrics  

def vertex_normals(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of vertices, 3]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    normals = torch.zeros(bs * nv, 3).to(device)

    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]  # expanded faces
    vertices_faces = vertices.reshape((bs * nv, 3))[faces.long()]

    faces = faces.view(-1, 3)
    vertices_faces = vertices_faces.view(-1, 3, 3)

    normals.index_add_(0, faces[:, 1].long(),
                       torch.cross(vertices_faces[:, 2] - vertices_faces[:, 1],
                                   vertices_faces[:, 0] - vertices_faces[:, 1]))
    normals.index_add_(0, faces[:, 2].long(),
                       torch.cross(vertices_faces[:, 0] - vertices_faces[:, 2],
                                   vertices_faces[:, 1] - vertices_faces[:, 2]))
    normals.index_add_(0, faces[:, 0].long(),
                       torch.cross(vertices_faces[:, 1] - vertices_faces[:, 0],
                                   vertices_faces[:, 2] - vertices_faces[:, 0]))

    normals = F.normalize(normals, eps=1e-6, dim=1)
    normals = normals.reshape((bs, nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return normals  