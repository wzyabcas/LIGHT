from utils.nn import mean_flat, sum_flat
import torch
import numpy as np
import torch.nn.functional as F


def huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0):
    """
    Elementwise Huber loss for tensors shaped [B, D, 1, T].
    Returns same shape as input (no reduction).
    """
    err = pred - target                  # [B, D, 1, T]
    abs_err = torch.abs(err)
    quadratic = 0.5 * err**2
    linear = delta * (abs_err - 0.5 * delta)
    return torch.where(abs_err < delta, quadratic, linear)

def huber_loss2(model_pred, target, delta = 1.0):
    """
    Elementwise Huber loss for tensors shaped [B, D, 1, T].
    Returns same shape as input (no reduction).
    """
    # print('ABCD')
    # print(delta.shape,'KSKS')
    return 2*delta * (torch.sqrt((model_pred - target) ** 2 + delta**2) - delta)
                        

def angle_l2(angle1, angle2):
    a = angle1 - angle2
    a = (a + (torch.pi/2)) % torch.pi - (torch.pi/2)
    return a ** 2

def diff_l1(a, b,delta=0):
    return abs(a - b)

def diff_l2(a, b,delta=0):
    return (a - b) ** 2

def masked_l2(a, b, mask, loss_fn=diff_l2, epsilon=1e-4, entries_norm=True,w=1.0,delta=0):
    # assuming a.shape == b.shape == bs, J, Jdim, seqlen
    # assuming mask.shape == bs, 1, 1, seqlen
    loss = loss_fn(a, b,delta)
    loss = sum_flat(loss * w* mask.to(loss.dtype))  # gives \sigma_euclidean over unmasked elements
    n_entries = a.shape[1]
    if len(a.shape) > 3:
        n_entries *= a.shape[2]
    non_zero_elements = sum_flat(mask)
    if entries_norm:
        # In cases the mask is per frame, and not specifying the number of entries per frame, this normalization is needed,
        # Otherwise set it to False
        non_zero_elements *= n_entries
    # print('mask', mask.shape)
    # print('non_zero_elements', non_zero_elements)
    # print('loss', loss)
    mse_loss_val = loss / (non_zero_elements + epsilon)  # Add epsilon to avoid division by zero
    # print('mse_loss_val', mse_loss_val)
    return mse_loss_val


def masked_goal_l2(pred_goal, ref_goal, cond, all_goal_joint_names):
    all_goal_joint_names_w_traj = np.append(all_goal_joint_names, 'traj')
    target_joint_idx = [[np.where(all_goal_joint_names_w_traj == j)[0][0] for j in sample_joints] for sample_joints in cond['target_joint_names']]
    loc_mask = torch.zeros_like(pred_goal[:,:-1], dtype=torch.bool)
    for sample_idx in range(loc_mask.shape[0]):
        loc_mask[sample_idx, target_joint_idx[sample_idx]] = True
    loc_mask[:, -1, 1] = False  # vertical joint of 'traj' is always masked out
    loc_loss = masked_l2(pred_goal[:,:-1], ref_goal[:,:-1], loc_mask, entries_norm=False)
    
    heading_loss = masked_l2(pred_goal[:,-1:, :1], ref_goal[:,-1:, :1], cond['is_heading'].unsqueeze(1).unsqueeze(1), loss_fn=angle_l2, entries_norm=False)

    loss =  loc_loss + heading_loss
    return loss
