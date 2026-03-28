import torch

def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask
    

def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b['inp'] for b in notnone_batches]
    if 'lengths' in notnone_batches[0]:
        lenbatch = [b['lengths'] for b in notnone_batches]
    else:
        lenbatch = [len(b['inp'][0][0]) for b in notnone_batches]


    databatchTensor = collate_tensors(databatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    if 'pad_lengths' in notnone_batches[0]: ## prefix_pad
        pad_lenbatchTensor = torch.as_tensor([b['pad_lengths'] for b in notnone_batches])
        maskbatchTensor = lengths_to_mask(lenbatchTensor, databatchTensor.shape[-1]).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting
        pad_maskbatchTensor = ~lengths_to_mask(pad_lenbatchTensor, databatchTensor.shape[-1]).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting
        maskbatchTensor = maskbatchTensor & pad_maskbatchTensor
    else:
        maskbatchTensor = lengths_to_mask(lenbatchTensor, databatchTensor.shape[-1]).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting

    motion = databatchTensor
    cond = {'y': {'mask': maskbatchTensor, 'lengths': lenbatchTensor}}

    if 'text' in notnone_batches[0]:
        textbatch = [b['text'] for b in notnone_batches]
        cond['y'].update({'text': textbatch})
    cond['y'].update({'human_gt':databatchTensor[:,:52*3]})
    cond['y'].update({'obj_gt':databatchTensor[:,52*3:]})
    if 'obj_bps' in notnone_batches[0]:
        cond['y'].update({'obj_bps': collate_tensors([b['obj_bps'] for b in notnone_batches])})
    if 'beta' in notnone_batches[0]:
        cond['y'].update({'beta': collate_tensors([b['beta'] for b in notnone_batches])})
    # if 'human_marker' in notnone_batches[0]:
    #     cond['y'].update({'human_marker': collate_tensors([b['human_marker'] for b in notnone_batches])})
    # if 'text_embed' in notnone_batches[0]:
    #     cond['y'].update({'text_embed': collate_tensors([b['text_embed'] for b in notnone_batches]).unsqueeze(0).float()})
    # if 'obj_marker' in notnone_batches[0]:
    #     cond['y'].update({'obj_marker': collate_tensors([b['obj_marker'] for b in notnone_batches])})
    if 'obj_points' in notnone_batches[0]:
        cond['y'].update({'obj_points': collate_tensors([b['obj_points'] for b in notnone_batches])})
    # if 'obj_points_1024' in notnone_batches[0]:
    #     cond['y'].update({'obj_points_1024': collate_tensors([b['obj_points_1024'] for b in notnone_batches])})
    # if 'obj_feat_pc' in notnone_batches[0]:
    #     cond['y'].update({'obj_feat_pc': collate_tensors([b['obj_feat_pc'] for b in notnone_batches])})
    # if 'obj_points_can_1024' in notnone_batches[0]:
    #     cond['y'].update({'obj_points_can_1024': collate_tensors([b['obj_points_can_1024'] for b in notnone_batches])})
    # if 'bone_length' in notnone_batches[0]:
    #     cond['y'].update({'bone_length': collate_tensors([b['bone_length'] for b in notnone_batches])})
    
    
    
    # if 'markers' in notnone_batches[0]:
    #     cond['y'].update({'markers': collate_tensors([b['markers'] for b in notnone_batches])})
    if 'relative_ids' in notnone_batches[0]:
        cond['y'].update({'relative_ids': collate_tensors([b['relative_ids'] for b in notnone_batches])})
    # if 'bb' in notnone_batches[0]:
    #     cond['y'].update({'bb': collate_tensors([b['bb'] for b in notnone_batches])})
    # if 'bb_can' in notnone_batches[0]:
    #     cond['y'].update({'bb_can': collate_tensors([b['bb_can'] for b in notnone_batches])})
    if 'contact_weight' in notnone_batches[0]:
        cond['y'].update({'contact_weight': collate_tensors([b['contact_weight'] for b in notnone_batches])})
    if 'contact_label' in notnone_batches[0]:
        cond['y'].update({'contact_label': collate_tensors([b['contact_label'] for b in notnone_batches])})
    if 'foot_contact' in notnone_batches[0]:
        cond['y'].update({'foot_contact': collate_tensors([b['foot_contact'] for b in notnone_batches])})
    # if 'full_motion' in notnone_batches[0]:
    #     cond['y'].update({'full_motion': collate_tensors([b['full_motion'] for b in notnone_batches])})
    
    
    # if 'full_points' in notnone_batches[0]:
    #     cond['y'].update({'full_points': collate_tensors([b['full_points'] for b in notnone_batches])})
        
    if 'seq_name' in notnone_batches[0]:
        seqname_batch = [b['seq_name'] for b in notnone_batches]
        cond['y'].update({'seq_name': seqname_batch})
    
    if 'tokens' in notnone_batches[0]:
        textbatch = [b['tokens'] for b in notnone_batches]
        cond['y'].update({'tokens': textbatch})

    if 'action' in notnone_batches[0]:
        actionbatch = [b['action'] for b in notnone_batches]
        cond['y'].update({'action': torch.as_tensor(actionbatch).unsqueeze(1)})

    # collate action textual names
    if 'action_text' in notnone_batches[0]:
        action_text = [b['action_text']for b in notnone_batches]
        cond['y'].update({'action_text': action_text})
    
    if 'prefix' in notnone_batches[0]:
        cond['y'].update({'prefix': collate_tensors([b['prefix'] for b in notnone_batches])})
    
    if 'orig_lengths' in notnone_batches[0]:
        cond['y'].update({'orig_lengths': torch.as_tensor([b['orig_lengths'] for b in notnone_batches])})

    if 'key' in notnone_batches[0]:
        cond['y'].update({'db_key': [b['key'] for b in notnone_batches]})

    return motion, cond

# an adapter to our collate func
def t2m_collate(batch, target_batch_size):
    repeat_factor = -(-target_batch_size // len(batch))  # Ceiling division
    repeated_batch = batch * repeat_factor 
    full_batch = repeated_batch[:target_batch_size]  # Truncate to the target batch size
    # batch.sort(key=lambda x: x[3], reverse=True)
    adapted_batch = [{
        'inp': torch.tensor(b[4].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
        'text': b[2], #b[0]['caption']
        'tokens': b[6],
        'lengths': b[5],
        'key': b[7] if len(b) > 7 else None,
    } for b in full_batch]
    return collate(adapted_batch)


def t2m_prefix_collate(batch, pred_len):
    # batch.sort(key=lambda x: x[3], reverse=True)
    adapted_batch = [{
        'inp': torch.tensor(b[4].T).float().unsqueeze(1)[..., -pred_len:], # [seqlen, J] -> [J, 1, seqlen]
        'prefix': torch.tensor(b[4].T).float().unsqueeze(1)[..., :-pred_len],
        'text': b[2], #b[0]['caption']
        'tokens': b[6],
        'lengths': pred_len,  # b[5],
        'orig_lengths': b[5][0], #  For evaluation
        'key': b[7] if len(b) > 7 else None,
    } for b in batch]
    return collate(adapted_batch)

def t2hoi_prefix_collate(batch, pred_len):
    # batch.sort(key=lambda x: x[3], reverse=True)
    adapted_batch = [{
        'inp': torch.tensor(b[1].T).float().unsqueeze(1)[..., -pred_len:], # [seqlen, J] -> [J, 1, seqlen]
        'prefix': torch.tensor(b[1].T).float().unsqueeze(1)[..., :-pred_len],
        'text': b[0], #b[0]['caption']
        'lengths': pred_len,  # b[5],
        'orig_lengths': b[2][0], #  For evaluation
        'seq_name':b[4],
        'obj_points':torch.tensor(b[5]).float(),
        'obj_bps':torch.tensor(b[3]).float(),
        'key': b[7] if len(b) > 7 else None,
    } for b in batch]
    return collate(adapted_batch)

def t2hoi_padprefix_collate(batch, pred_len):
    # batch.sort(key=lambda x: x[3], reverse=True)
    adapted_batch = [{
        'inp': torch.tensor(b[1].T).float().unsqueeze(1)[..., -pred_len:], # [seqlen, J] -> [J, 1, seqlen]
        'prefix': torch.tensor(b[1].T).float().unsqueeze(1)[..., :-pred_len],
        'text': b[0], #b[0]['caption']
        'lengths': pred_len,  # b[5],
        'orig_lengths': b[2][0], #  For evaluation
        'pad_lengths': b[2][2] if len(b[2]==3) else 0,
        'seq_name':b[4],
        'obj_points':torch.tensor(b[5]).float(),
        'obj_bps':torch.tensor(b[3]).float(),
        'key': b[7] if len(b) > 7 else None,
    } for b in batch]
    return collate(adapted_batch)

# def t2hoi_collate_orig(batch, target_batch_size):
#     repeat_factor = -(-target_batch_size // len(batch))  # Ceiling division
#     repeated_batch = batch * repeat_factor 
#     full_batch = repeated_batch[:target_batch_size]  # Truncate to the target batch size
#     # batch.sort(key=lambda x: x[3], reverse=True)
#     adapted_batch = [{
#         'inp': torch.tensor(b[1].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
#         'text': b[0], #b[0]['caption']
#         'lengths': b[2],
#         'seq_name':b[4],
#         'obj_points':torch.tensor(b[5]).float(),
#         'obj_bps':torch.tensor(b[3]).float(),
#         'markers': torch.tensor(b[7].T).float().unsqueeze(1) if len(b) > 7 else torch.zeros((300,1)),
#         'relative_ids': torch.tensor(b[8]).long() if len(b) > 8 else torch.zeros((300,1)),
#         'beta': torch.tensor(b[6]).float() if len(b) > 6 else torch.zeros((13)),
#         'human_marker': torch.tensor(b[9]).float() if len(b) > 9 else torch.zeros((109*3)),
#         'obj_marker': torch.tensor(b[10]).float() if len(b) > 10 else torch.zeros((72)),
#         'text_embed': torch.tensor(b[11]).float() if len(b) > 11 else torch.zeros((1)),
#         'contact_weight': torch.tensor(b[12]).float() if len(b) > 12 else torch.zeros((300,52)),
#         'contact_label': torch.tensor(b[13]).float() if len(b) > 13 else torch.zeros((300,52)),
#         'foot_contact': torch.tensor(b[14]).float() if len(b) > 14 else torch.zeros((300,4)),
#         'bb': torch.tensor(b[15]).float() if len(b) > 15 else torch.zeros((6)),
#         'bb_can':torch.tensor(b[16]).float() if len(b) > 16 else torch.zeros((6)),
#         'bone_length':torch.tensor(b[17]).float() if len(b) > 17 else torch.zeros((15)),
#         'obj_points_1024':torch.tensor(b[18]).float() if len(b) > 18 else torch.zeros((1024,3)),
#         'obj_points_can_1024':torch.tensor(b[19]).float() if len(b) > 19 else torch.zeros((1024,3)),
#         'obj_feat_pc':torch.tensor(b[20]).float() if len(b) > 20 else torch.zeros((2052)),
#         'full_motion': torch.tensor(b[21].T).float().unsqueeze(1) if len(b) > 21 else torch.zeros((1)),
#         'full_points':torch.tensor(b[22]).float() if len(b) > 22 else torch.zeros((1))
#     } for b in full_batch]
#     return collate(adapted_batch)


def t2hoi_collate(batch, target_batch_size):
    repeat_factor = -(-target_batch_size // len(batch))  # Ceiling division
    repeated_batch = batch * repeat_factor 
    full_batch = repeated_batch[:target_batch_size]  # Truncate to the target batch size
    # batch.sort(key=lambda x: x[3], reverse=True)
    adapted_batch = [{
        'inp': torch.tensor(b[1].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
        'text': b[0], #b[0]['caption']
        'lengths': b[2],
        'seq_name':b[4],
        'obj_points':torch.tensor(b[5]).float(),
        'obj_bps':torch.tensor(b[3]).float(),
        'relative_ids': torch.tensor(b[7]).long() if len(b) > 7 else torch.zeros((300,1)),
        'beta': torch.tensor(b[6]).float() if len(b) > 6 else torch.zeros((13)),
        'contact_weight': torch.tensor(b[8]).float() if len(b) > 8 else torch.zeros((300,52)),
        'contact_label': torch.tensor(b[9]).float() if len(b) > 9 else torch.zeros((300,52)),
        'foot_contact': torch.tensor(b[10]).float() if len(b) > 10 else torch.zeros((300,4)),
    } for b in full_batch]
    return collate(adapted_batch)


# return caption, motion, length,obj_bps,seq_name, obj_points

# return caption, motion, m_length,obj_bps,seq_name
# return word_embeddings, pos_one_hots, caption, sent_len, motion, length, '_'.join(tokens)


