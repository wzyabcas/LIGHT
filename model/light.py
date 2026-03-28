import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import sys

sys.path.append('.')
from model.rope_decoder import FullMultimodalDecoder
from model.BERT.BERT_encoder import load_bert

from pytorch3d.transforms import *
from model.ffn import FFN


def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask

class LIGHT(nn.Module):
    def __init__(self, njoints, nfeats, translation,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                  activation="gelu",  data_rep='rot6d',
                 arch='trans_dec', foot=0,cond_mask_uniform=0,**kargs):
        
        
        super().__init__()
        # print(use_bone,'USE_BONE')
        # self.rope=rope
        # self.learnable_pe = learnable_pe
        # self.use_obj_feat=use_obj_feat
        # # print(self.use_obj_feat,'USE_OBJ_FEAT')
        # self.dit_split= dit_split
        # self.use_hand_scalar_rot= use_hand_scalar_rot
        # self.pointnet = pointnet
        self.cond_mask_uniform = cond_mask_uniform
        # self.adaln_t = adaln_t
        # self.repre_bb = repre_bb
        # self.use_bone= use_bone
        self.use_dec=1
        
        # self.hand_split = hand_split
        # self.only_marker = only_marker
        # self.dynamic_bps = dynamic_bps
        # self.dynamic_marker = dynamic_marker
        # self.use_bb = use_bb
        # self.task = task
        # self.ome = ome
        # self.clean=clean
        # self.bps_mode=bps_mode
        # self.marker_mode=marker_mode
        # self.cid_num=cid_num
        self.foot = foot
        # self.cid = cid
        # self.ho_pe = ho_pe
        # self.beta_mode = beta_mode
        # self.cat = cat
        # self.dit = dit
        # self.double_dit = self.dit in ['dit2','dit3']
        # self.use_rot = use_rot
        # self.norm_first = norm_first
        # self.joint_nums = joint_nums
        # self.embed_shape = embed_shape
        # self.split_t = split_t
        # self.zero_init = zero_init
        # self.use_mask = use_mask
        # self.unet_config = unet_config
        # self.use_bps = use_bps
        # self.use_joint = use_joint
        # self.use_marker = use_marker
        # self.st_gcn = st_gcn
        # self.original_df = original_df
        # self.split_condition = split_condition
        # self.use_gd = use_gd
        # self.use_beta = use_beta
        # self.st_attn_fuse = st_attn_fuse
        # self.st_dim_repre = st_dim_repre
        # self.st_att=st_att
        # self.st_depth = st_depth
        # self.local_window = local_window
        # self.split_hand = split_hand
        # self.add_dec = add_dec
        # self.split_ho_emb=split_ho_emb
        # self.split_pe=split_pe
        # self.repre = repre
        # self.process_v1 = process_v
        # self.predict_prefix = predict_prefix
        # print(repre,predict_prefix,'REPRE')
        # self.legacy = legacy
        # self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        # self.num_actions = num_actions
        self.data_rep = data_rep
        # self.dataset = dataset

        # self.pose_rep = pose_rep
        # self.glob = glob
        # self.glob_rot = glob_rot
        self.translation = translation

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        # self.ablation = ablation
        self.activation = activation
        # self.clip_dim = clip_dim
        # self.action_emb = kargs.get('action_emb', None)
        self.input_feats = self.njoints * self.nfeats

        # self.normalize_output = kargs.get('normalize_encoder_output', False)
        
        self.cond_mode = kargs.get('cond_mode', 'no_cond')
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.mask_frames = kargs.get('mask_frames', False)
        self.arch = arch
        # self.gru_emb_dim = self.latent_dim if self.arch == 'gru' else 0
        # self.emb_trans_dec = emb_trans_dec
        
        # if self.use_rot:
        #     print('UR')
        #     self.body_index = list(range(22*3))+list(range(52*3,52*3+3+22*6))+list(range(52*3+3+52*6,52*3+3+52*6+self.foot))
        #     self.hand_index = list(range(22*3,52*3))+list(range(52*3+3+22*6,52*3+3+52*6))
        # else:
        #     print('UR2')
        self.body_index = list(range(22*3))+list(range(52*3,52*3+self.foot))
        self.hand_index = list(range(22*3,52*3))
    
        self.beta_embedding = nn.Linear(in_features=13, out_features=self.latent_dim)  
     
        self.human_beta = nn.Linear(2*self.latent_dim,self.latent_dim)
        self.obj_beta = nn.Linear(2*self.latent_dim,self.latent_dim)
        self.hand_beta = nn.Linear(2*self.latent_dim,self.latent_dim)
                
        self.emb_policy = kargs.get('emb_policy', 'add')
        

        self.number_per_frames = 3
        self.sequence_pos_encoder = Split_PositionalEncoding(self.latent_dim, self.dropout, max_len=kargs.get('pos_embed_max_len', 1000),number_per_frames=3,emb_first=0)
        self.sequence_pos_encoder_hoi = HOIEncoding(self.latent_dim, self.dropout, max_len=kargs.get('pos_embed_max_len', 5000),number_per_frames=3,emb_first=0)

            
        self.pred_len = kargs.get('pred_len', 0)
        self.context_len = kargs.get('context_len', 0)
        self.total_len = self.pred_len + self.context_len
        self.is_prefix_comp = self.total_len > 0
        self.all_goal_joint_names = kargs.get('all_goal_joint_names', [])
        
        self.multi_target_cond = kargs.get('multi_target_cond', False)
        self.multi_encoder_type = kargs.get('multi_encoder_type', 'multi')
        self.target_enc_layers = kargs.get('target_enc_layers', 1)
        bps_dim = 1024*3+1 + 1024*3
        self.bps_encoder = nn.Sequential(
                nn.Linear(in_features=bps_dim, out_features=512),
                nn.ReLU(),
                nn.Linear(in_features=512, out_features=self.latent_dim),
                )
        self.seqTransDecoder = FullMultimodalDecoder(num_layers=self.num_layers, d_model=self.latent_dim, n_head=self.num_heads, cond_dim=self.latent_dim, num_modalities=self.number_per_frames,ff_size = self.ff_size,dropout = self.dropout)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        self.t_adaln = FFN(self.latent_dim, self.ff_size, self.dropout)
        
        self.text_encoder_type = 'bert'
        self.use_feature = False
        print("Loading BERT...")
        bert_model_path = 'distilbert/distilbert-base-uncased'
        self.clip_model = load_bert(bert_model_path)  # Sorry for that, the naming is for backward compatibility
        self.encode_text = self.bert_encode_text
        self.clip_dim = 768
        self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
        self.numbera = 52
        rot_num = 30
        rot_num_body = 0
        rot_num_hand = 30 
        self.rot_num = rot_num
        obj_dim = 9
        cid_num = 0
        
        
        self.input_process = InputProcess(self.data_rep, 3*22+self.foot+rot_num_body, self.latent_dim)
        self.input_process_hand = InputProcess(self.data_rep, 3*30+rot_num_hand, self.latent_dim)
        self.input_process_obj = InputProcess(self.data_rep, obj_dim, self.latent_dim)
        self.output_process = OutputProcess(self.data_rep, 3*22+self.foot+rot_num_body, self.latent_dim, 3*22+self.foot+rot_num_body,
                                            self.nfeats)
        self.output_process_hand = OutputProcess(self.data_rep, 3*30+rot_num_hand, self.latent_dim, 3*30+rot_num_hand,
                                            self.nfeats)
        self.output_process_obj = OutputProcess(self.data_rep, obj_dim , self.latent_dim, obj_dim,
                                            self.nfeats)
            
        self._zero_inits([self.output_process,self.output_process_obj,self.output_process_hand])
        self._xavier_inits([self.input_process,self.input_process_obj,self.input_process_hand])
        
        

                


    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]
    def tma_encode_text(self, raw_text):
    
        device = next(self.parameters()).device
        
        return self.clip_model(raw_text).loc.unsqueeze(0)
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
    def _basic_init(self,module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight,0.02)
            # torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    def _zero_init(self,module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight,0.02)
            # torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    def _zero_inits(self,modules):
        for block in modules:
            self._zero_init(block)
    def _xavier_init(self,module):
        if isinstance(module, nn.Linear):
            # nn.init.normal_(module.weight,0.02)
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    def _xavier_inits(self,modules):
        for block in modules:
            self._xavier_init(block)
    
    def encode_obj(self, obj_bps):
      
        if len(obj_bps.shape) ==2:
            obj_points = obj_bps.view(obj_bps.shape[0], -1)
            obj_emb = self.bps_encoder(obj_points) # [bs, d]
            return obj_emb.unsqueeze(0)
        
        elif obj_bps.shape[1] !=1:
            obj_points = obj_bps.reshape(obj_bps.shape[0],obj_bps.shape[1] ,-1)
            obj_emb = self.bps_encoder(obj_points) # [bs, d]
            return obj_emb
        
        else:
            obj_points = obj_bps.view(obj_bps.shape[0], -1)
            obj_emb = self.bps_encoder(obj_points) # [bs, d]
            return obj_emb.unsqueeze(0)
        
    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16

        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def mask_cond(self, cond, force_mask=False,modes=None):
        seq_len, bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            if modes is None or not self.cond_mask_uniform:
                # print('SKT')
                mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(1, bs, 1)  # 1-> use null_cond, 0-> use real cond
                return cond * (1. - mask)
            else:
                mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(1, bs, 1)  # 1-> use null_cond, 0-> use real cond
                mask2 = (1-modes).reshape(1,bs,1)
                mask_final = (mask *mask2).to(cond.dtype)
        
                return cond * (1. - mask)
                
        else:
            # print('K',self.cond_mask_prob)
            return cond
    def set_mean_std_rt(self,mean_rt,std_rt):
        self.mean_rt = mean_rt
        self.std_rt = std_rt

    def clip_encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = next(self.parameters()).device
        max_text_len = 20 if self.dataset in ['humanml', 'kit'] else None  # Specific hardcoding for humanml dataset
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2 # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate
            # print('texts', texts.shape)
            zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
            texts = torch.cat([texts, zero_pad], dim=1)
            # print('texts after pad', texts.shape, texts)
        else:
            texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate
        return self.clip_model.encode_text(texts).float().unsqueeze(0)
    
    def bert_encode_text(self, raw_text):
        enc_text, mask = self.clip_model(raw_text)  # self.clip_model.get_last_hidden_state(raw_text, return_mask=True)  # mask: False means no token there
        enc_text = enc_text.permute(1, 0, 2)
        return enc_text, ~mask

    def forward(self, x, timesteps, y=None, is_causal=None,x2=None,x3=None,modes=None):
       
            
        bs, njoints, nfeats, nframes = x.shape
        time_emb = self.embed_timestep(timesteps.reshape(-1)).reshape(1,bs,-1,self.latent_dim)

        force_mask = y.get('uncond', False)
        if 'text' in self.cond_mode:
            enc_text = self.encode_text(y['text'])
            if type(enc_text) == tuple:
                enc_text, text_mask = enc_text
                if text_mask.shape[0] == 1 and bs > 1:  #
                    text_mask = torch.repeat_interleave(text_mask, bs, dim=0)
            text_emb = self.embed_text(self.mask_cond(enc_text, force_mask=force_mask,modes=modes))  # 
            obj_emb = self.encode_obj(y['obj_bps'])
            human_emb = self.beta_embedding(y['beta']).reshape(1,bs,-1)
            beta_emb = human_emb

            obj_emb = torch.cat([obj_emb,beta_emb],0)
            emb = text_emb
            
        
        x_human = self.input_process(x) # T,B,F
        x_obj = self.input_process_obj(x2)
        x_hand = self.input_process_hand(x3)
        x = torch.stack((x_human, x_obj,x_hand), dim=1).reshape(-1,x_obj.shape[1],x_obj.shape[2])
        Lmax = x.shape[0]
            
        frames_mask = None
        is_valid_mask = y['mask'].shape[-1] > 1  
        if self.mask_frames and is_valid_mask:
            frame_mask = lengths_to_mask(y['lengths']*self.number_per_frames, nframes*self.number_per_frames).unsqueeze(1).unsqueeze(1)
            frames_mask = torch.logical_not(frame_mask[..., :Lmax].squeeze(1).squeeze(1)).to(device=x.device)

        xseq = self.sequence_pos_encoder_hoi(x)

        te = time_emb[0]#.permute(1,0,2)
        emb_list = self.seqTransDecoder.merge_te_w_condition(te,obj_emb)
        output = self.seqTransDecoder(xseq.permute(1,0,2), emb.permute(1,0,2), emb_list, frames_mask,text_mask).permute(1,0,2)
        
        output = output.view(nframes,3,bs,self.latent_dim)
        output_human = self.output_process(output[:,0])
        output_obj = self.output_process_obj(output[:,1])
        output_hand = self.output_process_hand(output[:,2])
        output = torch.cat([output_human,output_obj,output_hand],1)
        return output
    
class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, dropout=0.1, height=312, width=80):
        super(PositionalEncoding2D, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                            -(np.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe = pe.permute(1, 2, 0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.shape[0], None, :x.shape[2],  :]
        return self.dropout(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_parameter('pe', nn.Parameter(pe, requires_grad=False))
        # self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)

class Split_PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000,number_per_frames=2,emb_first=0):
        super(Split_PositionalEncoding, self).__init__()
        self.number_per_frames = number_per_frames
        self.emb_first =emb_first
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_parameter('pe', nn.Parameter(pe, requires_grad=False))


        # self.register_buffer('pe', pe)

    def forward(self, x):
        # x[0] = x[0] + self.pe[0]
        N=self.number_per_frames
        if self.emb_first:
            x[0] = x[0] + self.pe[0]
            
        for i in range(self.emb_first,self.number_per_frames+self.emb_first):
            
            x[i::N] = x[i::N] + self.pe[self.emb_first:(x.shape[0]+N-1)//N]
        # x[1::2] = x[1::2] + self.pe[0:(x.shape[0]+1)//2]
        # x[3::3] = x[3::3] + self.pe[1:(x.shape[0]+2)//3]
        return self.dropout(x)

class HOIEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000,number_per_frames=2,emb_first=0):
        super(HOIEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.number_per_frames = number_per_frames
        self.emb_first = emb_first

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        # self.register_buffer('pe', pe)
        self.learnable_pe = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 1, d_model)) for _ in range(self.number_per_frames)
        ])
        # 标准的从 0 附近初始化
        for p in self.learnable_pe:
            nn.init.normal_(p, std=0.02)
        self.register_parameter('pe', nn.Parameter(pe, requires_grad=False))

        self.interval = len(self.pe)
            
        # m,0,d

    def forward(self, x):
        N = self.number_per_frames
        
        for i in range(N):
            x[i+self.emb_first::N] = x[i+self.emb_first::N] + self.pe[self.interval*i//N:self.interval*i//N+1]
                
    
        return self.dropout(x)
class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder,time_embed_dim=0):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder
        if time_embed_dim == 0:
            time_embed_dim = self.latent_dim
        
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
       

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)
    def get_cosemb(self, timesteps):
        return (self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)

        
        
        
        

class InputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim,skip=0):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.skip = skip
        if not skip:
            self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
            # w = self.poseEmbedding.weight.data
            
        if self.data_rep == 'rot_vel':
            self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)
            w = self.velEmbedding.weight.data
            

    def forward(self, x,batch_first=0):
        bs, njoints, nfeats, nframes = x.shape
        if batch_first:
            x = x.permute((0, 3, 1, 2)).reshape(bs, nframes, njoints*nfeats)
        else:
            x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats)
        if self.skip:
            return x 
        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            x = self.poseEmbedding(x)  # [seqlen, bs, d]
            return x
        elif self.data_rep == 'rot_vel':
            first_pose = x[[0]]  # [1, bs, 150]
            first_pose = self.poseEmbedding(first_pose)  # [1, bs, d]
            vel = x[1:]  # [seqlen-1, bs, 150]
            vel = self.velEmbedding(vel)  # [seqlen-1, bs, d]
            return torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, d]
        else:
            raise ValueError

            

class OutputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats,skip=0):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.skip = skip
        if not skip:
            self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
            
        if self.data_rep == 'rot_vel':
            self.velFinal = nn.Linear(self.latent_dim, self.input_feats)
           

    def forward(self, output,batch_first=0):
        # T,B,D OR B,T,D
        nframes, bs, d = output.shape
        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            if not self.skip:
                output = self.poseFinal(output)  # [seqlen, bs, 150]
        elif self.data_rep == 'rot_vel':
            first_pose = output[[0]]  # [1, bs, d]
            first_pose = self.poseFinal(first_pose)  # [1, bs, 150]
            vel = output[1:]  # [seqlen-1, bs, d]
            vel = self.velFinal(vel)  # [seqlen-1, bs, 150]
            output = torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, 150]
        else:
            raise ValueError
        if batch_first:                # B,T
            output = output.reshape(nframes, bs, self.njoints, self.nfeats)
            output = output.permute(0, 2, 3, 1)
            
        else:
            
            output = output.reshape(nframes, bs, self.njoints, self.nfeats)
            
                
            output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        return output


