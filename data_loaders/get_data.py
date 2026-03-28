from torch.utils.data import DataLoader
import torch
from data_loaders.tensors import collate as all_collate
from data_loaders.tensors import t2m_collate,t2hoi_collate, t2hoi_prefix_collate
import torch.utils.data.distributed as dist_utils
from collections import Counter
import numpy as np
def get_dataset_class(name):
    # if name == "amass":
    #     from .amass import AMASS
    #     return AMASS
    # elif name == "uestc":
    #     from .a2m.uestc import UESTC
    #     return UESTC
    # elif name == "humanact12":
    #     from .a2m.humanact12poses import HumanAct12Poses
    #     return HumanAct12Poses
    # elif name == "humanml" or name in []:
    from data_loaders.humanml.data.dataset import HumanML3D
    return HumanML3D
    # elif name == "kit":
    #     from data_loaders.humanml.data.dataset import KIT
    #     return KIT
    # else:
    #     raise ValueError(f'Unsupported dataset name [{name}]')

def get_collate_fn(name, hml_mode='train', pred_len=0, batch_size=1,pad_prefix=0):
    if hml_mode == 'gt':
        from data_loaders.humanml.data.dataset import collate_fn as t2m_eval_collate
        return t2m_eval_collate
    # if name in ["humanml", "kit"]:
    if pred_len > 0 and pad_prefix ==0:
        return lambda x: t2hoi_prefix_collate(x, pred_len=pred_len)
    elif pred_len > 0 :
        return lambda x: t2hoi_prefix_collate(x, pred_len=pred_len)
    return lambda x: t2hoi_collate(x, batch_size)
    # else:
    #     return all_collate


def get_dataset(args,name, num_frames, split='train', hml_mode='train', abs_path='.', fixed_len=0, 
                device=None, autoregressive=False, cache_path=None): 
    DATA = get_dataset_class(name)
    
    dataset = DATA(args,split=split, num_frames=num_frames, mode=hml_mode, abs_path=abs_path, fixed_len=fixed_len, 
                       device=device, autoregressive=autoregressive)
    # if name in ["humanml", "kit"]:
    #     dataset = DATA(split=split, num_frames=num_frames, mode=hml_mode, abs_path=abs_path, fixed_len=fixed_len, 
    #                    device=device, autoregressive=autoregressive)
    # else:
    #     dataset = DATA(split=split, num_frames=num_frames)
    return dataset
class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset

    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(
        self,
        dataset,
    ):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) 
        self.balance_type = dataset.t2m_dataset.balance_type

        
        self.num_samples = len(self.indices) 
        weights = dataset.t2m_dataset.objid_list
        
        # if self.balance_type:
        #     self.labels = [float(x) if isinstance(x, np.ndarray) else float(x)
        #         for x in dataset.t2m_dataset.objid_list]
        #     print('BT1')
        #     weights = self.labels
        #     # label_to_count = Counter(self.labels)
        #     # weights = [1.0 / label_to_count[label] for label in self.labels]
        # else:
        #     self.labels = [int(x) if isinstance(x, np.ndarray) else int(x)
        #         for x in dataset.t2m_dataset.objid_list]
        #     label_to_count = Counter(self.labels)
        #     weights = [1.0 / label_to_count[label] for label in self.labels]

        # distribution of classes in the dataset
        # df = pd.DataFrame()
        # df["label"] = self._get_labels(dataset) if labels is None else labels
        # df.index = self.indices
        # df = df.sort_index()

        # label_to_count = df["label"].value_counts()

        # weights = 1.0 / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights)

    def _get_labels(self, dataset):
        if self.callback_get_label:
            return self.callback_get_label(dataset)
        elif isinstance(dataset, torch.utils.data.TensorDataset):
            return dataset.tensors[1]
        elif isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels.tolist()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return [x[1] for x in dataset.imgs]
        elif isinstance(dataset, torchvision.datasets.DatasetFolder):
            return dataset.samples[:][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[:][1]
        elif isinstance(dataset, torch.utils.data.Dataset):
            return dataset.get_labels()
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples

def get_dataset_loader(args,name, batch_size, num_frames, split='train', hml_mode='train', fixed_len=0, pred_len=0, 
                       device=None, autoregressive=False,eval_mode=False):
    dataset = get_dataset(args,name, num_frames, split=split, hml_mode=hml_mode, fixed_len=fixed_len, 
                device=device, autoregressive=autoregressive)
    
    collate = get_collate_fn(name, hml_mode, pred_len, batch_size,args.pad_prefix)
    if not eval_mode:
        # if torch.distributed.is_initialized():
        #     # sampler = dist_utils.DistributedSampler(dataset,shuffle=True)
        #     # sampler = None
        #     # is_shuffle=True
            
        #     ## SGD
        #     # print('SAMPLER')
        #     # sampler = None
        #     if args.balance_type<2:
        #         print('AKAK')
        #         sampler = ImbalancedDatasetSampler(dataset)
        #         is_shuffle=False
        #     else:
        #         sampler = None
                
        #         is_shuffle=True
        # else:
        #     if args.balance_type<2:
        #         print('AKAK')
        #         sampler = ImbalancedDatasetSampler(dataset)
        #         is_shuffle=False
        #     else:
        #         sampler = None
                
        #         is_shuffle=True
            # print('INBALANCED')
            # sampler = ImbalancedDatasetSampler(dataset)
            # is_shuffle=False
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,sampler=None,
            num_workers=4, drop_last=False, collate_fn=collate,pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True
        )
        # print(batch_size,is_shuffle,'TYPE')
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, # change
            num_workers=2, drop_last=False, collate_fn=collate,pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True
        )
    # loader = DataLoader(
    #     dataset, batch_size=batch_size, shuffle=True,
    #     num_workers=8, drop_last=True, collate_fn=collate
    # )

    return loader


# def get_dataset_loader(conf: DatasetConfig):
#     # name, batch_size, num_frames, split='train', hml_mode='train'
#     dataset = get_dataset(conf)
#     collate = get_collate_fn(conf.name, conf.hml_mode, conf.training_stage)

#     if conf.hml_mode == 'train':
#         if torch.distributed.is_initialized():
#             sampler = dist_utils.DistributedSampler(dataset,shuffle=True)
#             is_shuffle=False
#         else:
#             sampler = None
#             is_shuffle=True
#         loader = DataLoader(
#             dataset, batch_size=conf.batch_size, shuffle=is_shuffle,sampler=sampler,
#             num_workers=32, drop_last=True, collate_fn=collate,
#         )
#     else:
#         loader = DataLoader(
#             dataset, batch_size=conf.batch_size, shuffle=True,
#             num_workers=8, drop_last=True, collate_fn=collate,
#         )
#     return loader