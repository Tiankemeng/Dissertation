import torch
import random
import numpy as np

def rank_elements(lst):
    sorted_lst = sorted((value, i) for i, value in enumerate(lst))
    rank_dict = {value: rank for rank, (value, _) in enumerate(sorted_lst, start=1)}
    return [rank_dict[value] for value in lst]

def compute_prototypes( support_features, labels,num_classes):
    prototypes = []
    for class_idx in range(num_classes):
        class_mask = (labels == class_idx)
        class_features = support_features[class_mask]
        prototype = class_features.mean(dim=0)
        prototypes.append(prototype)
    return torch.stack(prototypes)

class DictToAttr():
    def __init__(self, data):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, DictToAttr(value))
            else:
                setattr(self, key, value)

    def __repr__(self):
        return f"{self.__dict__}"
    


def set_seed(seed):
    """
    设置随机种子以确保实验的可重复性
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = True

