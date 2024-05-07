import torch
from torch import nn
from typing import Any
from typing_extensions import Self


class Estimator(nn.Module):
    
    def __init__(self, model, device):
        super().__init__()
        model.eval()
        self.model = model
        self.device = device
        self.sigma = 0.1
    
    
    def calc_prob_serial(self, x_batch: torch.Tensor, *args: Any) -> torch.Tensor:
        '''
        Calculate p_correct in a serial manner
        x_batch: batch of images, [b, C, H, W]
        output: p_correct probabilities, [b]
        Note: Should be overridden by all subclasses
        '''
        raise NotImplementedError
        
    def calc_prob_batched(self, x_batch: torch.Tensor, *args: Any) -> torch.Tensor:
        '''
        Calculate p_correct in a batched manner
        x_batch: batch of images, [b, C, H, W]
        output: p_correct probabilities, [b]
        Note: Should be overridden by all subclasses
        '''
        raise NotImplementedError
        
    def calc_prob_dl(self, dl: torch.utils.data.DataLoader, *args: Any) -> torch.Tensor:
        '''
        Calculate p_correct for a dataloader, where each batch is calculated using calc_prob_batched()
        dl: dataloader (with each batch of images being [b, C, H, W])
        output: p_correct probabilities, [n], where n=#points in dataloader
        Note: Should be overridden by all subclasses
        '''
        raise NotImplementedError