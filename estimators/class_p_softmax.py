import torch

from estimators.superclass_estimator import Estimator
from typing import Tuple



###class
class PSoftmax(Estimator):
    def __init__(self, model, device):
        super().__init__(model, device)
    
    
    def calc_prob_serial(self, x_batch: torch.Tensor) -> torch.Tensor:
        '''
        Calculate p_correct in a serial manner
        x_batch: batch of images, [b, C, H, W]
        output: p_correct probabilities, [b]
        '''
        
        batch_size = x_batch.size(0)
        p_batch = torch.zeros(batch_size, device=x_batch.device) #[batch_size], storage for p values of x_batch
        
        for i in range(batch_size):
            #get data point x_i
            x = x_batch[[i], :, :, :] #[1, C, H, W]
            
            #calculate p for x_i
            p_batch[i] = self.calc_prob_batched(x) #[1]
            
        return p_batch
    
    
    def calc_prob_batched(self, x_batch: torch.Tensor) -> torch.Tensor:
        '''
        Calculate p_correct in a batched manner
        x_batch: batch of images, [b, C, H, W]
        output: p_correct probabilities, [b]
        '''
        
        #calculate logits
        logits = self.model(x_batch) #[b, n_c]
        #calculate softmax probability vector
        softmax_probs = torch.nn.functional.softmax(logits, dim=1) #[b, n_c]
        #get softmax probability of predicted class
        p_batch = softmax_probs.max(dim=1)[0] #[b]
        
        return p_batch
    
        
    def calc_prob_dl(self, dl: torch.utils.data.DataLoader) -> torch.Tensor:
        '''
        Calculate p_correct for a dataloader, where each batch is calculated using calc_prob_batched()
        dl: dataloader (with each batch of images being [b, C, H, W])
        output: p_correct probabilities, [n], where n=#points in dataloader
        '''
        self.model = self.model.to(self.device)
        p_dl_lst = [] #storage for p_correct values for each x_batch
        
        #for each x_batch, calculate p
        for x_batch, _ in dl:
            x_batch = x_batch.to(self.device)
            p_batch = self.calc_prob_batched(x_batch)
            p_dl_lst.append(p_batch)

        #get p for full dl
        p_dl = torch.cat(p_dl_lst) #[n] where n=#points in dl
        
        return p_dl 