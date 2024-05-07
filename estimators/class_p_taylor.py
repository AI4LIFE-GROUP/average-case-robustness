import torch

from estimators.superclass_estimator import Estimator
from estimators.class_p_mmse import x_to_g_and_gradg, g_and_gradg_to_z_and_cov, z_and_cov_to_p 



###class
class PTaylor(Estimator):
    def __init__(self, model, device):
        super().__init__(model, device)
    
    
    def calc_prob_serial(self, x_batch: torch.Tensor, est_type: str, maxpts: int=10000, abseps: float=1e-4, releps: float=1e-4) -> torch.Tensor:
        '''
        Calculate p_correct in a serial manner
        x_batch: batch of images, [b, C, H, W]
        est_type: function used to calculate probability, ['mvncdf', 'mvsigmoid'] --> currently does not handle 'both' case (which returns a tuple of tensors)
        maxpts, abseps, releps: arguments for compute_mvn_cdf()
        
        output: p_correct probabilities, [b]
        Note: model and sigma (standard deviation of iid N(0, sigma^2) noise) are stored in self
        '''
            
        batch_size = x_batch.size(0)
        p_batch = torch.zeros(batch_size, device=x_batch.device) #[batch_size], storage for p values of x_batch
        
        #for each point in x_batch...
        for i in range(batch_size):
            #get data point x_i
            x = x_batch[[i], :, :, :] #[1, C, H, W]

            #calculate p
            p_batch[i] = self.calc_prob_batched(x, est_type, maxpts, abseps, releps)
            
        return p_batch #[b]
    
    
    def calc_prob_batched(self, x_batch: torch.Tensor, est_type: str, maxpts: int=10000, abseps: float=1e-4, releps: float=1e-4):
        '''
        Calculate p_correct in a batched manner
        
        x_batch: batch of images, [b, C, H, W]
        est_type: function used to calculate probability, ['mvncdf', 'mvsigmoid', 'both']
        maxpts, abseps, releps: arguments for compute_mvn_cdf()
        
        output: p_correct probabilities, [b]
        Note: model and sigma (standard deviation of iid N(0, sigma^2) noise) are stored in self
        '''
        g, gradg = x_to_g_and_gradg(x_batch, self.model, target=None) #[b, n_c-1], [b, n_c-1, CHW]
        z, cov = g_and_gradg_to_z_and_cov(g, gradg, self.sigma)

        #if est_type=='mvncdf' or 'mvsigmoid': p_batch is a tensor: [b]
        #if est_type=='both': p_batch is a tuple of two tensors: ( [b], [b] )
        p_batch = z_and_cov_to_p(z, cov, est_type, maxpts, abseps, releps)

        return p_batch #[b] or ( [b], [b] )
    
    
    def calc_prob_dl(self, dl: torch.utils.data.DataLoader, est_type: str, maxpts: int=10000, abseps: float=1e-4, releps: float=1e-4):
        '''
        Calculate p_correct for a dataloader, where each batch is calculated using calc_prob_batched()

        dl: dataloader (with each batch of images being [b, C, H, W])
        est_type: function used to calculate probability, ['mvncdf', 'mvsigmoid', 'both']
        maxpts, abseps, releps: arguments for compute_mvn_cdf()
        
        output: p_correct probabilities, [n], where n=#points in dataloader
        '''

        self.model = self.model.to(self.device)

        if est_type=='mvncdf' or est_type=='mvsigmoid':
            p_dl_lst = [] #storage for p_correct values for each x_batch
            
            #for each x_batch, calculate p
            for x_batch, _ in dl:
                x_batch = x_batch.to(self.device) #[b, C, H, W]
                p_batch = self.calc_prob_batched(x_batch, est_type, maxpts, abseps, releps) #[b]
                p_dl_lst.append(p_batch)
            
            #get p for full dl
            p_dl = torch.cat(p_dl_lst) #[n], where n=#points in dl
        
            return p_dl
        
        elif est_type=='both':
            p_dl_mvncdf_lst = [] #storage for p_correct values for each x_batch
            p_dl_mvsigmoid_lst = [] #storage for p_correct values for each x_batch
            
            #for each x_batch, calculate p
            for x_batch, _ in dl:
                x_batch = x_batch.to(self.device) #[b, C, H, W]
                p_batch_mvncdf, p_batch_mvsigmoid = self.calc_prob_batched(x_batch, est_type, maxpts, abseps, releps) #([b], [b])
                
                p_dl_mvncdf_lst.append(p_batch_mvncdf)
                p_dl_mvsigmoid_lst.append(p_batch_mvsigmoid)
            
            #get p for full dl
            p_dl_mvncdf = torch.cat(p_dl_mvncdf_lst) #[n], where n=#points in dl
            p_dl_mvsigmoid = torch.cat(p_dl_mvsigmoid_lst) #[n], where n=#points in dl

            return p_dl_mvncdf, p_dl_mvsigmoid
        
        