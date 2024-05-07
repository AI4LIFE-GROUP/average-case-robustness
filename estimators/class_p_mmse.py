import torch

from estimators.superclass_estimator import Estimator
from estimators.fns_add_noise import add_gaussian_noise, add_gaussian_noise_batched
from src.fns_sigma import f_to_g, x_to_gradf
from estimators.fns_mvn_cdf_torch import compute_mvn_cdf

from estimators.class_p_emp import get_list_batch_sizes
from typing import Tuple


###functions for class, start
def x_to_g(x: torch.Tensor, model: torch.nn.Module, target):
    '''
    Given x, calculate g(x)
    
    x: batch of images, [b, C, H, W]
    model: model
    target: target for g, if target=None, then target=predicted class
    '''
    
    #calculate logits
    f = model(x) #[b, n_c]
    
    #set target
    if target is None:
        target = f.max(dim=1)[1] #[b], target=predicted class
    
    #calculate g
    g = f_to_g(f, target) #[b, n_c-1]
    
    return g


def gradf_to_gradg(grad_f: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    '''
    Calculate gradient norm of g(x) and covariance matrices
    
    grad_f: gradient of f(x), [b, n_c, CHW]
    target: target class for g(x), [b]
    
    grad_g_norm: gradient norm of g(x), [b, n_c-1]
    cov: covariance matrices, [b, n_c-1, n_c-1]
    '''
    
    ###gradf_to_gradgnorm

    #g_i = f_target - f_i, therefore:
    #∇g_i = ∇f_target - ∇f_i
    #and 
    #|| ∇g_i ||^2 = || ∇f_target - ∇f_i ||^2 
    #             = || ∇f_target ||^2  +  || ∇f_i ||^2  -  2(∇f_target)^T(∇f_i)
    #             = a^2 + b^2 - 2ab

    #get dimensions
    b = grad_f.size(0)
    n_c = grad_f.size(1)
    CHW = grad_f.size(2)

    #get grad_f_target
    idx = target.reshape(b, 1) #[b, 1]
    idx = idx.repeat(1, CHW).view(b, 1, CHW)  #[b, 1, CHW]
    grad_f_target = torch.gather(grad_f, 1, idx) #[b, 1, CHW]

    #get grad_f_others
    mask_others_2d = torch.ones(b, n_c, device=target.device).scatter_(1, target.unsqueeze(dim=1), 0.) #[b, n_c]
    mask_others_3d = mask_others_2d.unsqueeze(dim=2).repeat(1, 1, CHW) #[b, n_c, CHW] #TODO: how to index directly instead of stacking?
    grad_f_others = grad_f[mask_others_3d.bool()].view(b, n_c-1, CHW) #[b, n_c-1, CHW]
    
    #calculate grad_g
    grad_g = grad_f_target - grad_f_others #[b, n_c-1, CHW]

    return grad_g


def x_to_g_and_gradg(x: torch.Tensor, model: torch.nn.Module, target):
    '''
    Calculate g and gradient of g
    
    x: batch of images, [b, C, H, W]
    model: model with logit outputs
    target: target for g, if target=None, then target=predicted class
    '''
    
    #set target
    if target is None:
        target = model(x).max(dim=1)[1] #[b], target=predicted class
    
    #calculate g
    g = x_to_g(x, model, target) #[b, n_c-1]

    #calculate gradg
    gradf = x_to_gradf(x, model) #[b, n_c, CHW]
    gradg = gradf_to_gradg(gradf, target) #[b, n_c-1, CHW]

    return g, gradg #[b, n_c-1], [b, n_c-1, CHW]


def calc_g_and_gradg_noisy(x: torch.Tensor, model: torch.nn.Module, sigma: float, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    x: one image, [1, C, H, W]
    model: torch model with logit outputs
    sigma: standard deviation of iid N(0, sigma^2) noise
    n_samples: number of noisy samples of x to generate
    '''
    #get predicted classes for x --> target should always be x's predicted class (not x_noisy's predicted class)
    target = model(x).max(dim=1)[1] #[1], target=predicted class of x
    target = target.repeat(n_samples) #[n_samples]

    #create 'n_samples' noisy samples
    x_noisy = add_gaussian_noise(x, sigma, n_samples) #[n_samples, C, H, W]

    #calculate g and gradg for this noisy sample
    g_noisy, gradg_noisy = x_to_g_and_gradg(x_noisy, model, target) #[n_samples, n_c-1], [n_samples, n_c-1, CHW]

    return g_noisy, gradg_noisy



###functions for class, start
def ensure_cov_psd(cov: torch.Tensor, lam=1e-5) -> torch.Tensor:
    '''
    Add small positive constant to diagonal of covariance matrices to make covariance matrices PSD
    
    cov: batch of covariance matrices, #[batch_size, dim, dim]
    '''
    #get values
    batch_size = cov.size(0)
    dim = cov.size(1)

    #create diagonal matrices
    I_2d = lam*torch.eye(dim, device=cov.device) #[dim, dim]
    I_2d = I_2d.unsqueeze(dim=0) #[1, dim, dim]
    I_3d = I_2d.repeat(batch_size, 1, 1) #[batch_size, dim, dim]

    return cov + I_3d #[batch_size, dim, dim]
###functions for class, end



# def g_and_gradg_to_p(g: torch.Tensor, gradg: torch.Tensor, sigma: float, maxpts: int, abseps: float, releps: float) -> torch.Tensor:
#     '''
#     g: g(x), [b, n_c-1] (where x: [b, C, H, W])
#     gradg: gradient of g(x), [b, n_c-1, CHW] (where x: [b, C, H, W])
#     sigma: standard deviation of iid N(0, sigma^2) noise
#     maxpts, abseps, releps: arguments for compute_mvn_cdf()
#     '''
#     #calculate gradgnorm, cov, z
#     gradgnorm = gradg.square().sum(dim=2).sqrt() #[b, n_c-1]
#     grad_g_unit = gradg / gradgnorm.unsqueeze(dim=2) #[b, n_c-1, CHW]

#     cov = torch.bmm(grad_g_unit, grad_g_unit.transpose(dim0=-2, dim1=-1)) #[b, n_c-1, n_c-1]
#     cov = ensure_cov_psd(cov)
#     z = g / (sigma*gradgnorm) #[b, n_c-1]
    
#     #calculate p
#     p = compute_mvn_cdf(z, cov, maxpts, abseps, releps) #[b]
    
#     return p


def g_and_gradg_to_z_and_cov(g: torch.Tensor, gradg: torch.Tensor, sigma: float) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    g: g(x), [b, n_c-1] (where x: [b, C, H, W])
    gradg: gradient of g(x), [b, n_c-1, CHW] (where x: [b, C, H, W])
    sigma: standard deviation of iid N(0, sigma^2) noise
    '''
    #calculate gradgnorm, cov, z
    gradgnorm = gradg.square().sum(dim=2).sqrt() #[b, n_c-1]
    grad_g_unit = gradg / gradgnorm.unsqueeze(dim=2) #[b, n_c-1, CHW]

    cov = torch.bmm(grad_g_unit, grad_g_unit.transpose(dim0=-2, dim1=-1)) #[b, n_c-1, n_c-1]
    cov = ensure_cov_psd(cov) #use 1e-1/1e-2/1e-3/1e-5 for cifar100 vit
    z = g / (sigma*gradgnorm) #[b, n_c-1]
    
    return z, cov


def mv_sigmoid(z: torch.Tensor) -> torch.Tensor:
    '''
    Calculate mv-sigmoid non-linearity
    
    z: [b, n_c-1]
    return: mv_sigmoid(-z), [b] --> '-z' because z=g(x) when used
    '''
    return 1 / (1 + z.neg().exp().sum(axis=1))


def z_and_cov_to_p(z: torch.Tensor, cov: torch.Tensor, est_type: str, maxpts: int, abseps: float, releps: float):
    '''
    z: [b, n_c-1]
    cov: [b, n_c-1, n_c-1]
    est_type: function used to calculate p, ['mvncdf', 'mvsigmoid', 'both']
    maxpts, abseps, releps: arguments for compute_mvn_cdf()

    Return either a tensor or a tuple of two tensors
    '''
    if est_type=='mvncdf':
        p = compute_mvn_cdf(z, cov, maxpts, abseps, releps) #[b]
        return p
    
    elif est_type=='mvsigmoid':
        p = mv_sigmoid(z) #[b]
        return p
    
    elif est_type=='both':
        p_mvncdf = compute_mvn_cdf(z, cov, maxpts, abseps, releps) #[b]
        p_mvsigmoid = mv_sigmoid(z) #[b]
        return p_mvncdf, p_mvsigmoid


    

###functions for class, end


###class
class PMMSE(Estimator):
    def __init__(self, model, device):
        super().__init__(model, device)
        
    
    def calc_prob_serial(self, x_batch: torch.Tensor, n_samples: int, est_type: str, maxpts: int=10000, abseps: float=1e-4, releps: float=1e-4) -> torch.Tensor:
        '''
        Calculate p_correct in a serial manner
        
        x_batch: batch of images, [b, C, H, W]
        n_samples: number of noisy samples for MMSE estimation
        est_type: function used to calculate probability, ['mvncdf', 'mvsigmoid'] --> currently does not handle 'both' case (which returns a tuple of tensors)
        maxpts, abseps, releps: arguments for compute_mvn_cdf()
        
        output: p_correct probabilities, [b]
        
        Note:
        - model and sigma (standard deviation of iid N(0, sigma^2) noise) are stored in self
        - same sigma (self.sigma) is used for both p_taylor calculation (in compute_mvn_cdf()) and mmse sampling (in add_gaussian_noise())
        '''
        
        batch_size = x_batch.size(0)
        p_batch = torch.zeros(batch_size, device=x_batch.device) #[batch_size], storage for p values of x_batch

        ###for each point x_i in x_batch...
        for i in range(batch_size):
            #get data point x_i
            x = x_batch[[i], :, :, :] #[1, C, H, W]

            ###calculate g and gradg over noisy samples
            #storage for average g and average gradg (will become a tensor)
            g = 0
            gradg = 0

            for _ in range(n_samples):
                g_noisy, gradg_noisy = calc_g_and_gradg_noisy(x, self.model, self.sigma, n_samples=1) #[1, n_c-1], [1, n_c-1, CHW]
                g += g_noisy
                gradg += gradg_noisy

            ###back to point level...
            #calculate average g and average gradg over the noisy samples
            g /= n_samples #[1, n_c-1]
            gradg /= n_samples #[1, n_c-1, CHW]

            #calculate z and cov
            z, cov = g_and_gradg_to_z_and_cov(g, gradg, self.sigma)

            #calculate p
            p_batch[i] = z_and_cov_to_p(z, cov, est_type, maxpts, abseps, releps) #[b]
                
        return p_batch #[batch_size]
    
    
    def calc_prob_batched(self, x_batch: torch.Tensor, n_samples: int, est_type: str, max_eff_batch_size: int=256, maxpts: int=10000, abseps: float=1e-4, releps: float=1e-4):
        '''
        Calculate p_correct in a batched manner
        
        x_batch: batch of images, [b, C, H, W]
        n_samples: number of noisy samples for MMSE estimation
        est_type: function used to calculate probability, ['mvncdf', 'mvsigmoid', 'both']
        max_eff_batch_size: batch size limit when generating noisy samples, here, effective batch size = b*n_samples
        maxpts, abseps, releps: arguments for compute_mvn_cdf()
        
        output: p_correct probabilities, [b]
        
        Note:
        - model and sigma (standard deviation of iid N(0, sigma^2) noise) are stored in self
        - same sigma (self.sigma) is used for both p_taylor calculation (in compute_mvn_cdf()) and mmse sampling (in add_gaussian_noise())
        - due to g_wrapper_full, calculations have an effective batch size of n_samples*b
          keep this in mind when choosing 1) batch size of x_batch and 2) n_samples
        '''
        
        b = x_batch.size(0) #batch size of x_batch
        
        #store g and gradg for each point
        g_lst = [] #list of g tensors that are each [1, n_c-1]
        gradg_lst = [] #list of gradg tensors that are each [1, n_c-1, CHW]
        
        ###for each point x_i in x_batch...
        for i in range(b):
            #get data point x_i
            x = x_batch[[i], :, :, :] #[1, C, H, W]
            
            ###calculate g and gradg for noisy samples in batched manner
            #if n_samples is small --> fit calculation in one batch
            if n_samples <= max_eff_batch_size:
                g_noisy, gradg_noisy = calc_g_and_gradg_noisy(x, self.model, self.sigma, n_samples) #[n_samples, n_c-1], [n_samples, n_c-1, CHW]
            
            #if n_samples is large --> split calculation into multiple batches
            else:
                #storage for g_noisy and gradg_noisy of each batch of noisy samples
                g_noisy_lst, gradg_noisy_lst = [], []
                
                #get list of batch_sizes: [max_eff_batch_size, max_eff_batch_size, ..., remainder]
                lst_batch_sizes = get_list_batch_sizes(n_samples, max_eff_batch_size)
                
                #calculate g_noisy and gradg_noisy
                for size in lst_batch_sizes:
                    g_noisy, gradg_noisy = calc_g_and_gradg_noisy(x, self.model, self.sigma, size) #[size, n_c-1], [size, n_c-1, CHW] #where size=max_eff_batch_size or remainder_samples
                    g_noisy_lst.append(g_noisy)
                    gradg_noisy_lst.append(gradg_noisy)

                #concatenate predicted classes for noisy samples of x_i
                g_noisy = torch.cat(g_noisy_lst) #[n_samples, n_c-1]
                gradg_noisy = torch.cat(gradg_noisy_lst) #[n_samples, n_c-1, CHW]
            
            ###back to point level...
            #calculate g and gradg for datapoint x_i: average g and average gradg over the noisy samples
            g = g_noisy.sum(dim=0, keepdim=True) / n_samples #[1, n_c-1]
            gradg = gradg_noisy.sum(dim=0, keepdim=True) / n_samples #[1, n_c-1, CHW]
            g_lst.append(g)
            gradg_lst.append(gradg)
            
        ###back to batch level...
        #compute p for batch
        g = torch.cat(g_lst) #[b, n_c-1]
        gradg = torch.cat(gradg_lst) #[b, n_c-1, CHW]

        z, cov = g_and_gradg_to_z_and_cov(g, gradg, self.sigma)
        #if est_type=='mvncdf' or 'mvsigmoid': p_batch is a tensor: [b]
        #if est_type=='both': p_batch is a tuple of two tensors: ( [b], [b] )
        p_batch = z_and_cov_to_p(z, cov, est_type, maxpts, abseps, releps)
        
        return p_batch #[b] or ( [b], [b] )
    
    
    def calc_prob_dl(self, dl: torch.utils.data.DataLoader, n_samples: int, est_type: str, max_eff_batch_size: int=256, maxpts: int=10000, abseps: float=1e-4, releps: float=1e-4):
        '''
        Calculate p_correct for a dataloader, where each batch is calculated using calc_prob_batched()
        
        dl: dataloader (with each batch of images being [b, C, H, W])
        n_samples: number of noisy samples for MMSE estimation
        est_type: function used to calculate probability, ['mvncdf', 'mvsigmoid', 'both']
        maxpts, abseps, releps: arguments for compute_mvn_cdf()
        
        output: p_correct probabilities, [n], where n=#points in dataloader
        
        Note: think about effective batch size! --> see calc_prob_batched() documentation
        '''
        
        self.model = self.model.to(self.device)


        if est_type=='mvncdf' or est_type=='mvsigmoid':
            p_dl_lst = [] #storage for p_correct values for each x_batch
            
            #for each x_batch, calculate p
            for x_batch, _ in dl:
                x_batch = x_batch.to(self.device) #[b, C, H, W]
                p_batch = self.calc_prob_batched(x_batch, n_samples, est_type, max_eff_batch_size, maxpts, abseps, releps) #[b]
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
                p_batch_mvncdf, p_batch_mvsigmoid = self.calc_prob_batched(x_batch, n_samples, est_type, max_eff_batch_size, maxpts, abseps, releps) #([b], [b])
                
                p_dl_mvncdf_lst.append(p_batch_mvncdf)
                p_dl_mvsigmoid_lst.append(p_batch_mvsigmoid)
            
            #get p for full dl
            p_dl_mvncdf = torch.cat(p_dl_mvncdf_lst) #[n], where n=#points in dl
            p_dl_mvsigmoid = torch.cat(p_dl_mvsigmoid_lst) #[n], where n=#points in dl

            return p_dl_mvncdf, p_dl_mvsigmoid
        
        
        
    