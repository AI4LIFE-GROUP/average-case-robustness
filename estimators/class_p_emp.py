import torch

from estimators.superclass_estimator import Estimator
from estimators.fns_add_noise import add_gaussian_noise

###functions for class, start
def calc_preds_noisy(x: torch.Tensor, model: torch.nn.Module, sigma: float, n_samples: int):
    '''
    Calculate predicted classes of noisy versions of x
    
    x: one image, [1, C, H, W]
    model: torch model
    sigma: standard deviation of gaussian noise
    n_samples: number of noisy versions of x to generate

    return: predicted classes of noisy versions of x
    '''
    #generate batch of noisy samples
    x_noisy = add_gaussian_noise(x, sigma, n_samples) #[n_samples, C, H, W]
    #get predictions for this noisy batch
    preds_noisy = model(x_noisy).max(dim=1)[1] #[n_samples]

    return preds_noisy


def get_list_batch_sizes(n_samples: int, max_eff_batch_size: int):
    '''
    Create a list of batch sizes: divide 'n_samples' into batches of size 'max_eff_batch_size' 
    Return: list of batch_sizes that looks like [max_eff_batch_size, max_eff_batch_size, ..., remainder]
    '''
    #calculate #batches and #remainder
    n_batches, remainder_batch = divmod(n_samples, max_eff_batch_size)

    #create list of batch sizes to iterate over
    lst_batch_sizes = [max_eff_batch_size]*n_batches
    
    if remainder_batch > 0:
        lst_batch_sizes.append(remainder_batch)

    return lst_batch_sizes
###functions for class, end


###class
class PEmpirical(Estimator):
    def __init__(self, model, device):
        super().__init__(model, device)
        
    def calc_prob_serial(self, x_batch: torch.Tensor, n_samples: int) -> torch.Tensor:
        '''
        Calculate p_correct in a serial manner
        
        x_batch: batch of images, [b, C, H, W]
        n_samples: number of noisy samples used to calculate p_empirical for each x
        
        return: p_correct probabilities, [b]
        Note: model and sigma (standard deviation of iid N(0, sigma^2) noise) are stored in self
        '''

        batch_size = x_batch.size(0)
        p_batch = torch.zeros(batch_size, device=x_batch.device) #[batch_size], storage for p values of x_batch

        ###for each data point in x_batch...
        for i in range(batch_size):
            #get data point x_i
            x = x_batch[[i], :, :, :] #[1, C, H, W]

            #calculate predicted class for x_i
            pred = self.model(x).max(dim=1)[1] #[1]

            ###for each noisy sample of x_i...
            preds_noisy = torch.zeros(n_samples, device=x_batch.device) #[n_samples], storage for predicted class of noisy samples

            for j in range(n_samples):
                #calculate predicted class for one noisy sample
                preds_noisy[j] = calc_preds_noisy(x, self.model, self.sigma, n_samples=1) #[1]

            #calculate p for x_i
            p_batch[i] = preds_noisy.eq(pred).sum()/n_samples 
        
        return p_batch #[batch_size]
    
        
    def calc_prob_batched(self, x_batch: torch.Tensor, n_samples: int, max_eff_batch_size: int=256) -> torch.Tensor:
        '''
        Calculate p_correct in a batched manner
        
        x_batch: batch of images, [b, C, H, W]
        n_samples: number of noisy samples used to calculate p_empirical for each x
        max_eff_batch_size: batch size limit when generating noisy samples, here, effective batch size = n_samples
        
        output: p_correct probabilities, [b]
        Note: model and sigma (standard deviation of iid N(0, sigma^2) noise) are stored in self
        '''

        ###for each data point in x_batch...
        batch_size = x_batch.size(0)
        p_batch = torch.zeros(batch_size, device=x_batch.device) #[batch_size], storage for p values of x_batch

        for i in range(batch_size):
            #get data point x_i
            x = x_batch[[i], :, :, :] #[1, C, H, W]

            # if i == 0:
            #     print(i)
            #     print(x)

            # print(i)
            # print(x[:, :, 0:5, 0:5])

            #calculate predicted class for x_i
            pred = self.model(x).max(dim=1)[1] #[1]

            ###calculate predicted class for noisy samples in batched manner
            #if n_samples is small --> fit calculation in one batch
            if n_samples <= max_eff_batch_size:
                preds_noisy = calc_preds_noisy(x, self.model, self.sigma, n_samples) #[n_samples]
            
            #if n_samples is large --> split calculation into multiple batches
            else:
                #storage for predicted class of each batch of noisy samples
                preds_noisy_lst = [] 

                #get list of batch_sizes: [max_eff_batch_size, max_eff_batch_size, ..., remainder]
                lst_batch_sizes = get_list_batch_sizes(n_samples, max_eff_batch_size)
                
                for size in lst_batch_sizes:
                    preds_noisy = calc_preds_noisy(x, self.model, self.sigma, size) #[size] where size=max_eff_batch_size or remainder_samples
                    preds_noisy_lst.append(preds_noisy)

                #concatenate predicted classes for noisy samples of x_i
                preds_noisy = torch.cat(preds_noisy_lst) #[n_samples]
            
            ###back to point level...
            #calculate p for datapoint x_i
            p_batch[i] = preds_noisy.eq(pred).sum()/n_samples

        return p_batch

        
    def calc_prob_dl(self, dl: torch.utils.data.DataLoader, n_samples: int, max_eff_batch_size: int=256) -> torch.Tensor:
        '''
        Calculate p_correct for a dataloader, where each batch is calculated using calc_prob_batched()
        
        dl: dataloader (with each batch of images being [b, C, H, W])
        n_samples: arg for self.calc_prob_batched()
        max_eff_batch_size: arg for self.calc_prob_batched()
        
        output: p_correct probabilities, [n], where n=#points in dataloader
        '''
        self.model = self.model.to(self.device)
        p_dl_lst = [] #storage for p_correct values for each x_batch
        
        #for each x_batch, calculate p
        for x_batch, _ in dl:
            x_batch = x_batch.to(self.device)
            p_batch = self.calc_prob_batched(x_batch, n_samples, max_eff_batch_size)
            p_dl_lst.append(p_batch)

        #get p for full dl
        p = torch.cat(p_dl_lst) #[n] where n=#points in dl
        
        return p
        

    