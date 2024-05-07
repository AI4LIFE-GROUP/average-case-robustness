import torch


# ----------------------------------

###function to generate noisy versions of x
def add_gaussian_noise(x, sigma, n_samples):
    '''
    Generate noisy samples of x, where noise is iid N(0, sigma^2)
    
    x: input image, tensor: [1, C, H, W]
    sigma: standard deviation of normal distribution for each noise term, int or float
    n_samples: number of noisy versions of x to make, int

    return: n_samples noisy versions of x, tensor: [n_samples, C, H, W]
    '''
    #get sizes
    C = x.size(1)
    H = x.size(2)
    W = x.size(3)
    
    #generate noise
    z = torch.randn(n_samples, C, H, W, device=x.device) #[n_samples, C, H, W] --> iid N(0, 1) noise
    noise = sigma*z #[n_samples, C, H, W] --> iid N(0, sigma^2) noise
    
    return x + noise #[n_samples, C, H, W]


###function to generate noisy versions of x, batched
def add_gaussian_noise_batched(x, sigma, n_samples):
    '''
    Generate noisy samples of x, where noise is iid N(0, sigma^2)
    
    x: batch of images, tensor: [b, C, H, W]
    sigma: standard deviation of normal distribution for each noise term, int or float
    n_samples: number of noisy versions of x to make for each point in batch, int

    return: n_samples noisy versions of x, tensor: [n_samples, b, C, H, W]
    '''
    #get sizes
    b = x.size(0)
    C = x.size(1)
    H = x.size(2)
    W = x.size(3)
    
    #generate noise
    z = torch.randn(n_samples, b, C, H, W, device=x.device) #[n_samples, b, C, H, W] --> iid N(0, 1) noise
    noise = sigma*z #[n_samples, b, C, H, W] --> iid N(0, sigma^2) noise
    
    return x.unsqueeze(dim=0) + noise #[n_samples, b, C, H, W]