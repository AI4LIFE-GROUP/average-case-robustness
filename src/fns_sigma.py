import torch
from typing import Tuple



def mv_sigmoid(z: torch.Tensor) -> torch.Tensor:
    '''
    mv-sigmoid non-linearity
    '''
    return 1 / (1 + z.neg().exp().sum(axis=1))



def f_to_g(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    '''
    Convert logits (f(x)) to difference of logits (g(x) where g_i = f_target - f_i)
    
    logits: f(x), [b, n_c]
    target: true or predicted labels, [b]

    output: g(x), [b, n_c-1]
    '''
    target = target.to(logits.device)
    
    #get f of target class
    f_target = logits.gather(1, target.unsqueeze(dim=1)) #[b, 1]
    
    #get f of other classes
    mask_others = torch.ones_like(logits).scatter_(1, target.unsqueeze(dim=1), 0.) #[b, n_c]
    n_others = logits.size(1)-1 #n_c-1
    f_others = logits[mask_others.bool()].view(-1, n_others) #[b, n_c-1]
    
    return f_target - f_others #[b, n_c-1]



def x_to_gradf(x: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
    '''
    Calculate gradient of f(x)
    
    x: input, [b, C, H, W]
    model: model that takes in x [b, C, H, W] and returns logits [b, n_c]
    grad_f: gradient of f(x), #[b, n_c, CHW]
    '''
    grad_f = torch.autograd.functional.jacobian(func=lambda inputs: model(inputs).sum(axis=0), 
                                                inputs=x, 
                                                vectorize=True) #[n_c, b, C, H, W]

    grad_f = grad_f.flatten(start_dim=2, end_dim=-1) #[n_c, b, CHW]
    grad_f = grad_f.swapaxes(0,1) #[b, n_c, CHW]
    
    return grad_f





def gradf_to_gradgnorm_and_cov(grad_f: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
    
    #calculate grad_g_norm
    a_square = grad_f_target.square().sum(dim=2) #[b, 1]
    b_square = grad_f_others.square().sum(dim=2) #[b, n_c-1]
    ab = torch.bmm(grad_f_target, grad_f_others.transpose(1, 2)) #[b, 1, n_c-1] ( =[b, 1, CHW]*[b, CHW, n_c-1] ) --> vector-matrix multiplication
    ab = ab.squeeze() #[b, n_c-1]
    grad_g_norm = (a_square + b_square - 2*ab).sqrt() #[b, n_c-1]
    
    #calculate covariance matrix
    grad_g_unit = grad_g/grad_g_norm.unsqueeze(dim=2) #[b, n_c-1, CHW]
    # cov = torch.bmm(grad_g_unit, grad_g_unit.mT) #[b, n_c-1, n_c-1]
    cov = torch.bmm(grad_g_unit, grad_g_unit.transpose(dim0=-2, dim1=-1)) #[b, n_c-1, n_c-1]

    return grad_g_norm, cov


def x_to_gradgnorm_and_cov(x: torch.Tensor, model: torch.nn.Module, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Calculate 1) gradient norm of g(x) and 2) covariance matrices
    x: inputs, [b, C, H, W]
    model: model that takes in x [b, C, H, W] and returns logits [b, n_c]
    target: target class for g(x) calculation, [b]
    '''
    #f to gradf
    grad_f = x_to_gradf(x, model)
    #gradf to gradgnorm
    grad_g_norm, cov = gradf_to_gradgnorm_and_cov(grad_f, target)

    return grad_g_norm, cov



