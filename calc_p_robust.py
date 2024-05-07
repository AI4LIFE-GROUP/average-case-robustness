import os
import torch
import torchvision
from torchvision import transforms
import models.resnet

from estimators.class_p_emp import PEmpirical
from estimators.class_p_taylor import PTaylor
from estimators.class_p_mmse import PMMSE
from estimators.class_p_softmax import PSoftmax


### load data
print('--> load data')
# load CIFAR10 test set
test_transform = transforms.Compose([
                           transforms.ToTensor()
                       ])
batch_size = 4
data_path = os.path.join(os.getcwd() , 'datasets', 'cifar10')
test_set = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=test_transform)

# get subset for demonstration purposes
subset_indices = list(range(8)) 
test_subset = torch.utils.data.Subset(test_set, subset_indices)
test_loader = torch.utils.data.DataLoader(test_subset, batch_size=batch_size, shuffle=False)


### load model
print('--> load model')
model_path = f'saved_models/resnet18_cifar10.pt'
model = models.resnet.ResNet18(activation=torch.nn.ReLU, num_classes=10)
model.load_state_dict(torch.load(model_path), strict=True)

cuda_available = torch.cuda.is_available()
device = torch.device('cuda' if cuda_available else 'cpu')
model.to(device)


### calculate p_robust
print('--> calculate p_robust')

def run_estimator_on_dataloader(estimator_class, estimator_args, sigma):
    estimator_class.sigma = sigma
    p = estimator_class.calc_prob_dl(**estimator_args)
    return p

def round_probust_values(p_robust):
    p_robust_lst = p_robust.tolist()
    rounded = [round(num, 2) for num in p_robust_lst]
    return rounded

torch.manual_seed(42)
sigma = 0.05

#instantiate classes
class_p_mc = PEmpirical(model, device)
class_p_mmse = PMMSE(model, device)
class_p_tay = PTaylor(model, device)
class_p_sm = PSoftmax(model, device)

#p_mc
p_mc = run_estimator_on_dataloader(
    estimator_class=class_p_mc, 
    estimator_args={'dl': test_loader, 'n_samples': 10000},
    sigma=sigma
    )
print(f'p_mc: {round_probust_values(p_mc)}')

#p_taylor and p_taylor_mvs
p_tay, p_tay_mvs = run_estimator_on_dataloader(
    estimator_class=class_p_tay, 
    estimator_args={'dl': test_loader, 'est_type': 'both'},
    sigma=sigma
    )
print(f'p_tay: {round_probust_values(p_tay)}')
print(f'p_tay_mvs: {round_probust_values(p_tay_mvs)}')

#p_mmse and p_mmse_mvs
p_mmse, p_mmse_mvs = run_estimator_on_dataloader(
    estimator_class=class_p_mmse, 
    estimator_args={'dl': test_loader, 'n_samples': 10, 'est_type': 'both'},
    sigma=sigma
    )
print(f'p_mmse: {round_probust_values(p_mmse)}')
print(f'p_mmse_mvs: {round_probust_values(p_mmse_mvs)}')

#####p_softmax: traditional softmax probability
p_sm = run_estimator_on_dataloader(
    estimator_class=class_p_sm, 
    estimator_args={'dl': test_loader},
    sigma=sigma)
print(f'p_softmax: {round_probust_values(p_sm)}')


print('complete!')



