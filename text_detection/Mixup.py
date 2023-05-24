import numpy as np
import torch

import numpy as np
import torch


def mixup_data(x, y, alpha=$number, runs=$number, use_cuda=True):
    output_x = torch.Tensor(0)
    output_x= output_x.numpy().tolist()
    output_y = torch.Tensor(0)
    output_y = output_y.numpy().tolist()
    batch_size = x.size()[0]    
    for i in range(runs):
        if alpha > 0.:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.

        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        mixed_y = lam * y + (1 - lam) * y[index]
        output_x.append(mixed_x)
        output_y.append(mixed_y)
    return torch.cat(output_x,dim=0), torch.cat(output_y,dim=0)
