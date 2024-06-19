import torch
import torch.nn as nn
import torch 
from torch.autograd import Variable
import torch.nn.functional as F

from .emd import SinkhornDistance_new

def cos_batch_torch(x, y):
    "Returns the cosine distance batchwise"
    # return: bs * n * m
    x = x.transpose(1,2)
    y = y.transpose(1,2)
    cos_sim = F.cosine_similarity(x[..., None, :, :], y[..., :, None, :], dim=-1)
    cos_dis = 1 - cos_sim # to minimize this value
    
    #sigma = 64.
    #cos_dis = 1.0 - torch.exp(sigma * (cos_sim - 1))
    '''
    # TODO:
    beta = 0.1
    min_score = cos_dis.min()
    max_score = cos_dis.max()
    threshold = min_score + beta * (max_score - min_score)
    res = cos_dis - threshold
    # res = torch.nn.ReLU()

    return torch.nn.functional.relu(res.transpose(2,1))
    '''
    return cos_dis.transpose(2,1)

def GW_distance(X, Y, p, q, lamda=0.5, iteration=5, OT_iteration=20):
    '''
    :param X, Y: Source and target embeddings , batchsize by embed_dim by n
    :param p, q: probability vectors
    :param lamda: regularization
    :return: GW distance
    '''
    Cs = cos_batch_torch(X, X).float().cuda()
    Ct = cos_batch_torch(Y, Y).float().cuda()
    bs = Cs.size(0)
    m = Ct.size(2)
    n = Cs.size(2)
    T, Cst = GW_torch_batch(Cs, Ct, bs, n, m, p, q, beta=lamda, iteration=iteration, OT_iteration=OT_iteration)
    #temp = torch.bmm(torch.transpose(Cst,1,2), T)
    #distance = batch_trace(temp, m, bs)
    #print("gw", T.transpose(1,2))
    distance = torch.sum(Cst * T, dim=(-2, -1))
    #distance = distance.mean()
    return distance, T.transpose(1,2), Cst.transpose(1,2)

def IPOT_torch_batch_uniform(C, beta=0.5, iteration=50):
    # C is the distance matrix
    # C: bs by n by m
    bs, n, m = C.shape
    sigma = torch.ones(bs, int(m), 1).cuda()/float(m)
    T = torch.ones(bs, n, m).cuda()
    A = torch.exp(-C/beta).float().cuda()
    for t in range(iteration):
        Q = A * T # bs * n * m
        for k in range(1):
            delta = 1 / (n * torch.bmm(Q, sigma))
            a = torch.bmm(torch.transpose(Q,1,2), delta)
            sigma = 1 / (float(m) * a)
        T = delta * Q * sigma.transpose(2,1)

    return T

def IPOT_torch_batch_uniform_v2(C, eps=5e-7, max_iter=100):
    B, S1, S2 = C.shape
    u = torch.zeros(B, S1).fill_(1. / S1).cuda()
    v = torch.zeros(B, S2).fill_(1. / S2).cuda()
    sinkhorn = SinkhornDistance_new(eps=eps, max_iter=max_iter)
    dist, P, C = sinkhorn(u, v, C)
    return P

def GW_torch_batch(Cs, Ct, bs, n, m, p, q, beta=0.5, iteration=5, OT_iteration=20):
    one_m = torch.ones(bs, m, 1).float().cuda()
    one_n = torch.ones(bs, n, 1).float().cuda()

    Cst = torch.bmm(torch.bmm(Cs**2, p), torch.transpose(one_m, 1, 2)) + \
          torch.bmm(one_n, torch.bmm(torch.transpose(q,1,2), torch.transpose(Ct**2, 1, 2))) # bs by n by m
    gamma = torch.bmm(p, q.transpose(2,1)) # outer product, init
    # gamma = torch.einsum('bi,bj->bij', (torch.squeeze(p), torch.squeeze(q))) # outer product, initialization
    for i in range(iteration):
        C_gamma = Cst - 2 * torch.bmm(torch.bmm(Cs, gamma), torch.transpose(Ct, 1, 2))
        '''
        # Sinkhorn iteration
        b = torch.ones(bs, m, 1).cuda()
        K = torch.exp(-C_gamma/beta)
        for i in range(50):cd
            a = p/(torch.bmm(K, b))
        	b = q/torch.bmm(K.transpose(1,2), a)
        gamma = a * K * b
        '''
        #gamma = IPOT_torch_batch_uniform(C_gamma, beta=beta, iteration=OT_iteration)
        gamma = IPOT_torch_batch_uniform_v2(C_gamma)
    Cgamma = Cst - 2 * torch.bmm(torch.bmm(Cs, gamma), torch.transpose(Ct, 1, 2))
    return gamma, Cgamma

def batch_trace(input_matrix, n, bs):
    a = torch.eye(n).cuda().unsqueeze(0).repeat(bs, 1, 1)
    b = a * input_matrix
    return torch.sum(torch.sum(b,-1),-1).unsqueeze(1)

def GW_distance_uniform(X, Y, lamda=1e-1, iteration=5, OT_iteration=20):
    m = X.size(2)
    n = Y.size(2)
    bs = X.size(0)
    p = (torch.ones(bs, m, 1)/m).cuda()
    q = (torch.ones(bs, n, 1)/n).cuda()
    return GW_distance(X, Y, p, q, lamda=lamda, iteration=iteration, OT_iteration=OT_iteration)

