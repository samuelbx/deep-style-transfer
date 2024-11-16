import torch
import ot.gmm
import sklearn.mixture

def style_transfer(method, alpha, cf, sf, K=None):
    assert method in ('gmmot-bary', 'gmmot-rand', 'gaussian', 'wct')
    if 'gmmot' in method:
        return gmm_transfer(alpha, cf, sf, method, K)
    elif method == 'gaussian':
        return gaussian_transfer(alpha, cf, sf)
    else:
        return wct(alpha, cf, sf)

def sqrtm(M):
    """Compute the square root of a positive semidefinite matrix"""
    _, s, v = M.svd()
    # truncate small components
    above_cutoff = s > s.max() * s.size(-1) * torch.finfo(s.dtype).eps
    s = s[..., above_cutoff]
    v = v[..., above_cutoff]
    return (v * s.sqrt().unsqueeze(-2)) @ v.transpose(-2, -1)

def gmm_transfer(alpha, cf, sf, method='gmmot-bary', K=1):
    """Delon & Desolneux (2020). A Wasserstein-type distance in the space of Gaussian mixture models"""
    cf, sf = cf.float(), sf.float()
    channels = cf.size(0)
    cfv, sfv = cf.view(channels, -1), sf.view(channels, -1) # c x (h x w)

    # Fit gaussian mixtures (reg_covar is added to covariances' diagonals to ensure SDP)
    c_gmm = sklearn.mixture.GaussianMixture(n_components=K, covariance_type='full', reg_covar=1e-3).fit(cfv.T)
    s_gmm = sklearn.mixture.GaussianMixture(n_components=K, covariance_type='full', reg_covar=1e-3).fit(sfv.T)

    # HACK: Convert GMM weights to torch Tensors
    wc, mc, Cc = torch.tensor(c_gmm.weights_, dtype=torch.float), torch.tensor(c_gmm.means_, dtype=torch.float), torch.tensor(c_gmm.covariances_, dtype=torch.float) 
    ws, ms, Cs = torch.tensor(s_gmm.weights_, dtype=torch.float), torch.tensor(s_gmm.means_, dtype=torch.float), torch.tensor(s_gmm.covariances_, dtype=torch.float)
    
    # Apply transport map to content data
    method = 'rand' if 'rand' in method else 'bary'
    pushed = (ot.gmm.gmm_ot_apply_map(cfv.T, mc, ms, Cc, Cs, wc, ws, method=method).T).view_as(cf)

    res = (1-alpha) * cf + alpha * pushed
    return res.float().unsqueeze(0)

def gaussian_transfer(alpha, cf, sf):
    """Mroueh, Y. (2019). Wasserstein style transfer"""
    # Approximate content features by Gaussian
    cf = cf.double()
    c_channels, c_width, c_height = cf.size(0), cf.size(1), cf.size(2)
    cfv = cf.view(c_channels, -1)

    c_mean = torch.mean(cfv, 1)
    c_mean = c_mean.unsqueeze(1).expand_as(cfv)
    cfv = cfv - c_mean

    c_covm = torch.mm(cfv, cfv.t()).div((c_width * c_height) - 1)
    c_covm_sqrt = sqrtm(c_covm)
    c_covm_sqrt_inv = torch.inverse(c_covm_sqrt)
    
    # Approximate style features by Gaussian
    sf = sf.double()
    _, s_width, s_heigth = sf.size(0), sf.size(1), sf.size(2)
    sfv = sf.view(c_channels, -1)

    s_mean = torch.mean(sfv, 1)
    s_mean = s_mean.unsqueeze(1).expand_as(sfv)
    sfv = sfv - s_mean
    s_covm = torch.mm(sfv, sfv.t()).div((s_width * s_heigth) - 1)

    A = c_covm_sqrt_inv @ sqrtm(c_covm_sqrt @ s_covm @ c_covm_sqrt) @ c_covm_sqrt_inv
    pushed = (s_mean + A @ cfv).view_as(cf)

    res = (1-alpha) * cf + alpha * pushed
    return res.float().unsqueeze(0)


def wct(alpha, cf, sf):
    """Li et al. (2017). Universal style transfer via feature transforms"""
    # content image whitening
    cf = cf.double()
    c_channels, c_width, c_height = cf.size(0), cf.size(1), cf.size(2)
    cfv = cf.view(c_channels, -1)  # c x (h x w)

    c_mean = torch.mean(cfv, 1) # perform mean for each row
    c_mean = c_mean.unsqueeze(1).expand_as(cfv) # add dim and replicate mean on rows
    cfv = cfv - c_mean # subtract mean element-wise

    c_covm = torch.mm(cfv, cfv.t()).div((c_width * c_height) - 1)  # construct covariance matrix
    _, c_e, c_v = torch.svd(c_covm, some=False) # singular value decomposition

    k_c = c_channels
    for i in range(c_channels):
        if c_e[i] < 0.00001:
            k_c = i
            break
    c_d = (c_e[0:k_c]).pow(-0.5)

    w_step1 = torch.mm(c_v[:, 0:k_c], torch.diag(c_d))
    w_step2 = torch.mm(w_step1, (c_v[:, 0:k_c].t()))
    whitened = torch.mm(w_step2, cfv)

    # style image coloring
    sf = sf.double()
    _, s_width, s_heigth = sf.size(0), sf.size(1), sf.size(2)
    sfv = sf.view(c_channels, -1)

    s_mean = torch.mean(sfv, 1)
    s_mean = s_mean.unsqueeze(1).expand_as(sfv)
    sfv = sfv - s_mean

    s_covm = torch.mm(sfv, sfv.t()).div((s_width * s_heigth) - 1)
    _, s_e, s_v = torch.svd(s_covm, some=False)

    s_k = c_channels # same number of channels ad content features
    for i in range(c_channels):
        if s_e[i] < 0.00001:
            s_k = i
            break
    s_d = (s_e[0:s_k]).pow(0.5)

    c_step1 = torch.mm(s_v[:, 0:s_k], torch.diag(s_d))
    c_step2 = torch.mm(c_step1, s_v[:, 0:s_k].t())
    colored = torch.mm(c_step2, whitened)

    cs0_features = (colored + s_mean.resize_as_(colored)).view_as(cf)

    ccsf = alpha * cs0_features + (1.0 - alpha) * cf
    return ccsf.float().unsqueeze(0)

def wct_mask(cf, sf):
    cf = cf.double()
    cf_sizes = cf.size()
    c_mean = torch.mean(cf, 1)
    c_mean = c_mean.unsqueeze(1).expand_as(cf)
    cf -= c_mean

    c_covm = torch.mm(cf, cf.t()).div(cf_sizes[1] - 1)
    c_u, c_e, c_v = torch.svd(c_covm, some=False)

    k_c = cf_sizes[0]
    for i in range(cf_sizes[0]):
        if c_e[i] < 0.00001:
            k_c = i
            break
    c_d = (c_e[0:k_c]).pow(-0.5)
    whitened = torch.mm(torch.mm(torch.mm(c_v[:, 0:k_c], torch.diag(c_d)), (c_v[:, 0:k_c].t())), cf)

    sf = sf.double()
    sf_sizes = sf.size()
    sfv = sf.view(sf_sizes[0], sf_sizes[1] * sf_sizes[2])
    s_mean = torch.mean(sfv, 1)
    s_mean = s_mean.unsqueeze(1).expand_as(sfv)
    sfv -= s_mean

    s_covm = torch.mm(sfv, sfv.t()).div((sf_sizes[1] * sf_sizes[2]) - 1)
    s_u, s_e, s_v = torch.svd(s_covm, some=False)

    s_k = sf_sizes[0]
    for i in range(sf_sizes[0]):
        if s_e[i] < 0.00001:
            s_k = i
            break
    s_d = (s_e[0:s_k]).pow(0.5)
    ccsf = torch.mm(torch.mm(torch.mm(s_v[:, 0:s_k], torch.diag(s_d)), s_v[:, 0:s_k].t()), whitened)

    ccsf += s_mean.resize_as_(ccsf)
    return ccsf.float()
