import torch
import numpy as np
import scipy
from scipy import stats
import math
def get_sigmas(config):
    if config.model.sigma_dist == 'geometric':
        sigmas = torch.tensor(
            np.exp(np.linspace(np.log(config.model.sigma_begin), np.log(config.model.sigma_end),
                               config.model.num_classes))).float().to(config.device)
    elif config.model.sigma_dist == 'uniform':
        sigmas = torch.tensor(
            np.linspace(config.model.sigma_begin, config.model.sigma_end, config.model.num_classes)
        ).float().to(config.device)

    else:
        raise NotImplementedError('sigma distribution not supported')

    return sigmas

def get_gammas(config):
    if config.model.gamma_dist == 'geometric':
        gammas = torch.tensor(
            np.exp(np.linspace(np.log(config.model.gamma_begin), np.log(config.model.gamma_end),
                               config.model.num_classes))).float().to(config.device)
    elif config.model.gamma_dist == 'uniform':
        gammas = torch.tensor(
            np.linspace(config.model.gamma_begin, config.model.gamma_end, config.model.num_classes)
        ).float().to(config.device)

    else:
        raise NotImplementedError('gamma distribution not supported')

    return gammas


@torch.no_grad()
def anneal_Langevin_dynamics(x_mod, scorenet, sigmas, n_steps_each=200, step_lr=0.000008,
                             final_only=False, verbose=False, denoise=True):
    images = []

    with torch.no_grad():
        for c, sigma in enumerate(sigmas):
            labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
            labels = labels.long()
            step_size = step_lr * (sigma / sigmas[-1]) ** 2
            for s in range(n_steps_each):
                grad = scorenet(x_mod, labels)

                noise = torch.randn_like(x_mod)
                grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
                noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()
                x_mod = x_mod + step_size * grad + noise * np.sqrt(step_size * 2)

                image_norm = torch.norm(x_mod.view(x_mod.shape[0], -1), dim=-1).mean()
                snr = np.sqrt(step_size / 2.) * grad_norm / noise_norm
                grad_mean_norm = torch.norm(grad.mean(dim=0).view(-1)) ** 2 * sigma ** 2

                if not final_only:
                    images.append(x_mod.to('cpu'))
                if verbose:
                    print("level: {}, step_size: {}, grad_norm: {}, image_norm: {}, snr: {}, grad_mean_norm: {}".format(
                        c, step_size, grad_norm.item(), image_norm.item(), snr.item(), grad_mean_norm.item()))


                    print("recovered value: class: {}, step_size: {}, mean {}, max {}".format(c, step_size, x_mod.abs().mean(),
                                                                         x_mod.abs().max()))
        if denoise:
            last_noise = (len(sigmas) - 1) * torch.ones(x_mod.shape[0], device=x_mod.device)
            last_noise = last_noise.long()
            x_mod = x_mod + sigmas[-1] ** 2 * scorenet(x_mod, last_noise)
            images.append(x_mod.to('cpu'))

        if final_only:
            return [x_mod.to('cpu')]
        else:
            return images

@torch.no_grad()
def anneal_Langevin_dynamics_linear_inverse(x_mod,y_observe, y_lower, y_upper, num_bits, delta, scorenet, sigmas, H, noise_var, image_size,
                                        n_steps_each=100, step_lr=0.000008, likelihood_scale = 1, quantization = False, NumIter_EP = 3, final_only=False):
    """
    pasteriror sampling for linear inverse, particularly including quantized CS.
    """
    images = []

    A_mat = torch.matmul(H,H.t())
    D_common, U_common = torch.linalg.eigh(A_mat)

    x_mod = x_mod.view(-1, x_mod.shape[2], image_size, image_size)
    with torch.no_grad():
        for c, sigma in enumerate(sigmas):
            labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
            labels = labels.long()
            step_size = step_lr * (sigma / sigmas[-1]) ** 2
           
            for s in range(n_steps_each):
                images.append(x_mod.to('cpu'))
                noise = torch.randn_like(x_mod) * np.sqrt(step_size * 2)
                grad = scorenet(x_mod, labels)

                if quantization:
                    if num_bits == 1:
                        # likelihood_grad = likelihood_score_onebit_diag(y_observe, x_mod, H, noise_var, sigma, diag_values) # this is QCS-SGM
                        likelihood_grad = likelihood_score_onebit_EP(y_observe, x_mod, H, noise_var, sigma, U_common, D_common,NumIter_EP)
                    else:
                        # likelihood_grad = likelihood_score_multibit_diag(y_observe, y_lower, y_upper, x_mod, H, noise_var, sigma, diag_values) # this is QCS-SGM
                        likelihood_grad = likelihood_score_multibit_EP(y_observe, y_lower, y_upper, x_mod, H, noise_var, sigma, U_common, D_common, NumIter_EP)
                else:
                    raise ValueError("quantization is needed")

                if torch.norm(likelihood_grad) > 0:
                    likelihood_scale1 = likelihood_scale*torch.norm(grad)/torch.norm(likelihood_grad)
                else:
                    likelihood_scale1 = 1
                x_mod = x_mod + step_size * (grad + likelihood_scale1 * likelihood_grad) + noise
                print("prior grad: class: {}, step_size: {}, mean {}, max {}".format(c, step_size, grad.abs().mean(),
                                                                         grad.abs().max()))
                print("likelihood grad: class: {}, step_size: {}, mean {}, max {}".format(c, step_size, likelihood_grad.abs().mean(),
                                                                         likelihood_grad.abs().max()))

                print("recovered value: class: {}, step_size: {}, mean {}, max {}".format(c, step_size, x_mod.abs().mean(),
                                                                         x_mod.abs().max()))

       
        if final_only:
            return [x_mod.to('cpu')]
        else:
            return images

def mat_by_vec(H, v):
    v1 = v.view(v.shape[0],-1,1)
    result = torch.matmul(H,v1)
    result = result.view(v.shape[0],H.shape[0])

    return result


def likelihood_score_onebit_diag(y, x_mod, H, noise_var,gamma,diag_values):
    z = mat_by_vec(H, x_mod)
    phi_z = torch.erfc(-z/torch.sqrt(2*(noise_var + diag_values * gamma**2)))/2
    phi_z = torch.clamp(phi_z, min=1e-6, max=1-1e-6)  # numerical protection
    Gauss_value = 1/torch.sqrt(2*(noise_var + diag_values * gamma**2)*np.pi)*torch.exp(-z**2/2/(noise_var + diag_values * gamma**2))
    temp = mat_by_vec(H.t(), ((1+y)/2/phi_z - (1-y)/2/(1-phi_z))*Gauss_value)
    grad = temp.view(-1, x_mod.shape[1], x_mod.shape[2], x_mod.shape[2])
    return grad

def likelihood_score_multibit_diag(y,y_lower, y_upper, x_mod, H, noise_var,gamma,diag_values):
    z = mat_by_vec(H, x_mod)
    u_tilde = (z - y_upper)/torch.sqrt(noise_var + diag_values * gamma**2)
    l_tilde = (z - y_lower)/torch.sqrt(noise_var + diag_values * gamma**2)

    Gauss_value = (1/np.sqrt(2*np.pi) * (torch.exp(-u_tilde**2/2) - torch.exp(-l_tilde**2/2))/torch.sqrt(noise_var + diag_values * gamma**2))
    Demominator_part = (torch.erfc(-u_tilde/np.sqrt(2))/2 - torch.erfc(-l_tilde/np.sqrt(2))/2)
    Demominator_part = torch.clamp(Demominator_part, min= -1, max=-1e-6)
    temp = mat_by_vec(H.t(), Gauss_value/Demominator_part)

    grad = temp.view(-1, x_mod.shape[1], x_mod.shape[2], x_mod.shape[2])
    return grad


def H(u):
    return 1 / 2 * torch.erfc(u / np.sqrt(2))

Min_value = 1e-6

def DlnH(u):
    ret = -torch.exp(-u ** 2 / 2) / np.sqrt(2 * np.pi) / torch.clip(H(u), Min_value, 1-Min_value)
    torch.nan_to_num(ret, nan=-float(u.cpu().numpy()[0]))
    return ret

def DlnHc(u):
    ret = torch.exp(-u ** 2 / 2) / np.sqrt(2 * np.pi) / torch.clip((1 - H(u)), Min_value, 1-Min_value)
    torch.nan_to_num(ret, nan=-float(u.cpu().numpy()[0]))
    return ret

def DDlnH(u):
    ret = u * torch.exp(-u ** 2 / 2) / np.sqrt(2 * np.pi) / torch.clip(H(u), Min_value, 1-Min_value) - DlnH(u) ** 2
    torch.nan_to_num(ret, nan=-1)
    return ret

def DDlnHc(u):
    ret = -u * torch.exp(-u ** 2 / 2) / np.sqrt(2 * np.pi) /torch.clip((1 - H(u)), Min_value, 1-Min_value) - DlnHc(u) ** 2
    torch.nan_to_num(ret, nan=-1)
    return ret

def phi(x, mu = 0, sigma = 1):
    return torch.erfc(-(x - mu) / sigma / np.sqrt(2)) / 2

def fun_positive(u,l):
    result = (torch.exp(-(u**2/2 - l**2/2)) - 1)/((1/l - 1/l**3 + 3/l**5) - torch.exp(-(u**2/2 - l**2/2))*(1/u - 1/u**3 + 3/u**5))
    return torch.nan_to_num(result, nan=1)

def fun2_positive(u,l): 
    result =  (u*torch.exp(-(u**2/2 - l**2/2)) - l)/((1/l - 1/l**3 + 3/l**5) - torch.exp(-(u**2/2 - l**2/2))*(1/u - 1/u**3 + 3/u**5))
    return torch.nan_to_num(result, nan=1)

def erfcx(x):
    result = torch.special.erfcx(x)
    result = torch.clip(result, 1e-6, 1e6)
    torch.nan_to_num(result, nan=1)   
    return result

def likelihood_score_onebit_EP(y_full, x_mod, H, noise_var, gamma, U_c, D_c, NumIter_EP = 10, damp =0.02):
    z_batch = mat_by_vec(H, x_mod)
    y_batch = y_full

    M = H.shape[0]
    Batch = x_mod.shape[0]

    D = 1/(gamma ** 2 * D_c + noise_var)
    U = U_c

    grad_value = torch.zeros(x_mod.shape).to(x_mod.device)

    for it in range(Batch):
        y = y_batch[it, :].reshape(-1,1)
        b = z_batch[it,:].reshape(-1,1)

        hF = torch.zeros((M, 1)).to(x_mod.device)
        lamF = torch.ones(1).to(x_mod.device)

        for iter in range(NumIter_EP):
            khF = hF
            klamF = lamF

            # Factorized part
            mF_new = hF / lamF - 1 / torch.sqrt(lamF) * ((1 + y) / 2 * DlnH(-torch.sqrt(lamF) * b - hF / torch.sqrt(lamF)) + (1 - y) / 2 * DlnHc(
                -torch.sqrt(lamF) * b - hF / torch.sqrt(lamF)))

            chiF_full = 1 / lamF + 1 / lamF * ((1 + y) / 2 * DDlnH(-torch.sqrt(lamF) * b - hF / torch.sqrt(lamF)) + (1 - y) / 2 * DDlnHc(
                -torch.sqrt(lamF) * b - hF / torch.sqrt(lamF)))

            chiF_new = chiF_full[torch.where(chiF_full>0)].mean()


            # damping
            if iter > 0:
                mF = (1-damp) * mF_new + damp * mF
                chiF= (1 - damp) * chiF_new + damp * chiF
            else:
                mF = mF_new
                chiF = chiF_new

            # Factorized part
            # Separator
            hG_new = mF / chiF - hF
            lamG_new = 1 / chiF - lamF
            lamG_new = max(Min_value, lamG_new)

            # damping
            if iter > 0:
                hG = (1 - damp) * hG_new + damp * hG
                lamG = (1 - damp) * lamG_new + damp * lamG
            else:
                hG = hG_new
                lamG = lamG_new

            # Gaussian part
            G_common = torch.matmul(U, torch.matmul(torch.diag(1 / (lamG + D)), U.t()))
            mG = torch.matmul(G_common, hG)
            chiG_mat = (1 / (lamG + D))
            chiG = chiG_mat[torch.where(chiG_mat>0)].mean()

            # Separator
            hF_new = mG / chiG - hG
            lamF_new = 1 / chiG - lamG

            # damping
            if iter > 0:
                hF = (1 - damp) * hF_new + damp * hF
                lamF = (1 - damp) * lamF_new + damp * lamF
            else:
                hF = hF_new
                lamF = lamF_new


            ERR = abs(hF - khF).mean() + abs(lamF - klamF).mean()
            if ERR < 1.0e-5:
                break
        grad = -torch.sqrt(lamF) * ((1 + y) / 2 * DlnH(-torch.sqrt(lamF) * b - hF / torch.sqrt(lamF)) + (1 - y) / 2 * DlnHc(
            -torch.sqrt(lamF) * b - hF / torch.sqrt(lamF)))
        grad1 = torch.matmul(H.t(), grad)
        grad1 = grad1.reshape(1, x_mod.shape[1], x_mod.shape[2], x_mod.shape[2])
        grad_value[it, :, :, :] = grad1

    return grad_value

def likelihood_score_multibit_EP(y_full, y_lower_full, y_upper_full, x_mod, H, noise_var, gamma, U_c, D_c,
                                 NumIter_EP=10, damp=0.02):
    z_batch = mat_by_vec(H, x_mod)
    y_batch = y_full

    M = H.shape[0]
    Batch = x_mod.shape[0]

    D = 1 / (gamma ** 2 * D_c + noise_var)
    U = U_c

    grad_value = torch.zeros(x_mod.shape).to(x_mod.device)
    
    for it in range(Batch):
        y = y_batch[it, :].reshape(-1, 1)
        b = z_batch[it, :].reshape(-1, 1)

        y_lower = y_lower_full[it, :].reshape(-1, 1)
        y_upper = y_upper_full[it, :].reshape(-1, 1)
        hF = torch.zeros((M, 1)).to(x_mod.device)
        lamF = torch.ones(1).to(x_mod.device)

        for it_ep in range(NumIter_EP):
            khF = hF
            klamF = lamF
            # Factorized part
            u_tilde = - torch.sqrt(lamF) * b - hF / torch.sqrt(lamF) + y_upper * torch.sqrt(lamF)
            l_tilde = - torch.sqrt(lamF) * b - hF / torch.sqrt(lamF) + y_lower * torch.sqrt(lamF)

            mF_new = hF / lamF - 1 / torch.sqrt(lamF * 2 * math.pi) * (
                        torch.exp(-u_tilde ** 2 / 2) - torch.exp(-l_tilde ** 2 / 2)) / torch.clamp(phi(u_tilde) - phi(l_tilde), Min_value, 1 - Min_value)
            chiF_full = 1 / lamF - (u_tilde * torch.exp(-u_tilde ** 2 / 2) - l_tilde * torch.exp(
                -l_tilde ** 2 / 2)) / lamF / np.sqrt(2 * math.pi) / torch.clamp(phi(u_tilde) - phi(l_tilde), Min_value, 1 - Min_value) - (mF_new - hF / lamF) ** 2

            chiF_new = chiF_full[torch.where(chiF_full>0)].mean()
            
            # damping
            if it_ep > 0:
                mF = (1 - damp) * mF_new + damp * mF
                chiF = (1 - damp) * chiF_new + damp * chiF
            else:
                mF = mF_new
                chiF = chiF_new

            # Factorized part
            # Separator
            hG_new = mF / chiF - hF
            lamG_new = 1 / chiF - lamF
            lamG_new = lamG_new.mean()
            lamG_new = max(Min_value, lamG_new)

            # damping
            if it_ep > 0:
                hG = (1 - damp) * hG_new + damp * hG
                lamG = (1 - damp) * lamG_new + damp * lamG
            else:
                hG = hG_new
                lamG = lamG_new

            # Gaussian part
            G_common = torch.matmul(U, torch.matmul(torch.diag(1 / (lamG + D)), U.t()))
            mG = torch.matmul(G_common, hG.float())
            chiG = (torch.diag(G_common)).mean()

            # Separator
            hF_new = mG / chiG - hG
            lamF_new = 1 / chiG - lamG

            # damping
            if it_ep > 0:
                hF = (1 - damp) * hF_new + damp * hF
                lamF = (1 - damp) * lamF_new + damp * lamF
            else:
                hF = hF_new
                lamF = lamF_new

            ERR = abs(hF - khF).mean() + abs(lamF - klamF).mean()
            if ERR < 1.0e-3:
                break
        grad = - torch.sqrt(lamF) / np.sqrt(2 * math.pi) * (
                    torch.exp(-u_tilde ** 2 / 2) - torch.exp(-l_tilde ** 2 / 2)) / torch.clamp(phi(u_tilde) - phi(l_tilde), Min_value, 1 - Min_value)

        grad1 = torch.matmul(H.t(), grad.float())
        grad1 = grad1.reshape(1, x_mod.shape[1], x_mod.shape[2], x_mod.shape[2])
        grad_value[it, :, :, :] = grad1

    return grad_value


@torch.no_grad()
def anneal_Langevin_dynamics_inpainting(x_mod, refer_image, scorenet, sigmas, image_size,
                                        n_steps_each=100, step_lr=0.000008):
    """
    Currently only good for 32x32 images. Assuming the right half is missing.
    """

    images = []

    refer_image = refer_image.unsqueeze(1).expand(-1, x_mod.shape[1], -1, -1, -1)
    refer_image = refer_image.contiguous().view(-1, refer_image.shape[2], image_size, image_size)
    x_mod = x_mod.view(-1, x_mod.shape[2], image_size, image_size)
    cols = image_size // 2
    half_refer_image = refer_image[..., :cols]
    with torch.no_grad():
        for c, sigma in enumerate(sigmas):
            labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
            labels = labels.long()
            step_size = step_lr * (sigma / sigmas[-1]) ** 2

            for s in range(n_steps_each):
                images.append(x_mod.to('cpu'))
                corrupted_half_image = half_refer_image + torch.randn_like(half_refer_image) * sigma
                x_mod[:, :, :, :cols] = corrupted_half_image
                noise = torch.randn_like(x_mod) * np.sqrt(step_size * 2)
                grad = scorenet(x_mod, labels)
                x_mod = x_mod + step_size * grad + noise
                print("class: {}, step_size: {}, mean {}, max {}".format(c, step_size, grad.abs().mean(),
                                                                         grad.abs().max()))

        return images

@torch.no_grad()
def anneal_Langevin_dynamics_interpolation(x_mod, scorenet, sigmas, n_interpolations, n_steps_each=200, step_lr=0.000008,
                             final_only=False, verbose=False):
    images = []

    n_rows = x_mod.shape[0]

    x_mod = x_mod[:, None, ...].repeat(1, n_interpolations, 1, 1, 1)
    x_mod = x_mod.reshape(-1, *x_mod.shape[2:])

    for c, sigma in enumerate(sigmas):
        labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
        labels = labels.long()
        step_size = step_lr * (sigma / sigmas[-1]) ** 2
        for s in range(n_steps_each):
            grad = scorenet(x_mod, labels)

            noise_p = torch.randn(n_rows, x_mod.shape[1], x_mod.shape[2], x_mod.shape[3],
                                  device=x_mod.device)
            noise_q = torch.randn(n_rows, x_mod.shape[1], x_mod.shape[2], x_mod.shape[3],
                                  device=x_mod.device)
            angles = torch.linspace(0, np.pi / 2., n_interpolations, device=x_mod.device)

            noise = noise_p[:, None, ...] * torch.cos(angles)[None, :, None, None, None] + \
                        noise_q[:, None, ...] * torch.sin(angles)[None, :, None, None, None]

            noise = noise.reshape(-1, *noise.shape[2:])
            grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()
            image_norm = torch.norm(x_mod.view(x_mod.shape[0], -1), dim=-1).mean()

            x_mod = x_mod + step_size * grad + noise * np.sqrt(step_size * 2)

            snr = np.sqrt(step_size / 2.) * grad_norm / noise_norm

            if not final_only:
                images.append(x_mod.to('cpu'))
            if verbose:
                print(
                    "level: {}, step_size: {}, image_norm: {}, grad_norm: {}, snr: {}".format(
                        c, step_size, image_norm.item(), grad_norm.item(), snr.item()))


    if final_only:
        return [x_mod.to('cpu')]
    else:
        return images
