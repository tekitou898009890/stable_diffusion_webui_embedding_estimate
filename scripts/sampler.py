import math

from scipy import integrate
import torch
from torch import nn
from torchdiffeq import odeint
import torchsde
from tqdm.auto import trange, tqdm

from modules import shared, devices, prompt_parser
from scripts.embedding import make_temp_embedding, get_conds_with_caching
from k_diffusion import utils

# from . import utils


def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7., device='cpu'):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)


def get_sigmas_exponential(n, sigma_min, sigma_max, device='cpu'):
    """Constructs an exponential noise schedule."""
    sigmas = torch.linspace(math.log(sigma_max), math.log(sigma_min), n, device=device).exp()
    return append_zero(sigmas)


def get_sigmas_polyexponential(n, sigma_min, sigma_max, rho=1., device='cpu'):
    """Constructs an polynomial in log sigma noise schedule."""
    ramp = torch.linspace(1, 0, n, device=device) ** rho
    sigmas = torch.exp(ramp * (math.log(sigma_max) - math.log(sigma_min)) + math.log(sigma_min))
    return append_zero(sigmas)


def get_sigmas_vp(n, beta_d=19.9, beta_min=0.1, eps_s=1e-3, device='cpu'):
    """Constructs a continuous VP noise schedule."""
    t = torch.linspace(1, eps_s, n, device=device)
    sigmas = torch.sqrt(torch.exp(beta_d * t ** 2 / 2 + beta_min * t) - 1)
    return append_zero(sigmas)


def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / utils.append_dims(sigma.to(devices.device), x.ndim)


def get_ancestral_step(sigma_from, sigma_to, eta=1.):
    """Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step."""
    if not eta:
        return sigma_to, 0.
    sigma_up = min(sigma_to, eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5)
    sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
    return sigma_down, sigma_up


def default_noise_sampler(x):
    return lambda sigma, sigma_next: torch.randn_like(x)


class BatchedBrownianTree:
    """A wrapper around torchsde.BrownianTree that enables batches of entropy."""

    def __init__(self, x, t0, t1, seed=None, **kwargs):
        t0, t1, self.sign = self.sort(t0, t1)
        w0 = kwargs.get('w0', torch.zeros_like(x))
        if seed is None:
            seed = torch.randint(0, 2 ** 63 - 1, []).item()
        self.batched = True
        try:
            assert len(seed) == x.shape[0]
            w0 = w0[0]
        except TypeError:
            seed = [seed]
            self.batched = False
        self.trees = [torchsde.BrownianTree(t0, w0, t1, entropy=s, **kwargs) for s in seed]

    @staticmethod
    def sort(a, b):
        return (a, b, 1) if a < b else (b, a, -1)

    def __call__(self, t0, t1):
        t0, t1, sign = self.sort(t0, t1)
        w = torch.stack([tree(t0, t1) for tree in self.trees]) * (self.sign * sign)
        return w if self.batched else w[0]


class BrownianTreeNoiseSampler:
    """A noise sampler backed by a torchsde.BrownianTree.

    Args:
        x (Tensor): The tensor whose shape, device and dtype to use to generate
            random samples.
        sigma_min (float): The low end of the valid interval.
        sigma_max (float): The high end of the valid interval.
        seed (int or List[int]): The random seed. If a list of seeds is
            supplied instead of a single integer, then the noise sampler will
            use one BrownianTree per batch item, each with its own seed.
        transform (callable): A function that maps sigma to the sampler's
            internal timestep.
    """

    def __init__(self, x, sigma_min, sigma_max, seed=None, transform=lambda x: x):
        self.transform = transform
        t0, t1 = self.transform(torch.as_tensor(sigma_min)), self.transform(torch.as_tensor(sigma_max))
        self.tree = BatchedBrownianTree(x, t0, t1, seed)

    def __call__(self, sigma, sigma_next):
        t0, t1 = self.transform(torch.as_tensor(sigma)), self.transform(torch.as_tensor(sigma_next))
        return self.tree(t0, t1) / (t1 - t0).abs().sqrt()

def sample_dpmpp_sde(model, x, step, sigmas, c=None,uc=None,cfg_scale=None,extra_args=None,image_conditioning=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, r=1 / 2, optimizer=None, loss_fn=None, input_tensor=None, gr_ptype="Prompt"):

    # xo = x
    # EMBEDDING_NAME = 'embedding_estimate'
    # cache = {}
    # uc_cache = [None,None]


    # optimizer.zero_grad() 

    # for i in trange(len(sigmas) - 1, disable=disable):
    # for i in trange(2, disable=disable):

    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    s_in = x.new_ones([x.shape[0]])

    def sampler(model,x,i,sigmas,denoised,noise_sampler,extra_args):
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0:
            # Euler method
            d = to_d(x, sigmas[i], denoised)
            dt = sigmas[i + 1] - sigmas[i]
            x = x + d * dt
        else:
            # DPM-Solver++
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t
            s = t + h * r
            fac = 1 / (2 * r)

            # Step 1
            sd, su = get_ancestral_step(sigma_fn(t), sigma_fn(s), eta)
            s_ = t_fn(sd)
            x_2 = (sigma_fn(s_) / sigma_fn(t)) * x - (t - s_).expm1() * denoised
            x_2 = x_2 + noise_sampler(sigma_fn(t), sigma_fn(s)) * s_noise * su
            denoised_2 = model(x_2, sigma_fn(s) * s_in, **extra_args)

            # Step 2
            sd, su = get_ancestral_step(sigma_fn(t), sigma_fn(t_next), eta)
            t_next_ = t_fn(sd)
            denoised_d = (1 - fac) * denoised + fac * denoised_2
            x = (sigma_fn(t_next_) / sigma_fn(t)) * x - (t - t_next_).expm1() * denoised_d
            x = x + noise_sampler(sigma_fn(t), sigma_fn(t_next)) * s_noise * su
        return x
    
    # def closure(model,x,xo,sigmas,optimizer,loss_fn,input_tensor,gr_ptype,c,uc,tc,tuc,image_conditioning,cfg_scale,noise_sampler,extra_args):

    """DPM-Solver++ (stochastic)."""
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args
    
    
    step_multiplier = 2 # for DPM
    gr_step = len(sigmas) - 1

        # 勾配を初期化

    # optimizer.zero_grad() 

    # nonlocal uc_cache
    # nonlocal cache
        
    # 入力データからembedingを作成
    # make_temp_embedding(EMBEDDING_NAME,input_tensor.squeeze(0).to(device=devices.device,dtype=torch.float16),cache) #ある番号ごとに保存機能も後で追加か

    # if gr_ptype == "Prompts":
    #     prompt = prompt_parser.get_multicond_learned_conditioning(shared.sd_model, [EMBEDDING_NAME], gr_step * step_multiplier) # 入力テンソルをモデルに通す -> embeding登録してプロンプトから通す
    #     empty_prompt = get_conds_with_caching(prompt_parser.get_learned_conditioning, [''], gr_step * step_multiplier,uc_cache)
    # else:
    #     prompt = prompt_parser.get_learned_conditioning(shared.sd_model, [EMBEDDING_NAME], gr_step * step_multiplier) # 入力テンソルをモデルに通す -> embeding登録してプロンプトから通す
    #     empty_prompt = get_conds_with_caching(prompt_parser.get_multicond_learned_conditioning, [''], gr_step * step_multiplier,uc_cache)
            
    
    output_c  = c
    output_uc = uc

    # output_c  = prompt if gr_ptype == "Prompts" else empty_prompt
    # output_uc = empty_prompt if gr_ptype == "Prompts" else prompt
    # target_c = tc 
    # target_uc = tuc 

    extra_args={
        'cond': output_c, 
        'image_cond': image_conditioning, 
        'uncond': output_uc, 
        'cond_scale': cfg_scale,
        's_min_uncond': 0
    }
    
    denoised = model(x, sigmas[step] * s_in, **extra_args)
    
    x  = sampler(model,x ,step,sigmas,denoised       ,noise_sampler,extra_args)

        # output = x
        # target = xo
        
        # loss = shared.sd_model.get_loss(output, target, mean=False).mean([1, 2, 3])
        # loss = loss_fn(denoised,denoised_target)

        # logvar_t = shared.sd_model.logvar[i]
        # loss = loss / torch.exp(logvar_t) + logvar_t

        # loss = shared.sd_model.l_simple_weight * loss.mean() 

        # loss_vlb = shared.sd_model.get_loss(loss, target, mean=False).mean(dim=(1, 2, 3))
        # loss_lvlb_weights = shared.sd_model.lvlb_weights.to(devices.device)
        # loss_vlb = (loss_lvlb_weights[i] * loss_vlb).mean()
        # loss += (shared.sd_model.original_elbo_weight * loss_vlb)

        # loss.backward()

        # if i == gr_step - 1:
        # print(f"\nIteration {i}, Loss: {loss.item()}")


        #     return x,xo #loss
        # x,xo = closure(model,x,xo,sigmas,optimizer,loss_fn,input_tensor,gr_ptype,c,uc,image_conditioning,cfg_scale,noise_sampler,extra_args)
        # optimizer.step(lambda:closure(model,x,xo,sigmas,optimizer,loss_fn,input_tensor,gr_ptype,c,uc,tc,tuc,image_conditioning,cfg_scale,noise_sampler,extra_args))
    
    # loss = loss_fn(x,xo)
        

    return x
