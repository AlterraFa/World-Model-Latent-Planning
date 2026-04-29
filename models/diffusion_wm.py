import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from einops import rearrange

from omegaconf import ListConfig
from utils.autoload_modules import instantiate_from_config

class DiffusionWM(nn.Module):
    def __init__(self, *, generator_config, encoder_config, timescale = 1.0):
        super().__init__()

        self.encoder = self._load_encoder(encoder_config)
        generator_config['params']['embed_dim'] = self.encoder.embed_dim
        self.diffuser = self._build_defuser(generator_config)
        self.timescale = timescale
        
    def _build_defuser(self, config) -> nn.Module:
        return instantiate_from_config(config)
        
    def _load_encoder(self, config) -> nn.Module:

        name = config.get('name', "Not found")
        repo = config.get('load_from', 'Not found')
        source = config.get('source', 'github')
        
        # -- Encoder has sdpa and grad checkpoint enabled
        encoder: nn.Module
        model = torch.hub.load(repo, name, trust_repo=True, source=source, pretrained=False, skip_validation=True)
        encoder = model[0]
        encoder.use_activation_checkpointing = config['use_activation_checkpointing']
        
        # -- Configure encoder with image/video parameters
        if hasattr(encoder, 'img_height'):
            encoder.img_height = config.get('crop_size', 224)
        if hasattr(encoder, 'img_width'):
            encoder.img_width = config.get('crop_size', 224)
        if hasattr(encoder, 'patch_size'):
            encoder.patch_size = config.get('patch_size', 16)
        if hasattr(encoder, 'tubelet_size'):
            encoder.tubelet_size = config.get('tubelet_size', 2)
        if hasattr(encoder, 'use_rope'):
            encoder.use_rope = config.get('use_rope', False)

        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad = False

        self.img_size = encoder.img_width
        self.patch_size = encoder.patch_size
        self.tubelet_size = encoder.tubelet_size

        return encoder
    
    @torch.no_grad()
    def encode_frames(self, images: torch.Tensor):
        orig_B, C, T, H, W = images.shape
        images = images.transpose(1, 2).flatten(0, 1).unsqueeze(2)
        z = self.encoder(images)
        B, N, D = z.shape
        T = B // orig_B
        
        H_patch = H // self.patch_size
        W_patch = W // self.patch_size

        z = rearrange(z, "(b t) (h w) d -> b t h w d", b = orig_B, t = T, h = H_patch, w = W_patch).permute(0, 1, 4, 2, 3)
        return z
        
    def forward(self, x: torch.Tensor, noise: torch.Tensor, t: torch.Tensor, frame_rate: torch.Tensor): 
        context = self.encode_frames(x)

        pred = self.diffuser(noise, context, t, frame_rate=frame_rate)
        return pred
        
    def roll_out(self, x_0, num_gen_frames=25, eta=0.0, NFE=20, num_samples=8):
        b, f = x_0.size(0), x_0.size(1)
        x_c = self.encode_frames(x_0)

        x_all = x_c.clone()
        for idx in tqdm(range(num_gen_frames), desc="Rolling out frames", leave=False):
            x_last = self.sample(images=x_c, latent=True, eta=eta, NFE=NFE, num_samples=num_samples)
            
            x_all = torch.cat([x_all, x_last.unsqueeze(1)], dim=1)
            x_c = torch.cat([x_c[:, 1:], x_last.unsqueeze(1)], dim=1)
        
        return x_all
    
    @torch.no_grad()
    def sample(self, images=None, latent=False, eta=0.0, NFE=20, num_samples=8, frame_rate=None):
        net = self.diffuser
        device = next(net.parameters()).device
        
        if images is not None:
            if not latent:
                context = self.encode_frames(images)
            else:
                context = images.clone()
        else:
            context = None

        if frame_rate is None:
            frame_rate = torch.full_like( torch.ones((num_samples,)), 5, device=device)
            
        input_h, input_w = net.input_size[0], net.input_size[1] if isinstance(net.input_size, (list, tuple, ListConfig)) else net.input_size
        target_t = torch.randn(num_samples, 1, net.embed_dim, input_h, input_w, device=device)
        
        t_steps = torch.linspace(1, 0, NFE + 1, device=device)

        with torch.no_grad():
            for i in range(NFE):
                t = t_steps[i].repeat(target_t.shape[0])
                neg_v = net(target_t, context, t=t * self.timescale, frame_rate=frame_rate)
                dt = t_steps[i] - t_steps[i+1] 
                dw = torch.randn(target_t.size()).to(target_t.device) * torch.sqrt(dt)
                diffusion = dt
                target_t  = target_t + neg_v * dt + eta *  torch.sqrt(2 * diffusion) * dw
        last_frame = target_t.clone()

        self.diffuser.train()
        return target_t.squeeze(1)
    
        
def add_noise(x, t, noise=None):
    noise = torch.randn_like(x) if noise is None else noise
    s = [x.shape[0]] + [1] * (x.dim() - 1)
    x_t = alpha(t).view(*s) * x + sigma(t).view(*s) * noise
    return x_t, noise

sigma_min = 1e-6
def alpha(t):
    return 1.0 - t

def sigma(t):
    return sigma_min + t * (1.0 - sigma_min)

def A_(t):
    return 1.0

def B_(t):
    return -(1.0 - sigma_min)

if __name__ == "__main__":
    device = torch.device('cuda')

    from omegaconf import OmegaConf
    
    cfg = OmegaConf.load("./cfgs/latent_dreaming/default.yaml")
    model_cfg = cfg['model']
    
    model = DiffusionWM(generator_config = model_cfg['diffuser'], encoder_config = model_cfg['encoder']).to(device)
    
    common_cfg = cfg['common']
    B = 2
    crop_size = common_cfg['crop_size']
    fpcs = common_cfg['fpcs']
    
    inp = torch.randn((B, 3, fpcs, crop_size, crop_size), device = device)
    inp_ctx = inp[:, :, :-1]
    inp_target = inp[:, :, -1:]

    with torch.no_grad(), torch.autocast('cuda', dtype = torch.bfloat16):

        # z_target = model.encode_frames(inp_target)
        # t = torch.rand((B,), device=inp_ctx.device)
        # frame_rate = torch.full((B, ), 5)
        # target_t, noise = add_noise(z_target, t)
        
        # pred = model(inp_ctx, target_t, t, frame_rate)

        # # -dxt/dt
        # target = A_(t) * z_target + B_(t) * noise

        # loss = ((pred.float() - target.float()) ** 2)
        
        # loss = loss.mean()
        # print(loss)

        model.roll_out(inp_ctx, num_samples = B)