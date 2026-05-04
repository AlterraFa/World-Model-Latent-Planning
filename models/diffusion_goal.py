import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from omegaconf import ListConfig
from tqdm.auto import tqdm
from .diffusion_wm import DiffusionWM

class DiffusionGoal(DiffusionWM):
    def __init__(self, *, diffuser_config, encoder_config, compile = False, timescale=1, goal_drop=0.0):
        super().__init__(diffuser_config=diffuser_config, encoder_config=encoder_config, compile = compile, timescale=timescale)
        self.goal_drop = goal_drop

    def drop_goal(self, goal):
        mask = torch.rand(goal.shape[0], device = goal.device) < self.goal_drop
        mask = mask.to(goal.device)
        goal[mask] = torch.full((goal.shape[1], ), torch.nan, device = goal.device, dtype = goal.dtype)
        return goal

    def forward(self, x: torch.Tensor, noise: torch.Tensor, goal: torch.Tensor, t: torch.Tensor, frame_rate: torch.Tensor):
        if self.goal_drop > 0.0 and self.training:
            goal = self.drop_goal(goal)
            
        context = self.encode_frames(x)
        pred = self.diffuser(noise, context, t, frame_rate = frame_rate, goal = goal)
        
        return pred
    
    def roll_out(self, images: torch.Tensor, n_hallucination: int, chunk_gen: int = 1, goal = None, eta = 0.0, NFE = 20):
        x_c = self.encode_frames(images)
        
        x_all = [x_c.clone()]
        if goal is None:
            goal = torch.full((images.shape[0], 2), torch.nan, device = images.device)

        num_steps = math.ceil(n_hallucination // chunk_gen)
        for idx in range(num_steps):
            x_last_t = self.sample(z = x_c, chunk_gen = chunk_gen, goal = goal, NFE = NFE, eta = eta, frame_rate = None)
            x_all.append(x_last_t)
            x_c = torch.cat([x_c[:, chunk_gen:], x_last_t], dim = 1)
        
        x_all = torch.cat(x_all, dim=1)[:, :chunk_gen+n_hallucination]

        return x_all
        
    @torch.no_grad()    
    def sample(self, z: torch.Tensor, chunk_gen: int, goal: torch.Tensor, eta = 0.0, NFE = 20, frame_rate = None):
        self.diffuser.eval()
        net = self.diffuser
        device = next(net.parameters()).device

        if frame_rate is None:
            frame_rate = torch.full_like( torch.ones((z.shape[0],)), 5, device=device)
        
        input_h, input_w = net.input_size[0], net.input_size[1] if isinstance(net.input_size, (list, tuple, ListConfig)) else net.input_size
        noise = torch.randn(z.shape[0], chunk_gen, net.embed_dim, input_h, input_w, device=device)
        
        t_steps = torch.linspace(1, 0, NFE + 1, device = device)
        with torch.no_grad():
            for i in range(NFE):
                t = t_steps[i].repeat(noise.shape[0])
                neg_v = net(noise, z, t = t * self.timescale, frame_rate = frame_rate, goal = goal)
                dt = t_steps[i] - t_steps[i + 1]
                dw = torch.randn(noise.size()).to(noise.device) * torch.sqrt(dt)
                diffusion = dt
                noise = noise + neg_v * dt + eta * torch.sqrt(2 * diffusion) * dw
                
        target_t = noise
        return target_t

if __name__ == "__main__":
    
    device = torch.device('cuda')

    from omegaconf import OmegaConf
    
    cfg = OmegaConf.load("./cfgs/latent_dreaming/diffusion-goal.yaml")
    OmegaConf.register_new_resolver("div", lambda x, y: int(x / y))
    model_cfg = cfg['model']['params']
    
    model = DiffusionGoal(diffuser_config = model_cfg['diffuser_config'], encoder_config = model_cfg['encoder_config'], timescale = model_cfg['timescale'], goal_drop = model_cfg['goal_drop']).to(device)
    
    common_cfg = cfg['common']
    B = 1
    crop_size = common_cfg['crop_size']
    fpcs = common_cfg['fpcs']
    
    inp = torch.randn((B, 3, 6, crop_size, crop_size), device = device)
    inp_goal = torch.randn((B, 2), device = device)

    with torch.no_grad(), torch.autocast(device.type, torch.bfloat16):
        # z_target = model.encode_frames(inp_target)
        
        # t = torch.rand((B,), device=inp_ctx.device)
        # frame_rate = torch.full((B, ), 5)
        # target_t, noise = add_noise(z_target, t)

        # pred = model(inp_ctx, target_t, inp_goal, t, frame_rate)
        
        # velocity = A_(t) * z_target + B_(t) * noise
        
        # loss = ((pred.float() - velocity.float()) ** 2).mean()
        model.roll_out(inp, n_hallucination = 25, chunk_gen = 5)