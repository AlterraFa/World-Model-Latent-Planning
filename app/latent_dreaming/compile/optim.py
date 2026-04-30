import torch
from utils.schedulers import CosineWSDSchedule, CosineWDSchedule, CosineSchedule
from utils.logger import Logger, log_parameters
logger = Logger(__name__)

def compile_opt(model: torch.nn.Module, optim_config: dict, mixed_precision = False):
    
    ema    = optim_config.get("ema", [0.99925, 0.99925])
    ipe    = optim_config.get("ipe", 200)
    warmup = optim_config.get("warmup", 10)
    anneal = optim_config.get('anneal', 30)
    num_epochs = optim_config.get('epochs', 50)
    betas = optim_config.get('betas', (0.9, 0.999))
    eps = optim_config.get('eps', 1e-8)

    lr_cfg = optim_config.get("learning_rate", {})
    lr_scale = lr_cfg.get('lr_scale', 1.0)
    ref_lr   = lr_cfg.get('lr', 0.000125)
    start_lr = lr_cfg.get('start_lr', 4.5e-05)
    final_lr = lr_cfg.get('final_lr', 0.0)
    
    wd_cfg = optim_config.get('weight_decay', {})
    wd_exclude = wd_cfg.get('wd_exclude', True)
    wd         = wd_cfg.get('weight_decay', 0.04)
    final_wd   = wd_cfg.get('final_weight_decay', 0.0)
    
    
    param_groups = [
        {
            "params": (p for n, p in model.diffuser.named_parameters() if ("bias" not in n) and (len(p.shape) != 1)),
            "lr_scale": lr_scale,
        },
        {
            "params": (p for n, p in model.diffuser.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
            "WD_exclude": wd_exclude,
            "weight_decay": 0,
        },
    ]
    
    optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=eps)
    scheduler = CosineWSDSchedule(
        optimizer,
        warmup_steps=int(warmup * ipe),
        anneal_steps=int(anneal * ipe),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(num_epochs * ipe),
    )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(num_epochs * ipe),
    )
    
    ema_scheduler = (
        ema[0] + i * (ema[1] - ema[0]) / (ipe * num_epochs)
        for i in range(int(ipe * num_epochs) + 1)
    )
    scaler = torch.amp.GradScaler() if mixed_precision else None
    
    log_parameters(logger, "Optimization", optim_config)

    return optimizer, scaler, scheduler, wd_scheduler, ema_scheduler