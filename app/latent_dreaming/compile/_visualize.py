"""Shared latent-space visualisation for latent_dreaming train & eval.

Call ``visualize_batch`` once per epoch (rank-0 only) or from eval.main().
It:
  1. Runs one forward pass on the supplied batch to obtain per-sample loss.
  2. Picks the best (lowest) and worst (highest) loss sample.
  3. For each, encodes the target frames and runs one sample() rollout.
  4. Projects encoded + diffused patches jointly through PCA → UMAP → RGB.
  5. Opens an interactive matplotlib figure for each sample.
"""
from __future__ import annotations

import io
import math
import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import umap as umap_lib

from utils.logger import Logger

logger = Logger(__name__)

# ImageNet normalisation constants (must match compile_transform defaults)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_VIZ_SEED = 12


# ─────────────────────────────── low-level helpers ───────────────────────────

def _denorm(img_t: torch.Tensor) -> np.ndarray:
    """(C, H, W) float tensor → (H, W, C) uint8 numpy in [0, 1]."""
    arr = img_t.cpu().float().numpy().transpose(1, 2, 0)
    return np.clip(arr * _STD + _MEAN, 0.0, 1.0)


def _to_patch_seq(z: torch.Tensor) -> np.ndarray:
    """(T, D, H, W) → (T, P, D) float32 numpy."""
    T, D, H, W = z.shape
    return z.float().permute(0, 2, 3, 1).reshape(T, H * W, D).cpu().numpy()


def _joint_umap_rgb(
    enc_np: np.ndarray,
    diff_np: np.ndarray,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    up_power: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project enc and diff latents into a shared UMAP RGB space.

    Parameters
    ----------
    enc_np, diff_np : (T, P, D)

    Returns
    -------
    enc_rgb, diff_rgb : (T, H_up, W_up, 3)  float32  [0, 1]
    """
    T, P, D = enc_np.shape
    grid = int(round(math.sqrt(P)))

    flat_enc  = enc_np.reshape(-1, D)
    flat_diff = diff_np.reshape(-1, D)
    flat_all  = np.concatenate([flat_enc, flat_diff], axis=0)

    pca  = PCA(n_components=min(50, D))
    red  = pca.fit_transform(flat_all)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        reducer = umap_lib.UMAP(
        n_components=3,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
            random_state=_VIZ_SEED,
        )
        proj = reducer.fit_transform(red)

    rgb_all  = MinMaxScaler().fit_transform(proj).astype(np.float32)
    N        = T * P
    enc_rgb  = rgb_all[:N].reshape(T, grid, grid, 3)
    diff_rgb = rgb_all[N:].reshape(T, grid, grid, 3)

    def _up(g: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(g).permute(0, 3, 1, 2)
        x = F.interpolate(x, scale_factor=up_power, mode="bilinear",
                          align_corners=False)
        return x.permute(0, 2, 3, 1).numpy()

    return _up(enc_rgb), _up(diff_rgb)


# ─────────────────────────────── interactive figure ──────────────────────────

def _fig_to_chw_tensor(fig: plt.Figure, dpi: int = 120) -> torch.Tensor:
    """Render a matplotlib figure to a (C, H, W) float32 tensor in [0, 1]."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    try:
        from PIL import Image as PILImage
        arr = np.array(PILImage.open(buf).convert("RGB"), dtype=np.float32) / 255.0
    except ImportError:
        import struct, zlib  # fallback: shouldn't normally be needed
        raise ImportError("Pillow is required for training logger figure rendering. pip install Pillow")
    return torch.from_numpy(arr).permute(2, 0, 1)   # (C, H, W)


def _show_figure(
    title: str,
    ctx_images: list[np.ndarray],
    tgt_images: list[np.ndarray],
    enc_rgb: np.ndarray,
    diff_rgb: np.ndarray,
    loss_val: float,
    n_frames: int = 4,
) -> plt.Figure:
    """
    Static grid layout:
      Row 0: n_frames context images
      Row 1: n_frames target images
      Row 2: n_frames encoded UMAP RGB patches
      Row 3: n_frames diffused UMAP RGB patches
    """
    T    = enc_rgb.shape[0]
    n_c  = len(ctx_images)

    # pick evenly-spaced frame indices
    tgt_idx = [int(round(i * (T - 1) / (n_frames - 1))) for i in range(n_frames)]
    ctx_idx = [int(round(i * (n_c - 1) / (n_frames - 1))) for i in range(n_frames)]

    fig, axes = plt.subplots(4, n_frames, figsize=(4 * n_frames, 12))
    fig.suptitle(f"{title}   loss={loss_val:.5f}", fontsize=11)

    row_labels = ["context", "target", "encoded (UMAP RGB)", "diffused (UMAP RGB)"]
    for r, lbl in enumerate(row_labels):
        axes[r, 0].set_ylabel(lbl, fontsize=9)

    for col, (ci, ti) in enumerate(zip(ctx_idx, tgt_idx)):
        axes[0, col].imshow(ctx_images[ci])
        axes[0, col].set_title(f"t={ci}", fontsize=8)

        axes[1, col].imshow(tgt_images[min(ti, len(tgt_images) - 1)])
        axes[1, col].set_title(f"t={ti}", fontsize=8)

        axes[2, col].imshow(enc_rgb[ti],  interpolation="nearest")
        axes[2, col].set_title(f"t={ti}", fontsize=8)

        axes[3, col].imshow(diff_rgb[ti], interpolation="nearest")
        axes[3, col].set_title(f"t={ti}", fontsize=8)

    for ax in axes.flat:
        ax.axis("off")

    fig.tight_layout()
    return fig


# ─────────────────────────────── public API ──────────────────────────────────

@torch.no_grad()
def visualize_batch(
    *,
    world_model: torch.nn.Module,
    images: torch.Tensor,
    frames,
    device: torch.device,
    dtype: torch.dtype,
    mixed_precision: bool,
    context_length: float,
    duration: float,
    fpcs: int,
    gen_chunksz: int,
    loss_exp: float,
    randomize_goal: bool,
    sigma_min: float = 1e-6,
    nfe: int = 20,
    eta: float = 0.0,
    n_neighbors: int = 30,
    up_power: int = 16,
    epoch: int | None = None,
    log_stats=None,
) -> None:
    """
    When ``log_stats`` is provided (training): renders each figure to a
    (C,H,W) float32 tensor and calls ``log_stats.log_media``, then closes
    the figure without blocking.

    When ``log_stats`` is None (eval): shows the figures interactively.
    """
    """
    Visualise encoded vs diffused latents for the best & worst sample in a batch.

    Parameters
    ----------
    world_model  : the raw (non-DDP) model; must have .encode_frames() and .sample()
    images       : (B, C, T, H, W) already on CPU or GPU
    frames       : NuplanFrame batch
    block        : if True, call plt.show(block=True) – set False during training
                   to avoid blocking (figures appear, training resumes immediately).
    """
    from datasets.utils.coordinate_transform import ego2local


    images = images.to(device, non_blocking=True, dtype=dtype)
    B = images.shape[0]

    ego_loc    = ego2local(frames.ego_pose).to(device, dtype=dtype)
    dt         = frames.frame_timestamps.diff(n=1, dim=1).float().mean(-1) / 1e6
    frame_rate = (1.0 / (dt + 1e-6)).to(dtype=dtype, device=device)

    split_fpcs     = math.ceil(context_length / duration * fpcs)
    context_image  = images[:, :, :split_fpcs]
    target_image   = images[:, :, split_fpcs:]
    chosen_target  = target_image[:, :, :gen_chunksz]

    if randomize_goal:
        goal_idx = torch.randint(
            gen_chunksz + split_fpcs, fpcs, (B,), dtype=torch.long, device=device
        )
        goal = ego_loc[torch.arange(B, device=device), goal_idx]
    else:
        goal = ego_loc[:, gen_chunksz + split_fpcs]

    # ── per-sample loss ───────────────────────────────────────────────────
    alpha_fn = lambda t: 1.0 - t
    sigma_fn = lambda t: sigma_min + t * (1.0 - sigma_min)

    diffuse_time = torch.rand((B,), device=device, dtype=dtype)
    noise = torch.randn_like(chosen_target)  # placeholder shape, overridden below

    with torch.autocast(device.type, dtype=dtype, enabled=mixed_precision):
        latent_target = world_model.encode_frames(chosen_target)
        _noise        = torch.randn_like(latent_target)
        s             = [B] + [1] * (latent_target.dim() - 1)
        noisy_target  = (alpha_fn(diffuse_time).view(*s) * latent_target
                         + sigma_fn(diffuse_time).view(*s) * _noise)

        velocity = 1.0 * latent_target + (-(1.0 - sigma_min)) * _noise
        pred_vel = world_model(
            x=context_image,
            noise=noisy_target,
            goal=goal,
            t=diffuse_time,
            frame_rate=frame_rate,
        )
        loss_per_sample = (
            (pred_vel.float() - velocity.float()).abs() ** loss_exp
        ).mean(dim=tuple(range(1, pred_vel.ndim)))   # (B,)

    best_idx  = int(loss_per_sample.argmin().item())
    worst_idx = int(loss_per_sample.argmax().item())
    tag = f"epoch {epoch}" if epoch is not None else "batch"

    # ── per-sample rollout + UMAP figure ─────────────────────────────────
    for label, idx in (("BEST", best_idx), ("WORST", worst_idx)):

        with torch.autocast(device.type, dtype=dtype, enabled=mixed_precision):
            ctx_lat = world_model.encode_frames(context_image[idx:idx + 1])
            diffused = world_model.sample(
                z=ctx_lat,
                chunk_gen=gen_chunksz,
                goal=goal[idx:idx + 1],
                NFE=nfe,
                eta=eta,
                frame_rate=frame_rate[idx:idx + 1],
            )   # (1, T_gen, D, H_p, W_p)

        enc_np  = _to_patch_seq(latent_target[idx].float())   # (T, P, D)
        diff_np = _to_patch_seq(diffused[0].float())           # (T, P, D)

        enc_rgb, diff_rgb = _joint_umap_rgb(
            enc_np, diff_np,
            n_neighbors=n_neighbors,
            up_power=up_power,
        )

        ctx_imgs = [_denorm(context_image[idx, :, t].float())
                    for t in range(context_image.shape[2])]
        tgt_imgs = [_denorm(chosen_target[idx, :, t].float())
                    for t in range(gen_chunksz)]

        fig = _show_figure(
            title=f"{label}  (idx={idx})  {tag}",
            ctx_images=ctx_imgs,
            tgt_images=tgt_imgs,
            enc_rgb=enc_rgb,
            diff_rgb=diff_rgb,
            loss_val=loss_per_sample[idx].item(),
        )

        if log_stats is not None:
            img_tensor = _fig_to_chw_tensor(fig)
            log_tag = f"Viz/{label.capitalize()}_latent"
            log_stats.log_media(
                tag=log_tag,
                media=img_tensor,
                media_type="image",
                step=epoch,
                caption=f"{label} (idx={idx}) loss={loss_per_sample[idx]:.5f}",
            )
            plt.close(fig)

    if log_stats is None:
        plt.show(block=True)
