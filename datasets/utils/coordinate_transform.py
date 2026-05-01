import torch

def quaternion_yaw(qw, qx, qy, qz):
    """Compute yaw (heading) from quaternion components using PyTorch."""
    # Using torch.atan2 (standard torch alias for arctan2)
    return torch.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy ** 2 + qz ** 2),
    )

def ego2local(ego_pose_seq):
    """
    Converts nuPlan ego poses to a local PyTorch tensor.
    Input: torch.Tensor of shape (B, T, 7) or (T, 7)
           Columns: [x, y, z, qw, qx, qy, qz]
    Output: torch.Tensor of shape (B, T, 2) or (T, 2) -> [x_local, y_local]
    """
    # 1. Ensure input is a tensor and keep it on its original device (GPU/CPU)
    if not isinstance(ego_pose_seq, torch.Tensor):
        data = torch.as_tensor(ego_pose_seq)
    else:
        data = ego_pose_seq

    # 2. Extract Dimensions using ellipses to handle both (B, T, 7) and (T, 7)
    x = data[..., 0]
    y = data[..., 1]
    # Skip z at index 2
    qw = data[..., 3]
    qx = data[..., 4]
    qy = data[..., 5]
    qz = data[..., 6]
    
    # Calculate yaw (heading) for all frames
    yaw = quaternion_yaw(qw, qx, qy, qz)
    
    # 3. Determine Reference (First frame of the sequence)
    if data.ndim == 3:
        # Batched input: (B, T, 7)
        # Use [:, 0:1] to keep dimensions (B, 1) for automatic broadcasting
        ref_x = x[:, 0:1]
        ref_y = y[:, 0:1]
        ref_yaw = yaw[:, 0:1]
    else:
        # Single sequence: (T, 7)
        ref_x = x[0]
        ref_y = y[0]
        ref_yaw = yaw[0]

    # 4. Rotation Logic (Vectorized)
    # Target: Local +y axis points toward the initial heading
    angle = -ref_yaw + (torch.pi / 2.0)
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)

    # Translation relative to the first frame
    dx = x - ref_x
    dy = y - ref_y

    # Rotation matrix transformation:
    # x' = x*cos - y*sin
    # y' = x*sin + y*cos
    x_local = cos_a * dx - sin_a * dy
    y_local = sin_a * dx + cos_a * dy

    # 5. Stack into final tensor [..., 2]
    # Result stays on the same device as input (e.g., CUDA)
    return torch.stack([x_local, y_local], dim=-1)