import numpy as np
from dataclasses import fields

def quaternion_yaw(qw, qx, qy, qz):
    """Compute yaw (heading) from quaternion components."""
    return np.arctan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy ** 2 + qz ** 2),
    )


def ego2local(ego_pose_seq):
    """
    Project global ego poses into a local frame.
    Supports: 
      - List of EgoPose dataclass objects
      - NumPy array of shape (N, 17) following the EgoPose field order
    """
    if ego_pose_seq is None or len(ego_pose_seq) == 0:
        return []

    # 1. Convert to NumPy array for uniform processing
    # If it's a list of dataclasses, extract values. 
    # If it's already a numpy array, use it directly.
    if isinstance(ego_pose_seq, list):
        # Create a mapping of field names to ensure order if necessary, 
        # but a simple list comprehension works given the dataclass definition.
        data = []
        for ep in ego_pose_seq:
            if ep is None:
                data.append([np.nan] * 17) # 17 is the number of fields in EgoPose
            else:
                # Extracts attributes in the order defined in the dataclass
                data.append([getattr(ep, f.name) for f in fields(ep)])
        data = np.array(data)
    else:
        data = np.asarray(ego_pose_seq)

    # 2. Extract relevant columns based on EgoPose ordering
    # Index 1:x, 2:y | Index 4:qw, 5:qx, 6:qy, 7:qz
    x_coords = data[:, 1]
    y_coords = data[:, 2]
    q_w, q_x, q_y, q_z = data[:, 4], data[:, 5], data[:, 6], data[:, 7]

    # 3. Find first valid index (not NaN)
    valid_mask = ~np.isnan(x_coords)
    if not np.any(valid_mask):
        return [None] * len(ego_pose_seq)
    
    first_idx = np.where(valid_mask)[0][0]

    # 4. Calculate reference heading and rotation
    ref_x, ref_y = x_coords[first_idx], y_coords[first_idx]
    
    # Calculate yaw using the quaternion_yaw logic
    qw, qx, qy, qz = q_w[first_idx], q_x[first_idx], q_y[first_idx], q_z[first_idx]
    yaw0 = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy**2 + qz**2))
    
    # Local frame: +y is heading. Angle to rotate is (-yaw0 + pi/2)
    angle = -yaw0 + np.pi / 2
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    # 5. Vectorized transformation
    dx = x_coords - ref_x
    dy = y_coords - ref_y
    
    x_local = cos_a * dx - sin_a * dy
    y_local = sin_a * dx + cos_a * dy

    # 6. Format output back to list of (x, y) or None
    transformed = []
    for i in range(len(data)):
        if not valid_mask[i]:
            transformed.append(None)
        else:
            transformed.append((float(x_local[i]), float(y_local[i])))

    return transformed