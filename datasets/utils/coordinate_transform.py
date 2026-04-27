import numpy as np

def quaternion_yaw(qw, qx, qy, qz):
    """Compute yaw (heading) from quaternion components."""
    return np.arctan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy ** 2 + qz ** 2),
    )


def transform_ego_trajectory_local(ego_pose_seq):
    """Project global ego poses into a local frame.

    Local frame convention:
      - origin at first valid pose
      - +y aligned with first valid pose heading

    Returns a list with the same length as ego_pose_seq where missing poses
    are kept as None and valid poses are (x_local, y_local) tuples.
    """
    if not ego_pose_seq:
        return []

    transformed = [None] * len(ego_pose_seq)
    first_idx = next((i for i, ep in enumerate(ego_pose_seq) if ep is not None), None)
    if first_idx is None:
        return transformed

    ep0 = ego_pose_seq[first_idx]
    ref_x, ref_y = ep0.x, ep0.y
    yaw0 = quaternion_yaw(ep0.qw, ep0.qx, ep0.qy, ep0.qz)
    cos_a, sin_a = np.cos(-yaw0 + np.pi / 2), np.sin(-yaw0 + np.pi / 2)

    for idx, ep in enumerate(ego_pose_seq):
        if ep is None:
            continue
        dx, dy = ep.x - ref_x, ep.y - ref_y
        transformed[idx] = (
            cos_a * dx - sin_a * dy,
            sin_a * dx + cos_a * dy,
        )
    return transformed