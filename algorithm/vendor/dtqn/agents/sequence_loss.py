import torch

def valid_td_steps(q_values, episode_lengths, history):

    lengths = torch.as_tensor(
        episode_lengths, dtype=torch.long, device=q_values.device
    ).reshape(-1)
    if (q_values.ndim != 2 or lengths.numel() != q_values.shape[0]
            or (lengths < 1).any() or (lengths > q_values.shape[1]).any()
            or history < 1):
        raise ValueError("invalid sequence lengths or TD history")
    positions = torch.arange(q_values.shape[1], device=q_values.device)[None, :]
    return (positions < lengths[:, None]) & (
        positions >= (lengths - history).clamp_min(0)[:, None]
    )
