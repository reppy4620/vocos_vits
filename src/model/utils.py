import torch


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(int(max_length), dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def generate_path(duration, mask):
    device = duration.device
    b, t_x, t_y = mask.shape
    cum_duration = torch.cumsum(duration, 1)
    path = torch.zeros(b, t_x, t_y, dtype=mask.dtype).to(device=device)
    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - torch.nn.functional.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
    path = path * mask
    return path


def slice_segments(x, start_indices, segment_size=4):
    ret = torch.zeros_like(x[:, :, :segment_size])
    for i in range(x.size(0)):
        start = start_indices[i]
        end = start + segment_size
        ret[i] = x[i, :, start:end]
    return ret


def rand_slice_segments(x, x_lengths=None, segment_size=4):
    B, _, T = x.size()
    if x_lengths is None:
        x_lengths = T
    start_max = x_lengths - segment_size + 1
    idx_start = (torch.rand([B]).to(device=x.device) * start_max).to(dtype=torch.long)
    ret = slice_segments(x, idx_start, segment_size)
    return ret, idx_start


def to_log_scale(x: torch.Tensor):
    x[x != 0] = torch.log(x[x != 0])
    return x
