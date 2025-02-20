import torch
from torch.utils.data import Dataset

class DivBoundaryOneHotDataset(Dataset):
    """
    Expects:
      x_data.shape = [N, 48,48,48, 4] 
        - x_data[..., 0] = divergence (float)
        - x_data[..., 1] = free domain mask (0 or 1)
        - x_data[..., 2] = wall mask (0 or 1)
        - x_data[..., 3] = inlet mask (0 or 1)

      y_data.shape = [N, 48,48,48, 1] (pressure)
    """
    def __init__(self, x_data, y_data):
        super().__init__()
        assert x_data.shape[0] == y_data.shape[0], "x_data and y_data differ in #samples"
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return self.x_data.shape[0]

    def __getitem__(self, idx):
        # x: [48,48,48, 4], y: [48,48,48, 1]
        return self.x_data[idx], self.y_data[idx]


class UnitGaussianNormalizer:
    """
    Normalizes continuous channels to zero mean and unit variance.
    We have 4 input channels:
      0 => divergence (continuous) => we DO want to normalize
      1 => free domain (binary)    => no normalization
      2 => wall (binary)           => no normalization
      3 => inlet (binary)          => no normalization
    By default, channel_ids_to_normalize=(0,) so only channel 0 is normalized.
    """
    def __init__(self, tensor, channel_ids_to_normalize=(0,), eps=1e-5):
        """
        tensor: shape [N, 48,48,48, 4]
        """
        self.eps = eps
        self.channel_ids_to_normalize = channel_ids_to_normalize
        
        n_channels = tensor.shape[-1]
        self.mean = torch.zeros(n_channels)
        self.std  = torch.ones(n_channels)

        # Flatten to [N * 48 * 48 * 48, 4]
        reshaped = tensor.view(-1, n_channels)
        
        for ch in channel_ids_to_normalize:
            ch_values = reshaped[:, ch]  # all values of that channel
            self.mean[ch] = ch_values.mean()
            self.std[ch]  = ch_values.std()

    def encode(self, x):
        # x shape: [N, 48,48,48, 4]
        x_out = x.clone()
        for ch in self.channel_ids_to_normalize:
            x_out[..., ch] = (x_out[..., ch] - self.mean[ch]) / (self.std[ch] + self.eps)
        return x_out

    def decode(self, x):
        x_out = x.clone()
        for ch in self.channel_ids_to_normalize:
            x_out[..., ch] = x_out[..., ch] * (self.std[ch] + self.eps) + self.mean[ch]
        return x_out

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std  = self.std.to(device)
        return self
