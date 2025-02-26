import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv3d(nn.Module):
    """
    3D Fourier layer: rfftn -> learned freq multiplication -> irfftn
    """
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.in_channels = in_channels
        self.out_channels = out_channels

        scale = 1/(in_channels * out_channels)
        self.weights1 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))

    def compl_mul3d(self, input, weights):
        # input:  (batch, in_channel, x, y, z)
        # weights: (in_channel, out_channel, x, y, z)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        # x: (batch, in_channel, Nx, Ny, Nz)
        batchsize = x.shape[0]
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])

        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-3),
            x.size(-2),
            x.size(-1)//2 + 1,
            dtype=torch.cfloat,
            device=x.device
        )

        # corners
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

class MLP(nn.Module):
    """
    Pointwise feed-forward in channel dimension (1x1x1 conv).
    """
    def __init__(self, in_channels, out_channels, mid_channels):
        super().__init__()
        self.mlp1 = nn.Conv3d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv3d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

class FNO3d(nn.Module):
    """
    3D Fourier Neural Operator.

    Now in_channels=4 => [divergence, free, wall, inlet].
    out_channels=1 => [pressure].
    We'll also append (x,y,z) => total in_channels + 3 => fc0 => 'width'.
    Then apply multiple spectral conv layers + final projection.
    """
    def __init__(self, modes1, modes2, modes3, width, in_channels, out_channels, depth=4):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.depth = depth

        # Lift from (in_channels + 3) -> width
        self.fc0 = nn.Linear(in_channels + 3, width)

        # spectral conv + mlp + skip
        self.convs = nn.ModuleList([
            SpectralConv3d(width, width, modes1, modes2, modes3) for _ in range(depth)
        ])
        self.mlps = nn.ModuleList([
            MLP(width, width, width) for _ in range(depth)
        ])
        self.skips = nn.ModuleList([
            nn.Conv3d(width, width, 1) for _ in range(depth)
        ])

        # final projection
        self.fc1 = MLP(width, out_channels, mid_channels=width*4)

    def forward(self, x):
        """
        x: [batch, Nx, Ny, Nz, in_channels=4].
        We add (x,y,z) => shape becomes (4 + 3).
        """
        b, Nx, Ny, Nz, Cin = x.shape
        grid = self.get_grid(b, Nx, Ny, Nz, x.device)  # [b, Nx, Ny, Nz, 3]

        x = torch.cat([x, grid], dim=-1)               # => [b, Nx, Ny, Nz, Cin+3]
        x = self.fc0(x)                                # => [b, Nx, Ny, Nz, width]
        x = x.permute(0, 4, 1, 2, 3)                    # => [b, width, Nx, Ny, Nz]

        for i in range(self.depth):
            x1 = self.convs[i](x)    # spectral conv
            x1 = self.mlps[i](x1)    # 1x1x1 conv in channel dim
            x2 = self.skips[i](x)    # skip connection
            x = x1 + x2
            if i < self.depth - 1:
                x = F.gelu(x)

        x = self.fc1(x)  # => [b, out_channels=1, Nx, Ny, Nz]
        x = x.permute(0, 2, 3, 4, 1)  # => [b, Nx, Ny, Nz, 1]
        return x

    def get_grid(self, b, Nx, Ny, Nz, device):
        """
        Create (x,y,z) coords in [0,1].
        """
        x_coords = torch.linspace(0, 1, Nx, device=device)
        y_coords = torch.linspace(0, 1, Ny, device=device)
        z_coords = torch.linspace(0, 1, Nz, device=device)

        x_coords = x_coords.view(1, Nx, 1, 1, 1).expand(b, -1, Ny, Nz, -1)
        y_coords = y_coords.view(1, 1, Ny, 1, 1).expand(b, Nx, -1, Nz, -1)
        z_coords = z_coords.view(1, 1, 1, Nz, 1).expand(b, Nx, Ny, -1, -1)

        return torch.cat((x_coords, y_coords, z_coords), dim=-1)

class LpLoss(nn.Module):
    """
    Relative L2: ||pred - true||2 / ||true||2.
    """
    def __init__(self, p=2):
        super().__init__()
        self.p = p

    def forward(self, pred, target):
        # shape: [b, Nx, Ny, Nz, 1]
        diff_norm = torch.norm(pred - target, p=self.p, dim=(-1, -2, -3, -4))
        return diff_norm.mean()


import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedLoss(nn.Module):
    """
    Combines:
      (1) Data loss: LpLoss (difference between predicted pressure and target pressure)
      (2) PDE loss: MSE( Laplacian(pred_dec), div_dec ), enforcing \nabla^2 p = div(u)
          in *physical* space (decoded).
    """
    def __init__(
        self, 
        data_loss_weight=1.0, 
        pde_loss_weight=0.1, 
        p=2,
        x_normalizer=None,   # used to decode divergence from x
        y_normalizer=None    # used to decode predicted pressure and target
    ):
        """
        data_loss_weight: multiplier for the data (pressure) loss
        pde_loss_weight:  multiplier for the PDE consistency term
        p: norm for LpLoss

        x_normalizer: normalizer for input data (divergence) if needed
        y_normalizer: normalizer for pressure output if needed
        """
        super().__init__()
        self.data_loss_fn = LpLoss(p=p)    # your LpLoss class
        self.data_loss_weight = data_loss_weight
        self.pde_loss_weight  = pde_loss_weight

        # Normalizers (optional). If not provided, PDE loss will remain in normalized space.
        self.x_normalizer = x_normalizer
        self.y_normalizer = y_normalizer

    def forward(self, pred, target, x):
        """
        pred:   predicted *normalized* pressure, shape [batch, Nx, Ny, Nz, 1]
        target: ground-truth *normalized* pressure, shape [batch, Nx, Ny, Nz, 1]
        x:      input tensor (normalized), shape [batch, Nx, Ny, Nz, C]
                - We assume x[..., 0] is the *normalized* divergence channel (phi).

        Returns a scalar: data_loss + pde_loss (with weights).
        """

        # 1) Data loss (in normalized space, or decode both if you prefer).
        #    Currently, we do the LpLoss in normalized space.
        data_loss = self.data_loss_fn(pred, target)  # => scalar

        # 2) PDE loss in *decoded* (physical) space
        #    a) Decode predicted pressure
        if self.y_normalizer is not None:
            pred_dec = self.y_normalizer.decode(pred)   # => shape [b, Nx, Ny, Nz, 1]
        else:
            pred_dec = pred

        #    b) Decode the ground-truth divergence from x
        #       x[..., 0] is the *normalized* divergence
        if self.x_normalizer is not None:
            # We'll decode the entire x, then take the first channel
            x_dec = self.x_normalizer.decode(x)         # shape [b, Nx, Ny, Nz, C]
            phi_dec = x_dec[..., 0]                     # decoded divergence
        else:
            phi_dec = x[..., 0]

        # p_pred_dec => shape [b, Nx, Ny, Nz]
        p_pred_dec = pred_dec[..., 0]

        # Laplacian in physical space
        lap_p_dec = self.laplacian_3d(p_pred_dec)

        # PDE mismatch: MSE( lap_p_dec, phi_dec )
        pde_loss = F.mse_loss(lap_p_dec, phi_dec)

        # Combine the two losses
        total_loss = self.data_loss_weight * data_loss + self.pde_loss_weight * pde_loss
        return total_loss

    def laplacian_3d(self, p):
        """
        Approximate 3D Laplacian of scalar field p using torch.gradient.
        p: shape [batch, Nx, Ny, Nz]
        Returns shape [batch, Nx, Ny, Nz] for ∇²p.
        """

        # First derivatives
        dpdx = torch.gradient(p, dim=1)[0]
        dpdy = torch.gradient(p, dim=2)[0]
        dpdz = torch.gradient(p, dim=3)[0]

        # Second derivatives
        dp2dx2 = torch.gradient(dpdx, dim=1)[0]
        dp2dy2 = torch.gradient(dpdy, dim=2)[0]
        dp2dz2 = torch.gradient(dpdz, dim=3)[0]

        # Laplacian
        lap_p = dp2dx2 + dp2dy2 + dp2dz2
        return lap_p
