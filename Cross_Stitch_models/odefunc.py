import torch
import torch.nn as nn

class ODEFunc(nn.Module):
    def __init__(self, feature_dim):
        super(ODEFunc, self).__init__()
        # Define the neural network representing the dynamics
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.Tanh(),
            nn.Linear(64, feature_dim)
        )

    def forward(self, t, x):
        """
        Compute the rate of change dx/dt.

        Args:
            t (torch.Tensor): Current time step.
            x (torch.Tensor): Current state (batch_size, feature_dim).

        Returns:
            torch.Tensor: Rate of change dx/dt (batch_size, feature_dim).
        """
        return self.net(x)
    

class FourierODEFunc(nn.Module):
    def __init__(self, feature_dim, fourier_dim=16):
        super(FourierODEFunc, self).__init__()
        self.fourier_transform = nn.Linear(feature_dim, fourier_dim)
        self.net = nn.Sequential(
            nn.Linear(fourier_dim, 64),
            nn.Tanh(),
            nn.Linear(64, feature_dim)
        )

    def forward(self, t, x):
        # Transform input to Fourier space
        x_transformed = torch.sin(self.fourier_transform(x))
        return self.net(x_transformed)


import torch
import torch.nn as nn

class GARCHODEFunc(nn.Module):
    def __init__(self):
        super(GARCHODEFunc, self).__init__()
        # Define learnable parameters for GARCH dynamics
        self.omega = nn.Parameter(torch.tensor(0.01, requires_grad=True))  # Long-term variance
        self.alpha = nn.Parameter(torch.tensor(0.1, requires_grad=True))   # Impact of past shocks
        self.beta = nn.Parameter(torch.tensor(0.85, requires_grad=True))   # Persistence of volatility

    def forward(self, t, sigma2_t, epsilon_t_minus_1):
        """
        Compute the rate of change d(sigma^2)/dt based on GARCH-inspired dynamics.

        Args:
            t (torch.Tensor): Current time step.
            sigma2_t (torch.Tensor): Current volatility squared (\( \sigma_t^2 \)).
            epsilon_t_minus_1 (torch.Tensor): Previous return shocks (\( \epsilon_{t-1}^2 \)).

        Returns:
            torch.Tensor: Rate of change d(sigma^2)/dt.
        """
        # GARCH-inspired dynamics
        dsigma2_dt = self.omega + self.alpha * epsilon_t_minus_1**2 + self.beta * sigma2_t - sigma2_t
        return dsigma2_dt
