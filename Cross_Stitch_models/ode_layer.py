
import torch.nn as nn
from torchdiffeq import odeint

class ODE(nn.Module):
    def __init__(self, ode_func, time_span):
        super(ODE, self).__init__()
        self.ode_func = ode_func
        self.time_span = time_span

    def forward(self, x):
        # Integrate the ODE using odeint
        out = odeint(self.ode_func, x, self.time_span, method='rk4')
        return out[-1]