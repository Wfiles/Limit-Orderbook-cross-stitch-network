import torch.nn as nn
import torch
from Cross_Stitch_models.linear_cross_stitch_unit import TriangularCrossStitchUnit
from Cross_Stitch_models.ode_layer import ODE
from Cross_Stitch_models.odefunc import ODEFunc
from Cross_Stitch_models.conv_block import conv_block_network


class CrossStitchNetwork(nn.Module):
    def __init__(self, number_of_tasks, time_span):
        """
        Cross-Stitch Network for multi-task learning with Neural ODEs.
        Args:
            number_of_tasks (int): Number of tasks.
            time_span (torch.Tensor): Time span for the ODE integration.
        """
        super(CrossStitchNetwork, self).__init__()
        self.number_of_tasks = number_of_tasks
        self.conv_block1 = conv_block_network(number_of_tasks=self.number_of_tasks)

        # ODE and Fully Connected layers
        self.ode = nn.ModuleList([ODE(ODEFunc(feature_dim=94), time_span=time_span) for _ in range(number_of_tasks)])
        self.task_fcs = nn.ModuleList([
            nn.Linear(94, 1) # Map from ODE output size to final output
            for _ in range(number_of_tasks)
        ])

    def forward(self, *tasks):
        """
        Forward pass for Cross-Stitch Network.
        Args:
            tasks: Variable-length input tensors, one per task.
        Returns:
            tuple of outputs, one per task.
        """
        # Process tasks through the first loop layer
        conv_outputs = self.conv_block1(tasks)

        outputs = []
        for i, task in enumerate(conv_outputs):
            # Reshape task input for ODE
            task_out = self.ode[i](task.squeeze(0))
    
            # Map ODE output to task-specific output
            task_output = self.task_fcs[i](task_out[:, -1, :])  # Use final time step's output
            task_output = task_output.squeeze(0)
            task_output = nn.Softplus()(task_output)
            outputs.append(task_output)
             
        return tuple(outputs)