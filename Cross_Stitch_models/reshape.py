import torch
import torch.nn as nn

class ReshapeToOriginal(nn.Module):
    def __init__(self, number_of_tasks):
        """
        Module to reshape the output of the loop_layer to match the original input shape.

        Args:
            input_shape (tuple): Original input shape (C, H, W).
            number_of_tasks (int): Number of tasks to handle.
        """
        super(ReshapeToOriginal, self).__init__()
    
        self.number_of_tasks = number_of_tasks
        
        # Create task-specific upsampling layers
        self.task_reshapers = nn.ModuleList([
           nn.Sequential(
        nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(1, 10)),
        nn.BatchNorm2d(32),
        nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(1, 2), stride=(1, 2)),
        nn.BatchNorm2d(32),
        nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=(4, 1)),
        nn.BatchNorm2d(1),
        nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=(1, 2), stride=(1, 2)),
        nn.BatchNorm2d(1),
        nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=(4, 1), stride=(1, 1)), 
        )
                for _ in range(number_of_tasks)
        ])
        
    def forward(self, tasks):
        """
        Forward pass to reshape the output back to the original input shape.

        Args:
            tasks (list of Tensors): Output from the loop_layer, one tensor per task.

        Returns:
            reshaped_tasks (list of Tensors): Reshaped tensors matching the original input size.
        """
        reshaped_tasks = []
        for i, task in enumerate(tasks):
            reshaped_task = self.task_reshapers[i](task)
            reshaped_tasks.append(reshaped_task)
        
        return reshaped_tasks

