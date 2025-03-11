import torch
import torch.nn as nn

class TriangularCrossStitchUnit(nn.Module):
    def __init__(self, num_tasks, upper_triangular=True):
        """
        Args:
            num_tasks: Number of tasks.
            upper_triangular: If True, use an upper triangular cross-stitch matrix. Otherwise, use lower triangular.
        """
        super(TriangularCrossStitchUnit, self).__init__()
        self.num_tasks = num_tasks
        self.upper_triangular = upper_triangular

        # Initialize the cross-stitch matrix as trainable parameters
        self.cross_stitch = nn.Parameter(torch.eye(num_tasks, num_tasks))

        # Create a triangular mask
        if self.upper_triangular:
            self.mask = torch.triu(torch.ones(num_tasks, num_tasks))
        else:
            self.mask = torch.tril(torch.ones(num_tasks, num_tasks))

    def forward(self, *features):
        """
        Args:
            features: Variable-length input tensors, one per task, each of shape (batch_size, channels, width, height).

        Returns:
            A tuple of output tensors, one per task, each of shape (batch_size, channels, width, height).
        """
        
        batch_size, channels, width, height = features[0].shape  

        # Stack features along the task dimension
        stacked = torch.stack(features, dim=2)  # Shape: (batch_size, channels, num_tasks, width, height)
        
        # Flatten for matrix multiplication
        stacked_flat = stacked.view(-1, self.num_tasks)  # Shape: (batch_size * channels * width * height, num_tasks)

        # Apply triangular masking to the cross-stitch matrix
        cross_stitch = self.cross_stitch * self.mask.to(self.cross_stitch.device)  # Ensure mask is on the same device

        # Multiply with the masked cross-stitch matrix
        stitched_flat = stacked_flat.matmul(cross_stitch)  # Shape: (batch_size * channels * width * height, num_tasks)

        # Reshape back to original dimensions
        stitched = stitched_flat.view(batch_size, channels, self.num_tasks, width, height)

        # Split along task dimension and return as a tuple
        output = torch.unbind(stitched, dim=2)  # Returns a tuple of tensors, one per task

        return output
