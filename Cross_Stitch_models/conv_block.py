
import torch.nn as nn
from Cross_Stitch_models.linear_cross_stitch_unit import TriangularCrossStitchUnit
from Cross_Stitch_models.inception import InceptionModule

class conv_block_network(nn.Module) :
    def __init__(self, number_of_tasks) :

        super(conv_block_network, self).__init__()

        self.number_of_tasks = number_of_tasks
        # Task-specific convolutional blocks
        self.task_convs1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 2), stride=(1, 2)),
                nn.LeakyReLU(negative_slope=0.01),
                nn.BatchNorm2d(32),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
                nn.LeakyReLU(negative_slope=0.01),
                nn.BatchNorm2d(32),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
                nn.LeakyReLU(negative_slope=0.01),
                nn.BatchNorm2d(32),
            )
            for _ in range(number_of_tasks)
        ])

        self.task_convs2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 2), stride=(1, 2)),
                nn.Tanh(),
                nn.BatchNorm2d(32),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 10)),
                nn.LeakyReLU(negative_slope=0.01),
                nn.BatchNorm2d(32),
                InceptionModule(32)
            )
            for _ in range(number_of_tasks)
        ])

        # Cross-Stitch Unit
        self.cross_stitch_unit_1 = TriangularCrossStitchUnit(num_tasks=self.number_of_tasks, upper_triangular=True)
        self.cross_stitch_unit_2 = TriangularCrossStitchUnit(num_tasks=self.number_of_tasks, upper_triangular=True)

    def forward(self, tasks) :
            
        task_features1 = []
        for i, task in enumerate(tasks):
        
            task = self.task_convs1[i](task)  # Task-specific convolution

            task_features1.append(task)

        task_stitch1 = self.cross_stitch_unit_1(*task_features1)

        task_features2 = []
        for i, task in enumerate(task_stitch1):
        
            task = self.task_convs2[i](task)  # Task-specific convolution

            task_features2.append(task.unsqueeze(0))

        task_stitch2 = self.cross_stitch_unit_2(*task_features2)

        return task_stitch2
        
            