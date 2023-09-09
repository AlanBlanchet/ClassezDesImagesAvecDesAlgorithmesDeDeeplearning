import torch
import torch.nn as nn
import torch.nn.functional as F


class Test(nn.Module):
    def __init__(self, s_in: int, num_classes: int):
        super(Test, self).__init__()

        self.s_in = s_in

        self.conv1 = nn.Conv2d(3, 6, 5)  # B, 6, s_in-4, s_in-4
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        s_final = (((s_in - 4) // 2) - 4) // 2
        self.fc1 = nn.Linear(16 * s_final**2, num_classes)

    def __call__(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # B, 6, (s_in-4)//2, (s_in-4)//2
        x = self.pool(
            F.relu(self.conv2(x))
        )  # B, 16, (((s_in-4)//2)-4)//2, (((s_in-4)//2)-4)//2
        x = torch.flatten(x, 1)  # B, 16 *  (((s_in-4)//2)-4)//2 * (((s_in-4)//2)-4)//2
        x = self.fc1(x)
        return x
