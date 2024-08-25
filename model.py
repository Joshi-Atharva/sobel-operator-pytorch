from torch import nn
import torch


class Sobel(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, bias=False)

        Gx = torch.tensor([[2.0, 0.0, -2.0], [4.0, 0.0, -4.0], [2.0, 0.0, -2.0]]) # sobel operator for horizontal edge detection
        Gy = torch.tensor([[2.0, 4.0, 2.0], [0.0, 0.0, 0.0], [-2.0, -4.0, -2.0]]) # for vertical edge detection
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0) # 0th dimension (axis 0) represents number of filters = out_channels
        G = G.unsqueeze(1) # dimension 1 (axis 1) represents in_channels 
        self.filter.weight = nn.Parameter(G, requires_grad=False)

    def forward(self, img):
        x = self.filter(img)
        x = torch.mul(x, x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.sqrt(x)
        return x
