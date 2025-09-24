import torch.nn as nn
class TriFastPose(nn.Module):
    def __init__(self, in_ch_short=1, in_ch_mid=1, in_ch_long=1, hid=8, num_classes=38):
        super().__init__()
        self.head = nn.Linear(3, num_classes)  # placeholder

    def forward(self, x_short, x_mid, x_long):
        # x_* ignorados en el stub; devolvemos dummy [B,T,C] con T=1
        import torch
        b = x_short.size(0)
        y = self.head(torch.zeros(b,3))
        return y.unsqueeze(1)
