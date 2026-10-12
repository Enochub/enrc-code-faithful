import torch


class RotationLayer(torch.nn.Module):
    def __init__(self, V):
        super().__init__()
        self.v = torch.nn.Parameter(torch.tensor(
            V, dtype=torch.float), requires_grad=True)

    def rotate_back(self, z_rot):
        return torch.matmul(z_rot, self.v.t())

    def rotate(self, z):
        return torch.matmul(z, self.v)

    def forward(self, z):
        z_rot = torch.matmul(z, self.v)

        z_detached_rot = torch.matmul(z.detach(), self.v)
        z_detached_rot_back = torch.matmul(z_detached_rot, self.v.t())
        loss = self.rotation_layer_loss(z.detach(), z_detached_rot_back)

        return z_rot, z_detached_rot, loss

    @staticmethod
    def rotation_layer_loss(z, z_rot_back):
        return ((z - z_rot_back)**2).sum(1).sum()
