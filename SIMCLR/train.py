import torch
import torch.nn.functional as F

class Trainer:
    def __init__(self, model, optimizer, scheduler, device, T):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.T = T

    def ntxent_loss(self, z, N):
        z = F.normalize(z, dim=1)
        mask = torch.eye(N * 2, device=self.device).bool()
        tmp = torch.mm(z, z.T).masked_fill(mask, float('-inf'))
        loss_matrix = -F.log_softmax(tmp / self.T, dim=1)
        loss = sum(torch.diag(loss_matrix[:N, N:])) + sum(torch.diag(loss_matrix[N:, :N]))
        return loss / (2 * N)

    def train_step(self, xi, xj):
        xi, xj = xi.to(self.device), xj.to(self.device)
        _, z = self.model(xi, xj)
        self.optimizer.zero_grad()
        loss = self.ntxent_loss(z, xi.size(0))
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def valid_step(self, xi, xj):
        xi, xj = xi.to(self.device), xj.to(self.device)
        _, z = self.model(xi, xj)
        return self.ntxent_loss(z, xi.size(0)).item()
