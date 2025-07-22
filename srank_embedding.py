import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

# ================================
# Step 1. Spearman Rank Loss
# ================================

def spearman_loss(pred_rank, true_rank):
    return F.mse_loss(pred_rank, true_rank)

# ================================
# Step 2. Permutahedron Projection (SoftRank)
# ================================

def soft_rank(x, reg=1.0, p=1.5, n_iter=50):
    """
    Differentiable approximation to ranking operator using p-norm permutahedron projection.

    Args:
        x: (B, N) score vector (vᵗf_i)
        reg: regularization strength (controls smoothness)
        p: norm parameter (e.g. 4/3 or 2)
        n_iter: number of projected gradient iterations (for simplicity)

    Returns:
        soft_ranked: (B, N) soft rank vectors
    """
    B, N = x.size()
    device = x.device

    # Canonical rank vector ρ = (N, N-1, ..., 1)
    rho = torch.arange(N, 0, -1, device=device).float().unsqueeze(0).expand(B, -1)

    # Initialize with rho
    y = rho.clone().detach().requires_grad_(True)

    # Gradient descent-style projection (simplified)
    for _ in range(n_iter):
        loss = -torch.sum(x * y) + (1 / p) * torch.sum(y.abs() ** p)
        grad, = torch.autograd.grad(loss, y, create_graph=True)
        y = y - reg * grad
        y = y - ((y.sum(dim=1, keepdim=True) - rho.sum(dim=1, keepdim=True)) / N)  # sum constraint
    return y

# ================================
# Step 3. RankAxis Learner
# ================================

class RankAxisModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.v = nn.Parameter(torch.randn(dim))
        
    def forward(self, x):
        v_unit = self.v / self.v.norm(p=2)  # Unit norm constraint
        s = torch.matmul(x, v_unit)        # Projected score: vᵗf_i
        return s

# ================================
# Step 4. Main Training Loop
# ================================

def train_rank_axis(embeddings, attributes, epochs=500, lr=1e-3, reg=1.0, p=1.5):
    """
    embeddings: (N, D) torch.Tensor of chemical embeddings
    attributes: (N,) torch.Tensor of toxicity scores (e.g., AC50)
    """
    device = embeddings.device
    N, D = embeddings.size()

    model = RankAxisModel(D).to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    # Precompute true rank
    true_rank = torch.argsort(torch.argsort(attributes, descending=False)).float()
    true_rank = true_rank.to(device)

    for epoch in range(epochs):
        optimizer.zero_grad()

        s = model(embeddings)                   # (N,)
        s = s.unsqueeze(0)                      # add batch dim
        pred_rank = soft_rank(s, reg=reg, p=p)  # (1, N)

        loss = spearman_loss(pred_rank[0], true_rank)

        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"[Epoch {epoch}] Spearman Loss: {loss.item():.4f}")

    return model.v.detach() / model.v.norm(p=2)  # Return final unit vector v
