import torch
import torch.nn as nn
import torch.nn.functional as F

def soft_rank_pnorm(scores, p=4/3, eps=1e-6, max_iter=50):
    """
    Differentiable p-norm projection onto the permutahedron
    Args:
        scores: (n,) projected values s_v = f @ v
        p: p-norm regularizer (e.g., 4/3 or 2)
    Returns:
        soft_rank: (n,) relaxed rank values
    """
    n = scores.shape[0]
    rho = torch.arange(n, 0, -1, dtype=scores.dtype, device=scores.device)  # [n, n-1, ..., 1]
    y = rho.clone()
    y.requires_grad = True

    optimizer = torch.optim.SGD([y], lr=1e-1)

    for _ in range(max_iter):
        optimizer.zero_grad()
        loss = -torch.dot(scores, y) + (1 / p) * torch.norm(y, p=p) ** p
        loss.backward()
        optimizer.step()
        # Projection to permutahedron constraints (approximate): sum = n(n+1)/2, 1 ≤ y ≤ n
        with torch.no_grad():
            y.clamp_(min=1.0, max=float(n))
            y -= (y.sum() - n * (n + 1) / 2) / n  # approximate projection to affine set

    return y.detach()

def spearman_loss(r_pred, r_true):
    return F.mse_loss(r_pred, r_true)

class RankAxisLearner(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.v = nn.Parameter(torch.randn(dim))
    
    def forward(self, F_emb):  # F_emb: (n, d)
        v_normalized = self.v / (self.v.norm(p=2) + 1e-8)
        scores = F_emb @ v_normalized  # shape: (n,)
        return scores

def compute_rank(a):
    """
    Compute integer ranks from real values. Output ∈ [1, n]
    """
    sorted_idx = torch.argsort(-a)  # descending
    rank = torch.empty_like(sorted_idx, dtype=torch.float32)
    rank[sorted_idx] = torch.arange(1, len(a) + 1, dtype=torch.float32, device=a.device)
    return rank

# ==== Example Training ====
def train_rank_axis(F_emb, attr, num_epochs=200, lr=1e-2, p=4/3):
    """
    Args:
        F_emb: (n, d) embedding matrix
        attr:  (n,) continuous attribute (age, count, etc.)
    """
    n, d = F_emb.shape
    device = F_emb.device
    model = RankAxisLearner(d).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    attr_rank = compute_rank(attr)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        scores = model(F_emb)
        r_soft = soft_rank_pnorm(scores, p=p)
        loss = spearman_loss(r_soft, attr_rank)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"[Epoch {epoch+1}] Spearman Loss: {loss.item():.4f}")

    return model.v.detach() / model.v.detach().norm(p=2)
