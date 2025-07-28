n, d = 200, 64
F = torch.randn(n, d)
true_v = torch.randn(d)
true_v = true_v / true_v.norm()
a = F @ true_v + 0.1 * torch.randn(n)

v_hat = estimate_rank_axis(F, a, p=1.5)
print("cosine similarity to ground truth:", F.cosine_similarity(v_hat, true_v, dim=0).item())

import torch
import torch.nn as nn
import torch.nn.functional as F

def soft_rank(s, p=1.5, n_iter=50):
    """
    Compute soft ranks via projection onto the permutahedron with p-norm regularization.
    s: (n,) score vector (e.g., s = v^T f_i)
    Returns: (n,) soft rank vector
    """
    n = s.shape[0]
    rho = torch.arange(n, 0, -1, dtype=s.dtype, device=s.device)  # [n, ..., 1]
    
    # Initialize y as the hard rank
    y = rho.clone().requires_grad_(True)
    optimizer = torch.optim.SGD([y], lr=0.1)
    
    for _ in range(n_iter):
        optimizer.zero_grad()
        # Objective: -<s, y> + (1/p) * ||y||_p^p
        loss = -torch.dot(s, y) + (1/p) * torch.norm(y, p=p) ** p
        loss.backward()
        optimizer.step()
        
        # Project y onto permutahedron constraints: sorted and sum-preserving
        with torch.no_grad():
            y.data = y.data - y.data.mean() + rho.mean()  # preserve mean
            y.data = torch.sort(y.data, descending=True)[0]

    return y.detach()

def estimate_rank_axis(F, a, p=1.5, lr=1e-2, n_steps=1000):
    """
    Estimate rank axis v given embeddings F (n x d) and attributes a (n,)
    """
    n, d = F.shape
    v = torch.randn(d, device=F.device)
    v = v / v.norm()  # normalize
    v = nn.Parameter(v)

    optimizer = torch.optim.Adam([v], lr=lr)
    rho = torch.argsort(torch.argsort(-a.float())).float() + 1  # ground-truth rank

    for step in range(n_steps):
        optimizer.zero_grad()
        s = F @ v  # score
        r_v = soft_rank(s, p=p)  # soft rank
        loss = F.mse_loss(r_v, rho)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            v.data = v.data / v.data.norm()

        if step % 100 == 0:
            print(f"[{step}] Loss: {loss.item():.4f}")
    
    return v.detach()
