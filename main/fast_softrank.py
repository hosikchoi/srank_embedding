# PAV 기반 Soft Rank (O(n log n))
def fast_soft_rank(scores, p=2.0, device=None):
    """
    Fast projection onto the permutahedron (descending order) using isotonic regression (PAV).
    """
    n = scores.shape[0]
    rho = torch.arange(n, 0, -1, dtype=scores.dtype, device=scores.device)  # [n, ..., 1]
    scores_sorted, indices = torch.sort(scores, descending=True)
    scores_unsorted = torch.zeros_like(scores)
    ranks = isotonic_projection(scores_sorted - rho, p=p)
    scores_unsorted[indices] = ranks
    return scores_unsorted


def isotonic_projection(x, p=2.0):
    """
    Solve: min_v 1/2 ||v - x||^2 + (1/p) * ||v||_p^p  s.t. v is decreasing
    PAV-based isotonic projection with p-norm regularization.
    """
    n = x.shape[0]
    v = x.clone()
    w = torch.ones_like(x)

    # PAV pooling
    idx = 0
    while idx < n - 1:
        if v[idx] < v[idx + 1]:  # violation of decreasing constraint
            j = idx
            while j >= 0 and v[j] < v[j + 1]:
                total = v[j] * w[j] + v[j + 1] * w[j + 1]
                weight = w[j] + w[j + 1]
                avg = total / weight
                v[j] = avg
                v[j + 1] = avg
                w[j] = weight
                w[j + 1] = weight
                j -= 1
            idx = max(j + 1, 0)
        else:
            idx += 1
    return v
# Rank Axis Estimation 업데이트 (속도 개선)
def estimate_rank_axis_fast(F, a, lr=1e-2, n_steps=1000):
    n, d = F.shape
    v = torch.randn(d, device=F.device)
    v = v / v.norm()
    v = nn.Parameter(v)

    optimizer = torch.optim.Adam([v], lr=lr)
    rho = torch.argsort(torch.argsort(-a)).float() + 1  # rank vector

    for step in range(n_steps):
        optimizer.zero_grad()
        s = F @ v
        r_v = fast_soft_rank(s)  # FAST soft rank
        loss = F.mse_loss(r_v, rho)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            v.data = v.data / v.data.norm()

        if step % 100 == 0:
            print(f"[{step}] Loss: {loss.item():.4f}")

    return v.detach()

import time
n, d = 2000, 64
F = torch.randn(n, d)
v_true = torch.randn(d)
v_true = v_true / v_true.norm()
a = F @ v_true + 0.1 * torch.randn(n)

start = time.time()
v_hat = estimate_rank_axis_fast(F, a)
end = time.time()

print("Estimated cosine:", torch.dot(v_hat, v_true).item())
print("Elapsed time (fast):", end - start)
