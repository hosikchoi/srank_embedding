import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def visualize_rank_axis_projection(embeddings, attributes, v, title='Embedding Projection by Toxicity'):
    """
    Args:
        embeddings: (N, D) torch.Tensor, chemical embeddings
        attributes: (N,) torch.Tensor, continuous toxicity values (e.g. AC50)
        v: (D,) torch.Tensor, learned rank axis (unit vector)
    """
    device = embeddings.device
    v = v / v.norm(p=2)

    # Step 1. Project embeddings onto rank axis
    s = torch.matmul(embeddings, v)  # (N,)

    # Step 2. Prepare dataframe for plotting
    df = pd.DataFrame({
        'Projection': s.cpu().numpy(),
        'Toxicity': attributes.cpu().numpy()
    })

    # Step 3. Sort by projection value
    df_sorted = df.sort_values('Projection', ascending=True).reset_index(drop=True)

    # Step 4. Visualization: Projection vs Toxicity
    plt.figure(figsize=(10, 5))
    sns.scatterplot(data=df_sorted, x='Projection', y='Toxicity', palette='viridis', hue='Toxicity', legend=False)
    plt.title(title)
    plt.xlabel('vᵗ · embedding')
    plt.ylabel('Toxicity (e.g., AC50)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
