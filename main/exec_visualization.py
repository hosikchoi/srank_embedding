# v는 학습된 rank axis (RankAxisModel에서 반환된 것)
# embeddings: (N, D) torch.Tensor
# toxicity_scores: (N,) torch.Tensor

visualize_rank_axis_projection(embeddings, toxicity_scores, v)
| 기능              | 설명                                                     |
| --------------- | ------------------------------------------------------ |
| hue 옵션 변경       | `hue='Toxicity'` 대신 chemical class, cluster ID 등 사용 가능 |
| log scale       | 독성값이 log scale (e.g. pIC50)인 경우 `plt.yscale('log')`    |
| chemical 이름 라벨링 | `SMILES`, `CAS`, `Name` 등을 함께 표시하려면 dataframe 확장 가능    |
