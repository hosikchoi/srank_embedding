F_emb = torch.randn(128, 512).cuda()  # (n=128, d=512)
attr = torch.linspace(0, 100, 128).cuda() + 5 * torch.randn(128).cuda()  # 예시 속성값
v_rank = train_rank_axis(F_emb, attr)
