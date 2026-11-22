# 模型权重（未入仓）

按提交说明第 ② 条（模型权重不塞 GitHub），**3 个 `.pt` 文件已通过校内门户「大文件分片上传」单独提交**，共 17.8 MB。

| 文件 | 用途 |
|---|---|
| model_crosslighting_none.pt | 跨光照基线（无归一化），对应 94.6% |
| model_crosslighting_clahe.pt | 跨光照 + CLAHE，对应 100% |
| model_all_clahe.pt | 部署模型（全量训练），检测器在用 |

## 仓库内怎么复现

```bash
python src/train.py --mode crosslighting --norm none    # 复现基线
python src/train.py --mode crosslighting --norm clahe   # 复现 +CLAHE
python src/train.py --mode all --norm clahe             # 复现部署模型
python src/eval_fill.py                                  # 补光对照
```

CPU 上单次训练约 3–9 分钟，输出直接落到本目录。

## 指标文件都还在

`metrics_*.json` 与混淆矩阵 PNG 全部保留在仓库里，不依赖权重也能看到每个数字。
