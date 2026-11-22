# -*- coding: utf-8 -*-
"""项目二：多光照物件分类训练与评估。

用法：
    python train.py --mode random            # 分层随机划分
    python train.py --mode crosslighting     # 用 L1/L2 训练，L3/L4 测试
    python train.py --mode crosslighting --norm clahe
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import cv2

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

CLASSES = ["O1", "O2", "O3", "O4", "O5", "EMPTY"]
CN = {"O1": "白色按压瓶", "O2": "深色充电宝", "O3": "金属钥匙",
      "O4": "透明玻璃瓶", "O5": "亮面包装盒", "EMPTY": "空桌面"}
LIGHTS = ["L1", "L2", "L3", "L4"]
LIGHT_CN = {"L1": "均匀顶光", "L2": "弱光", "L3": "强反光", "L4": "阴影遮挡"}

FB = r"C:\Windows\Fonts\msyhbd.ttc"
FR = r"C:\Windows\Fonts\msyh.ttc"


def font(sz, bold=False):
    return ImageFont.truetype(FB if bold else FR, sz)


def collect():
    items = []
    for p in sorted(DATA.rglob("*.jpg")):
        light = p.parent.parent.name.split("_")[0]
        cls = "EMPTY" if p.parent.name.startswith("空桌面") else p.parent.name.split("_")[0]
        if light not in LIGHTS or cls not in CLASSES:
            continue
        items.append((p, light, cls))
    return items


class DS(Dataset):
    def __init__(self, items, tf):
        self.items = items
        self.tf = tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, light, cls = self.items[i]
        img = Image.open(p).convert("RGB")
        return self.tf(img), CLASSES.index(cls), i


def clahe_fn(img):
    a = np.asarray(img)
    lab = cv2.cvtColor(a, cv2.COLOR_RGB2LAB)
    l, a_, b_ = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return Image.fromarray(cv2.cvtColor(cv2.merge([l, a_, b_]), cv2.COLOR_LAB2RGB))


def build_tf(train, norm):
    t = []
    if norm == "clahe":
        t.append(transforms.Lambda(clahe_fn))
    if train:
        t += [transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
              transforms.RandomHorizontalFlip(),
              transforms.RandomRotation(10),
              transforms.ColorJitter(0.30, 0.30, 0.20, 0.05)]
    else:
        t += [transforms.Resize(256), transforms.CenterCrop(224)]
    t += [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    return transforms.Compose(t)


def split(items, mode):
    tr, va = [], []
    if mode == "all":
        tr = list(items)
        va = list(items)
        return tr, va
    if mode == "crosslighting":
        for it in items:
            (tr if it[1] in ("L1", "L2") else va).append(it)
        return tr, va
    by_cell = {}
    for it in items:
        by_cell.setdefault((it[1], it[2]), []).append(it)
    for cell, lst in by_cell.items():
        k = max(1, round(len(lst) * 0.25))
        va += lst[:k]
        tr += lst[k:]
    return tr, va


def make_model():
    try:
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        pretrained = True
    except Exception as e:
        print("预训练权重不可用，改为从头训练：", e)
        m = models.mobilenet_v3_small(weights=None)
        pretrained = False
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, len(CLASSES))
    return m, pretrained


def train(args):
    items = collect()
    tr, va = split(items, args.mode)
    print(f"训练 {len(tr)} 张，验证 {len(va)} 张 | 模式 {args.mode} | 归一化 {args.norm}")

    tf_tr = build_tf(True, args.norm)
    tf_va = build_tf(False, args.norm)
    dl_tr = DataLoader(DS(tr, tf_tr), batch_size=args.bs, shuffle=True, num_workers=0)
    dl_va = DataLoader(DS(va, tf_va), batch_size=args.bs, shuffle=False, num_workers=0)

    model, pretrained = make_model()
    counts = np.array([sum(1 for it in tr if it[2] == c) for c in CLASSES], dtype=float)
    w = torch.tensor((counts.sum() / (counts + 1e-6)) ** 0.5, dtype=torch.float32)
    w = w / w.mean()
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_acc, best_state, curve = 0.0, None, []
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        tot, corr, loss_sum = 0, 0, 0.0
        for x, y, _ in dl_tr:
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(y)
            corr += (out.argmax(1) == y).sum().item()
            tot += len(y)
        sched.step()
        model.eval()
        vtot, vcorr = 0, 0
        with torch.no_grad():
            for x, y, _ in dl_va:
                out = model(x)
                vcorr += (out.argmax(1) == y).sum().item()
                vtot += len(y)
        tr_acc, va_acc = corr / tot, vcorr / max(vtot, 1)
        curve.append({"epoch": ep + 1, "train_acc": tr_acc, "val_acc": va_acc, "loss": loss_sum / tot})
        if va_acc >= best_acc:
            best_acc, best_state = va_acc, {k: v.clone() for k, v in model.state_dict().items()}
        print(f"  ep{ep+1:02d} loss={loss_sum/tot:.3f} train={tr_acc:.3f} val={va_acc:.3f}")

    model.load_state_dict(best_state)
    model.eval()

    # 逐张推理，收集预测与耗时
    preds = []
    times = []
    with torch.no_grad():
        for x, y, idx in dl_va:
            for j in range(len(y)):
                one = x[j:j + 1]
                t1 = time.perf_counter()
                out = model(one)
                times.append((time.perf_counter() - t1) * 1000)
                preds.append((int(idx[j]), int(y[j]), int(out.argmax(1).item()),
                              float(torch.softmax(out, 1)[0, out.argmax(1)][0])))

    cm = np.zeros((6, 6), dtype=int)
    per_light = {}
    for i, y, p, c in preds:
        cm[y, p] += 1
        L = va[i][1]
        d = per_light.setdefault(L, {"n": 0, "ok": 0})
        d["n"] += 1
        d["ok"] += int(y == p)
    for L in per_light:
        per_light[L]["acc"] = per_light[L]["ok"] / per_light[L]["n"]

    acc = float(np.trace(cm)) / max(cm.sum(), 1)
    metrics = {
        "mode": args.mode, "norm": args.norm, "pretrained": pretrained,
        "n_train": len(tr), "n_val": len(va), "epochs": args.epochs,
        "best_val_acc": best_acc, "val_acc": acc,
        "per_light": per_light,
        "per_class": {CLASSES[i]: {"n": int(cm[i].sum()), "recall": float(cm[i, i] / max(cm[i].sum(), 1))}
                      for i in range(6)},
        "confusion": cm.tolist(),
        "infer_ms_mean": float(np.mean(times)), "infer_ms_p95": float(np.percentile(times, 95)),
        "train_seconds": round(time.time() - t0, 1),
        "curve": curve,
    }
    tag = f"{args.mode}_{args.norm}"
    (OUT / f"metrics_{tag}.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save({"state": best_state, "classes": CLASSES, "norm": args.norm}, OUT / f"model_{tag}.pt")
    draw_cm(cm, OUT / f"cm_{tag}.png", f"混淆矩阵 · {args.mode} / {args.norm}  (acc={acc:.3f})")
    draw_curve(curve, OUT / f"curve_{tag}.png")
    print(json.dumps({k: metrics[k] for k in ("val_acc", "per_light", "infer_ms_mean", "train_seconds")},
                     ensure_ascii=False, indent=2))
    return metrics


def draw_cm(cm, path, title):
    n = cm.shape[0]
    cell, left, top = 78, 150, 92
    W, H = left + n * cell + 40, top + n * cell + 80
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    d.text((24, 22), title, font=font(22, True), fill=(25, 25, 25))
    mx = max(cm.max(), 1)
    for i in range(n):
        for j in range(n):
            v = cm[i, j]
            shade = int(245 - 175 * (v / mx))
            x0, y0 = left + j * cell, top + i * cell
            d.rectangle([x0, y0, x0 + cell - 3, y0 + cell - 3], fill=(shade, shade, 250))
            col = "white" if v / mx > 0.6 else (30, 30, 30)
            d.text((x0 + cell / 2 - 10, y0 + cell / 2 - 12), str(v), font=font(20, True), fill=col)
    for i, c in enumerate(CLASSES):
        d.text((left + i * cell + 12, top - 26), c, font=font(13), fill=(60, 60, 60))
        d.text((16, top + i * cell + 26), c, font=font(13), fill=(60, 60, 60))
    d.text((left, H - 46), "预测 →", font=font(15), fill=(80, 80, 80))
    d.text((16, top - 52), "真实 ↓", font=font(15), fill=(80, 80, 80))
    im.save(path)


def draw_curve(curve, path):
    W, H = 900, 420
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    d.text((30, 16), "训练曲线", font=font(20, True), fill=(25, 25, 25))
    x0, y0, x1, y1 = 80, 60, 860, 350

    def px(i):
        return x0 + (x1 - x0) * i / max(len(curve) - 1, 1)

    def py(v):
        return y1 - (y1 - y0) * v

    for v in (0, 0.25, 0.5, 0.75, 1.0):
        d.line([(x0, py(v)), (x1, py(v))], fill=(232, 232, 232))
        d.text((40, py(v) - 8), f"{v:.2f}", font=font(12), fill=(120, 120, 120))
    for key, col in (("train_acc", (40, 90, 170)), ("val_acc", (200, 60, 50))):
        pts = [(px(i), py(c[key])) for i, c in enumerate(curve)]
        if len(pts) > 1:
            d.line(pts, fill=col, width=3)
    d.text((x0 + 10, y1 + 14), "epoch", font=font(13), fill=(90, 90, 90))
    d.text((x1 - 190, y0 - 30), "蓝 训练 / 红 验证", font=font(13), fill=(90, 90, 90))
    im.save(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="random", choices=["random", "crosslighting", "all"])
    ap.add_argument("--norm", default="none", choices=["none", "clahe"])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    train(ap.parse_args())
