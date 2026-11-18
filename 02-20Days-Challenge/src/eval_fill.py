# -*- coding: utf-8 -*-
"""补光实验评估：用「只见过 L1/L2」的模型，分别测 L3（强反光）与 L5（强反光 + 补光）。

用法：python eval_fill.py
输出：results/fill_light.json、results/fill_light.png
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torchvision import models, transforms
import cv2

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
RES = BASE / "results"
CLASSES = ["O1", "O2", "O3", "O4", "O5", "EMPTY"]
CN = {"O1": "白色按压瓶", "O2": "深色充电宝", "O3": "金属钥匙",
      "O4": "透明玻璃瓶", "O5": "亮面包装盒", "EMPTY": "空桌面"}
FB = r"C:\Windows\Fonts\msyhbd.ttc"
FR = r"C:\Windows\Fonts\msyh.ttc"


def font(sz, b=True):
    return ImageFont.truetype(FB if b else FR, sz)


def clahe(img):
    a = np.asarray(img)
    lab = cv2.cvtColor(a, cv2.COLOR_RGB2LAB)
    l, aa, bb = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return Image.fromarray(cv2.cvtColor(cv2.merge([l, aa, bb]), cv2.COLOR_LAB2RGB))


def tf(norm):
    t = []
    if norm == "clahe":
        t.append(transforms.Lambda(clahe))
    t += [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    return transforms.Compose(t)


def build(norm):
    m = models.mobilenet_v3_small(weights=None)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, 6)
    ckpt = torch.load(RES / f"model_crosslighting_{norm}.pt", map_location="cpu", weights_only=False)
    m.load_state_dict(ckpt["state"])
    m.eval()
    return m


def collect(light):
    items = []
    d = DATA / light
    if not d.exists():
        return items
    for p in sorted(d.rglob("*.jpg")):
        cls = "EMPTY" if p.parent.name.startswith("空桌面") else p.parent.name.split("_")[0]
        if cls in CLASSES:
            items.append((p, cls))
    return items


def evaluate(model, items, norm):
    T = tf(norm)
    conf = np.zeros((6, 6), int)
    with torch.no_grad():
        for p, cls in items:
            x = T(Image.open(p).convert("RGB")).unsqueeze(0)
            pr = int(model(x).argmax(1))
            conf[CLASSES.index(cls), pr] += 1
    acc = np.trace(conf) / max(conf.sum(), 1)
    per = {CLASSES[i]: round(conf[i, i] / max(conf[i].sum(), 1), 3) for i in range(6) if conf[i].sum()}
    return acc, conf, per


def main():
    out = {}
    for norm in ("none", "clahe"):
        if not (RES / f"model_crosslighting_{norm}.pt").exists():
            continue
        model = build(norm)
        for light in ("L3_强反光", "L5_补光"):
            items = collect(light)
            if not items:
                print(f"{light}: 没有数据，跳过")
                continue
            acc, conf, per = evaluate(model, items, norm)
            out[f"{norm}|{light}"] = {"acc": acc, "n": len(items), "per_class": per,
                                     "confusion": conf.tolist(), "light": light, "norm": norm}
            print(f"{norm:5} {light:10} n={len(items):3} acc={acc:.3f} {per}")
    (RES / "fill_light.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # 画对比图
    keys = [("none", "L3_强反光"), ("none", "L5_补光")]
    vals = [out.get(f"{a}|{b}", {}).get("acc") for a, b in keys]
    if all(v is not None for v in vals):
        W, H = 900, 460
        im = Image.new("RGB", (W, H), "white")
        d = ImageDraw.Draw(im)
        d.text((30, 18), "补光实验：同一个模型，只改一盏灯", font=font(22), fill=(30, 30, 30))
        x0, y0, x1, y1 = 100, 90, W - 60, H - 90
        for i, ((norm, light), v) in enumerate(zip(keys, vals)):
            bh = (y1 - y0) * v
            cx = x0 + (x1 - x0) * (0.22 + i * 0.45)
            col = (190, 60, 50) if i == 0 else (60, 140, 90)
            d.rectangle([cx - 70, y1 - bh, cx + 70, y1], fill=col)
            d.text((cx - 40, y1 - bh - 34), f"{v*100:.1f}%", font=font(24), fill=col)
            name = "强反光（无补光）" if i == 0 else "强反光 + 补光"
            d.text((cx - 78, y1 + 14), name, font=font(15, False), fill=(40, 40, 40))
        d.line([(x0, y1), (x1, y1)], fill=(80, 80, 80), width=2)
        im.save(RES / "fill_light.png")
        print("figure saved")
    return out


if __name__ == "__main__":
    main()
