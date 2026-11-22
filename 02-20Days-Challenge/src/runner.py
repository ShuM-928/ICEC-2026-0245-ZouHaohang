# -*- coding: utf-8 -*-
"""项目二：连续运行检测器（用于采集 process_logs 里的检测日志）。

启动后自动连摄像头、持续推理并写日志：
    logs/detect_YYYYMMDD.csv   ts, pred, conf, ms      每 2 秒一行
    logs/fa_YYYYMMDD.csv       ts, last_ts, last_pred   按 F 记录误报

快捷键：F 标记误报   Esc 退出
"""
import csv
import datetime
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch
import tkinter as tk
from PIL import Image, ImageDraw, ImageTk
from torchvision import transforms, models
import torch.nn as nn
import cv2

from camera_shoot import Cam, list_devices, load_conf

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs"
LOGS.mkdir(exist_ok=True)
MODEL = ROOT / "results" / "model_all_clahe.pt"
CLASSES = ["O1", "O2", "O3", "O4", "O5", "EMPTY"]
CN = {"O1": "白色按压瓶", "O2": "深色充电宝", "O3": "金属钥匙",
      "O4": "透明玻璃瓶", "O5": "亮面包装盒", "EMPTY": "空桌面"}

W, H = 960, 620


def clahe(img):
    a = np.asarray(img)
    lab = cv2.cvtColor(a, cv2.COLOR_RGB2LAB)
    l, aa, bb = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return Image.fromarray(cv2.cvtColor(cv2.merge([l, aa, bb]), cv2.COLOR_LAB2RGB))


TF = transforms.Compose([
    transforms.Lambda(clahe),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def build_model():
    m = models.mobilenet_v3_small(weights=None)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, 6)
    ckpt = torch.load(MODEL, map_location="cpu", weights_only=False)
    m.load_state_dict(ckpt["state"])
    m.eval()
    return m


class Runner:
    def __init__(self, root):
        self.root = root
        self.model = build_model()
        self.count = 0
        self.fa = 0
        self.last = None
        self.last_conf = 0.0
        self.last_ms = 0.0
        self.last_logged = None
        self.day = datetime.date.today().isoformat()
        self.unc_dir = LOGS / "uncertain"
        self.unc_dir.mkdir(exist_ok=True)
        self.n_unc = 0
        self.last_unc = 0.0

        conf = load_conf()
        name = conf.get("camera_name") or ""
        real = [n for n, v in list_devices() if not v]
        usb = [n for n in real if "icspring" in n.lower()]
        if usb:
            name = usb[0]
        elif name not in real:
            name = real[0] if real else ""
        self.cam = Cam(name) if name else None
        self.cam_name = name

        root.title("项目二 · 连续检测运行器")
        root.configure(bg="#111")
        root.geometry(f"{W + 320}x{H}")
        root.resizable(False, False)

        self.canvas = tk.Canvas(root, width=W, height=H, bg="#000", highlightthickness=0)
        self.canvas.place(x=0, y=0)
        panel = tk.Frame(root, bg="#111")
        panel.place(x=W, y=0, width=320, height=H)

        tk.Label(panel, text="当前识别", bg="#111", fg="#888", font=("微软雅黑", 10)).place(x=16, y=14)
        self.lb_cls = tk.Label(panel, text="—", bg="#111", fg="#FFD24A", font=("微软雅黑", 22, "bold"))
        self.lb_cls.place(x=16, y=38)
        self.lb_conf = tk.Label(panel, text="", bg="#111", fg="#CCC", font=("微软雅黑", 12))
        self.lb_conf.place(x=16, y=88)
        self.lb_ms = tk.Label(panel, text="", bg="#111", fg="#777", font=("微软雅黑", 9))
        self.lb_ms.place(x=16, y=116)

        self.lb_cnt = tk.Label(panel, text="", bg="#111", fg="#DDD", font=("微软雅黑", 11))
        self.lb_cnt.place(x=16, y=160)
        self.lb_fa = tk.Label(panel, text="", bg="#111", fg="#DDD", font=("微软雅黑", 11))
        self.lb_fa.place(x=16, y=188)
        self.lb_time = tk.Label(panel, text="已运行 00:00", bg="#111", fg="#777", font=("微软雅黑", 9))
        self.lb_time.place(x=16, y=212)
        self.lb_unc = tk.Label(panel, text="低置信度截图 0", bg="#111", fg="#777", font=("微软雅黑", 9))
        self.lb_unc.place(x=16, y=234)
        self.lb_dev = tk.Label(panel, text=f"设备：{self.cam_name}", bg="#111", fg="#555", font=("微软雅黑", 8))
        self.lb_dev.place(x=16, y=256)

        tk.Button(panel, text="标记误报  (F)", command=self.mark_fa, bg="#7F1D1D", fg="#FFD7D7",
                  bd=0, font=("微软雅黑", 11), takefocus=0).place(x=16, y=230, width=280)
        self.txt = tk.Text(panel, width=38, height=12, bg="#1b1b1b", fg="#999", bd=0,
                           font=("Consolas", 8), state="disabled")
        self.txt.place(x=16, y=280)

        tk.Label(panel, text="日志 logs/detect_*.csv\n低置信截图 logs/uncertain/\nEsc 退出",
                 bg="#111", fg="#666", font=("微软雅黑", 8), justify="left").place(x=16, y=486)

        root.bind_all("<Key-f>", lambda e: (self.mark_fa(), "break")[1])
        root.bind_all("<Escape>", lambda e: self.close())
        self.t0 = time.time()
        self.next_log = 0.0
        self.loop()
        root.protocol("WM_DELETE_WINDOW", self.close)

    def log(self, s):
        self.txt.config(state="normal")
        self.txt.insert("end", s + "\n")
        self.txt.see("end")
        self.txt.config(state="disabled")

    def paths(self):
        d = datetime.date.today().isoformat()
        return LOGS / f"detect_{d}.csv", LOGS / f"fa_{d}.csv"

    def loop(self):
        img = self.cam.latest(1.0) if self.cam else None
        if img is not None:
            x = TF(img).unsqueeze(0)
            t1 = time.perf_counter()
            with torch.no_grad():
                out = self.model(x)
            self.last_ms = (time.perf_counter() - t1) * 1000
            prob = torch.softmax(out, 1)[0]
            i = int(prob.argmax())
            self.last = CLASSES[i]
            self.last_conf = float(prob[i])
            self.count += 1
            if self.last_conf < 0.6:
                self.lb_cls.config(text=f"未知  (可能是 {CN[self.last]})")
            else:
                self.lb_cls.config(text=f"{self.last}  {CN[self.last]}")
            self.lb_conf.config(text=f"置信度 {self.last_conf:.3f}")
            self.lb_ms.config(text=f"推理 {self.last_ms:.1f} ms")
            self.lb_cnt.config(text=f"累计检测 {self.count}")
            self.lb_fa.config(text=f"标记误报 {self.fa}")

            disp = img.copy()
            d = ImageDraw.Draw(disp)
            d.rectangle([0, 0, disp.width, 54], fill=(0, 0, 0))
            d.text((14, 12), f"{CN[self.last]}  {self.last_conf:.2f}", fill=(255, 210, 74))
            disp = disp.resize((W, int(disp.height * W / disp.width)))
            self.photo = ImageTk.PhotoImage(disp)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, image=self.photo, anchor="nw")

            self.check_uncertain(img)

            el = int(time.time() - self.t0)
            self.lb_time.config(text=f"已运行 {el//3600:02d}:{el%3600//60:02d}:{el%60:02d}")

            now = time.time()
            if now >= self.next_log:
                self.next_log = now + 2.0
                self.write_row()

        self.root.after(120, self.loop)

    def check_uncertain(self, img):
        """置信度低于 0.6 时自动截图存档，每 20 秒最多一张。"""
        if self.last_conf >= 0.6:
            return
        now = time.time()
        if now - self.last_unc < 20:
            return
        self.last_unc = now
        ts = datetime.datetime.now()
        fn = self.unc_dir / f"{ts:%Y%m%d_%H%M%S}_{self.last}_{self.last_conf:.2f}.jpg"
        try:
            img.save(fn, "JPEG", quality=85)
        except Exception:
            return
        f = LOGS / f"uncertain_{ts.date().isoformat()}.csv"
        new = not f.exists()
        with f.open("a", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["ts", "pred", "conf", "ms", "file"])
            w.writerow([ts.strftime("%Y-%m-%d %H:%M:%S"), self.last, f"{self.last_conf:.4f}",
                        f"{self.last_ms:.2f}", fn.name])
        self.n_unc += 1
        self.lb_unc.config(text=f"低置信度截图 {self.n_unc}")
        self.log(f"低置信 {self.last} {self.last_conf:.2f} → {fn.name}")

    def write_row(self):
        if self.last is None:
            return
        det, _ = self.paths()
        new = not det.exists()
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with det.open("a", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["ts", "pred", "conf", "ms", "reject"])
            w.writerow([ts, self.last, f"{self.last_conf:.4f}", f"{self.last_ms:.2f}",
                        1 if self.last_conf < 0.6 else 0])
        self.last_logged = (ts, self.last, self.last_conf)

    def mark_fa(self):
        if self.last_logged is None:
            return
        _, fa, _ = self.paths()
        new = not fa.exists()
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with fa.open("a", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["ts", "last_ts", "last_pred"])
            w.writerow([ts, self.last_logged[0], self.last_logged[1]])
        self.fa += 1
        self.lb_fa.config(text=f"标记误报 {self.fa}")
        self.log(f"误报 @ {ts}  pred={self.last_logged[1]}")

    def close(self):
        try:
            if self.cam:
                self.cam.stop()
        except Exception:
            pass
        self.root.destroy()


if __name__ == "__main__":
    guard = socket.socket()
    try:
        guard.bind(("127.0.0.1", 50993))
    except OSError:
        print("已经有一个运行器在跑了")
        raise SystemExit(0)
    if not MODEL.exists():
        print("缺少模型：", MODEL)
        raise SystemExit(1)
    root = tk.Tk()
    Runner(root)
    root.mainloop()
