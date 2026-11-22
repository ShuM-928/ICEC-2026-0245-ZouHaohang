# -*- coding: utf-8 -*-
"""项目二：无窗口后台检测器（不显示界面，只写日志）。

用法：
    python runner_headless.py            # 一直跑，Ctrl+C 结束
    python runner_headless.py --minutes 240   # 跑 4 小时后自动停

日志：logs/detect_YYYYMMDD.csv（每 2 秒一行）
      logs/fa_YYYYMMDD.csv（手动标记的误报，可在有界面模式下按 F）
"""
import argparse
import csv
import datetime
import time
from pathlib import Path

import torch

from runner import build_model, TF, CLASSES, CN, LOGS
from camera_shoot import Cam, list_devices, load_conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=0, help="0 表示一直跑")
    ap.add_argument("--interval", type=float, default=2.0, help="写日志间隔（秒）")
    args = ap.parse_args()

    name = load_conf().get("camera_name") or ""
    real = [n for n, v in list_devices() if not v]
    if name not in real:
        name = real[0] if real else ""
    if not name:
        print("没有可用摄像头")
        return 1

    model = build_model()
    cam = Cam(name)
    print(f"摄像头：{name}")
    print(f"日志目录：{LOGS}")
    print("开始检测，Ctrl+C 结束")

    t_end = time.time() + args.minutes * 60 if args.minutes else None
    n, err = 0, 0
    t_start = time.time()
    next_report = t_start + 60

    try:
        while True:
            if t_end and time.time() >= t_end:
                break
            img = cam.latest(1.0)
            if img is None:
                err += 1
                if err % 30 == 1:
                    print(f"[{datetime.datetime.now():%H:%M:%S}] 没有画面，等待重连…")
                time.sleep(0.5)
                continue

            x = TF(img).unsqueeze(0)
            t1 = time.perf_counter()
            with torch.no_grad():
                out = model(x)
            ms = (time.perf_counter() - t1) * 1000
            prob = torch.softmax(out, 1)[0]
            i = int(prob.argmax())
            pred, conf = CLASSES[i], float(prob[i])
            n += 1

            d = datetime.date.today().isoformat()
            f = LOGS / f"detect_{d}.csv"
            new = not f.exists()
            with f.open("a", newline="", encoding="utf-8-sig") as fh:
                w = csv.writer(fh)
                if new:
                    w.writerow(["ts", "pred", "conf", "ms"])
                w.writerow([datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), pred, f"{conf:.4f}", f"{ms:.2f}"])

            if time.time() >= next_report:
                next_report = time.time() + 60
                dur = (time.time() - t_start) / 60
                print(f"[{datetime.datetime.now():%H:%M:%S}] 已跑 {dur:.0f} 分钟，"
                      f"{n} 次检测，当前 {CN[pred]} {conf:.2f}，{ms:.1f} ms")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n手动停止")
    finally:
        cam.stop()

    dur = (time.time() - t_start) / 60
    print(f"结束：运行 {dur:.0f} 分钟，{n} 次检测，断流 {err} 次")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
