# -*- coding: utf-8 -*-
"""
项目二 · 多光照数据集拍摄器（ffmpeg 版）
按空格或点按钮拍照，自动落到 data/<光照>/<物件>/ 下，右上角显示该拍什么。
摄像头按设备名选择，可避开手机虚拟摄像头 / OBS 虚拟设备。
"""
import csv
import datetime
import io
import json
import os
import queue
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk

try:
    import imageio_ffmpeg
    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG = "ffmpeg"

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CONF = ROOT / "config.json"
LOG = ROOT / "session_log.csv"

DEFAULT_CONF = {
    "per_cell": 12,
    "empty_per_lighting": 5,
    "camera_name": "",
    "lights": [
        {"code": "L1", "name": "均匀顶光"},
        {"code": "L2", "name": "弱光"},
        {"code": "L3", "name": "强反光"},
        {"code": "L4", "name": "阴影遮挡"},
    ],
    "objects": [
        {"code": "O1", "name": "白瓷杯"},
        {"code": "O2", "name": "深色充电宝"},
        {"code": "O3", "name": "金属钥匙"},
        {"code": "O4", "name": "玻璃杯"},
        {"code": "O5", "name": "亮面包装盒"},
    ],
    "empty_name": "空桌面",
}

PW, PH = 940, 700
CREATE_NO_WINDOW = 0x08000000


def load_conf():
    conf = dict(DEFAULT_CONF)
    if CONF.exists():
        try:
            conf.update(json.loads(CONF.read_text(encoding="utf-8")))
        except Exception:
            pass
    else:
        CONF.write_text(json.dumps(DEFAULT_CONF, ensure_ascii=False, indent=2), encoding="utf-8")
    return conf


def save_conf(conf):
    CONF.write_text(json.dumps(conf, ensure_ascii=False, indent=2), encoding="utf-8")


def list_devices():
    """返回 [(名字, 是否虚拟)]，顺序同 DirectShow。"""
    try:
        out = subprocess.run([FFMPEG, "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
                             capture_output=True, text=True, encoding="utf-8", errors="ignore",
                             creationflags=CREATE_NO_WINDOW)
        names = re.findall(r'"([^"]+)" \(video\)', out.stderr or "")
    except Exception:
        names = []
    res = []
    for n in names:
        low = n.lower()
        virtual = any(k in low for k in ("virtual", "virtualcamera", "obs", "webcastmate", "虚拟"))
        res.append((n, virtual))
    return res


class Cam:
    """常驻 ffmpeg 进程，把 MJPEG 流拆成 PIL 图像。"""

    def __init__(self, name, fps=10):
        self.name = name
        self.q = queue.Queue(maxsize=2)
        self.proc = None
        self.alive = False
        self.error = ""
        cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-f", "dshow", "-i", f"video={name}",
               "-video_size", "1280x720", "-framerate", "30",
               "-vf", f"fps={fps}", "-f", "mjpeg", "-q:v", "4", "-"]
        try:
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                         bufsize=0, creationflags=CREATE_NO_WINDOW)
        except Exception as e:
            self.error = str(e)
            return
        self.alive = True
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        buf = b""
        while self.alive:
            chunk = self.proc.stdout.read(8192)
            if not chunk:
                self.alive = False
                break
            buf += chunk
            while True:
                i = buf.find(b"\xff\xd8")
                j = buf.find(b"\xff\xd9", i + 2) if i >= 0 else -1
                if i < 0 or j < 0:
                    if len(buf) > 4_000_000:
                        buf = b""
                    break
                data, buf = buf[i:j + 2], buf[j + 2:]
                try:
                    img = Image.open(io.BytesIO(data)).convert("RGB")
                except Exception:
                    continue
                try:
                    if self.q.full():
                        self.q.get_nowait()
                    self.q.put(img)
                except Exception:
                    pass
        try:
            err = self.proc.stderr.read().decode("utf-8", "ignore").strip()
            if err:
                self.error = err.splitlines()[-1]
        except Exception:
            pass

    def latest(self, timeout=0.8):
        try:
            return self.q.get(timeout=timeout)
        except Exception:
            return None

    def stop(self):
        self.alive = False
        try:
            if self.proc and self.proc.poll() is None:
                self.proc.kill()
        except Exception:
            pass


class App:
    def __init__(self, root):
        self.root = root
        self.conf = load_conf()
        self.lights = self.conf["lights"]
        self.objects = self.conf["objects"]
        self.per_cell = int(self.conf["per_cell"])
        self.empty_n = int(self.conf["empty_per_lighting"])
        self.empty_name = self.conf["empty_name"]

        self.li, self.oi = 0, 0
        self.mode = "obj"
        self.frame = None
        self.last_path = None
        self.rect = (0, 0, 0, 0)
        self.devices = list_devices()

        self.cam = self.open_first()
        if self.cam is None:
            msg = "打不开摄像头。"
            if self.devices:
                msg += "\n\n检测到的设备：\n" + "\n".join(d[0] for d in self.devices)
            msg += "\n\n常见原因：摄像头被抖音/直播/会议软件占用。\n把它们关掉再重开程序。"
            if self.last_error:
                msg += f"\n\nffmpeg: {self.last_error[:200]}"
            messagebox.showerror("摄像头未就绪", msg)
            raise SystemExit(1)

        self.build_ui()
        usb = [d[0] for d in self.devices if "icspring" in d[0].lower()]
        if usb and self.cam.name != usb[0]:
            self.log_line(f"注意：{usb[0]} 打不开（被占用）")
            self.log_line("先关掉抖音/直播伴侣/会议软件，再重开本程序")
        else:
            self.log_line(f"已连接 {self.cam.name}")
        self.resume()
        self.poll()
        root.protocol("WM_DELETE_WINDOW", self.on_close)

    def open_first(self):
        self.last_error = ""
        real = [d[0] for d in self.devices if not d[1]]
        pref = self.conf.get("camera_name") or ""
        order = []
        if pref in real:
            order.append(pref)
        order += [n for n in real if "icspring" in n.lower() and n not in order]
        order += [n for n in real if n not in order]
        for name in order:
            cam = Cam(name)
            img = cam.latest(4.0)
            if img is not None:
                self.conf["camera_name"] = name
                save_conf(self.conf)
                return cam
            self.last_error = cam.error
            cam.stop()
        return None

    def switch_camera(self):
        real = [d[0] for d in self.devices if not d[1]]
        if len(real) < 2:
            self.log_line("只有一个可用设备")
            return
        cur = self.conf.get("camera_name")
        idx = real.index(cur) if cur in real else -1
        for k in range(1, len(real) + 1):
            name = real[(idx + k) % len(real)]
            cam = Cam(name)
            img = cam.latest(4.0)
            if img is not None:
                old = self.cam
                self.cam = cam
                self.conf["camera_name"] = name
                save_conf(self.conf)
                if old:
                    old.stop()
                self.lb_cam.config(text=name)
                self.log_line(f"已切到 {name}")
                return
            cam.stop()
        self.log_line("其他设备都打不开（可能被占用）")

    def build_ui(self):
        r = self.root
        r.title("项目二 · 多光照数据集拍摄器")
        r.configure(bg="#111")
        r.geometry("1280x780")
        r.resizable(False, False)

        self.canvas = tk.Canvas(r, width=PW, height=PH, bg="#000", highlightthickness=0)
        self.canvas.place(x=0, y=0)

        panel = tk.Frame(r, bg="#111", width=340)
        panel.place(x=PW, y=0, width=340, height=780)

        tk.Label(panel, text="当前拍摄", bg="#111", fg="#888", font=("微软雅黑", 11)).place(x=18, y=12)
        self.lb_target = tk.Label(panel, text="", bg="#111", fg="#FFD24A", font=("微软雅黑", 19, "bold"),
                                  wraplength=310, justify="left")
        self.lb_target.place(x=16, y=36)
        self.lb_sub = tk.Label(panel, text="", bg="#111", fg="#DDD", font=("微软雅黑", 12))
        self.lb_sub.place(x=18, y=116)

        self.pb = ttk.Progressbar(panel, length=300, mode="determinate")
        self.pb.place(x=18, y=150)
        self.lb_prog = tk.Label(panel, text="", bg="#111", fg="#888", font=("微软雅黑", 9))
        self.lb_prog.place(x=18, y=172)
        self.lb_cam = tk.Label(panel, text="", bg="#111", fg="#666", font=("微软雅黑", 8), anchor="w")
        self.lb_cam.place(x=18, y=192, width=300)

        tk.Label(panel, text="光照", bg="#111", fg="#666", font=("微软雅黑", 8)).place(x=18, y=210)
        self.light_btns = []
        for i, L in enumerate(self.lights):
            b = tk.Button(panel, text=f"{L['code']} {L['name']}", command=lambda i=i: self.goto_light(i),
                          bg="#333", fg="#CCC", bd=0, font=("微软雅黑", 8), takefocus=0, wraplength=70)
            b.place(x=18 + i * 76, y=226, width=72, height=30)
            self.light_btns.append(b)

        tk.Label(panel, text="物件", bg="#111", fg="#666", font=("微软雅黑", 8)).place(x=18, y=262)
        self.obj_btns = []
        for i, O in enumerate(self.objects):
            b = tk.Button(panel, text=f"{O['code']} {O['name']}", command=lambda i=i: self.goto_obj(i),
                          bg="#333", fg="#CCC", bd=0, font=("微软雅黑", 8), takefocus=0, wraplength=56)
            b.place(x=18 + i * 61, y=278, width=57, height=34)
            self.obj_btns.append(b)

        tk.Label(panel, text="拍摄序列", bg="#111", fg="#666", font=("微软雅黑", 8)).place(x=18, y=320)
        self.txt = tk.Text(panel, width=40, height=7, bg="#1b1b1b", fg="#AAA", bd=0,
                           font=("Consolas", 8), state="disabled")
        self.txt.place(x=18, y=338)

        tk.Button(panel, text="拍 照   (空格)", command=self.capture, bg="#2E7D32", fg="white",
                  activebackground="#388E3C", bd=0, font=("微软雅黑", 12, "bold"), height=2,
                  takefocus=0).place(x=18, y=470, width=300)
        tk.Button(panel, text="撤回上一张  (Backspace)", command=self.undo, bg="#37474F", fg="white",
                  bd=0, font=("微软雅黑", 9), takefocus=0).place(x=18, y=536, width=148)
        tk.Button(panel, text="跳过本组  (S)", command=self.skip, bg="#455A64", fg="white",
                  bd=0, font=("微软雅黑", 9), takefocus=0).place(x=172, y=536, width=146)
        tk.Button(panel, text="打开数据文件夹", command=self.open_folder, bg="#37474F", fg="white",
                  bd=0, font=("微软雅黑", 9), takefocus=0).place(x=18, y=574, width=148)
        tk.Button(panel, text="清空已拍照片", command=self.clear_all, bg="#7F1D1D", fg="#FFD7D7",
                  activebackground="#991B1B", bd=0, font=("微软雅黑", 9), takefocus=0).place(x=172, y=574, width=146)

        tk.Label(panel, text="点上面按钮选光照/物件；空格拍照\n← → 切物件   ↑ ↓ 切光照   C 切摄像头   Esc 退出",
                 bg="#111", fg="#666", font=("微软雅黑", 8), justify="left").place(x=18, y=612)

        self.txt.configure(takefocus=0)
        self.canvas.configure(takefocus=1)
        self.canvas.focus_set()

        def key(fn):
            def h(e):
                fn()
                return "break"
            return h

        r.bind_all("<space>", key(self.capture))
        r.bind_all("<Return>", key(self.capture))
        r.bind_all("<BackSpace>", key(self.undo))
        r.bind_all("<Escape>", key(self.on_close))
        r.bind_all("<Key-s>", key(self.skip))
        r.bind_all("<Key-c>", key(self.switch_camera))
        r.bind_all("<Right>", key(lambda: self.step_obj(1)))
        r.bind_all("<Left>", key(lambda: self.step_obj(-1)))
        r.bind_all("<Down>", key(lambda: self.step_light(1)))
        r.bind_all("<Up>", key(lambda: self.step_light(-1)))
        for i in range(4):
            r.bind_all(str(i + 1), key(lambda i=i: self.goto_light(i)))
        for i, ch in enumerate("abcde"):
            r.bind_all(ch, key(lambda i=i: self.goto_obj(i)))

    # ---------- 预览 ----------
    def poll(self):
        img = self.cam.latest()
        if img is not None:
            self.frame = img
            w, h = img.size
            s = min(PW / w, PH / h)
            nw, nh = int(w * s), int(h * s)
            im = img.resize((nw, nh), Image.LANCZOS)
            self.photo = ImageTk.PhotoImage(im)
            self.canvas.delete("vid")
            self.canvas.create_image(PW // 2, PH // 2, image=self.photo, tags="vid")
            newrect = ((PW - nw) // 2, (PH - nh) // 2, nw, nh)
            if self.rect != newrect:
                self.rect = newrect
                self.draw_guides()
        elif self.cam.alive is False:
            self.canvas.delete("vid")
            self.canvas.create_text(PW // 2, PH // 2, text="没有画面\n（设备被占用或已断开）", fill="#FF6B6B",
                                    font=("微软雅黑", 20), justify="center", tags="vid")
        self.root.after(40, self.poll)

    def draw_guides(self):
        self.canvas.delete("guide")
        x, y, w, h = self.rect
        if w == 0:
            return
        bw, bh = int(w * 0.55), int(h * 0.55)
        bx, by = x + (w - bw) // 2, y + (h - bh) // 2
        self.canvas.create_rectangle(bx, by, bx + bw, by + bh, outline="#FFD24A", width=2, dash=(6, 4), tags="guide")
        cx, cy = PW // 2, PH // 2
        self.canvas.create_line(cx - 12, cy, cx + 12, cy, fill="#FFD24A", tags="guide")
        self.canvas.create_line(cx, cy - 12, cx, cy + 12, fill="#FFD24A", tags="guide")

    # ---------- 状态 ----------
    def cell_dir(self, li=None, oi=None, mode=None):
        li = self.li if li is None else li
        oi = self.oi if oi is None else oi
        mode = self.mode if mode is None else mode
        L = self.lights[li]
        ldir = f"{L['code']}_{L['name']}"
        if mode == "empty":
            return DATA / ldir / self.empty_name
        O = self.objects[oi]
        return DATA / ldir / f"{O['code']}_{O['name']}"

    def count(self, li=None, oi=None, mode=None):
        d = self.cell_dir(li, oi, mode)
        return len(list(d.glob("*.jpg"))) if d.exists() else 0

    def cell_total(self):
        return self.per_cell if self.mode == "obj" else self.empty_n

    def resume(self):
        for li in range(len(self.lights)):
            for oi in range(len(self.objects)):
                if self.count(li, oi, "obj") < self.per_cell:
                    self.li, self.oi, self.mode = li, oi, "obj"
                    self.refresh()
                    return
            if self.count(li, None, "empty") < self.empty_n:
                self.li, self.mode = li, "empty"
                self.refresh()
                return
        self.refresh()

    def refresh(self):
        L = self.lights[self.li]
        n = self.count()
        total = self.cell_total()
        if self.mode == "empty":
            self.lb_target.config(text=f"{L['code']}  {L['name']}\n空桌面")
        else:
            O = self.objects[self.oi]
            self.lb_target.config(text=f"{L['code']}  {L['name']}\n{O['code']}  {O['name']}")
        self.lb_sub.config(text=f"第 {min(n + 1, total)} / {total} 张")
        self.pb["maximum"] = total
        self.pb["value"] = n
        done = sum(self.count(li, oi, "obj") for li in range(len(self.lights)) for oi in range(len(self.objects)))
        done += sum(self.count(li, None, "empty") for li in range(len(self.lights)))
        total_all = len(self.lights) * (len(self.objects) * self.per_cell + self.empty_n)
        self.lb_prog.config(text=f"总计 {done} / {total_all} 张")
        if hasattr(self, "light_btns"):
            for i, b in enumerate(self.light_btns):
                on = i == self.li
                b.configure(bg="#FFD24A" if on else "#333", fg="#111" if on else "#CCC")
            for i, b in enumerate(self.obj_btns):
                on = (i == self.oi and self.mode == "obj")
                b.configure(bg="#FFD24A" if on else "#333", fg="#111" if on else "#CCC")

    def log_line(self, s):
        self.txt.config(state="normal")
        self.txt.insert("end", s + "\n")
        self.txt.see("end")
        self.txt.config(state="disabled")

    # ---------- 拍照 ----------
    def capture(self):
        img = self.frame
        if img is None:
            self.log_line("没有画面，拍不了")
            return
        d = self.cell_dir()
        d.mkdir(parents=True, exist_ok=True)
        n = self.count() + 1
        if n > self.cell_total():
            self.next_cell()
            return
        lcode = self.lights[self.li]["code"]
        oname = self.objects[self.oi]["code"] if self.mode == "obj" else "EMPTY"
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = d / f"{lcode}_{oname}_{n:02d}_{ts}.jpg"

        exif = Image.Exif()
        now = datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S")
        exif[271] = "UVC Camera"
        exif[272] = "USB Camera"
        exif[274] = 1
        exif[306] = now
        exif[36867] = now
        exif[36868] = now
        img.save(path, "JPEG", quality=95, exif=exif)
        self.last_path = path

        with LOG.open("a", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow([now, lcode, self.lights[self.li]["name"], oname, n, path.name, str(path)])
        self.flash()
        self.refresh()
        if n >= self.cell_total():
            self.next_cell()

    def flash(self):
        self.canvas.create_rectangle(0, 0, PW, PH, outline="#4CAF50", width=8, tags="flash")
        self.root.after(90, lambda: self.canvas.delete("flash"))

    def undo(self):
        if self.last_path is None or not self.last_path.exists():
            return
        try:
            self.last_path.unlink()
            self.log_line(f"已撤回 {self.last_path.name}")
        except Exception as e:
            self.log_line(f"撤回失败 {e}")
        self.last_path = None
        self.refresh()

    def next_cell(self):
        if self.mode == "obj":
            if self.oi + 1 < len(self.objects):
                self.oi += 1
            else:
                self.mode = "empty"
        else:
            if self.li + 1 < len(self.lights):
                self.li += 1
                self.oi = 0
                self.mode = "obj"
            else:
                self.log_line("全部拍完")
        self.refresh()

    def skip(self):
        self.next_cell()

    def set_mode(self, m):
        self.mode = m
        self.refresh()

    def goto_light(self, li):
        self.li = li
        self.oi = 0
        self.mode = "obj"
        self.refresh()

    def goto_obj(self, oi):
        if oi < len(self.objects):
            self.oi = oi
            self.mode = "obj"
            self.refresh()

    def step_obj(self, d):
        self.mode = "obj"
        self.oi = (self.oi + d) % len(self.objects)
        self.refresh()

    def step_light(self, d):
        self.li = (self.li + d) % len(self.lights)
        self.refresh()

    def open_folder(self):
        DATA.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(DATA))
        except Exception as e:
            self.log_line(f"打开失败：{e}")

    def clear_all(self):
        import shutil
        files = list(DATA.rglob("*.jpg")) if DATA.exists() else []
        n = len(files)
        if n == 0:
            messagebox.showinfo("清空", "还没有拍过照片。")
            return
        if not messagebox.askyesno("确认清空",
                                   f"将删除 {n} 张照片，不可恢复。\n\n确定要清空吗？",
                                   icon="warning", default="no"):
            return
        if n >= 50 and not messagebox.askyesno("再次确认",
                                               f"真的要删掉 {n} 张？\n\n删了就找不回来了。",
                                               icon="warning", default="no"):
            return
        failed = 0
        for f in files:
            try:
                f.unlink()
            except Exception:
                failed += 1
        for d in sorted([x for x in DATA.rglob("*") if x.is_dir()], key=lambda x: -len(x.parts)):
            try:
                d.rmdir()
            except OSError:
                pass
        if LOG.exists():
            bak = LOG.with_name(f"session_log_bak_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv")
            try:
                shutil.move(str(LOG), str(bak))
            except Exception:
                pass
        self.last_path = None
        self.li, self.oi, self.mode = 0, 0, "obj"
        self.refresh()
        self.log_line(f"已清空 {n} 张照片" + (f"，{failed} 张失败" if failed else ""))

    def on_close(self):
        try:
            self.cam.stop()
        except Exception:
            pass
        self.root.destroy()


def selftest():
    devs = list_devices()
    print("devices:")
    for n, v in devs:
        print("  ", n, "(virtual)" if v else "")
    real = [n for n, v in devs if not v]
    for n in real:
        cam = Cam(n)
        img = cam.latest(4.0)
        if img is not None:
            p = ROOT / "_selftest"
            p.mkdir(exist_ok=True)
            f = p / "t.jpg"
            img.save(f, "JPEG", quality=92)
            print("CAPTURED", n, img.size, f)
            cam.stop()
            return 0
        print("FAIL", n, cam.error[:80])
        cam.stop()
    return 1


if __name__ == "__main__":
    guard = socket.socket()
    try:
        guard.bind(("127.0.0.1", 50992))
    except OSError:
        r0 = tk.Tk()
        r0.withdraw()
        messagebox.showwarning("已在运行", "已经开着一个拍摄窗口了，先关掉它再打开新的。")
        raise SystemExit(0)

    if "--test" in sys.argv:
        raise SystemExit(selftest())

    root = tk.Tk()
    app = App(root)
    if "--demo" in sys.argv:
        root.after(4000, app.on_close)
        root.mainloop()
        print("GUI_OK")
        raise SystemExit(0)
    root.mainloop()
