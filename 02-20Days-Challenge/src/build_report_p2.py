# -*- coding: utf-8 -*-
"""生成项目二报告（docx）。"""
import csv
import json
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from PIL import Image, ImageDraw, ImageFont

BASE = Path(r"D:\项目三")
LOGS = BASE / "logs"
RES = BASE / "results"
FIG = BASE / "figs_report"
FIG.mkdir(exist_ok=True)
OUT = BASE / "02_项目二_报告.docx"

BODY, HEAD = "宋体", "微软雅黑"
INK, GREY = (35, 35, 35), (120, 120, 120)
BLUE, RED, ORANGE = (40, 90, 170), (198, 60, 50), (222, 140, 40)
FB = r"C:\Windows\Fonts\msyhbd.ttc"
FR = r"C:\Windows\Fonts\msyh.ttc"


def font(sz, bold=True):
    return ImageFont.truetype(FB if bold else FR, sz)


def load_day(d):
    p = LOGS / f"detect_{d}.csv"
    rows = list(csv.DictReader(p.open(encoding="utf-8-sig")))
    for r in rows:
        r["c"] = float(r["conf"])
        r["m"] = float(r["ms"])
        r["r"] = r.get("reject", "0") == "1"
        r["h"] = int(r["ts"][11:13]) + int(r["ts"][14:16]) / 60.0
    return rows


DAYS = ["2026-11-16", "2026-11-17", "2026-11-18"]
DATA = {d: load_day(d) for d in DAYS}


def stats(rows):
    n = len(rows)
    return {
        "n": n,
        "conf": sum(r["c"] for r in rows) / n,
        "ms": sum(r["m"] for r in rows) / n,
        "rej": sum(1 for r in rows if r["r"]),
    }


ALL = [r for d in DAYS for r in DATA[d]]

import datetime as _dt


def _span(rows):
    a = _dt.datetime.strptime(rows[0]["ts"], "%Y-%m-%d %H:%M:%S")
    b = _dt.datetime.strptime(rows[-1]["ts"], "%Y-%m-%d %H:%M:%S")
    return (b - a).total_seconds() / 60.0


TOTAL_MIN = sum(_span(DATA[d]) for d in DAYS)


def fig_timeline(path):
    """三天置信度随时间的走势（5 分钟窗口）。"""
    W, H = 1500, 620
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    d.text((36, 20), "三天连续运行：置信度随时间的走势（5 分钟窗口均值）", font=font(26), fill=INK)
    x0, y0, x1, y1 = 90, 90, W - 60, H - 110

    def px(hh):
        return x0 + (x1 - x0) * (hh - 12.5) / 8.0

    def py(v):
        return y1 - (y1 - y0) * v

    for hh in range(13, 21):
        d.line([(px(hh), y0), (px(hh), y1)], fill=(238, 238, 238))
        d.text((px(hh) - 12, y1 + 8), f"{hh}:00", font=font(14, False), fill=GREY)
    for v in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        d.line([(x0, py(v)), (x1, py(v))], fill=(240, 240, 240))
        d.text((44, py(v) - 8), f"{v:.1f}", font=font(13, False), fill=GREY)
    d.line([(x0, py(0.6)), (x1, py(0.6))], fill=(220, 60, 50), width=2)

    cols = {"2026-11-16": (60, 110, 180), "2026-11-17": (60, 150, 110), "2026-11-18": (200, 120, 40)}
    for day in DAYS:
        rows = DATA[day]
        buckets = {}
        for r in rows:
            k = round(r["h"] * 12) / 12
            buckets.setdefault(k, []).append(r["c"])
        pts = [(px(k), py(sum(v) / len(v))) for k, v in sorted(buckets.items())]
        d.line(pts, fill=cols[day], width=3)
    d.text((x0 + 10, y0 - 26), "红线 = 拒识阈值 0.6", font=font(15, False), fill=(200, 60, 50))
    y = y0 + 4
    for day in DAYS:
        d.line([(W - 250, y), (W - 210, y)], fill=cols[day], width=4)
        d.text((W - 200, y - 9), day[5:], font=font(15, False), fill=INK)
        y += 26
    im.save(path)


def fig_segments(path):
    """三段工况的拒识率与平均置信度。"""
    W, H = 1200, 560
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    d.text((36, 20), "同一模型、同一机位：只改一个条件，拒识率从 0.2% 到 78.8%", font=font(24), fill=INK)
    segs = [("常规物件\n14:24–15:33", 0.20, 0.9966, (70, 150, 90)),
            ("外观变体\n（白蓝盒）", 13.43, 0.8222, (222, 150, 40)),
            ("光照失效\n（17–18 时关灯）", 78.82, 0.5061, (190, 60, 50)),
            ("恢复光照\n同一白蓝盒", 17.36, 0.8152, (60, 130, 190))]
    x0, y0, x1, y1 = 110, 100, W - 70, H - 90
    bw = (x1 - x0) / len(segs)
    for i, (name, rj, conf, col) in enumerate(segs):
        hgt = (y1 - y0) * rj / 100.0
        cx = x0 + bw * i + bw * 0.22
        d.rectangle([cx, y1 - hgt, cx + bw * 0.56, y1], fill=col)
        d.text((cx + 6, y1 - hgt - 30), f"{rj:.1f}%", font=font(22), fill=col)
        d.text((cx + 6, y1 - hgt - 60), f"置信度 {conf:.2f}", font=font(15, False), fill=GREY)
        for j, line in enumerate(name.split("\n")):
            d.text((x0 + bw * i + 18, y1 + 14 + j * 22), line, font=font(15, False), fill=INK)
    d.line([(x0, y1), (x1, y1)], fill=(80, 80, 80), width=2)
    d.text((40, y0 - 10), "拒识率", font=font(16), fill=GREY)
    im.save(path)


def fig_dist(path):
    """三天置信度分布对比。"""
    import collections
    W, H = 1300, 500
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    d.text((36, 18), "置信度分布：三天合计", font=font(24), fill=INK)
    bins = [i / 20 for i in range(20)]
    x0, y0, x1, y1 = 90, 80, W - 50, H - 70
    hist = collections.Counter()
    for r in ALL:
        hist[min(int(r["c"] * 20), 19)] += 1
    mx = max(hist.values())
    bw = (x1 - x0) / 20
    for i in range(20):
        h = hist.get(i, 0) / mx * (y1 - y0)
        col = (70, 150, 90) if bins[i] >= 0.8 else ((222, 150, 40) if bins[i] >= 0.6 else (190, 60, 50))
        d.rectangle([x0 + i * bw + 2, y1 - h, x0 + (i + 1) * bw - 2, y1], fill=col)
    for v in (0.6, 0.8):
        d.line([(x0 + v * 20 * bw, y0), (x0 + v * 20 * bw, y1)], fill=(120, 120, 120), width=2)
        d.text((x0 + v * 20 * bw - 16, y0 - 22), f"{v:.1f}", font=font(14), fill=GREY)
    d.line([(x0, y1), (x1, y1)], fill=(80, 80, 80), width=2)
    for i in range(0, 21, 2):
        d.text((x0 + i * bw - 8, y1 + 8), f"{i/20:.1f}", font=font(12, False), fill=GREY)
    d.text((W - 330, H - 40), "绿 ≥0.8 / 橙 0.6–0.8 / 红 <0.6", font=font(14, False), fill=GREY)
    im.save(path)


TL = FIG / "timeline.png"
SG = FIG / "segments.png"
DS = FIG / "dist.png"
fig_timeline(TL)
fig_segments(SG)
fig_dist(DS)


def set_font(run, name=BODY, size=10.5, bold=False, color=None):
    run.font.name = name
    run.font.size = Pt(size)
    run.font.bold = bold
    if color:
        run.font.color.rgb = color
    rPr = run._element.get_or_add_rPr()
    rf = rPr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rPr.append(rf)
    for a in ("w:ascii", "w:hAnsi", "w:eastAsia"):
        rf.set(qn(a), name)


def para(doc, text, size=10.5, indent=0.74, space=6):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space)
    p.paragraph_format.line_spacing = 1.45
    if indent:
        p.paragraph_format.first_line_indent = Cm(indent)
    set_font(p.add_run(text), BODY, size)
    return p


def head(doc, text, size=13.5):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(5)
    p.paragraph_format.keep_with_next = True
    set_font(p.add_run(text), HEAD, size, True, RGBColor(0x1F, 0x3A, 0x5F))


def table(doc, header, rows, widths, fsize=9):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]
        c.text = ""
        r = c.paragraphs[0].add_run(h)
        set_font(r, HEAD, 9.5, True)
        c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            r = cells[i].paragraphs[0].add_run(str(v))
            set_font(r, BODY, fsize)
            cells[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    for i, w in enumerate(widths):
        for row in t.rows:
            row.cells[i].width = Cm(w)
    for row in t.rows:
        row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))


doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21), Cm(29.7)
for a in ("left_margin", "right_margin", "top_margin"):
    setattr(sec, a, Cm(2.2))
sec.bottom_margin = Cm(2.0)
normal = doc.styles["Normal"]
normal.font.name = BODY
normal.font.size = Pt(10.5)
normal.element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), BODY)

p = doc.add_paragraph()
set_font(p.add_run("项目二　低成本桌面级智能视觉状态检测与抗光照干扰实证"), HEAD, 18, True,
         RGBColor(0x1F, 0x3A, 0x5F))
p.paragraph_format.space_after = Pt(3)
p = doc.add_paragraph()
set_font(p.add_run(f"260 张多光照数据集 ｜ 跨光照评估 ｜ 三天连续运行 {len(ALL):,} 次检测 ｜ 2026-11-15 至 11-18"),
         BODY, 9, color=RGBColor(0x77, 0x77, 0x77))

head(doc, "一、场景与任务")
para(doc, "本工作把题目里的「药盒核验」做成桌面版本：桌上放一件物品，系统回答两个问题——"
          "第一，桌上有没有东西；第二，如果有，是哪一件。对应 5 类物件加 1 个「空桌面」类。")
para(doc, "选这个口径而不是坐姿监测，是因为它天然包含「拒识」：桌上空着、或者放了一件不认识的东西，"
          "系统都应该说不知道，而不是硬塞进 5 类里的某一类。这一点决定了后面所有的评估方式。")

head(doc, "二、数据采集")
para(doc, "5 件物件按表面光学性质挑选，而不是按品类：白色按压瓶（漫反射曲面）、深色充电宝（低反光）、"
          "金属钥匙（镜面）、透明玻璃瓶（折射）、亮面包装盒（高光）。这五种表面恰好对应五种失效模式："
          "弱光下白件过曝、深色件暗部丢失、金属高光、透明件边缘消失、亮面件高光盖轮廓。")
para(doc, "四种光照：L1 均匀顶光、L2 弱光（灯打墙反射）、L3 强反光（灯直射物件）、L4 阴影遮挡（侧光加挡板）。"
          "每格 12 张，共 240 张，另拍 20 张空桌面用于标定拒识。全部由本人实拍，JPEG 内含 EXIF 拍摄时间"
          "（例如 2026:11:15 20:54:58），与拍摄脚本写出的 session_log.csv 逐张对得上。")
para(doc, "一个关键细节：相机默认开启自动曝光，会把四种光照「抹平」——实测平均亮度 L1/L3/L4 是 149/143/138，"
          "几乎没差别。关掉自动曝光与低照度补偿后重新拍摄，光照差异才真正进入数据。")

head(doc, "三、模型与训练")
para(doc, "MobileNetV3-Small 预训练权重微调，输入 224×224，AdamW，25 epoch，CPU 训练约 3–10 分钟。"
          "训练时用随机裁剪、翻转、旋转与颜色抖动做增强。")
para(doc, "评估采用「跨光照」协议：只用 L1/L2 训练，用完全没见过的 L3/L4 测试。"
          "这比随机划分严格得多——同一格 12 张几乎重复，随机划分会泄漏，测出来的 100% 不可信。")

head(doc, "四、离线结果")
m1 = json.loads((RES / "metrics_crosslighting_none.json").read_text(encoding="utf-8"))
m2 = json.loads((RES / "metrics_crosslighting_clahe.json").read_text(encoding="utf-8"))
table(doc, ["方案", "L3 强反光", "L4 阴影", "整体"],
      [["基线（无归一化）", f"{m1['per_light']['L3']['acc']*100:.1f}%", f"{m1['per_light']['L4']['acc']*100:.1f}%",
        f"{m1['val_acc']*100:.1f}%"],
       ["＋CLAHE 局部对比度归一化", f"{m2['per_light']['L3']['acc']*100:.1f}%",
        f"{m2['per_light']['L4']['acc']*100:.1f}%", f"{m2['val_acc']*100:.1f}%"]],
      [6.5, 3.4, 3.0, 2.8])
para(doc, "基线在强反光下只有 90.8%，看混淆矩阵，错的 6 张全部是透明玻璃瓶被判成亮面包装盒——"
          "强光下玻璃边缘消失，只剩一块高光。在 LAB 空间对亮度通道做 CLAHE（clipLimit=2，8×8 网格）后，"
          "L3/L4 全部答对。")
doc.add_picture(str(RES / "cm_crosslighting_none.png"), width=Cm(11.4))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
para(doc, "图 1　跨光照混淆矩阵（基线）：6 个错误集中在 O4→O5", 9, indent=0)

head(doc, "五、连续运行")
st = {d: stats(DATA[d]) for d in DAYS}
table(doc, ["日期", "时段", "检测次数", "平均置信度", "拒识率", "推理耗时"],
      [[d[5:], f"{DATA[d][0]['ts'][11:16]}–{DATA[d][-1]['ts'][11:16]}", f"{st[d]['n']:,}",
        f"{st[d]['conf']:.4f}", f"{st[d]['rej']/st[d]['n']*100:.2f}%", f"{st[d]['ms']:.1f} ms"] for d in DAYS],
      [2.2, 3.6, 2.8, 3.0, 2.4, 2.8])
para(doc, "注：11-16 的日志是早期 4 列格式，没有拒识标记，拒识率按 0 计；11-17 起日志增加 reject 列。", 9, indent=0.74)
para(doc, f"三天累计 {len(ALL):,} 次检测、{int(TOTAL_MIN // 60)} 小时 {int(TOTAL_MIN % 60)} 分运行时长，平均置信度 "
          f"{sum(r['c'] for r in ALL)/len(ALL):.4f}。")

doc.add_picture(str(TL), width=Cm(15.4))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
para(doc, "图 2　三天置信度走势：11-18 傍晚整体下坠，对应关灯", 9, indent=0)

head(doc, "六、四种工况：拒识率的阶梯")
para(doc, "11-18 一天里恰好出现了四种工况，同模型、同机位、同一张桌子，每次只改一个条件：")
table(doc, ["工况", "说明", "次数", "拒识率", "平均置信度"],
      [["A 常规物件", "训练集里见过的五件", "1969", "0.20%", "0.9966"],
       ["B 外观变体", "白蓝包装盒（训练集是黄盒）", "2516", "13.43%", "0.8222"],
       ["C 光照失效", "17–18 时关灯，画面平均亮度 19.9", "1700", "78.82%", "0.5061"],
       ["D 恢复光照", "同一白蓝盒，灯重新打开", "3012", "17.36%", "0.8152"]],
      [3.0, 7.4, 2.2, 2.4, 2.8])
doc.add_picture(str(SG), width=Cm(14.4))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
para(doc, "图 3　四段工况的拒识率：光照一撤，拒识率立刻上一个数量级", 9, indent=0)

para(doc, "C 段最有价值。灯关掉以后画面几乎是黑的，模型没有乱认，而是持续输出低置信度的「空桌面」，"
          "78.82% 被拒识拦下。如果没有这个阈值，这一小时会变成几千条随机标签，整份日志直接作废。"
          "换句话说，系统在看不见的时候会说「不知道」，而不是硬猜——这是拒识机制真正的价值。")
para(doc, "D 段是天然对照组：灯重新打开、物件还是那个白蓝盒，拒识率立刻从 78.82% 回落到 17.36%，"
          "置信度从 0.51 回到 0.82。说明 C 段的下坠完全是光照造成的，不是模型坏了。")

doc.add_picture(str(DS), width=Cm(14.4))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
para(doc, "图 4　三天全部检测的置信度分布：0.99 附近一个高峰，0.5–0.6 一条拖尾", 9, indent=0)

head(doc, "七、阈值取舍")
para(doc, "拒识阈值定在 0.6，是用数据选的：已知物件几乎都在 0.99 以上，换件瞬间（手入镜、运动模糊）落在 0.30–0.59，"
          "而外观变体稳定贴在 0.51–0.60。")
para(doc, "代价也很清楚：阈值把「外观变体」和「未知物体」一起挡住了，于是正确判定被误拒。"
          "白蓝盒子那两个小时里，模型其实每一帧都判对了，只是不敢确认。"
          "把阈值降到 0.5 能让它通过，但手部遮挡那批 0.4–0.5 的帧会一起放进来——这是一笔明账。")
para(doc, "报告里保留 0.6，是因为这份数据的用途是「记录可信判定」，宁可少说，不可说错。")

head(doc, "八、改进方向")
para(doc, "① 补样本：把白蓝盒子按四种光照各拍 12 张补进训练集，重训后可直接检验「类内差异」是否被消除。", indent=0.74)
para(doc, "② 补光实验：在 L3 强反光下加一盏侧向补光，比较补光前后的准确率。题目要求的「补光/算法改进」"
          "目前只做了算法改进（CLAHE），补光这一支还没有实测数据。", indent=0.74)
para(doc, "③ 阈值分场景：已知类别用 0.6，未知物体检测用更低的阈值配合时序平滑（连续 N 帧一致才输出）。", indent=0.74)

head(doc, "九、局限")
para(doc, "① 数据规模小：5 类共 240 张训练图，测试集只有 130 张，单次评估的置信区间较宽。", indent=0.74)
para(doc, "② 拍摄环境是固定的书桌隔层，属于受控暗箱，不是开放桌面；换个背景需要重新评估。", indent=0.74)
para(doc, "③ 连续运行的场景单一：三天里主要是单件摆放，没有覆盖多物件同框与计数。", indent=0.74)
para(doc, "④ 自动曝光关闭后，环境变暗会直接导致失效；实际部署需要补光或自动增益补偿。", indent=0.74)

head(doc, "十、结论")
para(doc, "① 在受控光照下，5 类物件 + 空桌面可以被稳定区分，分布内置信度 0.99 以上。", indent=0.74)
para(doc, "② 真正让模型失效的是「没见过的外观」和「没光」，不是强弱光本身；跨光照掉 5 个点，"
          "CLAHE 可以补回来，而外观差异需要补样本。", indent=0.74)
para(doc, "③ 拒识机制是这套系统里最值得保留的部分：它把两类失效都变成了「未知」，"
          "而不是错误答案。代价是误拒，这个代价目前是值得的。", indent=0.74)

doc.save(OUT)
print("saved", OUT)

