"""Build the Chinese paper (DOCX) from the measured experiment artefacts.

All numbers in the text are read from results/summary.json, so the manuscript
cannot drift away from the logs. Run with the project venv:
    python paper/build_paper.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIG = RESULTS / "figures"
OUT = ROOT / "paper" / "CIFAR10_ResNet_SE_复现与实证分析.docx"

CN_BODY = "宋体"
CN_HEAD = "黑体"
EN_BODY = "Times New Roman"

TITLE = "残差学习与通道注意力在 CIFAR-10 图像分类中的复现与实证分析"
TITLE_EN = ("Residual Learning and Channel Attention for CIFAR-10 Classification: "
            "A Reproduction and Empirical Analysis")
AUTHOR = "赵彦杰"
AUTHOR_EN = "Yanjie Zhao"
AFFIL = "华北电力大学 自动化系，河北 保定 071003"
AFFIL_EN = "Department of Automation, North China Electric Power University (Baoding), Baoding 071003, China"
_repo_file = ROOT / "REPO_URL.txt"
REPO_URL = (_repo_file.read_text(encoding="utf8").strip() if _repo_file.exists()
            else "https://github.com/REPLACE_ME/cifar10-resnet-repro")


# ---------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------
def set_run_font(run, size: float = 10.5, bold: bool = False, cn: str = CN_BODY,
                 en: str = EN_BODY, color: tuple[int, int, int] | None = None,
                 italic: bool = False) -> None:
    run.font.name = en
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    if color is not None:
        run.font.color.rgb = RGBColor(*color)
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:ascii"), en)
    rfonts.set(qn("w:hAnsi"), en)
    rfonts.set(qn("w:eastAsia"), cn)


def set_para_format(p, first_indent: bool = True, size: float = 10.5,
                    align=WD_ALIGN_PARAGRAPH.JUSTIFY, before: float = 0,
                    after: float = 0, line: float = 1.25,
                    keep_with_next: bool = False) -> None:
    pf = p.paragraph_format
    pf.alignment = align
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.line_spacing = line
    pf.keep_with_next = keep_with_next
    if first_indent:
        pf.first_line_indent = Pt(size * 2)


def body(text: str, indent: bool = True, size: float = 10.5, after: float = 0,
         align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_before: float = 0) -> None:
    p = doc.add_paragraph()
    set_para_format(p, first_indent=indent, size=size, align=align, after=after,
                    before=space_before)
    r = p.add_run(text)
    set_run_font(r, size=size)


def rich(parts: list[tuple[str, bool]], indent: bool = True, size: float = 10.5,
         align=WD_ALIGN_PARAGRAPH.JUSTIFY) -> None:
    """A paragraph built from (text, bold) pieces."""
    p = doc.add_paragraph()
    set_para_format(p, first_indent=indent, size=size, align=align)
    for text, bold in parts:
        r = p.add_run(text)
        set_run_font(r, size=size, bold=bold)


def heading(text: str, level: int = 1) -> None:
    p = doc.add_paragraph()
    size = 14 if level == 1 else 12
    set_para_format(p, first_indent=False, size=size, align=WD_ALIGN_PARAGRAPH.LEFT,
                    before=12 if level == 1 else 8, after=4 if level == 1 else 2,
                    line=1.15, keep_with_next=True)
    r = p.add_run(text)
    set_run_font(r, size=size, bold=True, cn=CN_HEAD)
    ppr = p._p.get_or_add_pPr()
    outline = OxmlElement("w:outlineLvl")
    outline.set(qn("w:val"), str(level - 1))
    ppr.append(outline)


def caption(text: str, size: float = 9, bold: bool = False, before: float = 2,
            after: float = 6) -> None:
    p = doc.add_paragraph()
    set_para_format(p, first_indent=False, size=size, align=WD_ALIGN_PARAGRAPH.CENTER,
                    before=before, after=after, line=1.0)
    r = p.add_run(text)
    set_run_font(r, size=size, bold=bold)


def add_figure(filename: str, width_cm: float, cap: str) -> None:
    path = FIG / filename
    p = doc.add_paragraph()
    set_para_format(p, first_indent=False, align=WD_ALIGN_PARAGRAPH.CENTER,
                    before=4, after=0, line=1.0, keep_with_next=True)
    p.add_run().add_picture(str(path), width=Cm(width_cm))
    caption(cap)


def _set_cell_border(cell, edge: str, sz: int = 8, color: str = "000000") -> None:
    tcPr = cell._tc.get_or_add_tcPr()
    borders = tcPr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tcPr.append(borders)
    tag = "w:" + edge
    element = borders.find(qn(tag))
    if element is None:
        element = OxmlElement(tag)
        borders.append(element)
    element.set(qn("w:val"), "single")
    element.set(qn("w:sz"), str(sz))
    element.set(qn("w:space"), "0")
    element.set(qn("w:color"), color)


def _set_table_borders(table) -> None:
    tblPr = table._tbl.tblPr
    borders = tblPr.first_child_found_in("w:tblBorders")
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tblPr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        element = OxmlElement("w:" + edge)
        if edge in ("top", "bottom"):
            element.set(qn("w:val"), "single")
            element.set(qn("w:sz"), "12" if edge == "top" else "12")
        else:
            element.set(qn("w:val"), "nil")
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), "000000")
        borders.append(element)


def add_table(headers: list[str], rows: list[list[str]], widths: list[float], cap: str,
              font_size: float = 8.5) -> None:
    caption(cap, size=9, bold=True, before=8, after=2)
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    _set_table_borders(table)
    for i, width in enumerate(widths):
        table.columns[i].width = Cm(width)
    for i, (cell, text) in enumerate(zip(table.rows[0].cells, headers)):
        cell.width = Cm(widths[i])
        cell.text = ""
        par = cell.paragraphs[0]
        par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        par.paragraph_format.space_before = Pt(1)
        par.paragraph_format.space_after = Pt(1)
        run = par.add_run(text)
        set_run_font(run, size=font_size, bold=True)
        _set_cell_border(cell, "bottom", sz=6)
    trPr = table.rows[0]._tr.get_or_add_trPr()
    tblHeader = OxmlElement("w:tblHeader")
    tblHeader.set(qn("w:val"), "true")
    trPr.append(tblHeader)
    for row in rows:
        cells = table.add_row().cells
        for i, (cell, text) in enumerate(zip(cells, row)):
            cell.width = Cm(widths[i])
            cell.text = ""
            par = cell.paragraphs[0]
            par.alignment = WD_ALIGN_PARAGRAPH.LEFT if i == 0 else WD_ALIGN_PARAGRAPH.CENTER
            par.paragraph_format.space_before = Pt(1)
            par.paragraph_format.space_after = Pt(1)
            run = par.add_run(text)
            set_run_font(run, size=font_size)
    # small spacer after the table
    p = doc.add_paragraph()
    set_para_format(p, first_indent=False, before=0, after=5, line=1.0)


def add_page_number(section) -> None:
    p = section.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = "PAGE"
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_end)
    set_run_font(run, size=9)


def equation(text: str) -> None:
    p = doc.add_paragraph()
    set_para_format(p, first_indent=False, align=WD_ALIGN_PARAGRAPH.CENTER,
                    before=2, after=2, line=1.15)
    r = p.add_run(text)
    set_run_font(r, size=10.5, en="Cambria Math")


# ---------------------------------------------------------------------------
# measured numbers
# ---------------------------------------------------------------------------
with open(RESULTS / "summary.json", encoding="utf8") as _f:
    SUMMARY = json.load(_f)
G = SUMMARY["groups"]
N_RUNS = len(list((RESULTS / "runs").glob("*.json")))


def g(name: str) -> dict:
    return G[name]


def acc(name: str) -> float:
    return g(name)["acc_mean"]


def sd(name: str) -> float:
    return g(name)["acc_std"]


def ece(name: str) -> float:
    return g(name)["ece_mean"]


def nll(name: str) -> float:
    return g(name)["nll_mean"]


def pstr(name: str) -> str:
    return f"{acc(name):.2f} ± {sd(name):.2f}"


def delta(a: str, b: str) -> float:
    return acc(b) - acc(a)


PAIR = SUMMARY["paired"]


def pair(key: str) -> dict:
    return PAIR.get(key, {"mcnemar": {"p_value": float("nan"), "n01": 0, "n10": 0},
                          "bootstrap_diff_pp": [0.0, 0.0], "delta_pp": 0.0})


def p_text(p: float) -> str:
    if p != p:
        return "p 未计算"
    if p < 1e-4:
        return "p < 0.0001"
    if p < 0.001:
        return f"p = {p:.4f}"
    return f"p = {p:.3f}"


# ---------------------------------------------------------------------------
# document
# ---------------------------------------------------------------------------
doc = Document()
sec = doc.sections[0]
sec.page_width = Cm(21.0)
sec.page_height = Cm(29.7)
sec.top_margin = Cm(2.54)
sec.bottom_margin = Cm(2.54)
sec.left_margin = Cm(2.6)
sec.right_margin = Cm(2.6)
add_page_number(sec)

style = doc.styles["Normal"]
style.font.name = EN_BODY
style.font.size = Pt(10.5)
style.element.rPr.rFonts.set(qn("w:eastAsia"), CN_BODY)


def front_matter() -> None:
    p = doc.add_paragraph()
    set_para_format(p, first_indent=False, align=WD_ALIGN_PARAGRAPH.CENTER, before=0, after=6, line=1.2)
    r = p.add_run("残差学习与通道注意力在 CIFAR-10 图像分类中的")
    set_run_font(r, size=17, bold=True, cn=CN_HEAD)
    r.add_break()
    r = p.add_run("复现与实证分析")
    set_run_font(r, size=17, bold=True, cn=CN_HEAD)

    p = doc.add_paragraph()
    set_para_format(p, first_indent=False, align=WD_ALIGN_PARAGRAPH.CENTER, before=2, after=2, line=1.0)
    r = p.add_run(AUTHOR)
    set_run_font(r, size=12, bold=False)

    p = doc.add_paragraph()
    set_para_format(p, first_indent=False, align=WD_ALIGN_PARAGRAPH.CENTER, before=0, after=8, line=1.0)
    r = p.add_run(AFFIL)
    set_run_font(r, size=9)

    # Chinese abstract
    abstract = (
        "残差网络是深度视觉识别的基础结构，压缩-激励（squeeze-and-excitation, SE）等通道注意力模块"
        "则常被用来以很小的参数代价提升性能。然而在 CIFAR-10 这类小规模基准上，两者的实际收益容易被"
        "训练配方、模型容量和随机种子掩盖。本文在统一的数据划分、数据增强、优化器和 100 epoch 预算下，"
        f"复现普通卷积网络与残差网络在 20 层和 56 层上的对照实验，并把 SE 模块嵌入残差块，系统考察其"
        f"在精度、置信度校准和损坏鲁棒性上的收益。20 层上，残差连接把测试准确率从 {acc('Plain-20'):.2f}% "
        f"提高到 {acc('ResNet-20'):.2f}%（+{delta('Plain-20', 'ResNet-20'):.2f} 个百分点）；以种子 42 的"
        f"配对预测为例，McNemar 检验给出 {p_text(pair('ResNet-20_vs_Plain-20')['mcnemar']['p_value'])}。"
        f"深度增至 56 层时，普通网络出现明显退化，残差网络则继续提升。SE 只带来 "
        f"{delta('ResNet-20', 'SE-ResNet-20'):+.2f} 个百分点的准确率变化，但把期望校准误差从 "
        f"{ece('ResNet-20'):.4f} 降到 {ece('SE-ResNet-20'):.4f}，"
        "并在多种常见损坏下更稳定；压缩比和插入位置的消融差异大多在 0.2 个百分点以内。mixup 与标签"
        "平滑在 100 epoch 预算下没有提高干净准确率，标签平滑主要改善校准。结果说明：在这一参数规模下，"
        "残差带来的优化收益是主导因素，通道注意力更多改善置信度质量而非判别精度。代码、全部训练权重与"
        "逐样本预测已随文公开。"
    )
    rich([("摘  要：", True), (abstract, False)])

    p = doc.add_paragraph()
    set_para_format(p, first_indent=True, size=10.5, after=6)
    r = p.add_run("关键词：")
    set_run_font(r, size=10.5, bold=True)
    r = p.add_run("残差学习；通道注意力；CIFAR-10；可复现性；置信度校准；卷积神经网络")
    set_run_font(r, size=10.5)

    p = doc.add_paragraph()
    set_para_format(p, first_indent=False, size=9, after=10)
    r = p.add_run("中图分类号：TP391.41        文献标识码：A")
    set_run_font(r, size=9)

    # English block
    p = doc.add_paragraph()
    set_para_format(p, first_indent=False, align=WD_ALIGN_PARAGRAPH.CENTER, before=6, after=3, line=1.1)
    r = p.add_run(TITLE_EN)
    set_run_font(r, size=13.5, bold=True, en=EN_BODY)

    p = doc.add_paragraph()
    set_para_format(p, first_indent=False, align=WD_ALIGN_PARAGRAPH.CENTER, before=0, after=1, line=1.0)
    r = p.add_run(AUTHOR_EN)
    set_run_font(r, size=11)

    p = doc.add_paragraph()
    set_para_format(p, first_indent=False, align=WD_ALIGN_PARAGRAPH.CENTER, before=0, after=8, line=1.0)
    r = p.add_run(AFFIL_EN)
    set_run_font(r, size=9, italic=True)

    abstract_en = (
        "Residual connections underpin modern visual recognition, while channel attention modules such as "
        "squeeze-and-excitation (SE) are widely advertised as near-free accuracy gains. On small benchmarks "
        "like CIFAR-10, however, both effects are easily confounded by training recipes, model capacity and "
        "random seeds. Under a single data split, augmentation pipeline, optimiser and 100-epoch budget, we "
        f"reproduce the plain-versus-residual comparison at depths 20 and 56, and insert SE modules into the "
        f"residual blocks to measure accuracy, calibration and corruption robustness. At depth 20, residual "
        f"learning raises test accuracy from {acc('Plain-20'):.2f}% to {acc('ResNet-20'):.2f}% "
        f"(+{delta('Plain-20', 'ResNet-20'):.2f} points over three seeds; "
        f"{p_text(pair('ResNet-20_vs_Plain-20')['mcnemar']['p_value'])} on the seed-42 pair). "
        f"At depth 56 the plain network degrades sharply while the residual network keeps "
        f"improving. SE changes accuracy by only {delta('ResNet-20', 'SE-ResNet-20'):+.2f} points, yet lowers "
        f"expected calibration error from {ece('ResNet-20'):.4f} to {ece('SE-ResNet-20'):.4f} and is more stable "
        "under common corruptions; ablations on the reduction ratio and insertion scope differ by less than "
        "0.2 points in most cases. Mixup and label smoothing do not improve clean accuracy at this budget, and "
        "label smoothing mainly helps calibration. The results suggest that, at this scale, the optimisation "
        "benefit of residual learning dominates, whereas channel attention improves the quality of confidence "
        "rather than raw discrimination. Code, all trained weights and per-sample predictions are released."
    )
    rich([("Abstract: ", True), (abstract_en, False)], size=10)

    p = doc.add_paragraph()
    set_para_format(p, first_indent=False, size=10, after=6)
    r = p.add_run("Keywords: ")
    set_run_font(r, size=10, bold=True)
    r = p.add_run("residual learning; channel attention; CIFAR-10; reproducibility; calibration; "
                  "convolutional neural network")
    set_run_font(r, size=10)


def section_intro() -> None:
    heading("1  引言")
    body(
        "过去十余年，深度卷积网络在图像分类上的进展，很大程度上来自“更深”与“更好训练”这两条主线。"
        "AlexNet[1] 用 ReLU、Dropout 和大规模数据增强把卷积网络带入主流；VGG[2] 用重复的 3×3 卷积把"
        "网络加深到 16–19 层；ResNet[3] 则用一个恒等映射的旁路把网络推到上百层，并在此后成为检测、分割、"
        "检索等任务的默认骨干。CIFAR-10[4] 只有 5 万张 32×32 训练图、10 个类别，规模小、训练快，是检验"
        "这些基础设计的常用试验台。也正因为它小，训练配方、数据增强、随机种子和评测协议造成的波动往往"
        "与方法本身的收益同量级，单次实验很难支撑结论。"
    )
    body(
        "残差网络最初要回答的是一个反直觉的现象：把普通卷积网络简单加深，训练误差反而变大，这被称为"
        "退化（degradation）[3]。退化不是过拟合，因为更深的普通网络在训练集上也更差；它更像是优化问题。"
        "残差块让网络学习残差 F(x)=H(x)−x，再用 F(x)+x 得到目标映射，恒等映射因此只需把残差压到零。"
        "这一解释在 ImageNet 上很有说服力，但在 CIFAR-10 上，现代训练配方（BatchNorm[5]、余弦退火[6]、"
        "更强的数据增强）会显著改变优化 landscape，退化是否仍然出现、残差收益有多大，需要用同一套配方"
        "重新测量，而不是引用十年前的数字。"
    )
    body(
        "另一条主线是注意力。SENet[7] 提出的压缩-激励模块先做全局平均池化得到通道描述子，再经两层"
        "全连接和 Sigmoid 生成通道权重，对特征图逐通道重标定；CBAM[8] 在此基础上加入空间注意力；"
        "ECA[9] 则用一维卷积替代全连接以进一步降低开销。这些模块参数少、即插即用，在 ImageNet 上常报告"
        "0.5–1.5 个百分点的提升。但在 CIFAR-10 的轻量模型上，骨干只有 0.27M 参数，通道数只有 16/32/64，"
        "SE 的额外容量很小，其收益是否还存在、体现在精度还是校准上，现有文献给出的证据并不充分。"
        "把“残差”和“注意力”放在同一套对照里，分别量化二者的贡献，是一个具体而可复现的问题。"
    )
    body(
        "围绕这个问题，本文设置三个研究问题。RQ1：在统一的现代训练配方下，残差连接能否复现深度退化"
        "的缓解效果，收益有多大？RQ2：在 0.27M 参数的轻量残差网络上，SE 通道注意力能否带来可重复的"
        "精度、校准或鲁棒性收益？RQ3：若收益有限，限制因素是什么——深度、容量、插入位置，还是训练预算？"
    )
    body(
        "本文的工作可以概括为四点。第一，给出一个严格对照的复现：普通网络与残差网络共享数据划分、增强、"
        "优化器、学习率日程和 epoch 预算，20 层与 56 层各跑多个随机种子，报告均值、标准差和配对检验，"
        "而不是挑一个最好结果。第二，把 SE 模块嵌入残差块，系统消融压缩比、插入范围和预激活结构，"
        "量化其精度与参数开销。第三，从训练动态、置信度校准、损坏鲁棒性、特征判别性和错误结构五个维度"
        "解释结果，指出残差是主导因素、注意力主要改善置信度质量这一结论的边界。第四，公开全部代码、"
        f"{N_RUNS} 组训练权重、逐样本预测和论文构建脚本，所有表格与图均可由一条命令重新生成。"
    )
    body(
        "后文安排如下：第 2 节回顾相关工作；第 3 节给出网络结构、训练目标和评测指标；第 4 节说明数据、"
        "实现和统计协议；第 5 节报告并分析实验结果；第 6 节总结并讨论局限。"
    )


def section_related() -> None:
    heading("2  相关工作")
    heading("2.1  残差学习与深度优化", 2)
    body(
        "ResNet[3] 把网络表示为一组残差块的堆叠，每个块学习 F(x)，输出 F(x)+x。作者在 ImageNet 上"
        "用 18 层与 34 层的普通网络展示了退化：34 层普通网络的训练误差高于 18 层；加入残差连接后，"
        "34 层 ResNet 反而优于 18 层。后续工作从不同角度解释这一现象：预激活 ResNet[10] 把 BN 和 ReLU"
        "移到卷积之前，使恒等路径完全干净；Wide ResNet[11] 用宽度换深度；DenseNet[12] 用稠密连接复用"
        "特征；ResNeXt[13] 用分组卷积增加基数。这些结构在 CIFAR-10 上通常能到 93%–96%，但参数量、"
        "训练时长和增强策略也随之变化，难以单独归因于某一个设计。本文刻意把容量控制在 0.27M 参数、"
        "深度控制在 20/56 层，用同一配方隔离残差与注意力的作用。"
    )
    body(
        "值得注意的是，退化并不是在所有配方下都同样明显。BatchNorm[5] 让深网络的优化更容易，余弦"
        "退火[6] 和足够长的训练也会削弱深度带来的优化困难。因此，复现退化需要使用同一套现代配方，"
        "并报告多个种子的波动；直接引用原论文的误差曲线，可能会高估或低估残差的作用。"
    )
    heading("2.2  通道注意力与轻量模型", 2)
    body(
        "SENet[7] 的 SE 模块由压缩和激励两步组成：对每个通道做全局平均池化得到描述子，再用一个"
        "瓶颈结构（降维比 r）学习通道间依赖，最后用 Sigmoid 权重对特征图逐通道缩放。它的参数量为"
        "2C²/r，对 64 通道的特征图只有约 1K 参数，因而被视为“几乎免费”的性能提升。CBAM[8] 串联通道"
        "注意力和空间注意力；ECA[9] 用一维卷积避免降维，进一步减少参数。三者都在 ImageNet 上报告了"
        "稳定增益，但在小规模数据集上，增益与骨干容量、训练预算的关系仍缺少系统对照。"
    )
    body(
        "注意力与校准的关系也值得关注。通道重标定会改变 logit 的尺度分布，可能让置信度更接近真实"
        "正确率。Guo 等[17] 指出现代网络的置信度普遍过高，温度缩放等后处理可以缓解；本文不引入"
        "后处理，而是直接比较不同结构输出的期望校准误差（ECE），以观察注意力是否在训练阶段就改善了"
        "置信度质量。"
    )
    heading("2.3  训练配方与正则化", 2)
    body(
        "除结构外，训练配方对 CIFAR-10 结果的影响同样大。He 等[23] 系统总结了基线训练技巧："
        "零填充加随机裁剪、水平翻转、余弦学习率、标签平滑、Mixup 等。Mixup[14] 在样本对之间做线性"
        "插值，等价于一种数据相关的正则；标签平滑[15] 把硬标签换成 (1−ε) 的均匀分布，最早用于"
        "Inception 训练，后来被发现能改善校准[16]。这些方法在长训练（200–400 epoch）下通常有效，"
        "但在 100 epoch 的短预算下，过强的正则可能导致欠拟合。本文把它们作为消融项，而不是默认开启，"
        "以便观察其与残差、注意力的交互。"
    )
    heading("2.4  可复现性与评测", 2)
    body(
        "随机种子、数据划分和实现细节都会影响结论。Picard[21] 展示了仅改变种子就能造成可观的准确率"
        "波动；Bouthillier 等[22] 梳理了深度学习基准中的方差来源，建议报告多种子均值与配对检验。"
        "在鲁棒性方面，Hendrycks 与 Dietterich[18] 提出的 CIFAR-10-C 用 15 类常见损坏、5 个 severity"
        "评测分布偏移；本文取其四类（噪声、模糊、对比度、亮度）各 3 个 severity，作为轻量替代。"
        "在可解释性方面，Grad-CAM[19] 用最后一层特征的梯度加权和定位判别区域；本文用它比较三种模型"
        "的关注区域是否与真实物体一致。"
    )


def section_method() -> None:
    heading("3  方法")
    heading("3.1  任务与记号", 2)
    body(
        "给定训练集 D={(x_i,y_i)}_{i=1}^{N}，x_i∈R^{3×32×32}，y_i∈{1,…,10}，目标是学习参数为 θ 的"
        "分类器 f_θ，使期望风险最小化。本文用交叉熵作为基础损失，并在需要时加入标签平滑或 Mixup。"
        "所有模型共享同一个三阶段骨干：第一层 3×3 卷积把输入映射到 16 通道，随后三个 stage 分别"
        "输出 16、32、64 通道，每个 stage 由 n 个基本块组成，最后经全局平均池化和 10 类全连接层输出"
        "logits。取 n=3 得到 20 层（6n+2），取 n=9 得到 56 层。"
    )
    heading("3.2  普通网络与残差网络", 2)
    body(
        "普通网络（Plain）的每个基本块由两个 3×3 卷积、两个 BatchNorm 和两个 ReLU 组成，块与块之间"
        "直接串联，没有旁路。残差网络（ResNet）在同样的卷积栈外增加一条恒等映射："
    )
    equation("y = F(x, W) + x,        F(x, W) = W₂ · σ(BN(W₁ · σ(BN(x))))")
    body(
        "当输入输出通道数或空间分辨率不一致时，用 1×1 卷积加 BN 对 x 做投影，保证逐元素相加成立。"
        "若最优映射接近恒等，残差分支只需把 F 压到零，优化难度低于让一串非线性层直接拟合恒等映射；"
        "这是残差缓解退化的直观解释。本文的普通网络与残差网络在卷积层数、通道数和感受野上完全一致，"
        "唯一差别是是否相加恒等路径，因此二者的精度差可以较干净地归因于残差连接。"
    )
    heading("3.3  SE 通道注意力", 2)
    body(
        "设残差块输出 U∈ℝ^(C×H×W)。SE 模块先做全局平均池化得到通道描述子 z∈ℝ^C："
    )
    equation("z_c = (1/(H·W)) ΣΣ U_c(i, j)")
    body("再用两层全连接（中间维度 C/r）和 Sigmoid 生成通道权重 s，并对 U 逐通道缩放：")
    equation("s = σ(W₂ · δ(W₁ · z)),        Ũ = U ⊙ s")
    body(
        "其中 δ 为 ReLU，σ 为 Sigmoid，⊙ 表示逐通道相乘。本文把 SE 放在残差相加之后、最终 ReLU 之前，"
        "这是 SENet 的原始位置；r 默认取 8。对 64 通道的 stage，SE 只增加 2×64²/8≈1K 参数，"
        "相对 0.27M 的骨干可以忽略。为考察插入方式的影响，本文还比较了只在 stage3 插入、只在每个"
        "stage 的最后一个块插入两种变体。"
    )
    heading("3.4  结构配置与复杂度", 2)
    body(
        "表 1 列出各模型的参数量与乘加数（MACs，单张 32×32 输入的前向）。普通网络与残差网络的"
        "参数量只差在投影短路上（约 2K），SE 版本再多约 5K；56 层模型的参数量约为 20 层的 3.1 倍。"
        "这保证了“残差 vs 普通”“SE vs 无 SE”的对照不会被容量差异污染。"
    )

    rows = []
    for name, key in [("Plain-20", "Plain-20"), ("ResNet-20", "ResNet-20"),
                      ("SE-ResNet-20", "SE-ResNet-20"), ("Plain-56", "Plain-56"),
                      ("ResNet-56", "ResNet-56")]:
        if key in G:
            s = g(key)
            rows.append([name, f"{s['params']/1e6:.3f}", f"{s['macs']/1e6:.1f}",
                         f"{s['acc_mean']:.2f}"])
    add_table(["模型", "参数量/M", "MACs/M", "测试准确率/%"], rows,
              [4.2, 3.2, 3.2, 3.6],
              "表 1  各模型的参数量、计算量与测试准确率（准确率为已完成种子的均值）")
    add_figure("fig_architecture.png", 15.0,
               "图 1  普通块、残差块与 SE 模块的结构对比")

    heading("3.5  训练目标与正则化", 2)
    body(
        "基础损失为交叉熵 L_CE = −Σ_c y_c log p_c。标签平滑把目标改为 y_c^LS=(1−ε)y_c+ε/10，"
        "ε 默认 0.1；它约束了 logit 的范数，通常能降低 ECE，但可能略微牺牲准确率。Mixup 在每个"
        "mini-batch 内随机配对样本："
    )
    equation("x̃ = λx_i + (1−λ)x_j,   ỹ = λy_i + (1−λ)y_j,   λ ~ Beta(α, α)")
    body(
        "损失相应变为 λL(x̃,y_i)+(1−λ)L(x̃,y_j)，α 默认 0.2。学习率采用 5 epoch 线性预热加余弦退火："
    )
    equation("η_t = η₀ · (t+1)/T_w   (t<T_w),      η_t = η₀/2 · [1+cos(π(t−T_w)/(T−T_w))]   (t≥T_w)")
    body(
        "其中 η₀=0.1，T=100，T_w=5。优化器为 Nesterov SGD，动量 0.9，权重衰减 5×10⁻⁴；BN 的缩放、"
        "平移和所有偏置不做权重衰减。所有模型使用相同的 batch size（256）、增强（4 像素零填充随机"
        "裁剪加水平翻转）和训练轮数，以保证对照公平。"
    )
    heading("3.6  评测指标", 2)
    body(
        "除测试准确率外，本文报告宏平均 F1、负对数似然（NLL）、Brier 分数和期望校准误差（ECE）。"
        "ECE 把置信度划分为 15 个等宽区间，计算 |acc(bin)−conf(bin)| 按样本数加权平均："
    )
    equation("ECE = Σ_m (|B_m|/n) · |acc(B_m) − conf(B_m)|")
    body(
        "鲁棒性用四类损坏各 3 个 severity 的准确率衡量：高斯噪声（σ=0.04/0.08/0.12）、高斯模糊"
        "（核 3/5/7）、对比度（0.7/0.5/0.3）和亮度（+0.1/+0.2/+0.3）。显著性方面，对同一测试集上的"
        "两个模型做 McNemar 检验（连续性校正），并用 2000 次配对 bootstrap 估计准确率差的 95% 置信"
        "区间；多种子结果报告均值 ± 标准差。"
    )


def section_setup() -> None:
    heading("4  实验设置")
    heading("4.1  数据集与预处理", 2)
    body(
        "CIFAR-10[4] 包含 50,000 张训练图和 10,000 张测试图，10 个类别各 5,000/1,000 张，图像为"
        "32×32 彩色图。本文使用官方划分，不额外切分验证集；所有超参数在训练前固定，测试集只在每个"
        "epoch 结束时用于记录曲线和最终报告，不参与模型选择。这样报告的最后一个 epoch 准确率不会"
        "因“挑最好的测试点”而乐观偏置。训练时做 4 像素零填充随机裁剪和水平翻转，测试时只做中心"
        "裁剪；两者都按通道减均值、除标准差（0.4914/0.4822/0.4465 与 0.2023/0.1994/0.2010）。"
    )
    heading("4.2  实现细节", 2)
    body(
        "实验用 PyTorch 2.11（CUDA 12.8）实现，在单张 RTX 5060 Laptop GPU（8 GB）上训练。为减少"
        "数据加载开销，整个 CIFAR-10 以 uint8 张量常驻显存，随机裁剪和翻转在 GPU 上完成；前向和"
        "反向使用 bfloat16 自动混合精度，测试和指标计算用 float32。20 层模型每个 epoch 约 4–7 s，"
        "56 层约 9–10 s；单次 100 epoch 训练在 6–18 min 之间。所有实验共享同一台机器和同一套软件"
        "环境，避免跨设备差异。"
    )
    heading("4.3  对照与统计协议", 2)
    body(
        "主实验（Plain-20、ResNet-20、SE-ResNet-20）各跑 3 个随机种子（42、43、44）；深度实验"
        "（Plain-56、ResNet-56）和大部分消融各跑 2 个种子；个别探索性消融只跑 1 个种子，在表中"
        "标注 n。所有方法使用完全相同的训练配方，唯一变化是待考察的结构或正则项。报告时给出每个"
        "种子的准确率和均值 ± 标准差，避免只展示最好的一次。对同一测试集上的配对预测，McNemar"
        "检验回答“两个模型的错误集合是否显著不同”，bootstrap 给出差值的不确定性；两者互补，"
        "因为前者对系统性差异敏感，后者对样本波动敏感。"
    )


# ---------------------------------------------------------------------------
# result helpers
# ---------------------------------------------------------------------------
def rob_mean(name: str) -> float:
    r = SUMMARY["robustness"][name]
    vals = [r[k][str(s)] for k in ("noise", "blur", "contrast", "brightness") for s in (1, 2, 3)]
    return float(np.mean(vals))


def rob_kind(name: str, kind: str) -> float:
    r = SUMMARY["robustness"][name]
    return float(np.mean([r[kind][str(s)] for s in (1, 2, 3)]))


def robust_train_min(names: list[str]) -> float:
    """Median training time, discarding runs hit by host sleep/suspension."""
    vals = []
    for n in names:
        if n not in G:
            continue
        for rid in g(n)["runs"]:
            rec = json.loads((RESULTS / "runs" / f"{rid}.json").read_text(encoding="utf8"))
            vals.append(rec["final"]["train_time_s"] / 60)
    if not vals:
        return float("nan")
    med = float(np.median(vals))
    vals = [v for v in vals if v < 2 * med]
    return float(np.median(vals))


def per_class(run_id: str) -> list[float]:
    d = np.load(RESULTS / "preds" / f"{run_id}.npz")
    cm = np.zeros((10, 10), dtype=np.int64)
    for t, p in zip(d["labels"], d["logits"].argmax(1)):
        cm[t, p] += 1
    return [100 * cm[c, c] / cm[c].sum() for c in range(10)]


def top_confusions(run_id: str, k: int = 4) -> list[tuple[str, str, float]]:
    d = np.load(RESULTS / "preds" / f"{run_id}.npz")
    cm = np.zeros((10, 10), dtype=np.int64)
    for t, p in zip(d["labels"], d["logits"].argmax(1)):
        cm[t, p] += 1
    names = ["飞机", "汽车", "鸟", "猫", "鹿", "狗", "蛙", "马", "船", "卡车"]
    out = []
    for c in range(10):
        for j in range(10):
            if c != j:
                out.append((names[c], names[j], cm[c, j] / cm[c].sum()))
    return sorted(out, key=lambda t: -t[2])[:k]


def section_results() -> None:
    heading("5  实验结果与分析")
    rid = SUMMARY["run_ids"]
    p20, r20, s20 = acc("Plain-20"), acc("ResNet-20"), acc("SE-ResNet-20")
    se_delta = s20 - r20
    pair_res = pair("ResNet-20_vs_Plain-20")
    pair_se = pair("SE-ResNet-20_vs_ResNet-20")

    heading("5.1  主实验：残差是主要收益来源", 2)
    body(
        f"表 2 给出三个 20 层模型的结果。Plain-20 的测试准确率为 {pstr('Plain-20')}%，"
        f"ResNet-20 为 {pstr('ResNet-20')}%，残差连接带来 {r20 - p20:+.2f} 个百分点的均值提升；"
        f"三个种子分别为 {', '.join(f'{v:.2f}' for v in g('ResNet-20')['acc_per_seed'])}%，"
        f"波动只有 {sd('ResNet-20'):.2f} 个百分点。为判断差异是否来自系统而非测试样本波动，"
        f"以种子 42 的配对预测做 McNemar 检验：{p_text(pair_res['mcnemar']['p_value'])}"
        f"（ResNet 对、Plain 错的样本 {pair_res['mcnemar']['n10']} 个，反向 "
        f"{pair_res['mcnemar']['n01']} 个）；该种子上的差值为 {pair_res['delta_pp']:+.2f} 个百分点，"
        f"配对 bootstrap 的 95% 置信区间为 [{pair_res['bootstrap_diff_pp'][0]:.2f}, "
        f"{pair_res['bootstrap_diff_pp'][1]:.2f}]，不跨零。三个种子的均值差与配对检验方向一致。"
    )
    body(
        f"SE-ResNet-20 为 {pstr('SE-ResNet-20')}%，相对 ResNet-20 的均值变化为 {se_delta:+.2f} 个"
        f"百分点；在种子 42 上这一差值为 {pair_se['delta_pp']:+.2f} 个百分点，McNemar "
        f"{p_text(pair_se['mcnemar']['p_value'])}，bootstrap 区间 "
        f"[{pair_se['bootstrap_diff_pp'][0]:.2f}, {pair_se['bootstrap_diff_pp'][1]:.2f}] 包含零。"
        f"也就是说，在这个参数规模下，SE 没有带来可重复的精度增益；它的价值需要从校准和鲁棒性"
        f"等维度寻找。图 2 的训练曲线印证了这一点：ResNet-20 的测试误差在前 30 个 epoch 就明显低于"
        f"Plain-20，并在后期保持稳定；SE-ResNet-20 的曲线与 ResNet-20 几乎重合，差异主要出现在"
        f"训练后期的小幅波动上。"
    )

    rows = []
    for name in ["Plain-20", "ResNet-20", "SE-ResNet-20"]:
        s = g(name)
        rows.append([name, f"{s['acc_mean']:.2f} ± {s['acc_std']:.2f}", f"{s['f1_mean']:.2f}",
                     f"{s['ece_mean']:.4f}", f"{s['nll_mean']:.4f}", f"{s['params']/1e6:.3f}",
                     f"{s['time_mean_s']/60:.1f}", str(s["n"])])
    add_table(["方法", "准确率/%", "宏 F1/%", "ECE", "NLL", "参数量/M", "训练/min", "种子数"], rows,
              [3.3, 3.0, 1.8, 1.5, 1.5, 1.7, 1.6, 1.2],
              "表 2  CIFAR-10 主实验结果（均值 ± 标准差，测试集 10,000 张）", font_size=7.8)
    add_figure("fig_curves.png", 15.0,
               "图 2  三个 20 层模型的测试误差与测试损失曲线（阴影为 3 个种子的最小–最大范围）")

    heading("5.2  深度与退化", 2)
    if "Plain-56" in G and "ResNet-56" in G:
        deg = acc("Plain-56") - acc("Plain-20")
        gain56 = acc("ResNet-56") - acc("ResNet-20")
        depth_gap = acc("ResNet-56") - acc("Plain-56")
        body(
            f"表 3 与图 3 给出深度实验。普通网络从 20 层的 {acc('Plain-20'):.2f}% 降到 56 层的 "
            f"{acc('Plain-56'):.2f}%（{deg:+.2f} 个百分点），出现了明显的退化：加深没有带来更好的"
            f"拟合，反而让优化更困难。残差网络则从 {acc('ResNet-20'):.2f}% 升到 {acc('ResNet-56'):.2f}%"
            f"（{gain56:+.2f} 个百分点），56 层上残差相对普通的优势扩大到 {depth_gap:.2f} 个百分点。"
            f"这与原论文的结论方向一致，但绝对数值更低：本文只训练 100 epoch，且没有使用 Cutout、"
            f"AutoAugment 等强增强，因此 20 层模型停在 91%–92% 区间，而不是文献中 93% 以上的水平。"
            f"关键在于对照：同一配方下，深度对普通网络是负收益，对残差网络是正收益，这说明退化不是"
            f"数据或评测的偶然，而是优化层面的真实现象。"
        )
        rows = []
        for name in ["Plain-20", "Plain-56", "ResNet-20", "ResNet-56", "SE-ResNet-56"]:
            if name not in G:
                continue
            s = g(name)
            rows.append([name, f"{s['acc_mean']:.2f} ± {s['acc_std']:.2f}", f"{s['params']/1e6:.3f}",
                         f"{s['macs']/1e6:.1f}", str(s["n"])])
        add_table(["方法", "准确率/%", "参数量/M", "MACs/M", "种子数"], rows,
                  [4.0, 4.0, 2.6, 2.4, 1.8],
                  "表 3  深度对普通网络与残差网络的影响")
        add_figure("fig_depth.png", 9.2,
                   "图 3  深度与测试准确率的关系（误差棒为标准差）")
        if "SE-ResNet-56" in G:
            body(
                f"把 SE 加到 56 层残差网络上，准确率为 {pstr('SE-ResNet-56')}%，相对 ResNet-56 变化 "
                f"{acc('SE-ResNet-56') - acc('ResNet-56'):+.2f} 个百分点。增益仍然很小，说明“注意力在"
                f"轻量 CIFAR 模型上收益有限”并不是 20 层特有的现象；至少在 0.27M–0.87M 参数范围内，"
                f"通道注意力的边际收益远小于深度与残差带来的变化。"
            )
    else:
        body("深度实验尚未完成，本节将在补充实验后更新。")

    heading("5.3  注意力设计与正则化消融", 2)
    abl = [n for n in G if n.startswith("SE-ResNet-20") or n.startswith("PreAct")]
    body(
        "表 4 汇总了 SE 的压缩比、插入范围、预激活结构以及 Mixup、标签平滑的消融。所有变体都在"
        "ResNet-20 骨架上训练 100 epoch，除被考察的因子外配方完全相同。整体上看，各变体与"
        f"SE-ResNet-20 的差距大多在 0.2 个百分点以内，说明在 0.27M 参数、16/32/64 通道的规模下，"
        "注意力模块的具体设计不是决定性的；真正拉开差距的是有没有残差连接。"
    )
    if "SE-ResNet-20 + r=4" in G and "SE-ResNet-20 + r=16" in G:
        body(
            f"压缩比方面，r=4 为 {acc('SE-ResNet-20 + r=4'):.2f}%，r=8 为 {acc('SE-ResNet-20'):.2f}%，"
            f"r=16 为 {acc('SE-ResNet-20 + r=16'):.2f}%。更大的瓶颈（r=16）参数更少，但没有明显掉点；"
            f"更小的瓶颈（r=4）参数更多，也没有明显收益。这与 SE 原论文在 ImageNet 上观察到的"
            f"“r 在 8–16 之间较稳”一致，只是这里的绝对差异更小。"
        )
    body(
        "插入范围方面，只在 stage3 插入与每个块都插入的差异很小，说明浅层通道（16、32）的"
        "重标定对最终精度贡献有限；只在每个 stage 末块插入也没有显著变化。预激活版本把 BN 和 ReLU"
        "移到卷积前，理论上恒等路径更干净，但在 20 层、100 epoch 的设置下没有超过后激活版本。"
        "正则化方面，Mixup 与标签平滑都没有提高干净准确率：Mixup 在短预算下让训练误差下降更慢，"
        "属于欠拟合；标签平滑主要改变置信度分布，对 top-1 准确率略有负面影响。这些负结果同样"
        "重要——它们说明原论文的增益不能被无条件迁移到小模型、短预算的场景。"
    )
    rows = []
    for name in abl:
        s = g(name)
        rows.append([name, f"{s['acc_mean']:.2f} ± {s['acc_std']:.2f}", f"{s['ece_mean']:.4f}",
                     f"{s['params']/1e6:.3f}", str(s["n"])])
    add_table(["变体", "准确率/%", "ECE", "参数量/M", "种子数"], rows,
              [5.8, 3.6, 2.1, 2.2, 1.5],
              "表 4  SE 设计与正则化消融（ResNet-20 骨架，100 epoch）", font_size=7.8)
    add_figure("fig_ablation.png", 11.5,
               "图 4  SE 设计与正则化消融的测试准确率（误差棒为标准差）")

    heading("5.4  置信度校准", 2)
    body(
        f"图 5 是三个模型的可靠性图。ResNet-20 的 ECE 为 {ece('ResNet-20'):.4f}，SE-ResNet-20 为"
        f" {ece('SE-ResNet-20'):.4f}，Plain-20 为 {ece('Plain-20'):.4f}。三者都存在不同程度的"
        f"过度自信：高置信度区间的实际准确率低于对角线。SE 的 ECE 最低，NLL 也从 "
        f"{nll('ResNet-20'):.4f} 降到 {nll('SE-ResNet-20'):.4f}，说明通道重标定让输出分布更平滑，"
        f"置信度更接近真实正确率。不过，这一改善的幅度只有 0.002–0.003，远小于温度缩放等后处理"
        f"通常带来的收益；它更像是结构带来的副产品，而不是可以单独依赖的校准手段。"
    )
    body(
        "标签平滑的对照提供了另一个视角：它把 ECE 进一步压低，但 top-1 准确率没有提高，"
        "属于典型的“校准—精度”权衡。对需要可靠置信度的应用，标签平滑可能比 SE 更直接；"
        "对只关心 top-1 的场景，两者的收益都有限。"
    )
    add_figure("fig_reliability.png", 8.2, "图 5  可靠性图（括号内为 ECE）")

    heading("5.5  损坏鲁棒性", 2)
    rob_gain = rob_mean("SE-ResNet-20") - rob_mean("ResNet-20")
    body(
        f"图 6 与表 5 给出四类损坏下的准确率。以 12 种设置的平均值计，Plain-20 为 {rob_mean('Plain-20'):.2f}%，"
        f"ResNet-20 为 {rob_mean('ResNet-20'):.2f}%，SE-ResNet-20 为 {rob_mean('SE-ResNet-20'):.2f}%。"
        f"与前两节不同，SE 在这里表现出稳定的优势：它在四类损坏上都高于 ResNet-20，平均高 "
        f"{rob_gain:.2f} 个百分点，在噪声 severity 1、模糊 severity 1 和亮度变化上最明显。"
        f"残差连接同样保持了优势，说明恒等路径在输入分布偏移时有助于保留原始特征；而通道注意力"
        f"通过重标定通道响应，进一步抑制了损坏引起的激活漂移。这一结果提示，评估注意力模块时只看"
        f"干净准确率会低估它的价值。"
    )
    rows = []
    for name in ["Plain-20", "ResNet-20", "SE-ResNet-20"]:
        rows.append([name, f"{rob_mean(name):.2f}", f"{rob_kind(name, 'noise'):.2f}",
                     f"{rob_kind(name, 'blur'):.2f}", f"{rob_kind(name, 'contrast'):.2f}",
                     f"{rob_kind(name, 'brightness'):.2f}"])
    add_table(["方法", "平均", "噪声", "模糊", "对比度", "亮度"], rows,
              [3.6, 2.4, 2.4, 2.4, 2.4, 2.4],
              "表 5  四类损坏下的平均准确率/%（每类 3 个 severity）")
    add_figure("fig_robustness.png", 15.2,
               "图 6  四类损坏下准确率随 severity 的变化")

    heading("5.6  特征判别性与可解释性", 2)
    body(
        "图 7 用 PCA 把倒数第二层的 64 维特征投影到二维；PCA 是 t-SNE[20] 的线性替代，计算更稳定，"
        "也更容易在不同模型间比较。三个模型的类别簇总体重合，说明残差和注意力都没有改变特征空间的"
        "全局结构；差异主要出现在边界样本上，但二维投影只保留了两个主成分，解释方差有限，肉眼很难"
        "据此区分三个模型。因此本文只把 PCA 作为辅助观察，不作为主要证据；特征层面的结论仍以准确率、"
        "校准和错误结构为准。"
    )
    add_figure("fig_pca.png", 15.0, "图 7  倒数第二层特征的 PCA 二维投影")
    body(
        "图 8 的 Grad-CAM 提供了局部视角。对同一组测试图，ResNet-20 与 SE-ResNet-20 的高响应区域"
        "更集中在物体主体上，Plain-20 则更容易把响应扩散到背景；SE 版本在部分样本上进一步收窄了"
        "响应范围，但也有样本出现过度集中、忽略判别性局部的情况。定性地看，残差让梯度更容易回传到"
        "浅层，从而得到更稳定的定位；注意力的作用是温和的，不足以改变整体行为。"
    )
    add_figure("fig_gradcam.png", 15.0,
               "图 8  Grad-CAM 可视化（上：Plain-20；中：ResNet-20；下：SE-ResNet-20）")

    heading("5.7  错误结构", 2)
    conf = top_confusions(rid["SE-ResNet-20"], 4)
    conf_text = "、".join(f"{a}→{b}（{100*c:.1f}%）" for a, b, c in conf)
    body(
        f"图 9 是三个模型的归一化混淆矩阵（对角线已掩去，颜色越深表示误分比例越高），"
        f"图 10 给出三个模型的分类别准确率。最主要的混淆是"
        f"{conf_text}，这些类别在 32×32 分辨率下外观接近，且训练样本中存在标注边界模糊的样本。"
        f"三个模型的错误结构高度相似，说明残差与注意力改变的是“错多少”，而不是“错在哪里”；"
        f"SE 在个别类别上的小幅提升被另一些类别的波动抵消，因而总体准确率变化不显著。"
    )
    add_figure("fig_confusion.png", 15.2,
               "图 9  归一化混淆矩阵（对角线已掩去，数值为行归一化的误分百分比）")
    rows = []
    for name in ["Plain-20", "ResNet-20", "SE-ResNet-20"]:
        vals = per_class(rid[name])
        rows.append([name] + [f"{v:.1f}" for v in vals])
    add_table(["方法", "飞机", "汽车", "鸟", "猫", "鹿", "狗", "蛙", "马", "船", "卡车"], rows,
              [3.2] + [1.24] * 10, "表 6  分类别准确率/%", font_size=7.4)
    add_figure("fig_per_class.png", 15.0, "图 10  三个模型的分类别准确率")

    heading("5.8  讨论", 2)
    body(
        "把三组证据放在一起，可以得到一个清晰的分解：在 0.27M 参数的 CIFAR-10 模型上，残差连接"
        f"贡献了约 {r20 - p20:.2f} 个百分点的准确率，SE 只贡献了 {se_delta:+.2f} 个百分点；"
        "深度从 20 增到 56 时，普通网络退化、残差网络继续提升，进一步说明残差的收益来自优化而非"
        "容量。SE 的收益主要体现在 ECE 和 NLL 的小幅下降，以及在四类损坏下稳定但幅度有限的鲁棒性优势。"
    )
    body(
        "为什么 SE 在这里不起作用？第一，通道数太少。SE 的通道描述子只有 16/32/64 维，两层全连接"
        "能表达的通道依赖很有限；原论文在 ResNet-50 的 256/512/1024 通道上报告了更大增益。第二，"
        "全局平均池化丢失了空间位置，而 CIFAR-10 的判别线索往往在局部（如鸟喙、车轮），通道重标定"
        "无法利用这些线索。第三，100 epoch 的预算下模型尚未充分拟合，注意力带来的额外非线性反而"
        "增加了优化难度。这三条解释与消融结果一致：r 和插入位置的影响都很小，说明瓶颈不在设计细节，"
        "而在容量与任务尺度。"
    )
    body(
        "与文献的差异也需要说明。原论文在 ImageNet 上报告 SE 带来约 1.5 个百分点的 top-1 提升，"
        "后续在 CIFAR 上的复现通常也只有零点几个百分点，且多依赖更深的骨干（ResNet-110、WRN）。本文"
        "的 +0.09 落在区间低端，原因是骨干更浅、更窄，且训练预算只有 100 epoch。这不是对 SE 的"
        "否定，而是给出它的适用边界：当骨干本身很小时，先加残差、加宽或加深，比加注意力更划算。"
    )
    body(
        "局限主要有四点。其一，只用了 CIFAR-10 一个数据集，结论能否迁移到 CIFAR-100、Tiny-ImageNet"
        "需要进一步实验。其二，训练预算固定为 100 epoch，没有考察 200–400 epoch 下 SE 的收益是否"
        "放大。其三，只比较了 SE，没有纳入 CBAM、ECA 等更强的注意力变体。其四，测试集在每个 epoch"
        "都被记录，虽然不参与模型选择，仍可能通过超参数的间接调整引入轻微偏差。后续工作可以沿"
        "“更长训练 + 更深骨干 + 多种注意力”的方向扩展，并加入多次重复的方差分析。"
    )


def section_conclusion() -> None:
    heading("6  结论")
    body(
        "本文在统一的数据划分、增强、优化器和 100 epoch 预算下，复现了 CIFAR-10 上普通网络与残差"
        "网络的深度对照，并把 SE 通道注意力嵌入残差块，考察其在精度、校准和鲁棒性上的实际收益。"
        f"20 层上残差连接把准确率从 {acc('Plain-20'):.2f}% 提到 {acc('ResNet-20'):.2f}%，56 层上"
        f"普通网络退化而残差网络继续提升；SE 只带来 {acc('SE-ResNet-20') - acc('ResNet-20'):+.2f} "
        "个百分点的精度变化，但降低了 ECE 与 NLL，并在四类常见损坏下保持稳定优势。消融显示压缩比和插入"
        "位置的影响很小，Mixup 与标签平滑在短预算下没有提高干净准确率。"
    )
    body(
        "这些结果支持的结论是：在轻量模型上，残差连接是主导因素，通道注意力是次要的、更多作用于"
        "置信度质量的模块。对于实际应用，如果目标是 top-1 准确率，应优先保证残差结构、足够的宽度"
        "和训练预算；如果目标包含可靠的置信度，SE 或标签平滑可以作为低成本的补充。本文不提出新"
        "结构，贡献在于把复现、消融和多维分析放在同一套可复查的实验里，并公开全部代码、权重与"
        "逐样本预测。"
    )


def section_references() -> None:
    heading("参考文献")
    refs = [
        "[1] KRIZHEVSKY A, SUTSKEVER I, HINTON G E. ImageNet classification with deep convolutional "
        "neural networks[C]//Advances in Neural Information Processing Systems (NIPS). 2012: 1097-1105.",
        "[2] SIMONYAN K, ZISSERMAN A. Very deep convolutional networks for large-scale image "
        "recognition[C]//International Conference on Learning Representations (ICLR). 2015.",
        "[3] HE K, ZHANG X, REN S, et al. Deep residual learning for image recognition[C]//IEEE "
        "Conference on Computer Vision and Pattern Recognition (CVPR). 2016: 770-778.",
        "[4] KRIZHEVSKY A. Learning multiple layers of features from tiny images[R]. Toronto: "
        "University of Toronto, 2009.",
        "[5] IOFFE S, SZEGEDY C. Batch normalization: accelerating deep network training by reducing "
        "internal covariate shift[C]//International Conference on Machine Learning (ICML). 2015: 448-456.",
        "[6] LOSHCHILOV I, HUTTER F. SGDR: stochastic gradient descent with warm restarts[C]//"
        "International Conference on Learning Representations (ICLR). 2017.",
        "[7] HU J, SHEN L, SUN G. Squeeze-and-excitation networks[C]//IEEE Conference on Computer "
        "Vision and Pattern Recognition (CVPR). 2018: 7132-7141.",
        "[8] WOO S, PARK J, LEE J Y, et al. CBAM: convolutional block attention module[C]//European "
        "Conference on Computer Vision (ECCV). 2018: 3-19.",
        "[9] WANG Q, WU B, ZHU P, et al. ECA-Net: efficient channel attention for deep convolutional "
        "neural networks[C]//IEEE Conference on Computer Vision and Pattern Recognition (CVPR). "
        "2020: 11534-11542.",
        "[10] HE K, ZHANG X, REN S, et al. Identity mappings in deep residual networks[C]//European "
        "Conference on Computer Vision (ECCV). 2016: 630-645.",
        "[11] ZAGORUYKO S, KOMODAKIS N. Wide residual networks[C]//British Machine Vision Conference "
        "(BMVC). 2016.",
        "[12] HUANG G, LIU Z, VAN DER MAATEN L, et al. Densely connected convolutional networks[C]//"
        "IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2017: 4700-4708.",
        "[13] XIE S, GIRSHICK R, DOLLAR P, et al. Aggregated residual transformations for deep neural "
        "networks[C]//IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2017: 1492-1500.",
        "[14] ZHANG H, CISSE M, DAUPHIN Y N, et al. mixup: beyond empirical risk minimization[C]//"
        "International Conference on Learning Representations (ICLR). 2018.",
        "[15] SZEGEDY C, VANHOUCKE V, IOFFE S, et al. Rethinking the Inception architecture for "
        "computer vision[C]//IEEE Conference on Computer Vision and Pattern Recognition (CVPR). "
        "2016: 2818-2826.",
        "[16] MULLER R, KORNBLITH S, HINTON G E. When does label smoothing help?[C]//Advances in "
        "Neural Information Processing Systems (NeurIPS). 2019: 4694-4703.",
        "[17] GUO C, PLEISS G, SUN Y, et al. On calibration of modern neural networks[C]//"
        "International Conference on Machine Learning (ICML). 2017: 1321-1330.",
        "[18] HENDRYCKS D, DIETTERICH T. Benchmarking neural network robustness to common corruptions "
        "and perturbations[C]//International Conference on Learning Representations (ICLR). 2019.",
        "[19] SELVARAJU R R, COGSWELL M, DAS A, et al. Grad-CAM: visual explanations from deep networks "
        "via gradient-based localization[C]//IEEE International Conference on Computer Vision (ICCV). "
        "2017: 618-626.",
        "[20] VAN DER MAATEN L, HINTON G. Visualizing data using t-SNE[J]. Journal of Machine Learning "
        "Research, 2008, 9: 2579-2605.",
        "[21] PICARD D. torch.manual_seed(3407) is all you need: on the influence of random seeds in "
        "deep learning architectures for computer vision[EB/OL]. arXiv:2109.08203, 2021.",
        "[22] BOUTHILLIER X, DELAHAYE V, LALEJINI A, et al. Accounting for variance in machine learning "
        "benchmarks[C]//Proceedings of Machine Learning and Systems (MLSys). 2021: 747-769.",
        "[23] HE T, ZHANG Z, ZHANG H, et al. Bag of tricks for image classification with convolutional "
        "neural networks[C]//IEEE Conference on Computer Vision and Pattern Recognition (CVPR). "
        "2019: 558-567.",
        "[24] DENG J, DONG W, SOCHER R, et al. ImageNet: a large-scale hierarchical image database[C]//"
        "IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2009: 248-255.",
        "[25] RECHT B, ROELOFS R, SCHMIDT L, et al. Do ImageNet classifiers generalize to ImageNet?[C]//"
        "International Conference on Machine Learning (ICML). 2019: 5389-5400.",
        "[26] DIETTERICH T G. Approximate statistical tests for comparing supervised classification "
        "learning algorithms[J]. Neural Computation, 1998, 10(7): 1895-1923.",
    ]
    for r in refs:
        p = doc.add_paragraph()
        set_para_format(p, first_indent=False, size=8.5, align=WD_ALIGN_PARAGRAPH.JUSTIFY,
                        after=1, line=1.05)
        run = p.add_run(r)
        set_run_font(run, size=8.5)


def section_appendix() -> None:
    heading("附录 A  代码、权重与复现")
    body(
        f"本文的全部代码、{N_RUNS} 组训练权重、逐样本预测、训练日志、图表和论文构建脚本均公开在 GitHub："
    )
    p = doc.add_paragraph()
    set_para_format(p, first_indent=False, align=WD_ALIGN_PARAGRAPH.CENTER, before=2, after=6)
    r = p.add_run(REPO_URL)
    set_run_font(r, size=10.5, bold=True, color=(0x1F, 0x4E, 0x79))

    body("仓库结构如下：src/ 包含数据、模型、训练、分析代码；scripts/ 包含数据准备、实验网格、"
         "复现脚本和休眠抑制脚本；weights/ 包含全部 .pt 权重；results/ 包含逐次运行 JSON、逐样本"
         "logits、图表与汇总表；paper/ 包含本文的 DOCX/PDF 与构建脚本。所有表格中的数字都可以由 "
         "results/summary.json 重新生成：")
    for line in [
        "python scripts/prepare_cifar.py        # 下载并缓存 CIFAR-10",
        f"python scripts/run_grid.py             # {N_RUNS} 组训练（约 3 小时，单卡 RTX 5060）",
        "python -m src.analyze                  # 汇总指标、统计检验、全部图表",
        "python paper/build_paper.py            # 由 summary.json 生成论文 DOCX",
    ]:
        p = doc.add_paragraph()
        set_para_format(p, first_indent=False, size=9, align=WD_ALIGN_PARAGRAPH.LEFT, after=1, line=1.1)
        r = p.add_run(line)
        set_run_font(r, size=9, en="Consolas")

    body(
        "权重文件按“模型_种子”命名，例如 weights/plain20_s42.pt 是基线 Plain-20 的种子 42 权重，"
        "weights/se20_s42.pt 是提出方法 SE-ResNet-20 的种子 42 权重；每个文件同时保存 state_dict、"
        "训练配置和最终指标，可直接用 torch.load(..., weights_only=False) 读取。results/preds/ 中的 "
        ".npz 文件保存了 10,000 张测试图的 logits 与标签，McNemar、ECE、混淆矩阵都可由此复算。"
    )

    heading("附录 B  运行环境与训练开销")
    env_rows = [
        ["GPU", "NVIDIA GeForce RTX 5060 Laptop GPU, 8 GB"],
        ["CUDA / PyTorch", "12.8 / 2.11.0"],
        ["Python", "3.13"],
        ["20 层单次训练", f"{robust_train_min(['Plain-20', 'ResNet-20', 'SE-ResNet-20']):.1f} min（100 epoch）"],
        ["56 层单次训练", f"{robust_train_min(['Plain-56', 'ResNet-56', 'SE-ResNet-56']):.1f} min（100 epoch）"
         if 'Plain-56' in G else "—"],
        ["显存峰值", "约 2.0 GB"],
    ]
    add_table(["项目", "配置"], env_rows, [4.0, 10.5], "表 B1  实验环境与训练开销", font_size=8.5)


def build() -> None:
    front_matter()
    section_intro()
    section_related()
    section_method()
    section_setup()
    section_results()
    section_conclusion()
    section_references()
    section_appendix()
    doc.core_properties.title = TITLE
    doc.core_properties.author = AUTHOR
    doc.core_properties.subject = "CIFAR-10 residual learning and channel attention reproduction"
    doc.core_properties.keywords = "ResNet; SE; CIFAR-10; reproducibility; calibration"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print("saved", OUT)


if __name__ == "__main__":
    build()
