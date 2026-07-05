# PDF 论文转 Markdown 流程

这个文档用于处理只有 PDF、没有 LaTeX 源码的论文。目标是整理出中文译文 Markdown：按论文原文的章节和段落顺序翻译正文，保留论文信息、图表、公式含义和参考文献编号，而不是写学习总结。

PDF 比 LaTeX 源码难处理，因为 PDF 更像“最终排版结果”，不一定保留结构化语义。标题、作者、章节、图表、参考文献、公式都可能只是页面上的文字和绘图对象。因此 PDF 转 Markdown 通常是“工具自动抽取 + 人工校正 + 半自动裁图”的流程。

最终推荐目录结构仍然是：

```text
论文/
└── Paper Title/
    ├── Paper Title.zh.md
    └── figures/
        ├── fig1.png
        ├── fig2.png
        └── ...
```

确认 Markdown 和图片都整理完毕后，可以按需要保留或删除原始 PDF。若 PDF 是唯一原始来源，建议先保留；若仓库空间有限，确认中文译文版自洽后也可以删除。

## 0. 核心难点：翻译，而不是 PDF 解析

PDF 转 Markdown 的工程难点是抽正文、裁图、修引用；但最终质量的核心难点仍然是翻译。PDF 解析只是把原文材料整理出来，真正要交付的是“按原论文结构排列的中文译文”，不是理解式转述，也不是学习总结。

翻译时遵守这些原则：

1. 逐段翻译原文正文，尽量保留原论文的信息量、语气强弱、限定条件和论证顺序。
2. 不要因为 `pdftotext` 抽出的句子断行混乱，就把内容压缩成摘要；必须先还原自然段，再翻译。
3. 原文中的 citation 编号要跟随对应论断保留，例如 `[12]`、`[3, 7]`，不能只在文末保留参考文献。
4. 图表说明要翻译，但图号、表号和子图编号必须和原论文一致。
5. 术语优先保持一致；第一次出现可以写成“中文（English）”，后文统一。
6. 对不确定的系统名、协议名、算法名、会议名、产品名，不要硬翻。
7. 公式附近的文字要特别小心，变量含义、前提条件、量纲和复杂度不要翻错。
8. 翻译完成后必须回看原 PDF 或抽取文本，检查有没有漏段、串栏、重复段、漏 cite、图表解释错位。

因此 PDF 流程实际分两层：

1. 解析层：把 PDF 里的正文、图、表、参考文献可靠地抽出来。
2. 翻译层：按原论文顺序逐段翻译，并做术语、引用、图表位置和中文可读性的二次校对。

## 1. 先判断 PDF 类型

PDF 大致分两类：

1. **文本型 PDF**：可以复制文字，`pdftotext` 能直接抽出正文。大部分 arXiv/ACM/USENIX 论文属于这类。
2. **扫描型 PDF**：页面本质是图片，`pdftotext` 抽不出正文或只有很少文字，需要 OCR。

先运行：

```bash
pdfinfo paper.pdf
pdftotext -layout -f 1 -l 1 paper.pdf -
```

如果第一页能抽出标题、摘要、正文，说明是文本型 PDF。若输出几乎为空，或者只有页眉页脚，就按扫描版处理。

## 2. 可用工具

当前本机已经可用的基础工具：

```bash
pdfinfo      # PDF 元信息、页数、尺寸、创建时间
pdftotext    # 抽正文，-layout 可保留双栏排版的大致位置
pdftoppm     # 把整页 PDF 渲染为 PNG
pdftocairo   # 另一种高质量 PDF 页面渲染器
pdfimages    # 提取 PDF 内嵌位图
mutool       # 查看 PDF 内部对象、字体、图片，也可渲染页面
python3 + fitz(PyMuPDF)  # 读取 metadata、目录、文本块、图片对象、按区域裁图
python3 + pypdf          # 读取 metadata、outline、页数
```

可选但当前环境未安装的增强工具：

- `ocrmypdf` / `tesseract`：扫描版 PDF OCR。
- `GROBID`：抽标题、作者、摘要、参考文献，输出 TEI XML。
- `pdffigures2`：自动识别论文中的图、表和 caption。
- `Marker`、`Docling`、`MinerU`：更高级的 PDF 到 Markdown/Layout 解析工具。
- `camelot`、`tabula`：抽 PDF 表格，适合有线框或规则排列的表。

基础工具已经足够处理大多数论文，但图表和双栏顺序通常仍需要人工校对。

## 3. 元信息提取

优先顺序：

1. `pdfinfo` / `pypdf` / PyMuPDF metadata。
2. PDF outline / bookmarks。
3. 首页标题区。
4. ACM/IEEE/USENIX reference format 或 DOI 行。

示例：

```bash
pdfinfo paper.pdf
```

Python 检查 metadata 和目录：

```bash
python3 - <<'PY'
import fitz

doc = fitz.open("paper.pdf")
print("pages:", doc.page_count)
print("metadata:", doc.metadata)
print("toc:")
for level, title, page in doc.get_toc():
    print("  " * (level - 1) + f"{title} -> page {page}")
PY
```

注意：

- PDF metadata 可能缺失、乱码或过旧，必须和首页交叉检查。
- `CreationDate` 不一定等于论文发表年份。发表年份优先从会议行、引用格式、DOI 页面或论文首页判断。
- 作者单位通常不在 metadata 中，通常要从首页版面人工整理。
- 多机构论文继续按“机构：作者列表”的方式写入 Markdown。

论文信息区推荐格式：

```md
# Paper Title

## 论文信息

- 标题：Paper Title
- 年份：2025
- 会议/期刊：SIGCOMM 2025
- 作者：...
- 机构：
  - 机构 1：作者 A、作者 B
  - 机构 2：作者 C
```

## 4. 正文抽取

文本型 PDF 优先用：

```bash
pdftotext -layout paper.pdf paper.layout.txt
pdftotext -raw paper.pdf paper.raw.txt
```

经验：

- `-layout` 更适合双栏论文，可以保留左右栏相对位置，方便人工判断图表位置。
- `-raw` 更适合后续脚本处理，但双栏顺序可能乱。
- 如果有 outline，可以先按 outline 确认章节顺序，再从 `paper.layout.txt` 中整理正文。
- 页眉、页脚、页码、版权声明通常要删除。
- 断词要修复，例如 `redun- dant` 应改成 `redundant`。
- 双栏换行要重排成自然段。

用 PyMuPDF 看文本块位置：

```bash
python3 - <<'PY'
import fitz

doc = fitz.open("paper.pdf")
for i, page in enumerate(doc):
    print("PAGE", i + 1)
    for b in page.get_text("blocks"):
        x0, y0, x1, y1, text, *_ = b
        text = text.replace("\\n", " ")[:100]
        print(round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1), repr(text))
PY
```

这个方法适合判断：标题块、作者块、caption 块、左右栏正文块、参考文献块。

## 5. 图片与图表处理

PDF 里的图有两种：

1. **内嵌位图**：可以用 `pdfimages` 抽出。
2. **矢量图**：由 PDF 绘图指令组成，`pdfimages` 抽不出完整图，只能渲染页面后裁剪。

因此不要只依赖 `pdfimages`。

### 5.1 检查内嵌图片

```bash
pdfimages -list paper.pdf
```

如果看到很多很小的 image/smask，通常只是图的一部分、阴影或透明 mask，不一定是完整 figure。对于论文中的矢量图，推荐走“页面渲染 + 裁剪”。

### 5.2 整页渲染

```bash
mkdir -p pages
pdftoppm -png -r 200 paper.pdf pages/page
```

或：

```bash
mkdir -p pages
pdftocairo -png -r 200 paper.pdf pages/page
```

说明：

- `-r 200` 通常够 Markdown 阅读；复杂图可用 `-r 300`。
- 整页图只作为中间文件，不放入最终译文版，最终只放裁好的 figure。

### 5.3 按区域裁图

最稳的方式是用 PyMuPDF 按 PDF 坐标裁图。先用文本块找到 caption 坐标，再人工估计 figure 区域。

示例：从第 3 页裁两个图。

```bash
python3 - <<'PY'
import fitz

doc = fitz.open("paper.pdf")
page = doc[2]  # 第 3 页，0-based

crops = {
    "fig1": fitz.Rect(55, 65, 300, 230),
    "fig2": fitz.Rect(315, 65, 560, 210),
}

for name, rect in crops.items():
    pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), clip=rect, alpha=False)
    pix.save(f"figures/{name}.png")
PY
```

裁剪经验：

- PDF 坐标单位是 point，页面左上附近通常是 `(0, 0)`。
- 先用 `page.get_text("blocks")` 找到 `Figure 1:` 的 caption 坐标。
- 图通常在 caption 上方，也可能在下方；需要看页面预览确认。
- 裁图宁可略宽一点，避免切掉坐标轴、图例或公式。
- 检查裁图时不要只看曲线主体是否存在，还要逐张确认 x/y 轴标题、刻度数字、图例、子图标号是否完整；例如性能图很容易把左侧 `CDF (%)`、`Throughput (Gbps)` 或底部 `Flow rank` 裁掉。
- 多栏论文里，相邻 figure 可能挨得很近，区域裁剪可能会带出旁边图的一条边；这种情况可以用 PIL 再做二次裁边或用白底遮掉杂边。
- 最终 Markdown 里的单图仍然按 `width="50%"` 居中；很宽的性能图可以用 `70%`，但需要统一美观。
- 裁完所有图后，做一张 contact sheet 快速肉眼检查，不要只相信脚本。

示例 contact sheet 检查脚本：

```bash
python3 - <<'PY'
from PIL import Image, ImageDraw
from pathlib import Path

figdir = Path("Paper Title/figures")
imgs = []
for p in sorted(figdir.glob("fig*.png")):
    im = Image.open(p).convert("RGB")
    im.thumbnail((260, 180), Image.LANCZOS)
    canvas = Image.new("RGB", (280, 220), "white")
    canvas.paste(im, ((280 - im.width) // 2, 25))
    ImageDraw.Draw(canvas).text((10, 5), p.stem, fill="black")
    imgs.append(canvas)

cols = 4
rows = (len(imgs) + cols - 1) // cols
sheet = Image.new("RGB", (cols * 280, rows * 220), "white")
for i, im in enumerate(imgs):
    sheet.paste(im, ((i % cols) * 280, (i // cols) * 220))
sheet.save("contact_sheet.png")
PY
```

Markdown 图格式推荐如下：

```md
<p align="center">
  <img src="figures/fig1.png" alt="图 1：xxx。" width="50%">
  <br><em>图 1：xxx。</em>
</p>
```

多子图继续用 HTML table：

```md
<table align="center">
  <tr>
    <td align="center" width="50%"><img src="figures/fig1a.png" alt="图 1a：xxx。" width="100%"><br><em>图 1a：xxx。</em></td>
    <td align="center" width="50%"><img src="figures/fig1b.png" alt="图 1b：yyy。" width="100%"><br><em>图 1b：yyy。</em></td>
  </tr>
</table>

<p align="center"><em>图 1：xxx 与 yyy 对比。</em></p>
```

图注中不要放 `$...$` 公式，HTML 标签内很多 Markdown 渲染器不会处理公式。图注里写普通文本，公式放正文。

组合图处理细节：

- 如果一个 Figure 里有 (a)、(b) 等多个子图，不要默认裁成一张大图。
- 当原 PDF 中子图纵向堆叠，转成 Markdown 后显得左右太空、整体太小，可以拆成 `fig20a.png`、`fig20b.png` 这类子图文件，再用 HTML table 一行两列或两行两列排版。
- 子图图片本体尽量只保留曲线/坐标轴/图例；`(a)`、`(b)` 的说明可以放在 Markdown 的 `<em>图 20a：...</em>` 里。
- 拆子图时仍要保留总图注，例如 `<p align="center"><em>图 20：...</em></p>`，这样图号和原论文保持一致。
- 如果裁图带入相邻 caption、孤立小点、其他子图边缘，优先小幅调整裁剪区域；必要时可用白底遮掉明显的 PDF 残留噪声，但遮完必须复查，不能擦掉图框、坐标轴、刻度或图例。

## 6. 图表放置位置

PDF 论文里的图和表经常因为双栏排版被放到页顶、页底或下一页。Markdown 译文版不需要复刻这种浮动位置，应该服务阅读顺序：读者第一次看到“图 20 分析……”或“如表 1 所示……”时，图表最好就在附近。

排版原则：

1. 先有正文提到，再放图表。不要在章节标题后立刻堆一组图，导致读者还没看到解释就先看图。
2. 单图放在第一次明确提到它的段落后面。例如段落里第一次出现“图 14 显示……”，图 14 就放在这个段落后面。
3. 如果段落先总述图，再分句解释图中多个现象，也可以放在这个总述段落后面。
4. 多子图放在最后一个关键子图被第一次解释的段落后面；如果正文一段同时解释图 20a 和图 20b，就把整个图 20 的 HTML table 放在这段后面。
5. 表格同理，放在第一次出现“表 1”“如表 1 所示”“见表 1”的段落后面。
6. 如果原 PDF 把图表放在第一次提到之前，Markdown 中要主动移动到首次提到之后。
7. 如果图表很大，放在第一次提到之后会打断长段落，可以先结束当前段落，再插入图表，然后继续解释。
8. 不要让图表离解释段落太远；通常相隔不超过一两个自然段。
9. 不要连续堆很多图再集中解释。读者应该是“读一段，看一张/一组图”，而不是上下翻找。

示例：

```md
图 20 分析 8192 节点 FatTree 中 incast 规模的影响。图 20a 显示……
图 20b 显示……

<table align="center">
  ...
</table>

<p align="center"><em>图 20：8192 节点 FatTree 中 incast 规模带来的开销。</em></p>
```

表格示例：

```md
表 1 汇总了各组件的 bit 开销，可以看到每个连接只需要很小的状态。

<div align="center">
<table>
  ...
</table>
<br><em>表 1：REPS 每连接内存占用。</em>
</div>
```

## 7. 表格处理

PDF 表格比 LaTeX 表格更难抽。推荐优先级：

1. 如果表格小，人工重写为 Markdown 或 HTML table。
2. 如果表格有清晰线框，可尝试 `camelot` 或 `tabula`。
3. 如果表格其实是图片或复杂排版，直接裁成 PNG，并在正文中解释关键字段。

表格也要居中：

```md
<div align="center">
<table>
  <tr><th>列 1</th><th>列 2</th></tr>
  <tr><td>A</td><td>B</td></tr>
</table>
<br><em>表 1：xxx。</em>
</div>
```

如果原论文把表叫 Figure，就按原论文称呼保持一致。

## 8. 参考文献处理

PDF 没有 `.bib`，所以参考文献通常从 `pdftotext` 的 `REFERENCES` 段抽取。

步骤：

1. 找到 `References` 或 `REFERENCES` 后的文本。
2. 保留原论文编号，例如 `[1]`、`[2]`。
3. 正文引用也必须保留并统一成 `[1]`、`[2, 3]`。不要只在文末保留参考文献编号，正文里的 DCTCP、RoCEv2、MPTCP、P4 等对应引用也要留在原论断附近。
4. 检查编号连续。
5. 每条参考文献之间空一行，避免 Markdown 挤在一起。

检查：

```bash
rg -n "^\\[[0-9]+\\]" Paper.zh.md
```

PDF 论文中的引用可能写成 `[1, 17]`，也可能因为抽取问题变成 `v2[25]`，需要人工修复空格和格式。

正文 cite 校验建议：

```bash
# 看正文是否仍有引用编号，而不是只有参考文献列表有编号
rg -n "\\[[0-9]+(, *[0-9]+)*\\]" Paper.zh.md
```

如果 `rg` 输出几乎只出现在“参考文献”之后，说明正文 cite 被转写时丢掉了，需要回到 `pdftotext -layout` 或 PDF 原文逐段补回。

## 9. 扫描版 PDF / OCR

如果 `pdftotext` 抽不出正文：

1. 先渲染页面。
2. 用 OCR 生成可搜索 PDF 或文本。
3. 再按文本型 PDF 流程整理。

推荐命令：

```bash
ocrmypdf --sidecar paper.ocr.txt paper.pdf paper.ocr.pdf
```

如果没有 `ocrmypdf`，可以先用 `pdftoppm` 渲染页面，再用 `tesseract` 对每页 OCR。OCR 结果一定要人工校正，尤其是公式、英文缩写、参考文献和作者名。

## 10. 最终排版规则

最终 Markdown 的排版规则：

- 论文标题作为一级标题。
- 开头保留论文信息，必须包含年份。
- 摘要单独成节。
- 章节按原论文顺序组织，正文以段落级翻译为主，不写成提纲式总结。
- 图表放在第一次明确提到它的段落后面或附近，而不是照搬 PDF 浮动位置；不要章节开头先堆图再解释。
- 单图居中显示，默认 `width="50%"`。
- 多子图用 HTML table。
- 表格用居中 HTML table；复杂表格可裁成 PNG。
- 参考文献使用 `[1]` 编号，不贴 BibTeX；正文中的 citation 编号也要保留在对应论断附近。
- 参考文献每条之间空一行；PDF 抽出来的参考文献经常换行混乱，必须人工检查编号是否连续。
- 最终检查不要有临时页面图、`.pdf` 图片引用、OCR 草稿和未确认的 TODO。

交付前自检清单：

- 论文信息：标题、年份、会议/期刊、作者、机构都已经从首页或 PDF metadata 交叉确认。
- 正文结构：章节顺序和原论文一致，没有把论文改写成学习总结。
- 正文 citation：正文中的 `[n]` 没有丢，技术论断附近仍保留对应引用编号。
- 参考文献：编号连续，正文引用都能在文末找到，文末没有明显缺行或粘连。
- 图片路径：Markdown 引用的所有图片都存在，且没有继续引用 `.pdf`、临时整页截图或 `pages/` 中间文件。
- 图片裁剪：逐张看过 contact sheet，坐标轴标题、刻度、图例、子图标号、右侧/底部边界都没有被裁掉。
- 图片排版：单图默认居中 `width="50%"`；宽图可放宽到 `70%`；多子图优先拆开后用 HTML table 排版。
- 图表位置：每个图表都放在第一次明确提到它的段落之后或附近，不在章节开头无解释地堆图。
- 表格排版：表格居中，caption 和原论文图号/表号一致；原论文称 Figure 的表格不要擅自改成 Table。
- 最终目录：只保留最终 `.zh.md` 和 `figures/` 中实际用到的图片，中间页图和 OCR 草稿不要混进去。

推荐检查：

```bash
rg -n "TODO|MISSING|\\.pdf|pages/|page-[0-9]" Paper.zh.md
# 如果命中参考文献 URL 里的 pages/，人工确认即可；真正要清理的是临时整页截图或中间目录引用。
python3 - <<'PY'
from pathlib import Path
import re

md = Path("Paper.zh.md")
text = md.read_text(encoding="utf-8")
base = md.parent
lines = text.splitlines()

imgs = re.findall(r'<img[^>]+src="([^"]+)"', text)
missing = [p for p in imgs if not (base / p).exists()]
print("images:", len(imgs), "missing:", missing)

if "## 参考文献" in text:
    body, ref_part = text.split("## 参考文献", 1)
else:
    body, ref_part = text, ""

refs = [int(x) for x in re.findall(r'^\\[(\\d+)\\] ', ref_part, re.M)]
if refs:
    print("refs:", len(refs), "range:", min(refs), max(refs))
    print("missing ref numbers:", [i for i in range(min(refs), max(refs) + 1) if i not in refs])

body_cites = re.findall(r'\\[[0-9]+(?:, *[0-9]+)*\\]', body)
used = sorted({int(n) for cite in body_cites for n in re.findall(r'\\d+', cite)})
print("body citation count:", len(body_cites))
print("body cites not in refs:", [n for n in used if n not in refs])

# 检查 figN/figNa 是否出现在“图 N”第一次正文提到之后。
ref_line = next((i for i, line in enumerate(lines, 1) if line.strip() == "## 参考文献"), len(lines) + 1)
position_issues = []
for idx, line in enumerate(lines, 1):
    m = re.search(r'<img src="figures/(fig\\d+[a-z]?)\\.png"', line)
    if not m:
        continue
    fig = m.group(1)
    num = re.match(r'fig(\\d+)', fig).group(1)
    first = None
    for j, l in enumerate(lines, 1):
        if j >= ref_line:
            break
        if "<img " in l or "<em>" in l or "alt=" in l:
            continue
        if re.search(rf'图\\s*{num}(?!\\d)', l):
            first = j
            break
    if not first or idx < first:
        position_issues.append((fig, idx, first))
print("figure position issues:", position_issues)
PY
```

## 11. 用示例 PDF 练手的结论

示例文件：

```text
/home/chen/workplace/infra/docx/论文/3098822.3098825.pdf
```

探测结果：

- 标题：Re-architecting datacenter networks and stacks for low latency and high performance
- 作者：Mark Handley、Costin Raiciu、Alexandru Agache、Andrei Voinescu、Andrew W. Moore、Gianni Antichi、Marcin Wójcik
- 年份：2017
- 会议：SIGCOMM '17
- 页数：14
- PDF 类型：文本型 PDF，metadata 完整，并且有 outline。
- 工具结论：
  - `pdfinfo` 可以直接抽标题、作者、关键词、创建时间。
  - `pypdf` / PyMuPDF 可以抽 metadata 和目录。
  - `pdftotext -layout` 可以抽出正文，但双栏文本、断词和图注需要人工整理。
  - `pdfimages` 只能抽内嵌位图；这篇论文有大量矢量图，完整图需要用 PyMuPDF 或 Poppler 渲染裁剪。
  - Figure 1 到 Figure 23 都可以用 PyMuPDF 坐标裁成 PNG。
  - Figure 18、Figure 21 这种紧贴其他图注或相邻图的情况，需要二次清理裁边。
  - 参考文献共 41 条，必须检查 `[1]` 到 `[41]` 是否连续。

这说明 PDF 转 Markdown 是可行的，但和 LaTeX 源码相比，自动化程度低一些。实践上最稳的路线是：metadata 自动抽，正文用 `pdftotext -layout` 辅助整理，图片用“整页渲染 + caption 定位 + 区域裁剪”，参考文献从 PDF 文本末尾人工编号校验。
