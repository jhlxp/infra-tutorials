# LaTeX zip/tar.gz 论文源码转 Markdown 流程

这个文档用于处理从 arXiv 或作者主页下载的 LaTeX 源码压缩包，例如 `.zip`、`.tar.gz`。目标不是复刻 PDF 排版，而是得到一份中文译文 Markdown：按论文原文的章节和段落顺序翻译正文，保留图表、公式和参考文献编号，而不是写学习总结。

最终推荐目录结构如下：

```text
论文/
└── Paper Title/
    ├── Paper Title.zh.md
    └── figures/
        ├── fig1.png
        ├── fig2.png
        └── ...
```

源码压缩包、`.tex`、`.bib`、`.sty`、PDF 图等中间文件确认无用后可以删掉，只保留 Markdown 和 PNG 图片。

## 1. 解压源码

arXiv 源码常见格式是 `.tar.gz`，也可能是 `.zip`。

```bash
cd /home/chen/workplace/infra/docx/论文
mkdir arXiv-xxxx
tar -xzf arXiv-xxxx.tar.gz -C arXiv-xxxx

# 如果是 zip
unzip paper.zip -d paper-source
```

解压后先找主文件：

```bash
find arXiv-xxxx -maxdepth 2 -name '*.tex' -print
rg -n '\\\\documentclass|\\\\begin\\{document\\}|\\\\input|\\\\include|\\\\bibliography' arXiv-xxxx
```

一般主文件会包含：

- `\documentclass`
- `\begin{document}`
- `\title`
- `\author`
- 多个 `\input{section}`
- `\bibliography{xxx}`

## 2. 展开论文正文

很多论文不是一个 `.tex` 文件，而是：

```tex
\input{abstract}
\input{intro}
\input{design}
\input{experiments}
\input{relatedwork}
```

整理 Markdown 时要按主文件里的顺序递归展开这些 `\input`。如果只按文件名排序，参考文献编号和图顺序很容易错。

建议顺序：

1. 先读主 `.tex`。
2. 按 `\input` 顺序拼接正文。
3. 忽略 `\iffalse ... \fi` 中被注释掉的内容。
4. 保留标题、作者、摘要、正文、表格、附录、参考文献。
5. 标题区要保留论文年份、作者单位或机构信息。多机构论文按机构分组列作者；单机构论文也写成 `- 机构：作者列表`，方便以后快速判断论文来自哪里。

## 3. 图片处理

LaTeX 论文里的图常见是 PDF：

```tex
\includegraphics{figures/foo.pdf}
```

Markdown/GitHub 一般不能把 PDF 当图片直接渲染，所以要转 PNG。

```bash
pdftoppm -png -singlefile -r 200 figures/foo.pdf figures/foo
```

批量转换可以这样做：

```bash
find figures -name '*.pdf' -print0 | while IFS= read -r -d '' f; do
  out="${f%.pdf}"
  pdftoppm -png -singlefile -r 200 "$f" "$out"
done
```

Markdown 中统一引用 PNG：

```md
<p align="center">
  <img src="figures/foo.png" alt="图 2：xxx。" width="50%">
  <br><em>图 2：xxx。</em>
</p>
```

当前推荐排版：

- 单图：居中，`width="50%"`。
- 两张子图：用 HTML table，一行两列。
- 三张子图：一行三列。
- 四张子图：两行两列。
- 每张子图保留自己的 caption。
- 多子图整体再保留一个总 caption。

两张子图示例：

```md
<table align="center">
  <tr>
    <td align="center" width="50%"><img src="figures/a.png" alt="图 1a：xxx。" width="100%"><br><em>图 1a：xxx。</em></td>
    <td align="center" width="50%"><img src="figures/b.png" alt="图 1b：yyy。" width="100%"><br><em>图 1b：yyy。</em></td>
  </tr>
</table>

<p align="center"><em>图 1：xxx 与 yyy 对比。</em></p>
```

## 4. 图注如何对应

不要只看图片文件名，要从 LaTeX 里抽：

```tex
\begin{figure}
  \includegraphics{figures/foo.pdf}
  \caption{Startup losses without mapping out bad paths}
  \label{fig:startup}
\end{figure}
```

对应到 Markdown：

```md
<p align="center">
  <img src="figures/foo.png" alt="图 4：不预先剔除坏路径时的启动丢包。" width="50%">
  <br><em>图 4：不预先剔除坏路径时的启动丢包。</em>
</p>
```

多子图一般是：

```tex
\begin{subfigure}
  \includegraphics{figures/a.pdf}
  \caption{Subcaption A}
\end{subfigure}
\begin{subfigure}
  \includegraphics{figures/b.pdf}
  \caption{Subcaption B}
\end{subfigure}
\caption{Overall caption}
```

Markdown 中要同时保留子图注和总图注。

如果图注使用 HTML，例如 `<em>图 3：...</em>`，图注里尽量不要写 `$t$`、`$n$`、`$\alpha$` 这类 LaTeX 公式。很多 Markdown 渲染器不会在 HTML 标签内部渲染数学公式。图注里直接写普通文本即可，例如 `t`、`n`、`alpha`；正文段落里再保留正式公式写法。

## 5. 图表放置位置

LaTeX 的 `figure` 和 `table` 是浮动体，编译成 PDF 时经常漂到页顶、页底或下一页。Markdown 译文版不应该照搬这种位置，否则图表会离正文解释太远。

整理 Markdown 时采用这个原则：

1. 先有正文提到，再放图表。不要在章节标题后立刻堆图，导致读者还没看到解释就先看图。
2. 图表尽量放在第一次明确提到它的段落后面。
3. 如果一个组合图包含多个子图，例如图 9a、9b、9c、9d，就放在最后一个子图第一次被解释的段落后面。
4. 如果正文先总述图，再分段解释图内各部分，可以放在总述段落后面。
5. 表格同理，放在第一次出现“见表 1”“如表 1 所示”的段落后面。
6. 如果源码中的 `figure` 或 `table` 出现在首次提到之前，Markdown 中要主动移动到首次提到之后。
7. 不要连续堆很多图再集中解释。读者应该是“读一段，看一张/一组图”，而不是上下翻找。
8. 不要因为 LaTeX 源码中 `figure` 出现在某处，就机械保留那个位置。

例如：

```md
图 2 展示 SRv6 转发过程。当数据包到达时，交换机会比较目的地址前 48 位……

<p align="center">
  <img src="figures/srv6.png" alt="图 2：使用 uN uSID 的 SRv6 转发。" width="50%">
  <br><em>图 2：使用 uN uSID 的 SRv6 转发。</em>
</p>
```

组合图示例：

```md
图 9a 展示 NIC-T0 链路故障……

在图 9b 中，我们 flap 单个 CX8 NIC 上的四条链路……

图 9c 展示 T0-T1 链路故障……

图 9d 还展示了同时 flap 8 条 T0-T1 链路并恢复的影响……

<table align="center">
  ...
</table>

<p align="center"><em>图 9：使用双向 ib_write_bw 的 T0-local 与 Cross-T1 可靠性结果。</em></p>
```

这样读者读到解释后，马上就能看到对应图，不需要上下翻找。

## 6. 参考文献处理

不要把 BibTeX 原样整段贴到 Markdown 译文版里。译文版更适合使用论文常见编号引用：

```md
已有工作表明训练通信会受尾延迟影响 [1, 2, 3]。

## 参考文献

[1] Author。Title。Venue, Year。

[2] Author。Title。Venue, Year。
```

编号原则：

1. 按正文第一次出现的引用顺序编号。
2. 同一个 citation key 后面再次出现时复用同一个编号。
3. 文末参考文献顺序必须和正文编号一致。
4. 附录里的引用也要纳入编号。
5. 参考文献每条之间空一行，避免某些 Markdown 阅读器把连续条目挤在一起。

LaTeX 中常见引用形式：

```tex
\cite{foo}
\cite{foo,bar,baz}
```

转换后：

```md
[1]
[1, 2, 3]
```

检查项：

- 正文所有 `[n]` 都能在文末找到。
- 正文 citation 不能在翻译时丢失；不要只保留文末参考文献列表，DCTCP、RoCEv2、MPTCP、P4 等技术点后面的 `[n]` 也要跟着正文保留。
- 文末没有未被正文引用的编号。
- 编号连续，没有缺号和重号。
- 从原 `.tex` 递归展开 `\input` 后的 citation key 顺序，与 Markdown 文末条目标题一致。

## 7. 表格处理

LaTeX 表格需要转成 Markdown 表格。优先保持信息完整，不追求完全复刻线条。

LaTeX：

```tex
\begin{tabular}{l l l}
Cluster & NIC & Topology \\
A & CX8 & 2-Tier \\
\end{tabular}
```

Markdown：

```md
| Cluster | NIC | Topology |
| --- | --- | --- |
| A | CX8 | 2-Tier |
```

表格 caption 放在表格前或表格后都可以，但整个目录里尽量统一。当前习惯是放在表格前：

```md
表 1：实验平台配置。

| 集群 | NIC | 拓扑 |
| --- | --- | --- |
```

如果 LaTeX 里是 `figure` 环境包着一个 `tabular`，例如论文把参数表称为“Figure 2”，Markdown 也继续按图处理，不要改叫“表”。这类内容建议用 HTML table 居中显示，并在下方保留“图 n：...”图注，保持和原论文一致。

## 8. 公式和符号

简单行内公式保留 `$...$`，块公式保留：

```md
$$
...
$$
```

如果 Markdown 目标平台不支持 LaTeX 公式，再考虑改成纯文本解释。不要把复杂公式硬翻成不准确的中文。

## 9. 最终清理

整理完成后，单独建立译文版目录：

```bash
mkdir -p "Paper Title/figures"
cp source/Paper.zh.md "Paper Title/"
cp source/figures/*.png "Paper Title/figures/"
```

然后检查：

交付前自检清单：

- 论文信息：标题、年份、会议/期刊、作者、机构都已经从首页或主 `.tex` 交叉确认。
- 正文结构：严格按主 `.tex` 的 `\input` / `\include` 顺序展开，没有按文件名乱序拼接。
- 正文 citation：所有 `\cite{...}` 都已经转成 `[n]`，正文里的引用编号没有丢。
- 参考文献：编号连续，同一个 citation key 复用同一个编号，文末条目和正文编号对应。
- 图片路径：Markdown 引用的所有 PNG 都存在，且没有继续引用 PDF 图片。
- 图片排版：单图默认居中 `width="50%"`；多子图用 HTML table；caption 保留子图注和总图注。
- 图表位置：每个图表都放在第一次明确提到它的段落之后或附近，不机械保留 LaTeX 浮动体位置。
- 表格排版：表格居中，原论文称 Figure 的表格继续称图，避免改乱图号/表号。
- 公式排版：正文公式尽量保留 `$...$` 或 `$$...$$`；HTML 图注中不要放 LaTeX 公式。
- 最终目录：译文版目录包含 `.zh.md` 和 `figures/`，源码目录、压缩包、`.aux`、`.log` 等中间产物确认无用后再删。

```bash
# 不能再有 PDF 图片引用
rg -n '!\\[[^\\]]*\\]\\([^)]*\\.pdf\\)|<img src="[^"]*\\.pdf"' "Paper Title"
# 如果额外扫描 pages/、page-1 这类临时文件名，命中参考文献 URL 时人工确认即可。

# 检查图片、正文引用、参考文献编号和图表位置
python3 - <<'PY'
from pathlib import Path
import re

base = Path("Paper Title")
md = base / "Paper Title.zh.md"
text = md.read_text(encoding="utf-8")
lines = text.splitlines()

imgs = re.findall(r'<img src="([^"]+)"', text)
missing = [p for p in imgs if not (base / p).exists()]
print("images:", len(imgs))
print("missing:", missing)

if "## 参考文献" in text:
    body, ref_part = text.split("## 参考文献", 1)
else:
    body, ref_part = text, ""

refs = [int(x) for x in re.findall(r'^\[(\d+)\] ', ref_part, re.M)]
if refs:
    print("refs:", len(refs), "range:", min(refs), max(refs))
    print("missing ref numbers:", [i for i in range(min(refs), max(refs) + 1) if i not in refs])

body_cites = re.findall(r'\[[0-9]+(?:, *[0-9]+)*\]', body)
used = sorted({int(n) for cite in body_cites for n in re.findall(r'\d+', cite)})
print("body citation count:", len(body_cites))
print("body cites not in refs:", [n for n in used if n not in refs])

# 检查 figN/figNa 是否出现在“图 N”第一次正文提到之后。
ref_line = next((i for i, line in enumerate(lines, 1) if line.strip() == "## 参考文献"), len(lines) + 1)
position_issues = []
for idx, line in enumerate(lines, 1):
    m = re.search(r'<img src="figures/(fig\d+[a-z]?)\.png"', line)
    if not m:
        continue
    fig = m.group(1)
    num = re.match(r'fig(\d+)', fig).group(1)
    first = None
    for j, l in enumerate(lines, 1):
        if j >= ref_line:
            break
        if "<img " in l or "<em>" in l or "alt=" in l:
            continue
        if re.search(rf'图\s*{num}(?!\d)', l):
            first = j
            break
    if not first or idx < first:
        position_issues.append((fig, idx, first))
print("figure position issues:", position_issues)
PY
```

确认译文版目录完整后，可以删除源码解压目录和压缩包。判断标准是：Markdown 所在的最终译文版文件夹已经自洽，里面包含 `.md`、`figures/` 以及 Markdown 实际引用到的所有图片；并且已经完成图片路径、参考文献编号、图表位置和残留 `.pdf` 图片引用检查。

换句话说，真正要长期保留的是这个译文版目录：

```text
Paper Title/
├── Paper Title.zh.md
└── figures/
```

检查无误后，原始 `arXiv-xxxx.tar.gz` 和解压出来的 `arXiv-xxxx/` 只是中间产物，可以删除：

```bash
rm -rf arXiv-xxxx
rm -f arXiv-xxxx.tar.gz
```

## 10. 本次整理采用的约定

这次 `Resilient AI Supercomputer Networking using MRC and SRv6` 的整理采用了这些约定：

- 中文 Markdown 文件名：`Resilient AI Supercomputer Networking using MRC and SRv6.zh.md`
- 图片全部转成 PNG，放在 `figures/`
- 单图宽度：`50%`
- 多子图：HTML table 并排
- 引用格式：正文 `[1, 2]`，文末 `[1] Author。Title。...`
- 保留摘要、正文、表格、附录和参考文献
- 删除 LaTeX 源码目录和原始压缩包，只保留译文版
