# lens-diagram-to-dxf

把**镜头光学结构图**（厂商官网/说明书里的 lens construction 图，PNG / JPG / SVG）反推成逐面数据：
自动找光轴、分出每片镜片、按「圆心落在光轴上」拟合每个面，输出

- **Excel** 面数据表（面序号 / 面型 / 曲率半径 / 厚度 / 玻璃占位 / 标记 / 半口径 / 作图半径 / 拟合 RMS）
- **AutoCAD DXF**（光轴、各面 ARC、边缘线、光阑、尺寸标注，可把原图按比例贴在底层对照）
- **CODE V `.seq`**（最简序列，`in` 进去就能看 layout）

> Reverse-engineer per-surface lens data (radii, thicknesses, semi-diameters) from a published
> lens construction diagram, and export it as Excel / AutoCAD DXF / CODE V sequence.

你只需要给**一个尺度**：总长、最大镜片直径、或 px/mm，三选一。

![合成示例的拟合叠加图](examples/demo/demo_overlay.png)

## 精度

仓库自带一个**往返自测**：用已知处方画一张合成结构图，再反推回来比对。

```
$ cd examples/demo && python3 make_demo.py
$ python3 ../../scripts/diag_extract.py demo.svg -o demo.json --length 18.719 --svg-width 1800
$ python3 compare.py

  面       R 真值       R 反推       误差 |     D 真值     D 反推       误差
  1     22.014     21.986   -0.13% |    3.259    3.288   +0.029
  2   -435.760   -436.718   -0.22% |    6.008    5.984   -0.024
  3    -22.213    -22.188   +0.11% |    1.000    0.995   -0.005
  4     20.291     20.281   -0.05% |    4.750    4.740   -0.010
STO        INF        INF          |    0.750    0.728   -0.022
  6     79.684     79.514   -0.21% |    2.952    2.983   +0.031
  7    -18.395    -18.370   +0.14% |    0.000    0.000   +0.000

曲率半径：最大误差 0.22%，平均 0.14%
厚度/间隔：最大误差 0.031 mm，平均 0.017 mm
```

这是**最好情况**：合成图、无非球面、渲染干净、87 px/mm。真实厂商图上，
[适马 85mm F1.2 那个案例](examples/sigma85/)（26.5 px/mm）逐面偏差平均 0.66 px ≈ 25 µm。

弱曲率面（|R| > 150 mm）从图上只能拟到 ±15%，非球面报的是最佳拟合球面，只能当初值。

## 用法

```bash
pip install -r requirements.txt

# 1) 提取：图 → JSON + 叠加核对图
python3 scripts/diag_extract.py diagram.png -o lens.json \
        --crop WxH+X+Y --length 119.72 --overlay ov.png --debug dbg

# 2) 目检 ov.png（品红=拟合弧）和 dbg_parts.png（每片一色），确认分片数 = 图上镜片数

# 3) 建表出图
python3 scripts/diag_build.py lens.json -o OUT --seq --image diagram_crop.jpg

# 4) 验收：把拟合弧压回原图逐行量偏差，全局平均 < 1 px 算合格
python3 scripts/fit_check.py lens.json diagram_crop.png
```

裁图时只保留镜头本体，**去掉图例色块、标题文字、尺寸线、外框**——它们会被当成假镜片。

## 它解决了什么

从图上量镜头看着简单，坑全在细节里。这些都是被实际打回来之后才修对的：

- **描边就是透镜边缘本身，只是线画得粗。** 真实面在描边的**中心线**上，不是填充区的内边。
  取错的话每片都瘦掉一个线宽，相邻片之间凭空多出 ≈0.2 mm 的假气隙，在 CAD 里表现为口径打架。
- **别用「x 不再变化」去砍机械磨边。** 弱曲率面的 dx/dy 本来就小于 0.5 px/行，
  这么砍会把 40% 的真实弧当磨边扔掉，半径直接错 17%。改成「靠近光轴的一半当种子 + 逐轮吸收」的稳健拟合。
- **未填充的镜片 vs 被围起来的空气**，在线稿里长得一模一样。按**边缘**判：
  镜片一定有有限厚度的机械边缘，空气楔两端是两条面弧的交点，是真正的尖点。
- **作图半径要做穿透截断**，否则相邻两面的弧会互相穿透。
- 带 alpha 通道的 PNG 必须先合成到白底，否则透明区变纯黑，整幅图的背景判断全错。
- AutoCAD 贴图那一串坑（`ACAD_IMAGE_DICT` 的键不能带路径和扩展名、`dimlfac` 默认 100、
  组码 290 的位置……）见 [`SKILL.md`](SKILL.md) 的「踩过的坑」。

## 文件

```
scripts/diag_extract.py   图 → lens.json（+ 叠加图、分片调试图）
scripts/diag_build.py     lens.json → xlsx / dxf / seq / spec.json
scripts/fit_check.py      验收：拟合弧压回原图逐行量偏差
scripts/asph_from_diagram.py  专利缺非球面系数时：用印刷 R/D 标定结构图，量非球面偏离并拟合偶次系数初值
scripts/start_fix.py      自动优化第 1 级：玻璃占位的面数据 → 光线追得通、像差初步收敛的起点
scripts/cv_opt.py         第 2 级：CODE V AUT（生成 _setup.seq 设变量 + _aut.seq 纯约束宏）
scripts/cv_snap.py        第 3 级：虚拟玻璃逐片吸附到真实牌号
scripts/run_all.py        三级一步跑完；viewer.py / live.py 浏览器里实时看；layout_png.py 叠回原图
scripts/cv_session.py     CODE V COM 薄封装（Windows + CODE V + pywin32）
restore/                  还原优化器：按官方 MTF 反求玻璃牌号 / 非球面 / 后截距，出 CODE V 交付件（见 restore/README.md）
SKILL.md                  完整流程、算法说明、踩过的坑（也是一个 Claude skill）
examples/demo/            合成示例 + 往返精度自测
examples/sigma85/         真实案例：适马 85mm F1.2
examples/zhongyi35_cn120215088/  专利还原：中一 35mm F1.4，专利缺非球面系数 → 图测初值 → CODE V → 人工修正
examples/sigma50cl/       还原优化器实例：适马 50mm F1.4 CL DG，贴合官方 MTF 求出全模压方案
```

`SKILL.md` 本身是一个 [Claude Code](https://claude.com/claude-code) skill：把整个目录放进
skills 目录，描述这类活的时候就会自动触发。当然脚本也可以完全脱离 Claude 单独用。

## 专利还原模式（专利缺非球面系数）

专利面表印全了、唯独没印非球面系数时，专利结构图往往是 CAD 按比例导出的：用印刷 R/D 把图标定到
≈0.03 mm，再逐行量非球面墨线偏离「印刷顶点球面」多少，拟合出偶次系数当初值
（`scripts/asph_from_diagram.py`）。之后**只用 INF 一个组态**优化初始结构，再交给设计者在 CODE V 里
用一个简短的约束宏手工修正：球面 RDY ±5%，非球面顶点 R 同号且 |R| 30..500，宏里不设变量。
完整流程和踩过的坑见 [`examples/zhongyi35_cn120215088/`](examples/zhongyi35_cn120215088/) 和 SKILL.md §6。

## 黑底剖面图的胶合线

尼康官网、DC Watch 那种黑镜筒剖面图，胶合线画成背景色（白片之间一条黑线），默认的胶合判据永远不成立，
胶合面会被拆成假空气间隔。加 `--dark-cement <线宽+1>`（3× 放大后一般取 5）即可。
实测尼康 AF-S 58mm f/1.4G：9 片 15 面，全部面平均偏差 0.91 px ≈ 38 µm，详见 SKILL.md。

## 自动优化流水线（可选，Windows + CODE V）

```bash
python scripts/run_all.py SIGMA85.xlsx --efl 85 --fno 1.25 --ih 21.6
python scripts/viewer.py <目录>     # http://localhost:8765 实时看
```
共三级：起点修复 → CODE V AUT → 吸附真实玻璃。第 2 级写出的约束宏照手工写法：
- 球面：RDY ±5%；
- 非球面：顶点 RDY 同号，|R| 在 30..500；
- 宏里不设变量，变量另放在 `_setup.seq`；
- 只做 INF。

见 SKILL.md §7。

## 还原优化器（restore/）

结构图只给几何。`restore/` 把几何锁定在图上，求剩下的量：

- 每片的真实牌号；
- 非球面系数；
- 后截距。

目标是贴合厂商官方 MTF 图，同时满足 EFL、CRA、相对照度等交付条件，最后出 CODE V 交付件。

- 仿 CODE V AUT 的阻尼最小二乘：拉格朗日乘子精确约束、有效集、自动阻尼、自动微分；
- 玻璃图：模压片用 CODE V 式凸包，抛光片用多边形；
- 从虚构玻璃到真实牌号：退火吸附 → 逐片冻结 → 精修 → 单片换料；
- 成批并行：Rust 引擎（CPU 多线程）或 PyTorch（CPU / CUDA）；搜索不需要 CODE V，复核和交付用 CODE V；
- 一个 `config.json` 描述一支镜头，`tools/config_from_seq.py` 从起始 `.seq` 起草。

```bash
python restore/run_pipeline.py --config examples/sigma50cl/config.json --out runs/sigma50 --codev
```
见 [`restore/README.md`](restore/README.md) 和 [`examples/sigma50cl/`](examples/sigma50cl/)。

## 局限

- **不输出玻璃。** 结构图给不出折射率，玻璃一律占位 `BSC7`，要自己补。
- 非球面只给最佳拟合球面。
- 光阑如果画成上下两段短竖线（不贯穿），要手工定位后插进 `lens.json`，方法见 SKILL.md §4.5。
- 反推数据只适合当优化初值或做结构对照，**不是厂商的实际设计数据**。

## License

MIT
