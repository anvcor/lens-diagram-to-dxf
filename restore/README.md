# restore/：从结构图到可交付的还原镜头

前面的脚本只能把镜头结构图读成几何：半径、间隔、口径。玻璃全是占位，非球面只是从图上量出的初值。
`restore/` 接着做剩下的事：几何锁定在图上，然后找

- 每片的**真实牌号**（按规则限定：模压片只用 M- / L- 料，抛光片只用常规料）；
- **非球面系数**（矢高框住图上量出的轮廓）；
- **后截距**；

目标是让镜头的 MTF 贴合厂商公布的 MTF 图，同时满足 EFL 窗口、焦点、像方主光线角 CRA、边缘相对照度 RI 这些交付条件。
最后输出 CODE V 交付件（`.seq`、优化宏、`.xlsx`），交给设计者手工收尾。

搜索不需要 CODE V。几百个方案成批并行算，光追引擎有两个：

- Rust（CPU 多线程，默认）；
- PyTorch（CPU 或 CUDA）。

CODE V 只在最后用：渐晕、相对照度、MTF 复核。

第一个用例是适马 50mm F1.4 CL DG | Contemporary（C027）的还原，见 [`examples/sigma50cl/`](../examples/sigma50cl/)。

## 新镜头怎么做

1. **结构图 → 面数据**：走 SKILL.md §1–5（`diag_extract.py` / `diag_build.py`）。
   专利缺非球面系数时，用 `asph_from_diagram.py` 量初值（§6）。
   得到一份能 `in` 进 CODE V 的起始 `.seq`，带 `STO`；有保护玻璃的话，写成最后几片平板。
2. **起草 config**：
   ```bash
   python restore/tools/config_from_seq.py start.seq --out examples/mylens/config.json --name mylens --image-height 21.6
   ```
   写出来的是草稿，每一项都要核对（字段说明见下表），重点是：
   - 已知牌号的片：`elements` 里写成 `"HOYA:FCD100"`，这片就固定不动；
   - 非球面：`asph_var` 里的曲率范围；`sag_table` 只在系数是量出来的时候才留，系数是猜的就删掉；
   - 玻璃：`glass.rules` 和 `glass.modes`，例如「非球面片只能模压」，或者再加一个混合模式；
   - 限值：`limits` 里的 EFL 窗口、CRA、RI 按相机系统定；
   - 视场：`image.fields_mm`；
   - 光阑口径：草稿是按 F 数近轴估的，图上能量出光圈口径就用量的。
3. **官方 MTF**：把厂商的 MTF 图数字化成 `mtf_official.json`，放在 config 旁边（格式见下）。
4. **玻璃目录**：把 `HOYA.AGF`、`OHARA.AGF` 等放进 `restore/glass_catalog/`，或者用 `LENS_GLASS_DIR` 指过去。
   这些文件不在仓库里。
   - 可选：放 CODE V 的 `hoya.dat` / `ohara.dat`，停产牌号会被自动剔除。
   - 可选：有 CODE V 的话，跑一次 `tools/glassmodel.py sample` 和 `tools/glassmodel.py fit`，
     求解器里的虚构玻璃就和 CODE V 的 GLC 完全一致。只跑一次，换镜头不用重跑；换波长要重跑。
5. **冒烟测试**（几分钟）：
   ```bash
   python restore/run_pipeline.py --config examples/mylens/config.json --out runs/mylens_smoke --smoke
   ```
6. **正式跑**：
   ```bash
   python restore/run_pipeline.py --config examples/mylens/config.json --out runs/mylens_r1 --codev
   ```
   每个玻璃模式依次跑这几步：
   - c：连续搜索；
   - a：吸附到真实牌号；
   - p：精修；
   - s：单片换料；
   - t：收紧 CRA / RI 再精修；
   - cv：CODE V 复核排序（加 `--codev` 才跑）。

   某一步的输出已经存在就跳过，中断后重跑同一条命令即可续上。
   适马 50（8 片可变玻璃、2 个可变非球面）在 16 核 CPU 上用 Rust 引擎，`LENS_NRING=8`：
   - 从已有的 96 个连续解出发，每个模式约 10 分钟；
   - 从 512 个随机起点开始，还要加上快速筛选的时间。
7. **交付**：选定的方案在 `runs/.../cv/<mode>_c<k>.*` 里，已经过 setvig，`.len` 和 `.seq` 都有。
   也可以从任意结果文件单独生成：
   ```bash
   python restore/deliver.py runs/mylens_r1/hybrid_t.jsonl --rank 0 --out deliver/MyLens_INF
   python restore/cvtools.py finalize deliver MyLens_INF   # setvig、R1..R5 检查、CRA、RI、SAV
   python restore/cvtools.py mtf deliver/MyLens_INF.len    # 和官方曲线逐点比较
   python restore/cvtools.py layout deliver MyLens_INF     # layout PNG
   ```
   `_opt.seq` 是给设计者的约束宏，只有约束、不设变量、只做 INF，写法同 SKILL.md §6.4：
   - EFL 窗口；
   - 球面 RDY ±5%；
   - 非球面顶点 |R| 30..500；
   - CRA、后截距；
   - 加工约束、GLA。

## config.json

完整示例：[`examples/sigma50cl/config.json`](../examples/sigma50cl/config.json)。
路径相对于 config 所在目录。

| 字段 | 内容 |
|---|---|
| `name` / `title` / `note` | 名称，`title` 用在报告和宏里 |
| `wavelengths_nm`, `ref_wavelength_index`, `weights` | 波长（nm）、参考波长序号（0 起）、权重（CODE V WTW） |
| `surfaces[]` | 按顺序列出各面，字段见下 |
| `elements` | `"L1": "var"` 表示玻璃可变，`"L4": "HOYA:FCD100"` 表示固定牌号 |
| `asph_var.<面>` | 可变非球面：曲率范围 `c_range`；随机起点的顶点半径 `random_R`（可选）；矢高表 `sag_table`（`{h: 矢高}`，mm，可选） |
| `sag_band` | 矢高框：`tol`（±mm）、`edge_tol`（最外一点）、`hmin`（更低的高度不约束） |
| `image` | 像方，字段见下 |
| `aperture.fno` | 近轴 F 数；交付件用 `FNO` 定孔径，光阑不设固定半径 |
| `targets` | 目标，字段见下 |
| `limits` | 限值，字段见下 |
| `glass` | 玻璃规则，字段见下 |
| `codev` | 交付件，字段见下 |
| `pipeline` | 覆盖 `run_pipeline.py` 的默认设置（键名同命令行，如 `keep`、`iters`、`top`），`env` 给各步加环境变量（如 `{"LENS_NRING": "8"}`） |

`surfaces[]` 每个面：
- `name`；
- `z`：顶点位置（mm，任意原点）；
- `R`：0 = 平面；
- `cir`：半口径；
- `glass_after`：这个面之后的元件名；
- `stop`：光阑面写 `true`；
- `asph`：`"var"` = 可变非球面；或固定系数 `[A4, A6, A8, A10]`；
- `solver_only`：只给求解器用的面（例如挡光环），不写进 CODE V。

`image`：
- `cover_glass {thickness, glass, gap}`；
- `back_gap_range`：后截距（最后一面 → 保护玻璃）允许的范围；
- `back_gap_random`：随机起点取后截距的范围；
- `image_height`：像高；
- `fields_mm`：MTF 比对用的实际像高。

`targets`：
- `mtf_file`；
- `mtf_key`：`geometric` 或 `diffraction`，看厂商图是哪种；
- `freqs`：线对数；
- `w_mtf`、`w_spot`：权重。

`limits`：
- `efl [lo, hi]`；
- `cra_max`：角点 CRA；
- `ri_min`：角点相对照度；
- `vd_max`；
- `dz_max`：近轴焦点允许偏离像面多少；
- `cra_angles_deg`：求 CRA 时插值用的视场角，可选。

`glass`：
- `vendors`；
- `rules`：`{"default": "polished", "L1": "moulded"}`，可选值 `moulded` / `polished` / `any`；
- `modes`：玻璃模式，例如 `{"moulded": {}, "hybrid": {"L8": "any"}}`，每个模式单独跑一遍；
- `moulded_prefix`：模压料的牌号前缀，默认 HOYA `M-`，OHARA `L-`；
- `catalog_dir`。

`codev`：
- `title`、`notes`；
- `gla`：宏里 GLA 行的牌号；
- `asph_R_range`；
- `rdy_tol`；
- `yri_fractions`：交付件视场占像高的比例；
- `opt_lines`：宏尾部的命令；
- `macro_dir`；
- `setvig_args`、`relillum_args`；
- `ri_deliver`：CODE V 复核时 RI 的下限。

### mtf_official.json

```json
{"geometric":   {"10S": {"h": [0, 1, ...], "mtf": [0.95, ...]}, "10M": {...}, "30S": {...}, "30M": {...}},
 "diffraction": {"10S": ...}}
```

- 键名是线对数加方向：S 是弧矢，对应 CODE V `MTF_1FLD` 的方位角 0；M 是子午，方位角 90。
- `h` 是像高（mm）。
- 厂商只给一种曲线就只写那一种。适马的是 Zemax 几何 MTF，不乘衍射极限；判断方法见 examples/sigma50cl。

## 算法

- **评价函数**：
  - 各视场、各波长用六角网格采样光瞳，每个视场 1 + 3n(n+1) 条光线，`LENS_NRING=n`，默认 5，正式跑用 8；
  - 光线落点的弥散斑行；
  - 按落点直接算几何 MTF，和官方曲线逐个视场比较，组成 MTF 行。
- **视场**：取实际像高，每轮按当前设计解出各视场角（类似 CODE V `YRI`）。
- **渐晕**：用实光线找出各视场能通过全部口径的光瞳边界（类似 setvig）；求导时把这个边界固定。
- **优化器**（`dls.py`）：仿 CODE V AUT 的阻尼最小二乘。
  - EFL、焦点、矢高框、玻璃图、vd、轴向间隙、CRA、RI 都作为约束（c ≤ 0），用拉格朗日乘子和有效集处理；
  - 阻尼自动选：一批 λ 同时解，用线性化后的光线落点预测 MTF，挑最好的几个用真实光追验证；
  - 导数：自动微分，或 Rust 引擎里的前向对偶数；
  - 连续搜索用硬约束；吸附之后的精修用软约束，此时 loss = ‖f‖² + ‖relu(c)‖²。
- **玻璃变量**（`glassmap.py`，`LENS_GMAP`）：
  - 模压片用 CODE V 式的 (nd, ΔN) 凸包，每条边一行线性约束；
  - 抛光片用围住真实玻璃的非凸多边形；
  - 这是默认的 `mixed`，另有 `hull` 和 `poly`。
- **吸附**（`snap.py attract`）：
  - 给每片加一行「到可用牌号的软最近距离」，权重逐步加大、温度逐步降低；
  - 某片距离小于 0.15（1 = 0.01 nd 或 1 个阿贝数）就冻结到那个牌号；
  - 最后剩下的按距离从近到远逐片吸附，每吸一片重新优化其余变量；
  - 吸附过程中，各片的折射率按附近牌号的反常色散加权修正，所以落到牌号上时的损失很小。
- **换料**（`snap.py swap`）：每片试最近的 k 个牌号，每个组合都精修一遍，loss 降低超过 3% 才接受。

## 环境变量

| 变量 | 作用 |
|---|---|
| `RESTORE_CONFIG` | config 路径（各脚本也接受 `--config`） |
| `GLASS_MODE` | 玻璃模式（`glass.modes` 的键） |
| `LENS_ENGINE` / `LENS_DEVICE` | `rust` 或 `torch`；`cpu` 或 `cuda`。默认：有 `lenscore.dll` 就用 Rust + CPU |
| `LENS_THREADS` | torch 在 CPU 上的线程数 |
| `LENS_NRING` | 光瞳环数（5 → 91 条光线；8 → 217 条） |
| `LENS_FIELDS` | `yri`（实际像高，默认）或 `ang`（固定视场角） |
| `LENS_GMAP` | `mixed`（默认）/ `hull` / `poly` |
| `LENS_GLASSMODEL` | `codev`（有拟合文件时默认）或 `schott2` |
| `LENS_GLASS_DIR` | 玻璃目录 |
| `LENS_EFL`, `LENS_CRA_MAX`, `LENS_RI_MIN`, `LENS_FNO`, `LENS_WW`, `T16BAND` | 临时覆盖 config 里的对应值 |
| `CODEV_MACRO_DIR` | CODE V 宏目录（`setvig.seq`、`Relillum.seq`） |

## Rust 引擎

`rust/lenscore` 是一个纯 C ABI 的动态库，Python 用 ctypes 加载，不需要 PyO3：

- 用 rayon 在全部核心上并行；
- 导数用前向对偶数；
- 求导时光瞳边界固定，这一点和 torch 引擎一样。

它和 torch 引擎的结果一致到 1e-13，比 GPU 上的 torch 引擎快 2.3–3.4 倍（适马 50 实测）。编译：

```bash
bash restore/rust/build.sh        # Windows 用 GNU 工具链（rustup x86_64-pc-windows-gnu），不需要 Visual Studio
```

产物 `restore/lenscore.dll`（Linux 上是 `liblenscore.so`）不进仓库。没有这个文件时自动退回 torch 引擎。

## 文件

```
config.py        读 config，导出几何、变量布局、限值、玻璃规则
glass.py         AGF 目录、在产过滤、虚构玻璃模型
glassmap.py      玻璃变量 ↔ (nd, vd)，玻璃图约束（mixed / hull / poly），每片允许的牌号
core.py          批量光追核心（torch，float64），近轴
params.py        u ↔ x 变量变换，随机起点
objective.py     视场、渐晕光瞳、几何 MTF、CRA、RI，残差和报告；命令行 = 连续搜索
dls.py           CODE V AUT 式阻尼最小二乘
rcore.py         Rust 引擎的 ctypes 接口
snap.py          attract / polish / swap
run_pipeline.py  整条流水线
deliver.py       .seq / _opt.seq / .xlsx（不需要 CODE V）
cvtools.py       CODE V：finalize / mtf / rank / layout
tools/config_from_seq.py   起始 .seq → config 草稿
tools/glassmodel.py        采样并拟合 CODE V 虚构玻璃
tests/check_sigma50.py     回归测试：和原适马求解器逐位一致
rust/                      Rust 引擎源码和编译脚本
```

## 验证

- `tests/check_sigma50.py`：用 6 个设计，和最初为适马 50 写的专用求解器比较。
  - 比较内容：光瞳、报告、评价函数、雅可比列范数、DLS 的误差函数；
  - torch 引擎逐位相同；
  - Rust 引擎：评价函数和雅可比一致到 4e-15；光瞳搜索各自二分，差一个步长（4e-6 mm）；
  - 运行：`LENS_GLASS_DIR=<目录> python restore/tests/check_sigma50.py`。
- `snap.py` 三个子命令：吸附、精修、换料，和原脚本跑出同样的牌号和数值（换料一步差 1e-12）。
- `deliver.py` + `cvtools.py`：重新生成第四轮交付版，CODE V 结果和原交付一致。
  - EPD 34.2799，CRA 31.25°，RI 0.214；
  - 几何 / 衍射 MTF 偏差 0.0280 / 0.0265。
- `tools/glassmodel.py`：重新采样拟合，系数和原拟合完全相同（5 次多项式）。
- `tools/config_from_seq.py`：用交付的 `.seq` 起草 config，对齐几个人工设定后，评价结果和手写的 config 一致。

## 局限

- **几何 MTF 采样**：求解器每个视场只有 91 或 217 条光线，CODE V 用 64×64 网格。两边对方案的排序可能不同，
  例如适马 50 第四轮，求解器认为混合方案更好，CODE V 认为全模压更好。**选方案以 CODE V 复核为准**（`--codev`）。
- 只做 INF 一个组态；对焦组态在 CODE V 里交给设计者加。
- 复合非球面的树脂层不建模，非球面直接写在玻璃面上。
- 半径和间隔锁定在结构图上，求解器不动它们，留给设计者在 CODE V 里按 ±5% 收尾。
