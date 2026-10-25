# 适马 50mm F1.4 CL DG | Contemporary（C027）还原

`restore/` 的第一个用例。几何来自适马官网的结构图（SVG，1 单位 = 1 mm），并用 PetaPixel 的剖面照片核对过。
玻璃、两个非球面和后截距靠贴合适马官方 MTF 图求出。只做 INF。

| 文件 | 内容 |
|---|---|
| `config.json` | 镜头定义（字段说明见 [restore/README.md](../../restore/README.md)） |
| `mtf_official.json` | 官方 MTF 图数字化：10 / 30 lp/mm，S / M，几何和衍射两种 |
| `regression.json` | 原专用求解器对 6 个设计的输出，供 `restore/tests/check_sigma50.py` 回归 |

## config 里的几处设定

- **BAF**：一个只给求解器用的挡光面（`solver_only`），不写进 CODE V。
- **S4**：非球面系数直接取自 CAD 曲线，固定不动。
- **S1 / S13**：可变非球面。
  - SVG 里这两条线画成带折点的贝塞尔曲线，不能直接用；
  - 矢高改在剖面照片上逐点量，误差约 ±0.025 mm；
  - 矢高框 ±0.035 mm，最外一点 ±0.01 mm。
- **玻璃**：
  - L4（FLD）指定为 HOYA FCD100；
  - 非球面片 L1、L2、L8 用模压料（HOYA M- / OHARA L-）；
  - `hybrid` 模式允许 L8 用抛光料加树脂层（复合非球面）；
  - vd ≤ 72；只用 HOYA / OHARA 的在产牌号。
- **保护玻璃**：BSC7 2.5 mm，后接空气 1.0 mm。
- **交付条件**：
  - 角点 21.6 mm 处 CRA < 32°；
  - 相对照度 ≥ 0.21（CODE V 复核时 ≥ 0.2）；
  - EFL 48.9–50.5；
  - FNO 1.46，光阑不设固定半径。
- **目标 MTF**：适马官方 MTF 是 Zemax 的几何 MTF，不乘衍射极限（用 ZOS-API 核实过）。所以 `mtf_key` 用 `geometric`。

## 结果

下表的 MTF 偏差是和官方曲线逐点比较的平均 |偏差|，CODE V MTF_1FLD 按实际像高计算。
所有方案 L4 都是 FCD100。

| 方案 | L1 | L2 | L3 | L5 | L6 | L7 | L8 | L9 | EFL | CRA | RI | 几何 | 衍射 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **第四轮全模压（交付）** | M-TAC80 | M-LAC130 | S-TIH4 | S-TIM2 | TAF4 | TAFD30 | L-BSL7 | S-TIL1 | 50.05 | 31.25° | 0.214 | **0.028** | **0.027** |
| 第四轮混合 | M-TAC80 | L-LAL15 | FD140 | E-F3 | TAF48 | TAFD37A | S-TIL1* | S-TIL1 | 49.82 | 31.45° | 0.215 | 0.032 | 0.029 |
| 第三轮混合 | L-LAM60 | M-LAC8 | FD110 | E-F8 | TAF1 | TAFD33 | E-FEL2* | E-C3 | 49.35 | 31.98° | 0.203 | 0.037 | 0.035 |
| 第三轮全模压 | L-LAM69 | M-LAC8 | S-TIH11 | S-FTM16 | S-LAH96 | S-LAH58 | M-FCD500 | FC5 | 49.59 | 31.59° | 0.202 | 0.086 | 0.086 |

\* 复合非球面（树脂层未建模）。

第四轮交付版：

- 轴上 10 / 30 lp/mm 为 0.956 / 0.731，官方 0.941 / 0.715；
- 剩余偏差主要在中间视场（12.5–15 mm）的 30 lp/mm 弧矢方向，比官方低 0.05–0.1。

前三轮和第四轮的区别：

- 换成 CODE V AUT 式的 DLS；
- 玻璃图改成 mixed：模压片用凸包，抛光片用多边形；
- 加了单片换料。

## 重跑

```bash
export LENS_GLASS_DIR=<HOYA/OHARA AGF 所在目录>
python restore/tests/check_sigma50.py                                          # 回归测试
python restore/run_pipeline.py --config examples/sigma50cl/config.json --out runs/sigma50 --codev
```

第五、六轮用的是 `LENS_NRING=8`：在 config 里加 `"pipeline": {"env": {"LENS_NRING": "8"}}`，
再用 `--starts` 从已有的连续解出发（每个模式 96 个）。
从随机起点搜索更慢，最后落到哪组牌号也会不同。
