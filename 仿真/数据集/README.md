# 信号与背景lambda网格仿真数据集

每个质心位置使用一个子目录。横轴为信号`lambda_signal`（配置与数组字段名`signal_lambda`），纵轴为每个bin的背景`lambda_bg`（字段名`background_lambda`），每个网格点保存100条直方图及一一对应的ctd、完整va、csum、snr和sa_avg。每条记录累计 **5000周期**，每条直方图2048个bin，死时间120 bin，随机种子20261111。

## lambda定义

模型输入为：

```text
lambda_input[y, x, i] = signal_lambda[x] * q[i] + background_lambda[y]
sum(q) = 1
```

- `signal_lambda`：整条信号谱每周期的平均入射光子数，是归一化信号模板的乘数。
- `background_lambda`：每个bin、每周期的平均入射背景光子数，直接加到前向模型的每个bin，不除以周期数，也不除以bin数。
- 每条5000周期记录的期望入射信号总数为`5000 * signal_lambda`，每bin期望入射背景数为`5000 * background_lambda`。探测计数还受到死时间和饱和的影响，不等于入射光子数。

采用已有单SPAD、非瘫痪死时间、逐周期复位模型；每条随机谱独立生成，不是对确定性前向期望简单加噪声。

## 默认范围

| 参数 | 取值 |
| --- | --- |
| 名义质心位置（bin） | 256、512、768、1024、1280、1536、1792 |
| 横轴signal_lambda | 0.001、0.003、0.01、0.03、0.1、0.3、1、3、10、30 |
| 纵轴background_lambda | 0、0.000001、0.00001、0.0001、0.001、0.005 |
| 每个组合的随机记录数 | 100 |
| 每条累计周期数 | 5000 |

共7个质心位置，每个位置60个信号/背景组合，共 **42000条直方图**。信号按对数间隔从弱信号到明显饱和；背景最高为每bin每条记录25个期望入射光子。网格汇总另存最大条件触发概率，便于查看饱和程度。已移除质心1和2048的配置及数据目录。

质心、ctd和bin坐标在本数据集中统一采用 **1至2048**。已有基线输出的0起始ctd在导出时加1，va不变。未指定bin宽与距离标定系数，因此目录表示bin位置，不直接换算为米。

## 信号模板与边缘位置

模板来自无背景光3%实测数据前100条：逐条反演，按bin平均，保留主峰附近30-bin窗口（原数据0起始区间`[44,74)`），窗口外置0，再归一化为面积1。这样不会把实测远端低计数尾部作为信号一起放大；主峰窗口内未额外扣背景。模板源数据反演仍采用文件实际的40000周期，与新生成直方图的5000周期区分。

平移使用线性插值和零填充，不环绕。当前7个位置的信号均未截窗，实际输入信号质心等于目录标注位置。`metadata.json`保存名义位置、实际输入信号质心和截窗前保留比例。脚本仍支持自定义边缘位置：截窗后将窗口内信号重新归一化为面积1，因此signal_lambda始终表示观测窗口内的信号总量；此时目录名表示名义位置，不应当作截窗后的真实质心标签。

## 文件组织与读取

```text
仿真/数据集/
  config.json                  可修改的生成配置
  generate_dataset.py          生成脚本
  run_config.json              本次实际配置快照
  template_info.json           模板来源与处理口径
  source_template.npz          未平移的单位面积主峰模板
  manifest.json                目录清单和样本总数
  ctd_256/
    dataset.npz                100条/组合的直方图与特征张量
    samples.csv                每条记录的坐标、五项特征及状态
    grid_summary.csv           各组合的均值、有效数量及饱和程度
    grid_summary.png           横轴信号lambda、纵轴背景lambda的网格图
    metadata.json              参数、坐标、数组形状及边界规则
  ...
  ctd_1792/
```

`histograms`形状为 **`(6, 10, 100, 2048)`**，顺序为`[背景索引y, 信号索引x, 帧k, bin索引i]`。`ctd`、`va`、`csum`、`snr`、`sa_avg`及有效性字段形状均为`(6, 10, 100)`。`lambda_input`和确定性前向期望`expected_histograms`形状为`(6, 10, 2048)`。随机直方图是整数计数。

`sa_avg`采用原基线的背景估计：扫描原始0起始bin 1至512，取首个累计出现4次的计数值，单位为每条直方图的**探测计数/bin**。它是从随机探测谱估计的背景，不是输入`background_lambda`，也不直接设为`5000 * background_lambda`。低SNR记录仍保留能够计算的sa_avg；无法分析的记录为NaN。`grid_summary.csv`保存`sa_avg_mean`和`sa_avg_finite_frames`，`grid_summary.png`展示对应均值网格。

```python
from pathlib import Path
import numpy as np

path = Path("仿真/数据集/ctd_256/dataset.npz")
with np.load(path, allow_pickle=False) as data:
    x = int(np.flatnonzero(data["signal_lambda"] == 1)[0])
    y = int(np.flatnonzero(data["background_lambda"] == 0.00001)[0])
    histograms = data["histograms"][y, x]  # (100, 2048)
    ctd = data["ctd"][y, x]               # (100,)
    va = data["va"][y, x]
    csum = data["csum"][y, x]
    snr = data["snr"][y, x]
    sa_avg = data["sa_avg"][y, x]
    valid = data["peak_valid"][y, x]
```

所有100条都保留，低信噪比记录不被筛掉。主峰采用全局最大峰；内区复用原基线的30-bin窗口、背景估计、扣背景截零、完整va和SNR门限8。边界峰的窗口裁剪到观测范围，并标记`window_clipped`。SNR不合格时`ctd/va`为NaN（CSV中为空），`csum/snr`仍保留；全零谱等无法分析的记录标记`analysis_error`并记录原因。汇总图中的ctd/va均值只基于有效值，CSV同时提供有效数量，不能将其解释为全部100条都有效。

## 生成与检查

在仓库根目录、安装依赖的Python环境中运行：

```powershell
python -X utf8 "仿真/数据集/generate_dataset.py"
python -X utf8 -m unittest discover -s "仿真/数据集" -p "test_*.py" -v
```

调整`config.json`后可指定新输出目录：

```powershell
python -X utf8 "仿真/数据集/generate_dataset.py" --config "仿真/数据集/config.json" --output "仿真/数据集/新配置输出"
```

也可使用`--overwrite`在原目录重建配置中各质心目录内的生成文件；脚本不会自动删除配置之外的质心目录或其他文件。

每个组合的随机种子按基础种子、名义质心、信号lambda、背景lambda派生。调整坐标顺序不会改变同一组合的数据；`cell_seeds`保存实际使用的种子。
