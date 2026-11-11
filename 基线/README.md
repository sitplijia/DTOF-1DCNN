# 完整基线

本文件定义当前项目的完整基线，入口为[load_histogram.py](load_histogram.py)。输入一条原始直方图，执行全局最大候选峰定位、30-bin取窗、背景估计、扣背景及负值置0，再以SNR阈值判断峰有效性。有效峰输出`ctd`和完整`va`；无有效峰时二者为`None`，保留`csum`、`snr`等诊断信息。默认SNR阈值为8，可配置；全局最大候选峰不合格时不再查找其他峰。

默认读取`数据/有背景光/31_spad1_event40000_lux5000_3%_hist_dspfulldata.csv`的第1条记录。每条记录按数字顺序提取`hist0`至`hist2047`；其余字段作为原始元数据保留。

## 1 输入与输出

令原始计数为$H_i$，索引从0开始。背景规则要求数据包含bin 512，因此至少需要513个bin。原始计数必须有限且非负。

| 输出 | 定义 |
| --- | --- |
| `peak_bin` | 全谱最大候选峰的bin编号，相同最大值取编号最小者；不合格时仅供诊断 |
| `peak_count` | 峰值bin的原始计数 |
| `signal_start`、`signal_stop` | 30-bin窗口的左闭右开边界 |
| `sa_avg` | 全局峰后跳过160个bin，再取接下来的100个bin计数均值 |
| `csum` | 窗口内逐bin扣背景、负值置0后的总计数 |
| `ctd` | 有效峰以非负计数加权得到的质心，单位bin；无有效峰时为`None` |
| `va` | 有效峰包含部分bin计数的完整va，单位bin；无有效峰时为`None` |
| `snr` | $csum/\sqrt{csum+30\,sa\_avg}$，线性比值 |
| `snr_threshold` | 有效峰SNR下限，默认8 |
| `peak_valid` | 有净信号且`snr >= snr_threshold`时为`True`，否则为`False` |

`background_start`和`background_stop`记录sa_avg均值窗口的左闭右开边界。窗口碰到直方图末尾时使用剩余bin；没有可用bin时二者均为-1且sa_avg设为0。

## 2 完整计算流程

### 2.1 寻峰与30-bin窗口

$$
p=\min\operatorname*{arg\,max}_i H_i,\qquad
a=p-15,\quad b=p+15,\quad L=b-a=30.
$$

窗口为$W=[a,b)$，包含峰前15个、峰值1个、峰后14个bin。峰值过于靠近边界时提示无法取得完整窗口，不静默缩短统计长度。这里先得到候选位置，是否为有效峰由第2.8节的SNR判断决定。

### 2.2 背景基底sa_avg

令全局峰bin为$p$。背景估计窗口从$p+160$开始，取100个bin：

$$
\mathcal{B}=[p+160,p+260),\qquad sa\_avg=\frac{1}{|\mathcal{B}|}\sum_{i\in\mathcal{B}}H_i.
$$

若窗口超出直方图末尾，则对剩余可用bin取均值；若没有可用bin，设`sa_avg=0`。这里的160和100均相对于全局峰bin，bin索引从0开始。`sa_avg`是计数/bin，不是算术以外的众数估计，也不等同于输入的`background_lambda`。

### 2.3 扣背景并将负值置0

窗口内统一构造

$$
q_i=\max(H_i-\beta,0),\qquad i\in W.
$$

`csum`、`ctd`、`va`和`snr`均使用这一背景处理口径。去除背景的数学依据见[背景扣除推导第9节](../理论/背景扣除对ctd和完整va的影响推导.md#9-不考虑饱和时ctd和va均应扣背景的推导)。

### 2.4 总计数csum

$$
\boxed{csum=\sum_{i=a}^{b-1}q_i}.
$$

先逐bin截零再求和，与先计算$\sum_i H_i-30\beta$再将总和截零不同。候选窗口无净信号时，将SNR记为0、`peak_valid=False`，ctd和va均为`None`。

### 2.5 质心ctd

仅在SNR门限通过且净信号为正时计算ctd和va。程序以bin编号作为中心坐标，bin $i$覆盖$[i-0.5,i+0.5)$，因此

$$
\boxed{ctd=\frac{\sum_{i=a}^{b-1}i\,q_i}{csum}}.
$$

该坐标与绘图横轴一致。理论文档采用中心坐标$x_i=i+0.5$，对应理论质心为$ctd+0.5$；坐标整体平移不改变va和质心偏移量。

### 2.6 完整va

设质心所在bin及其左侧占比为

$$
P=\lfloor ctd+0.5\rfloor,\qquad
f=ctd-(P-0.5),\qquad 0\le f<1.
$$

在bin内均匀密度约定下，左翼有效计数与距离矩为

$$
M_L=\sum_{i=a}^{P-1}q_i+q_P f,
$$

$$
D_L=\sum_{i=a}^{P-1}q_i(ctd-i)+\frac12 q_P f^2.
$$

因此

$$
\boxed{va=\frac{D_L}{M_L}}.
$$

该定义保留质心所在bin的部分计数；va分母是左翼计数$M_L$，不是整窗csum。推导与[原理论文档](../理论/va补偿质心偏移推导与三种饱和状态分析10301338.md)式（7）一致，只是坐标原点相差0.5。

### 2.7 近似计数信噪比snr

$$
\boxed{snr=\frac{csum}{\sqrt{csum+L\beta}}
=\frac{csum}{\sqrt{csum+30\,sa\_avg}}}.
$$

分母同时计入信号及统计窗口背景的散粒噪声，保持线性比值，不作dB换算。$\beta=0$且净信号为正时退化为$snr=\sqrt{csum}$，不会因背景为0而输出无穷大。候选窗口净信号为0时，SNR记为0。

本式在非饱和、近似独立Poisson计数且暂不计背景估计误差时具有计数信噪比含义。当前背景选择及逐bin截零使其作为近似质量指标使用。它与CSV中原始`SNR`字段分别记录；存在饱和时也按本定义输出数值，但不将其解释为已校准的定位不确定度。

### 2.8 SNR门限与无有效峰

默认阈值$T=8$，有效性定义为

$$
peak\_valid=(csum>0)\ \land\ (snr\ge T).
$$

阈值采用大于或等于判断。只检验第2.1节的全局最大候选峰，不合格时返回`peak_valid=False`，`ctd=None`、`va=None`，不退回其他候选峰。`peak_bin`、窗口边界、csum和snr继续返回用于诊断，下游应先判断`peak_valid`。

8是当前数据下的可调工程初始值，不等于统计学上的8-sigma显著性，也不保证零虚警。用主峰之后的片段做探索检查时，3%和88%的可用片段最高SNR约为6.00和7.21，但3m的同类片段可达到57.69，不能认定这些片段都是纯背景。因此当前检查不能给出可靠的假峰率；正式阈值应由独立的无目标背景测量和允许的虚警率标定。按虚警概率选择背景相关阈值的研究可参考[Adaptive single photon detection under fluctuating background noise](https://doi.org/10.1364/OE.404681)。

阈值必须有限且非负。可设置`--snr-threshold 0`停用正阈值筛选，但无净信号的窗口仍无效。

### 2.9 实现顺序

```python
peak_bin = int(np.argmax(histogram))
a, b = peak_bin - 15, peak_bin + 15
sa_avg, background_start, background_stop = estimate_sa_avg(histogram, peak_bin)
window = histogram[a:b]
q = np.maximum(window - sa_avg, 0.0)
csum = float(q.sum())
noise_variance = csum + (b - a) * sa_avg
snr = csum / np.sqrt(noise_variance) if noise_variance > 0 else 0.0
peak_valid = csum > 0 and snr >= snr_threshold
if peak_valid:
    # 函数内部执行相同的扣背景和截零，再按完整va定义计算。
    ctd, va = compute_ctd_va(window, a, sa_avg)
else:
    ctd, va = None, None
```

上述代码展示计算关系，正式实现的输入、窗口与分母有效性检查以`load_histogram.py`为准。

## 3 运行与绘图

在项目根目录`D:\Projects\1DCNN`运行：

```powershell
conda run -n py310 python -X utf8 "基线/load_histogram.py"
```

点击运行或执行上述命令，默认会显示直方图，并在`基线/输出`保存PNG；关闭图窗后程序结束。运行所选Python解释器应为`D:\Software\Conda\miniconda3\envs\py310\python.exe`。

上图展示全谱，下图展示峰值附近，两幅图的纵轴均使用普通线性刻度。横轴是CSV列对应的bin编号，尚未换算为时间或距离。

区间采用左闭右开的表示方式。峰值附近不足完整30个bin时提示无法计算。图中橙色区域表示候选统计窗口，绿色区域表示`sa_avg`均值窗口，绿色虚线表示`sa_avg`。有效峰为红点并显示紫色ctd线；无有效峰时使用灰色候选点，标题显示`No valid peak`，不绘制ctd线。绘图仍展示原始计数。

指定其他记录、数据文件或仅保存图片：

```powershell
conda run -n py310 python -X utf8 "基线/load_histogram.py" --row 10
& "D:\Software\Conda\miniconda3\envs\py310\python.exe" -X utf8 "基线/load_histogram.py" --csv "数据/有背景光/31_spad1_event40000_lux5000_88%_hist_dspfulldata.csv"
conda run -n py310 python -X utf8 "基线/load_histogram.py" --no-show
conda run -n py310 python -X utf8 "基线/load_histogram.py" --snr-threshold 10 --no-show
```

`--row`从0开始，不包含CSV表头。默认数据文件有600条记录，可选择0至599；每次加载一条，不合并多行。默认路径根据脚本位置确定，因此也可从其他工作目录启动。

指定含`%`的CSV文件名时，以上示例直接调用conda环境内的Python，避免本机`conda run`经批处理传参时改写百分号。

其他Python代码可在脚本所在目录导入并调用完整基线：

```python
from load_histogram import analyze_histogram, load_histogram

histogram, metadata = load_histogram(row_index=0)
stats = analyze_histogram(histogram)
if stats.peak_valid:
    print(stats.peak_bin, stats.csum, stats.sa_avg, stats.ctd, stats.va, stats.snr)
else:
    print("无有效峰", stats.snr, stats.snr_threshold)
# histogram：长度为2048的NumPy数组，保留原始计数。
# metadata：该条记录的非直方图字段，保持CSV原始字符串。
```

## 4 当前实测示例

以下均为有背景光`lux5000`数据的第1条记录，所有指标按上述完整基线重新计算。

三条记录的SNR均通过默认阈值8，所以下表的ctd和va均为有效输出。

| 数据文件标识 | peak_bin | sa_avg | csum | ctd，bin | va，bin | snr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3% | 60 | 0 | 1556 | 60.239075 | 3.764433 | 39.446166 |
| 88% | 59 | 0 | 23102 | 59.187603 | 3.536090 | 151.993421 |
| 3m | 50 | 1 | 39685 | 50.400454 | 0.500756 | 199.135689 |

依赖为NumPy和Matplotlib，当前conda的`py310`环境已经具备。CSV中的`mp_peak_ctd`、`mp_peak_va`、`mp_peak_csum`、`mp_peak_sa_avg`、`SNR`和`real_distance`等字段仅作为原始元数据保留；其硬件量化与统计规则尚未确认，不直接与本基线同名指标视为相同结果。

## 5 与后续补偿的衔接

当前完整基线的输出为寻峰结果、背景基底、csum、ctd、完整va及snr。理论中的va系数补偿和1DCNN残差补偿是该基线之上的后续扩展。若增加这些阶段，应记录新的补偿输出，并保持窗口、背景处理、坐标及参考标签口径一致。

只有`peak_valid=True`时，ctd和va才能进入后续补偿或作为有效特征使用。回归检查可运行：`python -m unittest discover -s "基线" -p "test_*.py" -v`。

理论关系见[va补偿推导](../理论/va补偿质心偏移推导与三种饱和状态分析10301338.md)，后续扩展见[1DCNN饱和残差补偿方案](../理论/va补偿基础上的1DCNN饱和残差补偿方案.md)。

全量数据及公式独立校验结果见[基线检查报告](检查报告.md)。
