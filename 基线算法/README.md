# 完整基线算法

本文件定义当前项目的完整基线算法，入口为[load_histogram.py](load_histogram.py)。输入一条原始直方图，执行寻峰、30-bin取窗、背景估计、扣背景及负值置0，输出`csum`、`ctd`、完整`va`与近似计数`snr`，并绘制全谱和峰值局部图。后续va偏移补偿及1DCNN扩展以这套输入输出口径为基础。

默认读取`数据/有背景光/31_spad1_event40000_lux5000_3%_hist_dspfulldata.csv`的第1条记录。每条记录按数字顺序提取`hist0`至`hist2047`；其余字段作为原始元数据保留。

## 1 输入与输出

令原始计数为$H_i$，索引从0开始。背景规则要求数据包含bin 512，因此至少需要513个bin。原始计数必须有限且非负。

| 输出 | 定义 |
| --- | --- |
| `peak_bin` | 全谱最大原始计数所在的bin编号，相同最大值取编号最小者 |
| `peak_count` | 峰值bin的原始计数 |
| `signal_start`、`signal_stop` | 30-bin窗口的左闭右开边界 |
| `sa_avg` | bin 1至512中首个累计出现4次的计数值 |
| `background_match_bin` | 所选计数值第4次出现的bin编号 |
| `csum` | 窗口内逐bin扣背景、负值置0后的总计数 |
| `ctd` | 以上述非负计数加权得到的质心，单位bin |
| `va` | 包含质心所在bin部分计数的完整va，单位bin |
| `snr` | $csum/\sqrt{csum+30\,sa\_avg}$，线性比值 |

`background_start=1`、`background_stop=513`记录背景检索区间的左闭右开边界，不表示对整个区间取平均。

## 2 完整计算流程

### 2.1 寻峰与30-bin窗口

$$
p=\min\operatorname*{arg\,max}_i H_i,\qquad
a=p-15,\quad b=p+15,\quad L=b-a=30.
$$

窗口为$W=[a,b)$，包含峰前15个、峰值1个、峰后14个bin。峰值过于靠近边界时提示无法取得完整窗口，不静默缩短统计长度。

### 2.2 背景基底sa_avg

按bin 1到512的顺序扫描`histogram[1:513]`。每遇到一个计数值，就将该值的累计出现次数加1；首个累计出现4次的值直接作为$\beta=sa\_avg$，0也参与统计，不要求连续出现。

例如依次遇到`[7, 2, 2, 2, 2, 7, 7, 7]`，计数值2在第5步累计出现4次，返回2。该规则按达到4次的先后决定返回值，不取最终众数，也不按计数值从小到大选择。字段名沿用`sa_avg`，当前定义不计算算术平均。

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

先逐bin截零再求和，与先计算$\sum_i H_i-30\beta$再将总和截零不同。扣背景后无有效计数时，ctd和va未定义，程序提示无法计算。

### 2.5 质心ctd

程序以bin编号作为中心坐标，bin $i$覆盖$[i-0.5,i+0.5)$，因此

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

分母同时计入信号及统计窗口背景的散粒噪声，保持线性比值，不作dB换算。$\beta=0$时退化为$snr=\sqrt{csum}$，不会因背景为0而输出无穷大。

本式在非饱和、近似独立Poisson计数且暂不计背景估计误差时具有计数信噪比含义。当前背景选择及逐bin截零使其作为近似质量指标使用。它与CSV中原始`SNR`字段分别记录；存在饱和时也按本定义输出数值，但不将其解释为已校准的定位不确定度。

### 2.8 实现顺序

```python
peak_bin = int(np.argmax(histogram))
a, b = peak_bin - 15, peak_bin + 15
sa_avg, background_match_bin = estimate_sa_avg(histogram)
window = histogram[a:b]
q = np.maximum(window - sa_avg, 0.0)
csum = float(q.sum())
# compute_ctd_va内部执行相同的扣背景和截零，再按完整va定义计算。
ctd, va = compute_ctd_va(window, a, sa_avg)
snr = csum / np.sqrt(csum + (b - a) * sa_avg)
```

上述代码展示计算关系，正式实现的输入、窗口与分母有效性检查以`load_histogram.py`为准。

## 3 运行与绘图

在项目根目录`D:\Projects\1DCNN`运行：

```powershell
conda run -n py310 python -X utf8 "基线算法/load_histogram.py"
```

点击运行或执行上述命令，默认会显示直方图，并在`基线算法/输出`保存PNG；关闭图窗后程序结束。运行所选Python解释器应为`D:\Software\Conda\miniconda3\envs\py310\python.exe`。

上图展示全谱，下图展示峰值附近，两幅图的纵轴均使用普通线性刻度。横轴是CSV列对应的bin编号，尚未换算为时间或距离。

区间采用左闭右开的表示方式。峰值附近不足完整30个bin、输入不包含bin 512，或搜索区间内不存在出现至少4次的计数值时，程序会提示无法计算。图中橙色区域表示统计窗口，绿色区域表示`sa_avg`的搜索范围，绿色虚线表示`sa_avg`，紫色竖虚线表示扣背景后的`ctd`。绘图仍展示原始计数。

指定其他记录、数据文件或仅保存图片：

```powershell
conda run -n py310 python -X utf8 "基线算法/load_histogram.py" --row 10
& "D:\Software\Conda\miniconda3\envs\py310\python.exe" -X utf8 "基线算法/load_histogram.py" --csv "数据/有背景光/31_spad1_event40000_lux5000_88%_hist_dspfulldata.csv"
conda run -n py310 python -X utf8 "基线算法/load_histogram.py" --no-show
```

`--row`从0开始，不包含CSV表头。默认数据文件有600条记录，可选择0至599；每次加载一条，不合并多行。默认路径根据脚本位置确定，因此也可从其他工作目录启动。

指定含`%`的CSV文件名时，以上示例直接调用conda环境内的Python，避免本机`conda run`经批处理传参时改写百分号。

其他Python代码可在脚本所在目录导入并调用完整基线：

```python
from load_histogram import analyze_histogram, load_histogram

histogram, metadata = load_histogram(row_index=0)
stats = analyze_histogram(histogram)
print(stats.peak_bin, stats.csum, stats.sa_avg, stats.ctd, stats.va, stats.snr)
# histogram：长度为2048的NumPy数组，保留原始计数。
# metadata：该条记录的非直方图字段，保持CSV原始字符串。
```

## 4 当前实测示例

以下均为有背景光`lux5000`数据的第1条记录，所有指标按上述完整基线重新计算。

| 数据文件标识 | peak_bin | sa_avg | csum | ctd，bin | va，bin | snr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3% | 60 | 0 | 1556 | 60.239075 | 3.764433 | 39.446166 |
| 88% | 59 | 0 | 23102 | 59.187603 | 3.536090 | 151.993421 |
| 3m | 50 | 1 | 39685 | 50.400454 | 0.500756 | 199.135689 |

依赖为NumPy和Matplotlib，当前conda的`py310`环境已经具备。CSV中的`mp_peak_ctd`、`mp_peak_va`、`mp_peak_csum`、`mp_peak_sa_avg`、`SNR`和`real_distance`等字段仅作为原始元数据保留；其硬件量化与统计规则尚未确认，不直接与本基线同名指标视为相同结果。

## 5 与后续补偿的衔接

当前完整基线的输出为寻峰结果、背景基底、csum、ctd、完整va及snr。理论中的va系数补偿和1DCNN残差补偿是该基线之上的后续扩展。若增加这些阶段，应记录新的补偿输出，并保持窗口、背景处理、坐标及参考标签口径一致。

理论关系见[va补偿推导](../理论/va补偿质心偏移推导与三种饱和状态分析10301338.md)，后续扩展见[1DCNN饱和残差补偿方案](../理论/va补偿基础上的1DCNN饱和残差补偿方案.md)。

全量数据及公式独立校验结果见[基线算法检查报告](检查报告.md)。
