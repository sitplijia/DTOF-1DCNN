# 1DCNN直方图补偿研究

本项目包含实测光子计数直方图的完整基线算法、背景扣除与va补偿的数学推导，以及后续1DCNN饱和残差补偿方案。

当前Python基线完成：全局最大候选峰定位、30-bin取窗、背景基底选择、逐bin扣背景与负值置0，以及csum、近似计数SNR的计算。SNR通过门限后计算ctd和完整va；默认阈值为8，不合格时返回无有效峰，不查找其他候选峰。1DCNN网络设计记录在理论文档中，尚未训练模型。

## 项目目录

| 目录 | 内容 |
| --- | --- |
| [基线算法](基线算法/README.md) | Python实现、完整算法定义、检查报告及示例图片 |
| [数据](数据/) | 有背景光与无背景光的6组CSV实测数据 |
| [理论](理论/) | va补偿推导、背景扣除推导及1DCNN扩展方案 |

## 运行

已验证环境为Python 3.10.20、NumPy 2.2.5、Matplotlib 3.10.9。进入项目根目录，在所选conda环境中安装依赖并运行：

```powershell
python -m pip install -r requirements.txt
python -X utf8 "基线算法/load_histogram.py"
```

默认显示有背景光3%数据第1条记录的完整直方图和峰值局部图，并将PNG保存在`基线算法/输出`。关闭图窗后程序结束。仅保存图片可使用`--no-show`，选择记录可使用`--row`，指定文件可使用`--csv`。

```powershell
python -X utf8 "基线算法/load_histogram.py" --row 10 --no-show
python -X utf8 "基线算法/load_histogram.py" --snr-threshold 10 --no-show
python -X utf8 "基线算法/load_histogram.py" --csv "数据/有背景光/31_spad1_event40000_lux5000_88%_hist_dspfulldata.csv"
```

基线SNR定义为`csum / sqrt(csum + 30 * sa_avg)`；有效峰的ctd和完整va使用同一组扣背景、负值置0后的窗口计数。下游先检查`peak_valid`：无有效峰时ctd和va为`None`，候选位置与SNR仅供诊断。阈值8是可调初始值，不能保证不存在假峰。背景估计规则、坐标约定和公式见[完整基线说明](基线算法/README.md)，验证范围见[检查报告](基线算法/检查报告.md)。
