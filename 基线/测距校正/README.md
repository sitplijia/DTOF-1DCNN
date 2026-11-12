# Distance calibration interface

Run `启动界面.bat` or start `distance_calibration_gui.py` with Python. The interface takes one CSV for each target (3%, 88%, and 3m), lets you set the SNR threshold and VA coefficient, and selects a reference POS from the 3% CSV.

The GUI is presentation-only and calls the `algorithm` package. `algorithm/features.py` reads CSVs and extracts histogram features, `algorithm/interpolation.py` contains the POS/csum interpolation rules, `algorithm/pipeline.py` performs calibration and export, and `algorithm/models.py` defines the data structures. `distance_calibration.py` keeps a small compatibility import surface.

For every CSV row, the app reads `POS` and the contiguous `hist0`… columns, then computes the baseline peak, `sa_avg`, `csum`, and SNR. It calculates ctd and va only when SNR passes the selected threshold. The CSV's `real_distance` and vendor-calculated peak fields are not used; POS is treated as the true distance.

The distance sequence is:

```text
d_raw = ctd * 15 mm/bin
d_va = (ctd - K * va) * 15 mm/bin
b = mean(d_va for 3% rows at the selected POS) - selected POS
```

At each POS, calibration boundaries use the valid-frame mean csum for each target. `err_88` and `err_3m` are the corresponding mean VA-compensated distances minus the 3% mean. The csum interpolation follows the two-patch rule in `dtof_calib`: 3%→88% and 88%→3m. The interpolated error is subtracted from `d_va - b` for each valid row.

The `测距均值（按POS）` tab uses one row per POS with separate mean columns for 3%, 88%, and 3m. `标准差/极差/质心对比` also uses one row per POS, with side-by-side 3%, 88%, and 3m columns for distance standard deviation, max-minus-min range, and raw `ctd * 15 mm` standard deviation. `插值距离、csum和误差` displays the VA-compensated target distances, csum boundaries, `err_88`/`err_3m`, and the mean interpolated error applied to each target's frames. `导出统计 CSV` exports one row per POS, target-specific statistics, interpolation values, and calibration parameters.

The selected reference POS is treated as a distance in the same units as POS. K is dimensionless and corrects ctd in bin units before conversion to distance. The 15 mm/bin factor is currently fixed in the implementation.
# VA 插值标签页

运行 `启动界面.bat` 后，在原有“插值距离、csum和误差”旁可选择“插值距离、va和误差”。两个结果标签页共用上方的 CSV、SNR 门限、VA 补偿系数和 3% 截距标定 POS。选择对应插值结果页后点击“计算”，测距均值、标准差和导出使用该插值方式；切换方式时清空旧结果，避免混淆。

VA 模式先沿 POS 插值三种靶面的 VA 均值及 VA 补偿后的相对误差，再以当前帧的 VA 在 3%—88%—3m 锚点之间作分段线性插值。3% 锚点误差为 0，最终距离仍减去截距和插值误差。页面显示所用 VA 锚点，导出包含 VA、csum 和插值模式。

VA 模式先沿 POS 插值得到锚点，再将三组 `(VA, 误差)` 按 VA 数值排序，保留靶面与误差的对应关系，在相邻 VA 锚点之间线性插值，不要求靶面顺序与 VA 大小一致。数值间隔不超过 1e-6 时报告错误；越出锚点范围时夹取端点。此间隔只是数值保护，实际可分辨性还需根据 VA 的帧间波动评估。排序后形成的是经验校正曲线，不代表物理关系已验证为单值。当前结果属于用同批数据标定和统计的结果，需要独立数据评估泛化效果。
