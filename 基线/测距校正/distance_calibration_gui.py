"""Desktop UI for SNR-gated, VA-compensated distance calibration."""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path

from algorithm import (
    PROJECT_DIR,
    TARGETS,
    calibrate,
    read_positions,
    save_summary_csv,
)


def default_csv(target: str) -> str:
    candidates = sorted(PROJECT_DIR.rglob("*hist_dspfulldata.csv"))
    if target == "3%":
        matches = [p for p in candidates if "_3%_" in p.name]
    elif target == "88%":
        matches = [p for p in candidates if "_88%_" in p.name]
    else:
        matches = [p for p in candidates if "_3m_" in p.name]
    preferred = [p for p in matches if "lux0" in p.name]
    selected = preferred[0] if preferred else (matches[0] if matches else None)
    return str(selected) if selected else ""


class DistanceCalibrationApp:
    def __init__(self, root: tk.Misc, interpolation_axis: str = "csum") -> None:
        self.root = root
        self.interpolation_axis = interpolation_axis
        self.result = None
        self.path_vars = {target: tk.StringVar(value=default_csv(target)) for target in TARGETS}
        self.snr_var = tk.StringVar(value="12")
        self.va_var = tk.StringVar(value="0")
        self.pos_var = tk.StringVar()
        self.intercept_var = tk.StringVar(value="截距：—")
        self.status_var = tk.StringVar(value="选择 3%、88%、3m 靶面数据后开始计算。")
        self._build()
        self._refresh_positions()

    def _build(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)

        files = ttk.LabelFrame(outer, text="输入数据 CSV", padding=8)
        files.pack(fill="x")
        for row, target in enumerate(TARGETS):
            ttk.Label(files, text=f"{target} 靶面", width=12).grid(row=row, column=0, sticky="w", pady=3)
            ttk.Entry(files, textvariable=self.path_vars[target]).grid(row=row, column=1, sticky="ew", padx=5)
            ttk.Button(files, text="浏览…", command=lambda t=target: self._browse(t)).grid(row=row, column=2, padx=4)
        files.columnconfigure(1, weight=1)

        settings = ttk.LabelFrame(outer, text="校准参数", padding=8)
        settings.pack(fill="x", pady=(10, 0))
        ttk.Label(settings, text="SNR阈值").grid(row=0, column=0, sticky="w")
        ttk.Entry(settings, textvariable=self.snr_var, width=12).grid(row=0, column=1, sticky="w", padx=(6, 18))
        ttk.Label(settings, text="VA补偿系数 K（无量纲）").grid(row=0, column=2, sticky="w")
        ttk.Entry(settings, textvariable=self.va_var, width=12).grid(row=0, column=3, sticky="w", padx=(6, 18))
        ttk.Label(settings, text="3%截距标定 POS").grid(row=0, column=4, sticky="w")
        self.pos_combo = ttk.Combobox(settings, textvariable=self.pos_var, width=14, state="readonly")
        self.pos_combo.grid(row=0, column=5, sticky="w", padx=6)
        ttk.Button(settings, text="计算", command=self._calculate).grid(row=0, column=6, padx=(12, 0))
        ttk.Button(settings, text="导出统计 CSV…", command=self._save).grid(row=0, column=7, padx=6)
        settings.columnconfigure(7, weight=1)

        formula = (
            "VA补偿距离 = (ctd − K × va) × 15 mm/bin；"
            "截距 = 选定POS处3%靶面的VA补偿距离均值 − 该POS。最终距离再减截距和插值误差。"
        )
        ttk.Label(outer, text=formula, wraplength=1120, justify="left").pack(fill="x", pady=8)
        ttk.Label(outer, textvariable=self.intercept_var, font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 5))

        notebook = ttk.Notebook(outer)
        self.notebook = notebook
        self.interpolation_tabs = {}
        self.interpolation_tables = {}
        notebook.pack(fill="both", expand=True)

        means_tab = ttk.Frame(notebook, padding=5)
        notebook.add(means_tab, text="测距均值（按POS）")
        self.means_table = self._make_table(means_tab, {
            "pos": ("POS", 150), "mean_3": ("3%均值 (mm)", 220),
            "mean_88": ("88%均值 (mm)", 220), "mean_3m": ("3m均值 (mm)", 220),
        })

        stats_tab = ttk.Frame(notebook, padding=5)
        notebook.add(stats_tab, text="标准差/极差/质心对比")
        self.stats_table = self._make_table(stats_tab, {
            "pos": ("POS", 100),
            "std_3": ("3%测距标准差 (mm)", 145), "std_88": ("88%测距标准差 (mm)", 150),
            "std_3m": ("3m测距标准差 (mm)", 145),
            "range_3": ("3%测距极差 (mm)", 135), "range_88": ("88%测距极差 (mm)", 140),
            "range_3m": ("3m测距极差 (mm)", 135),
            "raw_3": ("3%原始质心标准差 (mm)", 155), "raw_88": ("88%原始质心标准差 (mm)", 160),
            "raw_3m": ("3m原始质心标准差 (mm)", 155),
        })

        for axis in ("csum", "va"):
            interpolation_tab = ttk.Frame(notebook, padding=5)
            notebook.add(interpolation_tab, text=f"插值距离、{axis}和误差")
            self.interpolation_tabs[str(interpolation_tab)] = axis
            ttk.Label(interpolation_tab, text="选择本页后点击上方“计算”；测距均值、标准差和导出随当前插值方式更新。").pack(anchor="w", pady=(0, 5))
            self.interpolation_tables[axis] = self._make_table(interpolation_tab, {
            "pos": ("POS", 85), "d3": ("3%补偿距离 (mm)", 130),
            "d88": ("88%补偿距离 (mm)", 135), "d3m": ("3m补偿距离 (mm)", 130),
            "cs3": (f"3% {axis}", 100), "cs88": (f"88% {axis}", 100),
            "cs3m": (f"3m {axis}", 100), "err88": ("err_88 (mm)", 110),
            "err3m": ("err_3m (mm)", 110), "used3": ("3%平均插值误差", 140),
            "used88": ("88%平均插值误差", 145), "used3m": ("3m平均插值误差", 140),
        })
        self.interpolation_table = self.interpolation_tables[self.interpolation_axis]
        notebook.bind("<<NotebookTabChanged>>", self._select_interpolation)
        ttk.Label(outer, textvariable=self.status_var, wraplength=1120).pack(fill="x", pady=(8, 0))

    def _select_interpolation(self, event=None) -> None:
        axis = self.interpolation_tabs.get(self.notebook.select())
        if axis is None or axis == self.interpolation_axis:
            return
        self.interpolation_axis = axis
        self.interpolation_table = self.interpolation_tables[axis]
        self.result = None
        for table in (self.means_table, self.stats_table, *self.interpolation_tables.values()):
            table.delete(*table.get_children())
        self.intercept_var.set("截距：—")
        self.status_var.set(f"已切换为 {axis} 插值，请点击“计算”。")

    @staticmethod
    def _make_table(parent: ttk.Frame, headings: dict[str, tuple[str, int]]) -> ttk.Treeview:
        frame = ttk.Frame(parent)
        frame.pack(fill="both", expand=True)
        table = ttk.Treeview(frame, columns=tuple(headings), show="headings")
        for key, (title, width) in headings.items():
            table.heading(key, text=title)
            table.column(key, width=width, anchor="center", stretch=True)
        yscroll = ttk.Scrollbar(frame, orient="vertical", command=table.yview)
        xscroll = ttk.Scrollbar(parent, orient="horizontal", command=table.xview)
        table.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        table.pack(side="left", fill="both", expand=True)
        yscroll.pack(side="right", fill="y")
        xscroll.pack(fill="x")
        return table

    def _browse(self, target: str) -> None:
        path = filedialog.askopenfilename(
            title=f"选择 {target} 靶面 CSV",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
        )
        if path:
            self.path_vars[target].set(path)
            if target == "3%":
                self._refresh_positions()

    def _refresh_positions(self) -> None:
        path = self.path_vars["3%"].get().strip()
        if not path or not Path(path).is_file():
            self.pos_combo["values"] = ()
            self.pos_var.set("")
            return
        try:
            positions = read_positions(path)
        except (OSError, ValueError):
            return
        values = [f"{pos:g}" for pos in positions]
        self.pos_combo["values"] = values
        if self.pos_var.get() not in values:
            self.pos_var.set(values[0])

    @staticmethod
    def _format(value: float | None) -> str:
        return "—" if value is None else f"{value:.6f}"

    def _calculate(self) -> None:
        try:
            snr_threshold = float(self.snr_var.get())
            coefficient = float(self.va_var.get())
            reference_pos = float(self.pos_var.get())
            files = {target: self.path_vars[target].get().strip() for target in TARGETS}
            if any(not path for path in files.values()):
                raise ValueError("Choose a CSV file for each of the three targets.")
            result = calibrate(files, reference_pos, coefficient, snr_threshold, self.interpolation_axis)
        except (OSError, ValueError) as exc:
            messagebox.showerror("计算失败", str(exc), parent=self.root)
            return
        self.result = result
        self.means_table.delete(*self.means_table.get_children())
        self.stats_table.delete(*self.stats_table.get_children())
        for table in self.interpolation_tables.values():
            table.delete(*table.get_children())
        # Pivot long-form summaries into one row per POS, with one mean per target.
        by_pos = {}
        for item in result.summaries:
            by_pos.setdefault(item.pos, {})[item.target] = item
        for pos in sorted(by_pos):
            items = by_pos[pos]
            self.means_table.insert("", "end", values=(
                f"{pos:g}",
                self._format(items.get("3%").corrected_mean if items.get("3%") else None),
                self._format(items.get("88%").corrected_mean if items.get("88%") else None),
                self._format(items.get("3m").corrected_mean if items.get("3m") else None),
            ))
            self.stats_table.insert("", "end", values=(
                f"{pos:g}",
                self._format(items.get("3%").corrected_std if items.get("3%") else None),
                self._format(items.get("88%").corrected_std if items.get("88%") else None),
                self._format(items.get("3m").corrected_std if items.get("3m") else None),
                self._format(items.get("3%").corrected_range if items.get("3%") else None),
                self._format(items.get("88%").corrected_range if items.get("88%") else None),
                self._format(items.get("3m").corrected_range if items.get("3m") else None),
                self._format(items.get("3%").raw_centroid_std if items.get("3%") else None),
                self._format(items.get("88%").raw_centroid_std if items.get("88%") else None),
                self._format(items.get("3m").raw_centroid_std if items.get("3m") else None),
            ))
        for point in result.interpolation_points:
            self.interpolation_table.insert("", "end", values=(
                f"{point.pos:g}", self._format(point.va_distance_3),
                self._format(point.va_distance_88), self._format(point.va_distance_3m),
                self._format(point.va_3 if self.interpolation_axis == "va" else point.csum_3), self._format(point.va_88 if self.interpolation_axis == "va" else point.csum_88),
                self._format(point.va_3m if self.interpolation_axis == "va" else point.csum_3m), self._format(point.err_88),
                self._format(point.err_3m), self._format(point.interpolated_err_3),
                self._format(point.interpolated_err_88), self._format(point.interpolated_err_3m),
            ))
        self.intercept_var.set(
            f"截距 b = {result.intercept_mm:.6f} mm  |  "
            f"参考 POS = {result.reference_pos:g}  |  K = {result.va_coefficient:g}（无量纲）"
        )
        self.status_var.set(
            f"当前插值方式：{self.interpolation_axis}；"
            f"已处理 {len(result.measurements)} 条记录；SNR有效帧："
            f"{sum(row.va is not None for row in result.measurements)}. "
            "原始 ctd 标准差以 mm 显示，用于对比。"
        )

    def _save(self) -> None:
        if self.result is None:
            messagebox.showinfo("暂无结果", "请先计算再导出。", parent=self.root)
            return
        path = filedialog.asksaveasfilename(
            title="保存测距统计", defaultextension=".csv",
            initialfile=f"distance_calibration_{self.interpolation_axis}_summary.csv",
            filetypes=(("CSV files", "*.csv"),),
        )
        if path:
            try:
                save_summary_csv(path, self.result)
            except OSError as exc:
                messagebox.showerror("保存失败", str(exc), parent=self.root)
                return
            self.status_var.set(f"统计结果已保存：{path}")


def main() -> None:
    root = tk.Tk()
    root.title("dToF 测距校准")
    root.geometry("1280x850")
    root.minsize(1000, 700)
    DistanceCalibrationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
