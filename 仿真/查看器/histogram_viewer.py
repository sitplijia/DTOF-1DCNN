"""桌面仿真直方图查看器：质心、信号lambda、背景lambda及帧号选择。"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from dataset_reader import DEFAULT_DATASET, FEATURES, discover_centroids, load_cube


def format_value(value: float, unit: str = "") -> str:
    return f"{value:.6f}{unit}" if np.isfinite(value) else "无有效值"


class HistogramViewer:
    def __init__(self, root: tk.Tk, dataset: Path):
        self.root, self.dataset = root, dataset
        self.cube = None
        self.sample = None
        self.future = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dataset-reader")
        self.closed = False
        self.refresh_job = None
        root.title("仿真直方图查看器")
        root.geometry("1480x930")
        root.minsize(1200, 780)
        root.configure(background="#f4f6f8")
        root.protocol("WM_DELETE_WINDOW", self.close)
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("TFrame", background="#f4f6f8")
        style.configure("TLabel", background="#f4f6f8", font=("Microsoft YaHei UI", 10))
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 19, "bold"))
        style.configure("Value.TLabel", foreground="#153e60", font=("Consolas", 15, "bold"))
        style.configure("TableValue.TLabel", foreground="#153e60", font=("Consolas", 12, "bold"))
        style.configure("Count.TLabel", foreground="#657384", font=("Microsoft YaHei UI", 9))
        style.configure("TButton", padding=(10, 5), font=("Microsoft YaHei UI", 10))
        outer = ttk.Frame(root, padding=16)
        outer.pack(fill="both", expand=True)
        header = ttk.Frame(outer)
        header.pack(fill="x")
        ttk.Label(header, text="仿真直方图查看器", style="Title.TLabel").pack(side="left")
        ttk.Button(header, text="打开数据集目录…", command=self.choose_dataset).pack(side="right")
        ttk.Label(outer, text="实测反演主峰模板 → 信号与背景 λ → 随机探测直方图；展示已保存的数据与特征").pack(anchor="w", pady=(5, 14))

        selectors = ttk.Frame(outer)
        selectors.pack(fill="x", pady=(0, 12))
        self.center_box = self.combo(selectors, "质心位置", 14, self.load_center)
        self.signal_box = self.combo(selectors, "信号 λ（全谱/周期）", 15, self.schedule_refresh)
        self.background_box = self.combo(selectors, "背景 λ（每bin/周期）", 16, self.schedule_refresh)
        frames = ttk.Frame(selectors)
        frames.pack(side="left", padx=(0, 14))
        ttk.Label(frames, text="帧号（从1开始）").pack(anchor="w")
        frame_row = ttk.Frame(frames)
        frame_row.pack(pady=(4, 0))
        self.frame_number = tk.StringVar(value="1")
        self.frame_box = ttk.Spinbox(frame_row, from_=1, to=100, width=6,
                                     textvariable=self.frame_number, command=self.schedule_refresh)
        self.frame_box.pack(side="left")
        ttk.Button(frame_row, text="上一帧", command=lambda: self.change_frame(-1)).pack(side="left", padx=(6, 3))
        ttk.Button(frame_row, text="下一帧", command=lambda: self.change_frame(1)).pack(side="left")
        self.frame_number.trace_add("write", self.schedule_refresh)
        self.frame_box.bind("<Return>", self.refresh)
        self.show_expected = tk.BooleanVar(value=False)
        ttk.Checkbutton(selectors, text="叠加模型期望", variable=self.show_expected,
                        command=self.schedule_refresh).pack(side="left", pady=(20, 0))

        body = ttk.Frame(outer)
        body.pack(fill="both", expand=True)
        plots = ttk.Frame(body)
        plots.pack(side="left", fill="both", expand=True)
        self.figure = Figure(figsize=(10, 7), dpi=100, layout="constrained")
        self.axes = self.figure.subplots(2, 1)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plots)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas, plots).update()
        self.sidebar = ttk.Frame(body, padding=(18, 6, 0, 0), width=370)
        self.sidebar.pack(side="right", fill="y")
        self.sidebar.pack_propagate(False)
        self.valid_text = tk.StringVar(value="正在加载数据…")
        ttk.Label(self.sidebar, textvariable=self.valid_text, wraplength=350).pack(anchor="w", pady=(0, 14))
        self.metric_values, self.mean_values, self.mean_counts = {}, {}, {}
        table = ttk.Frame(self.sidebar)
        table.pack(fill="x", pady=(0, 12))
        table.columnconfigure(1, weight=1)
        table.columnconfigure(2, weight=1)
        ttk.Label(table, text="指标").grid(row=0, column=0, sticky="w", padx=(0, 10))
        ttk.Label(table, text="当前帧").grid(row=0, column=1, sticky="e", padx=(0, 14))
        self.mean_heading = tk.StringVar(value="100帧均值")
        ttk.Label(table, textvariable=self.mean_heading).grid(row=0, column=2, sticky="e")
        for index, (name, label) in enumerate((("ctd", "ctd (bin)"), ("va", "va (bin)"),
                                               ("csum", "csum"), ("snr", "snr"), ("sa_avg", "sa_avg"))):
            row = 1 + index * 2
            ttk.Label(table, text=label).grid(row=row, column=0, sticky="w", pady=(12, 0), padx=(0, 10))
            self.metric_values[name] = tk.StringVar(value="—")
            self.mean_values[name] = tk.StringVar(value="—")
            self.mean_counts[name] = tk.StringVar(value="")
            ttk.Label(table, textvariable=self.metric_values[name], style="TableValue.TLabel").grid(row=row, column=1, sticky="e", pady=(12, 0), padx=(0, 14))
            ttk.Label(table, textvariable=self.mean_values[name], style="TableValue.TLabel").grid(row=row, column=2, sticky="e", pady=(12, 0))
            ttk.Label(table, textvariable=self.mean_counts[name], style="Count.TLabel").grid(row=row + 1, column=1, columnspan=2, sticky="e")
        ttk.Label(self.sidebar, text="均值范围：当前质心、信号λ和背景λ下的全部帧。各项只对有效数值取平均，不先平均直方图；sa_avg单位为计数/bin。",
                  style="Count.TLabel", wraplength=350).pack(anchor="w", pady=(0, 12))
        ttk.Separator(self.sidebar).pack(fill="x", pady=(0, 10))
        self.details = tk.StringVar(value="")
        ttk.Label(self.sidebar, textvariable=self.details, wraplength=350, justify="left").pack(anchor="w")
        self.status = tk.StringVar(value="请选择数据集")
        ttk.Label(outer, textvariable=self.status, wraplength=1280).pack(fill="x", pady=(10, 0))
        root.bind("<Control-Left>", lambda event: self.change_frame(-1))
        root.bind("<Control-Right>", lambda event: self.change_frame(1))
        self.set_dataset(dataset)

    @staticmethod
    def combo(parent, label, width, command):
        frame = ttk.Frame(parent)
        frame.pack(side="left", padx=(0, 14))
        ttk.Label(frame, text=label).pack(anchor="w")
        box = ttk.Combobox(frame, state="readonly", width=width)
        box.pack(pady=(4, 0))
        box.bind("<<ComboboxSelected>>", command)
        return box

    def choose_dataset(self):
        selected = filedialog.askdirectory(title="选择包含manifest.json或质心子目录的数据集", initialdir=self.dataset)
        if selected:
            self.set_dataset(Path(selected))

    def set_dataset(self, dataset):
        try:
            entries = discover_centroids(dataset)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.status.set(f"无法打开数据集：{exc}")
            messagebox.showerror("数据集加载失败", str(exc), parent=self.root)
            return
        self.dataset, self.entries = Path(dataset).resolve(), entries
        self.center_box.configure(values=[label for label, _ in entries])
        self.center_box.current(0)
        self.load_center()

    def load_center(self, event=None):
        index = self.center_box.current()
        if index < 0:
            return
        if self.future is not None:
            self.future.cancel()
        self.cube = self.sample = None
        for collection in (self.metric_values, self.mean_values, self.mean_counts):
            for value in collection.values():
                value.set("—")
        self.valid_text.set("正在解压该质心的数据…")
        self.status.set("首次切换质心需要读取数据；信号、背景及帧切换在内存中完成。")
        self.details.set("")
        for ax in self.axes:
            ax.clear()
            ax.text(.5, .5, "Loading dataset...", transform=ax.transAxes, ha="center", va="center")
        self.canvas.draw_idle()
        self.future = self.executor.submit(load_cube, self.entries[index][1])
        self.root.after(50, self.poll_load, self.future)

    def poll_load(self, future):
        if self.closed or future is not self.future:
            return
        if not future.done():
            self.root.after(50, self.poll_load, future)
            return
        try:
            self.cube = future.result()
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.valid_text.set("数据加载失败")
            self.status.set(str(exc))
            messagebox.showerror("质心数据加载失败", str(exc), parent=self.root)
            return
        for box, values, preferred in ((self.signal_box, self.cube.signals, 1.),
                                        (self.background_box, self.cube.backgrounds, 0.0001)):
            previous = box.get()
            labels = [f"{value:g}" for value in values]
            box.configure(values=labels)
            box.current(labels.index(previous) if previous in labels else int(np.argmin(abs(values - preferred))))
        self.frame_box.configure(to=self.cube.frames)
        self.mean_heading.set(f"{self.cube.frames}帧均值")
        self.frame_number.set("1")
        self.refresh()

    def schedule_refresh(self, *args):
        if self.refresh_job is not None:
            self.root.after_cancel(self.refresh_job)
        self.refresh_job = self.root.after(120, self.refresh)

    def change_frame(self, delta):
        if self.cube is None:
            return
        try:
            current = int(self.frame_number.get())
        except ValueError:
            current = 1
        self.frame_number.set(str(min(self.cube.frames, max(1, current + delta))))

    def refresh(self, event=None):
        self.refresh_job = None
        if self.cube is None:
            return
        try:
            sample = self.cube.select(self.background_box.current(), self.signal_box.current(), int(self.frame_number.get()))
        except ValueError as exc:
            self.status.set(f"选择无效：{exc}")
            return
        self.sample = sample
        statistics = self.cube.group_statistics(self.background_box.current(), self.signal_box.current())
        for name in FEATURES:
            self.metric_values[name].set(format_value(sample[name]))
            self.mean_values[name].set(format_value(statistics[name]["mean"]))
            self.mean_counts[name].set(f"均值有效帧：{statistics[name]['finite_count']}/{statistics[name]['total_count']}")
        state = {"ok": "有效回波", "low_snr": "低信噪比：ctd、va 无有效值", "analysis_error": "特征无法计算"}.get(sample["status"], sample["status"])
        if sample["window_clipped"]:
            state += "（分析窗口已截窗）"
        self.valid_text.set(state)
        hist, bins, meta = sample["histogram"], self.cube.bins, self.cube.metadata
        peak_index = int(np.argmax(hist))
        lo, hi = max(0, peak_index - 15), min(hist.size, peak_index + 15)
        center = float(meta["actual_signal_centroid_bin"])
        details = (
            f"全谱探测计数：{int(hist.sum())}\n观测最大bin：{bins[peak_index]}\n"
            f"最大bin计数：{int(hist[peak_index])}\n输入信号质心：{center:.5f} bin\n"
            f"理想信号va：{meta['ideal_signal_va_bin']:.5f} bin\n"
            f"累计周期：{meta['num_cycles']}\n死时间：{meta['dead_bins']} bin\n"
            f"SNR ? {meta['snr_threshold']:g}"
        )
        if sample["error"]:
            details += "\n\n原因：" + sample["error"]
        self.details.set(details)
        for ax in self.axes:
            ax.clear()
            ax.step(bins, hist, where="mid", color="#2474a5", linewidth=1., label="Simulated histogram")
            if self.show_expected.get():
                ax.plot(bins, sample["expected"], color="#dc8538", linewidth=1.2, label="Forward expectation")
            ax.axvline(center, color="#6f7781", linestyle=":", label="Input signal centroid")
            if np.isfinite(sample["ctd"]):
                ax.axvline(sample["ctd"], color="#b94765", linestyle="--", label="ctd")
            ax.axvspan(bins[lo] - .5, bins[hi - 1] + .5, color="#efb94f", alpha=.12, label="Analysis window")
            if np.isfinite(sample["sa_avg"]):
                ax.axhline(sample["sa_avg"], color="#52986c", linestyle="--", linewidth=.9, label="Estimated background")
            ax.set_xlabel("Bin index (1-based)")
            ax.set_ylabel("Detected counts / histogram")
            ax.grid(alpha=.2)
        self.axes[0].set_title(f"Full histogram | centroid {meta['nominal_centroid_bin']:g} | signal lambda={sample['signal_lambda']:g} | background lambda={sample['background_lambda']:g} | frame {sample['frame_number']}/{self.cube.frames}", fontsize=10)
        self.axes[0].set_xlim(bins[0] - .5, bins[-1] + .5)
        self.axes[1].set_title("Signal region | stored features; invalid features remain undefined", fontsize=10)
        self.axes[1].set_xlim(max(bins[0] - .5, center - 35), min(bins[-1] + .5, center + 35))
        self.axes[0].legend(loc="upper right", fontsize=8, ncol=2)
        self.canvas.draw_idle()
        self.status.set(f"{self.cube.folder / 'dataset.npz'}　|　帧号 {sample['frame_number']}（数组索引 {sample['frame_number'] - 1}）")

    def close(self):
        self.closed = True
        if self.future is not None:
            self.future.cancel()
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()
    root = tk.Tk()
    HistogramViewer(root, args.dataset)
    root.mainloop()


if __name__ == "__main__":
    main()
