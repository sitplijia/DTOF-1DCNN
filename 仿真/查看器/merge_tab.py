"""Histogram mixture selection UI; numerical work lives in merge_algorithms."""
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from merge_algorithms import merge_samples


class MergeTab:
    def __init__(self, parent):
        self.cube = None
        self.result = None
        self.count_vars = []
        self.on_background_change = None
        self.on_start_frame_change = None
        controls = ttk.Frame(parent, padding=(6, 2))
        controls.pack(fill='x')
        row = ttk.Frame(controls); row.pack(fill='x', pady=2)
        ttk.Label(row, text="背景λ（每bin/周期）").pack(side='left')
        self.background = ttk.Combobox(row, state='readonly', width=14)
        self.background.pack(side='left', padx=6)
        self.background.bind('<<ComboboxSelected>>', self.change_background)
        ttk.Label(row, text="起始帧（1起始）").pack(side='left')
        self.start = tk.StringVar(value='1')
        self.start_box = ttk.Spinbox(row, from_=1, to=100, textvariable=self.start, width=6)
        self.start_box.pack(side='left', padx=6)
        self.start.trace_add('write', self.change_start_frame)
        ttk.Button(row, text="合并并计算", command=self.calculate).pack(side='left', padx=6)
        ttk.Button(row, text="清空选择", command=self.clear).pack(side='left')
        ttk.Button(row, text="取样明细", command=self.show_sources).pack(side='left', padx=4)
        self.toggle_button = ttk.Button(row, text="收起选择表", command=self.toggle_grid)
        self.toggle_button.pack(side='left', padx=6)
        ttk.Label(controls, text="质心沿用上方选择；每列选1～8条，不选为0；连续取帧，末尾回绕。", font=('Microsoft YaHei UI', 8)).pack(anchor='w')
        self.grid_area = grid_area = ttk.Frame(parent)
        grid_area.pack(fill='x', padx=6)
        self.grid_visible = True
        self.grid_canvas = tk.Canvas(grid_area, height=175, highlightthickness=0)
        self.grid_canvas.pack(fill='x')
        scroll = ttk.Scrollbar(grid_area, orient='horizontal', command=self.grid_canvas.xview)
        scroll.pack(fill='x'); self.grid_canvas.configure(xscrollcommand=scroll.set)
        self.grid = ttk.Frame(self.grid_canvas)
        self.grid_canvas.create_window((0,0), window=self.grid, anchor='nw')
        self.grid.bind('<Configure>',lambda e:self.grid_canvas.configure(scrollregion=self.grid_canvas.bbox('all')))
        self.info = tk.StringVar(value="正在加载数据集…")
        self.info_label = tk.Label(parent, textvariable=self.info, wraplength=1300,
                                   padx=6, pady=2, height=2, anchor='w', justify='left',
                                   background='#f4f6f8', font=('Microsoft YaHei UI', 9))
        self.info_label.pack(fill='x')
        self.metrics = tk.StringVar(value='')
        ttk.Label(parent, textvariable=self.metrics, padding=(6, 2)).pack(fill='x')
        self.figure = Figure(figsize=(11,6),dpi=100,layout='constrained')
        self.ax, self.peak_ax = self.figure.subplots(1, 2)
        self.plot_layout = 'columns'
        self.canvas = FigureCanvasTkAgg(self.figure,master=parent)
        self.toolbar = NavigationToolbar2Tk(self.canvas,parent,pack_toolbar=False)
        self.toolbar.pack(side='bottom',fill='x')
        self.canvas.get_tk_widget().pack(fill='both',expand=True)
        self.canvas.get_tk_widget().bind('<Configure>', self.resize_plot, add='+')
        self.toolbar.update()

    def resize_plot(self, event):
        if event.height < 80 or event.width < 100:
            return
        layout = 'columns' if event.height < 360 and event.width >= 750 else 'rows'
        if layout == self.plot_layout:
            return
        self.plot_layout = layout
        spec = self.figure.add_gridspec(1, 2) if layout == 'columns' else self.figure.add_gridspec(2, 1)
        self.ax.set_subplotspec(spec[0])
        self.peak_ax.set_subplotspec(spec[1])
        self.canvas.draw_idle()

    def show_sources(self):
        if self.result is None:
            self.info.set('请先合并计算，再查看取样明细。')
            return
        window = tk.Toplevel(self.grid_area)
        window.title('合并取样明细')
        window.geometry('650x400')
        text = ScrolledText(window,wrap='word',font=('Microsoft YaHei UI',10))
        text.pack(fill='both',expand=True)
        text.insert('end',f"质心：{self.cube.metadata['nominal_centroid_bin']:g} bin\n"
                    f"背景λ：{self.result['background_lambda']:g}\n"
                    f"合并数量：{self.result['child_count']}\n\n")
        for source in self.result['sources']:
            text.insert('end',f"信号λ={source['signal_lambda']:g}：帧 {source['frames']}\n")
        text.configure(state='disabled')

    def toggle_grid(self):
        self.grid_visible = not self.grid_visible
        if self.grid_visible:
            self.grid_area.pack(fill='x', padx=6, before=self.info_label)
        else:
            self.grid_area.pack_forget()
        self.toggle_button.configure(text='收起选择表' if self.grid_visible else '展开选择表')

    def change_background(self, event=None):
        if self.on_background_change is not None:
            self.on_background_change(self.background.current())
        self.refresh_background()

    def refresh_background(self):
        self.invalidate()
        if self.cube is not None and any(var.get() for var in self.count_vars):
            self.calculate()

    def change_start_frame(self, *args):
        if self.on_start_frame_change is not None:
            self.on_start_frame_change(self.start.get())
        self.invalidate()
        if self.cube is not None and any(var.get() for var in self.count_vars):
            self.calculate()

    def invalidate(self, *args):
        self.result = None
        if hasattr(self,'info'):
            self.info.set('参数或选择已更改，请点击“合并并计算”。')
            self.metrics.set('')
            self.ax.clear(); self.peak_ax.clear(); self.canvas.draw_idle()
            self.toolbar.update()

    def set_cube(self, cube):
        self.cube = cube
        for widget in self.grid.winfo_children(): widget.destroy()
        self.count_vars = []
        self.grid_canvas.xview_moveto(0)
        self.invalidate()
        if cube is None:
            self.background.configure(values=()); self.background.set('')
            self.info.set('正在加载质心数据…')
            return
        self.background.configure(values=[f'{v:g}' for v in cube.backgrounds]); self.background.current(0)
        self.start_box.configure(to=cube.frames); self.start.set('1')
        small_font = ('Microsoft YaHei UI', 8)
        ttk.Label(self.grid,text='数量/λ',width=9,font=small_font).grid(row=0,column=0)
        for n in range(1,9): ttk.Label(self.grid,text=str(n),anchor='center',font=small_font).grid(row=n,column=0)
        for x, signal in enumerate(cube.signals):
            var = tk.IntVar(value=0); self.count_vars.append(var)
            ttk.Label(self.grid,text=f'{signal:g}',width=7,anchor='center',font=small_font).grid(row=0,column=x+1)
            for n in range(1,9):
                tk.Checkbutton(self.grid,text=str(n),variable=var,onvalue=n,offvalue=0,
                               indicatoron=False,width=5,font=small_font,padx=0,pady=0,borderwidth=1,
                               state='normal' if n <= cube.frames else 'disabled',
                               selectcolor='#9ac8ef',command=self.invalidate).grid(row=n,column=x+1,padx=1,pady=0)
        self.grid.update_idletasks()
        self.grid_canvas.configure(height=self.grid.winfo_reqheight())

    def clear(self):
        for var in self.count_vars: var.set(0)
        self.invalidate()

    def calculate(self):
        if self.cube is None: return
        try:
            start_frame = int(self.start.get())
        except ValueError:
            self.invalidate(); self.info.set('起始帧必须是整数。'); return
        try:
            result = merge_samples(self.cube,[var.get() for var in self.count_vars],
                                   self.background.current(),start_frame)
        except ValueError as exc:
            self.invalidate(); self.info.set(str(exc)); return
        self.result = result
        self.info.set(f"质心={self.cube.metadata['nominal_centroid_bin']:g} bin；背景λ={result['background_lambda']:g}；"
                      f"合并{result['child_count']}条；{result['state']}。具体帧号见“取样明细”。")
        self.metrics.set('    '.join(f"{name} = {result[name]:.6f}" if np.isfinite(result[name]) else f"{name} = 无有效值"
                                    for name in ('ctd','va','csum','snr','sa_avg')))
        peak = int(np.argmax(result['histogram']))
        window = result['feature_window']
        for axis in (self.ax, self.peak_ax):
            axis.clear()
            axis.plot(self.cube.bins,result['histogram'],label='Merged histogram',linewidth=1)
            axis.plot(self.cube.bins,result['expected'],label='Sum of forward expectations',alpha=.7)
            axis.set(xlabel='Bin (1-based)',ylabel='Detected counts')
            if window is not None:
                lo, hi = window
                axis.axvspan(self.cube.bins[lo]-.5, self.cube.bins[hi-1]+.5,
                             color='tab:orange',alpha=.12,label='Feature window')
                if np.isfinite(result['ctd']):
                    axis.axvline(result['ctd'],color='tab:red',linestyle='--',label='Centroid')
            axis.grid(alpha=.25);axis.legend(fontsize=7)
            axis.tick_params(labelsize=8)
            axis.xaxis.label.set_fontsize(9); axis.yaxis.label.set_fontsize(9)
        self.ax.set_title(f"Full histogram: sum of {result['child_count']} samples",fontsize=9)
        self.ax.set_xlim(self.cube.bins[0]-.5,self.cube.bins[-1]+.5)
        if np.any(result['histogram']):
            left, right = max(0,peak-40), min(len(self.cube.bins),peak+41)
            self.peak_ax.set_xlim(self.cube.bins[left]-.5,self.cube.bins[right-1]+.5)
            maximum = max(float(result['histogram'][left:right].max()),float(result['expected'][left:right].max()))
            self.peak_ax.set_ylim(0,max(1,maximum*1.12))
            suffix = 'shaded: feature window' if window is not None else 'no valid feature window'
            self.peak_ax.set_title(f"Peak bin {self.cube.bins[peak]:g}\n{suffix}",fontsize=9)
        else:
            self.peak_ax.set_title('No detected peak (all counts are zero)')
        self.toolbar.update()
        self.canvas.draw_idle()
