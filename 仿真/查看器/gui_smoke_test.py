"""实际启动Tk界面检查选择、无效帧、模型叠加及质心切换，保存应用窗口截图后退出。"""

import argparse
import ctypes
from ctypes import wintypes
from pathlib import Path
import time
import tkinter as tk

import numpy as np
from PIL import Image

from dataset_reader import DEFAULT_DATASET
from histogram_viewer import HistogramViewer, format_value


def capture_window(root, path):
    """只捕获目标Tk窗口，即使它被其他应用遮挡，也不采集桌面内容。"""
    class BitmapInfoHeader(ctypes.Structure):
        _fields_ = [("size", wintypes.DWORD), ("width", wintypes.LONG),
                    ("height", wintypes.LONG), ("planes", wintypes.WORD),
                    ("bits", wintypes.WORD), ("compression", wintypes.DWORD),
                    ("image_size", wintypes.DWORD), ("xppm", wintypes.LONG),
                    ("yppm", wintypes.LONG), ("colors", wintypes.DWORD), ("important", wintypes.DWORD)]

    user32, gdi32 = ctypes.WinDLL("user32", use_last_error=True), ctypes.WinDLL("gdi32", use_last_error=True)
    for dll, name, restype, argtypes in (
        (user32, "GetAncestor", wintypes.HWND, [wintypes.HWND, wintypes.UINT]),
        (user32, "GetClientRect", wintypes.BOOL, [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]),
        (user32, "GetDC", wintypes.HDC, [wintypes.HWND]),
        (user32, "ReleaseDC", ctypes.c_int, [wintypes.HWND, wintypes.HDC]),
        (user32, "PrintWindow", wintypes.BOOL, [wintypes.HWND, wintypes.HDC, wintypes.UINT]),
        (gdi32, "CreateCompatibleDC", wintypes.HDC, [wintypes.HDC]),
        (gdi32, "CreateCompatibleBitmap", wintypes.HBITMAP, [wintypes.HDC, ctypes.c_int, ctypes.c_int]),
        (gdi32, "SelectObject", wintypes.HANDLE, [wintypes.HDC, wintypes.HANDLE]),
        (gdi32, "DeleteObject", wintypes.BOOL, [wintypes.HANDLE]),
        (gdi32, "DeleteDC", wintypes.BOOL, [wintypes.HDC]),
        (gdi32, "GetDIBits", ctypes.c_int, [wintypes.HDC, wintypes.HBITMAP, wintypes.UINT,
                                           wintypes.UINT, ctypes.c_void_p, ctypes.POINTER(BitmapInfoHeader), wintypes.UINT]),
    ):
        function = getattr(dll, name)
        function.restype, function.argtypes = restype, argtypes
    hwnd = user32.GetAncestor(root.winfo_id(), 2)
    rect = wintypes.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(rect))
    width, height = rect.right, rect.bottom
    dc = user32.GetDC(hwnd)
    memory = gdi32.CreateCompatibleDC(dc)
    bitmap = gdi32.CreateCompatibleBitmap(dc, width, height)
    previous = gdi32.SelectObject(memory, bitmap)
    try:
        if not user32.PrintWindow(hwnd, memory, 3):
            raise OSError("无法捕获查看器窗口")
        gdi32.SelectObject(memory, previous)
        info = BitmapInfoHeader()
        info.size, info.width, info.height, info.planes, info.bits = ctypes.sizeof(info), width, -height, 1, 32
        pixels = ctypes.create_string_buffer(width * height * 4)
        if gdi32.GetDIBits(memory, bitmap, 0, height, pixels, ctypes.byref(info), 0) != height:
            raise OSError("无法读取查看器窗口截图")
        Image.frombytes("RGB", (width, height), pixels.raw, "raw", "BGRX").save(path)
    finally:
        gdi32.SelectObject(memory, previous)
        gdi32.DeleteObject(bitmap)
        gdi32.DeleteDC(memory)
        user32.ReleaseDC(hwnd, dc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--screenshot", type=Path, default=Path(__file__).with_name("gui_preview.png"))
    args = parser.parse_args()
    root = tk.Tk()
    app = HistogramViewer(root, args.dataset)
    started, errors = time.monotonic(), []
    state = 0

    def check():
        nonlocal state
        try:
            if time.monotonic() - started > 45:
                raise AssertionError("GUI加载超时")
            if app.cube is None:
                root.after(50, check)
                return
            if state == 0:
                app.signal_box.current(0)
                app.background_box.current(0)
                app.frame_number.set("1")
                app.refresh()
                assert not app.sample["peak_valid"]
                assert app.metric_values["ctd"].get() == "无有效值"
                assert app.mean_values["ctd"].get() == "无有效值"
                assert app.mean_counts["ctd"].get() == "均值有效帧：0/100"
                assert np.isfinite(app.sample["sa_avg"])
                app.signal_box.current(6)
                app.background_box.current(4)
                app.frame_number.set("100")
                app.show_expected.set(True)
                app.refresh()
                assert app.sample["peak_valid"]
                for name in ("ctd", "va", "csum", "snr", "sa_avg"):
                    assert app.metric_values[name].get() == format_value(app.sample[name])
                    group = app.cube.features[name][4, 6]
                    finite = group[np.isfinite(group)]
                    assert app.mean_values[name].get() == format_value(float(finite.mean()))
                    assert app.mean_counts[name].get() == f"均值有效帧：{finite.size}/100"
                means_before = {key: var.get() for key, var in app.mean_values.items()}
                line = next(line for line in app.axes[0].lines if line.get_label() == "Simulated histogram")
                np.testing.assert_array_equal(line.get_ydata(), app.cube.histograms[4, 6, 99])
                assert any(line.get_label() == "Forward expectation" for line in app.axes[0].lines)
                app.change_frame(-1)
                app.refresh()
                assert app.sample["frame_number"] == 99
                assert {key: var.get() for key, var in app.mean_values.items()} == means_before
                state = 1
                root.lift()
                root.after(500, check)
                return
            if state == 1:
                root.update_idletasks()
                capture_window(root, args.screenshot)
                app.center_box.current(1)
                app.load_center()
                state = 2
                root.after(50, check)
                return
            assert app.cube.metadata["nominal_centroid_bin"] == 512
            app.refresh()
            assert app.sample is not None
            print("GUI smoke test passed: current-frame and group-mean features, finite sample counts, invalid frames, frame-independent means, plots and centroid switching.")
            print(f"Screenshot: {args.screenshot.resolve()}")
            app.close()
        except Exception as exc:
            errors.append(exc)
            app.close()

    root.after(50, check)
    root.mainloop()
    if errors:
        raise errors[0]


if __name__ == "__main__":
    main()
