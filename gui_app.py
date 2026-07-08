"""
Cow Segmentation 图形界面。
- 自定义输入/输出文件夹
- 结果可视化窗口
- 零额外依赖（tkinter 内置）
"""

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageTk


def imread_cn(path: str) -> np.ndarray:
    """cv2.imread 不支持中文路径，用 imdecode 绕过。"""
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


class CowSegmentationApp:
    """主 GUI 应用"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Cow Segmentation — Detector-Guided SAM3")
        self.root.geometry("1280x800")
        self.root.minsize(1024, 600)

        self.pipeline = None
        self.image_files: list[str] = []
        self.current_index: int = -1
        self.current_result = None
        self._stop_flag = False
        self._busy = False

        self._build_ui()

        # 延迟初始化 pipeline，不阻塞 GUI 启动
        self.root.after(100, self._init_pipeline)

    # ── UI 构建 ──────────────────────────────────────────────

    def _build_ui(self):
        # 顶部：文件夹选择
        top_frame = ttk.Frame(self.root, padding=8)
        top_frame.pack(fill=tk.X)

        ttk.Label(top_frame, text="输入目录:").grid(row=0, column=0, sticky=tk.W, padx=(0, 4))
        self.input_var = tk.StringVar(value=os.getcwd())
        ttk.Entry(top_frame, textvariable=self.input_var, width=50).grid(row=0, column=1, padx=4)
        ttk.Button(top_frame, text="浏览...", command=self._browse_input).grid(row=0, column=2, padx=2)

        ttk.Label(top_frame, text="输出目录:").grid(row=0, column=3, sticky=tk.W, padx=(16, 4))
        self.output_var = tk.StringVar(value=os.path.join(os.getcwd(), "output"))
        ttk.Entry(top_frame, textvariable=self.output_var, width=50).grid(row=0, column=4, padx=4)
        ttk.Button(top_frame, text="浏览...", command=self._browse_output).grid(row=0, column=5, padx=2)

        # 控制按钮行
        ctrl_frame = ttk.Frame(self.root, padding=4)
        ctrl_frame.pack(fill=tk.X)
        ttk.Button(ctrl_frame, text="刷新文件列表", command=self._refresh_file_list).pack(side=tk.LEFT, padx=4)
        self.run_btn = ttk.Button(ctrl_frame, text="▶ 运行当前图像", command=self._run_current)
        self.run_btn.pack(side=tk.LEFT, padx=4)
        self.batch_btn = ttk.Button(ctrl_frame, text="▶ 批量运行全部", command=self._run_batch)
        self.batch_btn.pack(side=tk.LEFT, padx=4)
        self.stop_btn = ttk.Button(ctrl_frame, text="■ 停止", command=self._stop, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=4)
        self.progress = ttk.Progressbar(ctrl_frame, length=200, mode="indeterminate")
        self.progress.pack(side=tk.LEFT, padx=12)
        self.status_var = tk.StringVar(value="正在初始化 Pipeline...")
        ttk.Label(ctrl_frame, textvariable=self.status_var, foreground="blue").pack(side=tk.LEFT, padx=4)

        # 中部：文件列表 + 图像显示
        main_frame = ttk.Frame(self.root, padding=4)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 左侧：文件列表
        left_frame = ttk.LabelFrame(main_frame, text="图像列表", padding=4)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 4))
        self.file_listbox = tk.Listbox(left_frame, width=36, exportselection=False)
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.file_listbox.bind("<<ListboxSelect>>", self._on_file_select)
        list_scroll = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=self.file_listbox.yview)
        list_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.file_listbox.config(yscrollcommand=list_scroll.set)

        # 右侧：图像显示（上下两栏）
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        orig_frame = ttk.LabelFrame(right_frame, text="原始图像", padding=2)
        orig_frame.pack(fill=tk.BOTH, expand=True)
        self.orig_canvas = tk.Canvas(orig_frame, bg="#333")
        self.orig_canvas.pack(fill=tk.BOTH, expand=True)

        res_frame = ttk.LabelFrame(right_frame, text="分割结果（mask + 检测框 + id 标签）", padding=2)
        res_frame.pack(fill=tk.BOTH, expand=True)
        self.res_canvas = tk.Canvas(res_frame, bg="#333")
        self.res_canvas.pack(fill=tk.BOTH, expand=True)

        self.root.bind("<Configure>", self._on_resize)

    # ── Pipeline 初始化（后台线程）────────────────────────

    def _init_pipeline(self):
        def worker():
            try:
                from config import PipelineConfig
                from cow_segmentation_pipeline import CowSegmentationPipeline

                config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
                if os.path.exists(config_path):
                    cfg = PipelineConfig.from_yaml(config_path)
                else:
                    cfg = PipelineConfig()
                cfg.vis_config.enabled = False

                self.pipeline = CowSegmentationPipeline(cfg)
                self.root.after(0, lambda: self.status_var.set("Pipeline 已就绪 ✓"))
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.root.after(0, lambda: self.status_var.set(f"Pipeline 失败: {e}"))
                self.root.after(0, lambda: messagebox.showwarning(
                    "Pipeline 初始化失败",
                    f"{e}\n\n请检查 config.yaml、模型路径、以及 sam3-main 是否在 PYTHONPATH 中。"
                ))

        threading.Thread(target=worker, daemon=True).start()

    # ── 文件夹操作 ──────────────────────────────────────────

    def _browse_input(self):
        path = filedialog.askdirectory(title="选择输入图像目录")
        if path:
            self.input_var.set(path)
            self._refresh_file_list()

    def _browse_output(self):
        path = filedialog.askdirectory(title="选择输出目录")
        if path:
            self.output_var.set(path)

    def _refresh_file_list(self):
        self.file_listbox.delete(0, tk.END)
        self.image_files.clear()
        input_dir = self.input_var.get()
        if not os.path.isdir(input_dir):
            self.status_var.set("输入目录不存在")
            return
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        for f in sorted(os.listdir(input_dir)):
            if os.path.splitext(f)[1].lower() in exts:
                self.image_files.append(f)
                self.file_listbox.insert(tk.END, f)
        self.status_var.set(f"找到 {len(self.image_files)} 张图像")

    # ── 文件选择 ────────────────────────────────────────────

    def _on_file_select(self, event):
        sel = self.file_listbox.curselection()
        if not sel:
            return
        self.current_index = sel[0]
        self.current_result = None
        self.res_canvas.delete("all")
        self._show_original()

    def _show_original(self):
        if self.current_index < 0 or self.current_index >= len(self.image_files):
            return
        path = os.path.join(self.input_var.get(), self.image_files[self.current_index])
        img = imread_cn(path)
        if img is None:
            self.status_var.set(f"无法读取: {path}")
            return
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self._display_on_canvas(img_rgb, self.orig_canvas)
        self.status_var.set(f"已加载: {self.image_files[self.current_index]}")

    # ── 运行（后台线程，不阻塞 UI）─────────────────────────

    def _run_current(self):
        if self._busy:
            return
        if self.current_index < 0:
            messagebox.showinfo("提示", "请先在左侧列表中选择一张图像")
            return
        if self.pipeline is None:
            messagebox.showinfo("提示", "Pipeline 还在初始化中，请稍候...")
            return
        self._set_busy(True)
        threading.Thread(target=self._process_one, args=(self.current_index, False), daemon=True).start()

    def _run_batch(self):
        if self._busy:
            return
        if not self.image_files:
            messagebox.showinfo("提示", "请先选择输入目录并刷新文件列表")
            return
        if self.pipeline is None:
            messagebox.showinfo("提示", "Pipeline 还在初始化中，请稍候...")
            return
        self._set_busy(True, batch=True)
        self.progress.config(mode="determinate")
        self.progress["maximum"] = len(self.image_files)
        self.progress["value"] = 0

        threading.Thread(target=self._batch_worker, daemon=True).start()

    def _batch_worker(self):
        for i in range(len(self.image_files)):
            if self._stop_flag:
                break
            self.root.after(0, lambda idx=i: self.progress.config(value=idx + 1))
            self._process_one(i, save_only=True)
            self.root.after(0, lambda idx=i: self.status_var.set(
                f"批量处理: {idx + 1}/{len(self.image_files)}"
            ))
        self.root.after(0, self._batch_done)

    def _batch_done(self):
        self._stop_flag = False
        self._set_busy(False, batch=True)
        self.progress.config(mode="indeterminate")
        self.progress.stop()
        self.status_var.set("批量处理完成")

    def _stop(self):
        self._stop_flag = True
        self.status_var.set("正在停止...")

    def _set_busy(self, busy: bool, batch: bool = False):
        self._busy = busy
        state = tk.DISABLED if busy else tk.NORMAL
        self.run_btn.config(state=state)
        self.batch_btn.config(state=state)
        self.stop_btn.config(state=tk.NORMAL if busy and batch else tk.DISABLED)
        if busy:
            self.progress.start(10)
        else:
            self.progress.stop()

    def _process_one(self, index: int, save_only: bool = False):
        fname = self.image_files[index]
        path = os.path.join(self.input_var.get(), fname)
        image = imread_cn(path)
        if image is None:
            self.root.after(0, lambda: self.status_var.set(f"无法读取: {fname}"))
            return

        try:
            self.root.after(0, lambda: self.status_var.set(f"推理中: {fname} ..."))
            result = self.pipeline.process(image)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: self.status_var.set(f"处理失败 [{fname}]: {e}"))
            self.root.after(0, lambda: messagebox.showerror("推理失败", str(e)))
            if not save_only:
                self.root.after(0, lambda: self._set_busy(False))
            return

        vis = self._render_result(image, result)

        # 保存
        os.makedirs(self.output_var.get(), exist_ok=True)
        out_name = f"{Path(fname).stem}_seg.jpg"
        out_path = os.path.join(self.output_var.get(), out_name)
        cv2.imwrite(out_path, vis)

        if not save_only:
            vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            self.root.after(0, lambda: self._display_on_canvas(vis_rgb, self.res_canvas))
            self.current_result = result
            self.root.after(0, lambda: self._set_busy(False))

        self.root.after(0, lambda: self.status_var.set(
            f"已保存: {out_name}  ({len(result.boxes)} 头牛, {result.elapsed_ms:.0f}ms)"
        ))

    # ── 可视化渲染 ──────────────────────────────────────────

    def _render_result(self, image: np.ndarray, result) -> np.ndarray:
        import colorsys

        n = max(len(result.instances), 1)
        colors = []
        for i in range(n):
            hue = i / n
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 1.0)
            colors.append((int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255)))

        overlay = image.copy()
        for i, inst in enumerate(result.instances):
            mask = inst["mask"]
            if mask.sum() > 0:
                overlay[mask] = colors[i % len(colors)]
        out = cv2.addWeighted(image, 0.5, overlay, 0.5, 0)

        for i, inst in enumerate(result.instances):
            bbox = inst.get("bbox")
            if bbox is not None:
                x1, y1, x2, y2 = bbox.astype(int)
                color = colors[i % len(colors)]
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

                label = f"id: {i}"
                font_scale = 1.0
                thickness = 2
                (tw, th), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness
                )
                bg_x1, bg_y1 = x1, y1 - th - baseline - 8
                bg_x2, bg_y2 = x1 + tw + 8, y1
                bg_y1 = max(bg_y1, 0)
                cv2.rectangle(out, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), -1)
                cv2.rectangle(out, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), 1)
                cv2.putText(
                    out, label, (x1 + 4, y1 - baseline - 4),
                    cv2.FONT_HERSHEY_DUPLEX, font_scale, color, thickness, cv2.LINE_AA,
                )

        return out

    # ── 图像显示 ────────────────────────────────────────────

    def _display_on_canvas(self, img_rgb: np.ndarray, canvas: tk.Canvas):
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        if w < 10 or h < 10:
            w, h = 600, 300

        pil_img = Image.fromarray(img_rgb)
        iw, ih = pil_img.size
        scale = min(w / iw, h / ih, 1.0)
        new_w, new_h = int(iw * scale), int(ih * scale)
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)

        tk_img = ImageTk.PhotoImage(pil_img)
        canvas.delete("all")
        canvas.create_image(w // 2, h // 2, image=tk_img, anchor=tk.CENTER)
        canvas._tk_img = tk_img  # 挂在各自 canvas 上，互不覆盖

    def _on_resize(self, event):
        if self.current_index >= 0:
            self._show_original()
            if self.current_result is not None:
                path = os.path.join(self.input_var.get(), self.image_files[self.current_index])
                img = imread_cn(path)
                if img is not None:
                    vis = self._render_result(img, self.current_result)
                    self._display_on_canvas(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
                                            self.res_canvas)

    def run(self):
        self._refresh_file_list()
        self.root.mainloop()


def main():
    app = CowSegmentationApp()
    app.run()


if __name__ == "__main__":
    main()
