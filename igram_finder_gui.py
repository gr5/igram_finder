#!/usr/bin/env python3
"""
igram_finder_gui.py

Tkinter GUI for igram_finder.py — auto-detect mirror aperture in Bath
interferogram images and write DFTFringe .oln files.

Layout
------
Left panel  : directory list, detection options, Run/Cancel, scrolling log.
Right panel : image list (top) + zoomed circle preview (bottom).
              Click any image in the list to view it; the preview shows the
              circle region cropped with a 10 % margin on every side.
"""

import io
import json
import queue
import sys
import threading
import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

from igram_finder import (
    LOW_FRINGE_SCORE,
    SPOT_THRESHOLD,
    _second_pass,
    detect_mirror,
    load_oln_circle,
    load_oln_regions,
    write_oln,
)

# ── status symbols shown in the image list ────────────────────────────────────
_SYM = {
    "pending":    "[ ]",
    "running":    "[…]",
    "ok":         "[✓]",
    "low-fringe": "[~]",
    "skipped":    "[S]",
    "failed":     "[!]",
}


# ── persisted GUI state (remembers last-used directory) ───────────────────────

_CONFIG_PATH = Path.home() / ".igram_finder_gui.json"


def _load_last_dir() -> Optional[str]:
    """Return the last directory the user browsed to, if still valid."""
    try:
        data = json.loads(_CONFIG_PATH.read_text())
        last = data.get("last_dir")
        if last and Path(last).is_dir():
            return last
    except (OSError, ValueError):
        pass
    return None


def _save_last_dir(path: str) -> None:
    try:
        _CONFIG_PATH.write_text(json.dumps({"last_dir": path}))
    except OSError:
        pass


# ── mouse-wheel scrolling for Listbox widgets ─────────────────────────────────

def _bind_mousewheel(widget: tk.Listbox) -> None:
    """Make a Listbox scroll with the mouse wheel on X11, Windows and macOS.

    On X11 the wheel arrives as Button-4 (up) / Button-5 (down); on Windows and
    macOS it arrives as <MouseWheel> with a signed ``delta``.
    """
    def _on_wheel(event: tk.Event) -> str:
        if getattr(event, "num", None) == 4:
            widget.yview_scroll(-3, "units")
        elif getattr(event, "num", None) == 5:
            widget.yview_scroll(3, "units")
        elif getattr(event, "delta", 0):
            widget.yview_scroll(-1 if event.delta > 0 else 1, "units")
        return "break"

    widget.bind("<MouseWheel>", _on_wheel)   # Windows / macOS
    widget.bind("<Button-4>", _on_wheel)     # X11 wheel up
    widget.bind("<Button-5>", _on_wheel)     # X11 wheel down


# ── thread-safe stdout capture ────────────────────────────────────────────────

class _QueueStream(io.TextIOBase):
    """Write-only stream that forwards text lines to a queue as ('log', line)."""

    def __init__(self, q: queue.Queue) -> None:
        self._q = q
        self._buf = ""

    def write(self, s: str) -> int:
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._q.put(("log", line))
        return len(s)

    def flush(self) -> None:
        if self._buf:
            self._q.put(("log", self._buf))
            self._buf = ""


# ── multi-directory picker dialog ────────────────────────────────────────────

class _MultiDirDialog:
    """
    Modal dialog that lists JPEG-containing directories under a chosen root
    and lets the user select multiple of them at once.

    Usage::

        dlg = _MultiDirDialog(parent_widget, root_path, options)
        # options: list of (display_label, Path)
        for path in dlg.result:   # empty list if cancelled
            ...
    """

    def __init__(self, parent: tk.Tk, root_path: Path,
                 options: list[tuple[str, Path]]) -> None:
        self.result: list[Path] = []
        self._options = options

        dlg = tk.Toplevel(parent)
        dlg.title("Select directories")
        dlg.transient(parent)
        dlg.grab_set()
        dlg.resizable(True, True)
        dlg.minsize(520, 320)

        ttk.Label(dlg, text=f"Root:  {root_path}",
                  font=("TkDefaultFont", 9)).pack(anchor="w", padx=10, pady=(10, 2))
        ttk.Label(dlg, text="Select directories to add  "
                  "(Ctrl/Shift+click for multiple):").pack(anchor="w", padx=10)

        lf = ttk.Frame(dlg)
        lf.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)
        lf.columnconfigure(0, weight=1)
        lf.rowconfigure(0, weight=1)

        self._lb = tk.Listbox(lf, selectmode=tk.EXTENDED,
                               font=("Courier", 10), height=14,
                               activestyle="dotbox")
        self._lb.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(lf, orient=tk.VERTICAL, command=self._lb.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self._lb.config(yscrollcommand=sb.set)
        _bind_mousewheel(self._lb)

        for label, _ in options:
            self._lb.insert(tk.END, label)
        self._lb.select_set(0, tk.END)   # pre-select all

        bf = ttk.Frame(dlg)
        bf.pack(fill=tk.X, padx=10, pady=(0, 10))
        ttk.Button(bf, text="All",  width=6,
                   command=lambda: self._lb.select_set(0, tk.END)).pack(side=tk.LEFT, padx=2)
        ttk.Button(bf, text="None", width=6,
                   command=lambda: self._lb.select_clear(0, tk.END)).pack(side=tk.LEFT, padx=2)
        ttk.Button(bf, text="Cancel", width=8,
                   command=dlg.destroy).pack(side=tk.RIGHT, padx=2)
        ttk.Button(bf, text="Add Selected", width=14,
                   command=lambda: self._ok(dlg)).pack(side=tk.RIGHT, padx=2)

        dlg.bind("<Return>", lambda _e: self._ok(dlg))
        dlg.bind("<Escape>", lambda _e: dlg.destroy())

        # centre on parent
        parent.update_idletasks()
        px, py = parent.winfo_x(), parent.winfo_y()
        pw, ph = parent.winfo_width(), parent.winfo_height()
        dlg.geometry(f"+{px + (pw - 520) // 2}+{py + (ph - 320) // 2}")

        parent.wait_window(dlg)

    def _ok(self, dlg: tk.Toplevel) -> None:
        self.result = [self._options[i][1] for i in self._lb.curselection()]
        dlg.destroy()


# ── main application class ────────────────────────────────────────────────────

class IgramFinderGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("igram_finder")
        self.root.minsize(1000, 650)

        # per-image state: str(path) -> dict(status, cx, cy, radius,
        #                                     fringe_score, regions)
        self._results: dict[str, dict] = {}
        self._image_paths: list[Path] = []   # ordered for listbox index alignment
        self._directories: list[Path] = []
        self._last_dir: Optional[str] = _load_last_dir()  # for browse dialog

        # threading
        self._work_queue: queue.Queue = queue.Queue()
        self._cancel_flag = threading.Event()
        self._running = False
        self._run_included: set[str] = set()   # snapshot of paths to process

        # preview cache — avoid re-reading the JPEG on every canvas resize
        self._crop_bgr: Optional[np.ndarray] = None   # BGR crop with circle drawn
        self._current_img_path: Optional[Path] = None

        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashwidth=5,
                               sashrelief=tk.RAISED)
        paned.grid(row=0, column=0, sticky="nsew")

        # left panel — fixed-ish width controls
        left = ttk.Frame(paned, padding=8)
        paned.add(left, minsize=300, width=320)

        # right panel — image list + preview
        right = tk.PanedWindow(paned, orient=tk.VERTICAL, sashwidth=5,
                                sashrelief=tk.RAISED)
        paned.add(right, minsize=500)

        self._build_left(left)
        self._build_right(right)

    # ── left panel ────────────────────────────────────────────────────────────

    def _build_left(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        row = 0

        # directories
        ttk.Label(parent, text="Directories:").grid(
            row=row, column=0, sticky="w"); row += 1

        dir_frame = ttk.Frame(parent)
        dir_frame.grid(row=row, column=0, sticky="ew"); row += 1
        dir_frame.columnconfigure(0, weight=1)

        self._dir_lb = tk.Listbox(dir_frame, height=4, selectmode=tk.SINGLE)
        self._dir_lb.grid(row=0, column=0, sticky="ew")
        sb = ttk.Scrollbar(dir_frame, orient=tk.VERTICAL,
                           command=self._dir_lb.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self._dir_lb.config(yscrollcommand=sb.set)
        _bind_mousewheel(self._dir_lb)

        bf = ttk.Frame(parent)
        bf.grid(row=row, column=0, sticky="w", pady=(2, 6)); row += 1
        ttk.Button(bf, text="Add…", command=self._add_directory).pack(
            side=tk.LEFT, padx=2)
        ttk.Button(bf, text="Remove", command=self._remove_directory).pack(
            side=tk.LEFT, padx=2)

        # options
        ttk.Separator(parent, orient=tk.HORIZONTAL).grid(
            row=row, column=0, sticky="ew", pady=4); row += 1
        ttk.Label(parent, text="Options:").grid(
            row=row, column=0, sticky="w"); row += 1

        self._force_var   = tk.BooleanVar()
        self._dryrun_var  = tk.BooleanVar()
        self._compare_var = tk.BooleanVar()
        self._relook_var  = tk.BooleanVar()
        self._nospots_var = tk.BooleanVar()

        for text, var in [
            ("--force  (overwrite existing .oln)", self._force_var),
            ("--dry-run  (detect, no files written)", self._dryrun_var),
            ("--compare  (test vs existing .oln)", self._compare_var),
            ("--re-look  (second-pass consensus)", self._relook_var),
            ("--no-spots  (disable reflection detection)", self._nospots_var),
        ]:
            ttk.Checkbutton(parent, text=text, variable=var).grid(
                row=row, column=0, sticky="w"); row += 1

        nf = ttk.Frame(parent)
        nf.grid(row=row, column=0, sticky="ew", pady=6); row += 1
        nf.columnconfigure(1, weight=1)

        self._threshold_var = tk.IntVar(value=20)
        self._scale_var     = tk.IntVar(value=4)
        self._jumpwarn_var  = tk.DoubleVar(value=50.0)
        self._spotthresh_var = tk.DoubleVar(value=SPOT_THRESHOLD)

        for r, label, var, lo, hi, inc in [
            (0, "Threshold (0-255):", self._threshold_var, 0,   255, 1),
            (1, "Scale:",             self._scale_var,     1,    16, 1),
            (2, "Jump-warn (px):",    self._jumpwarn_var,  0,   500, 10),
            (3, "Spot-threshold:",    self._spotthresh_var, 0.0, 2.0, 0.05),
        ]:
            ttk.Label(nf, text=label).grid(row=r, column=0, sticky="w")
            ttk.Spinbox(nf, from_=lo, to=hi, increment=inc,
                        textvariable=var, width=7).grid(
                row=r, column=1, sticky="w", padx=6)

        # run / cancel
        ttk.Separator(parent, orient=tk.HORIZONTAL).grid(
            row=row, column=0, sticky="ew", pady=4); row += 1
        rcf = ttk.Frame(parent)
        rcf.grid(row=row, column=0, sticky="w", pady=4); row += 1

        self._run_btn = ttk.Button(rcf, text="Run", command=self._run, width=10)
        self._run_btn.pack(side=tk.LEFT, padx=2)
        self._cancel_btn = ttk.Button(rcf, text="Cancel", command=self._cancel,
                                      width=10, state=tk.DISABLED)
        self._cancel_btn.pack(side=tk.LEFT, padx=2)

        # log
        ttk.Separator(parent, orient=tk.HORIZONTAL).grid(
            row=row, column=0, sticky="ew", pady=4); row += 1
        ttk.Label(parent, text="Log:").grid(
            row=row, column=0, sticky="w"); row += 1
        parent.rowconfigure(row, weight=1)
        self._log_box = scrolledtext.ScrolledText(
            parent, width=38, height=12,
            state=tk.DISABLED, wrap=tk.WORD,
            font=("Courier", 9), bg="#f8f8f8")
        self._log_box.grid(row=row, column=0, sticky="nsew")

    # ── right panel ───────────────────────────────────────────────────────────

    def _build_right(self, paned: tk.PanedWindow) -> None:
        # top: image list
        top = ttk.Frame(paned, padding=(4, 4))
        top.columnconfigure(0, weight=1)
        top.rowconfigure(1, weight=1)
        paned.add(top, minsize=120, height=220)

        self._img_count_var = tk.StringVar(value="Images  (click to preview):")
        ttk.Label(top, textvariable=self._img_count_var).grid(
            row=0, column=0, columnspan=2, sticky="w")

        self._img_lb = tk.Listbox(
            top, selectmode=tk.SINGLE,
            font=("Courier", 9), activestyle="dotbox")
        self._img_lb.grid(row=1, column=0, sticky="nsew")
        sb2 = ttk.Scrollbar(top, orient=tk.VERTICAL,
                             command=self._img_lb.yview)
        sb2.grid(row=1, column=1, sticky="ns")
        self._img_lb.config(yscrollcommand=sb2.set)
        self._img_lb.bind("<<ListboxSelect>>", self._on_image_select)
        self._img_lb.bind("<space>", self._toggle_selected)
        self._img_lb.bind("<Double-Button-1>", self._toggle_selected)
        _bind_mousewheel(self._img_lb)

        # selection controls
        self_frame = ttk.Frame(top)
        self_frame.grid(row=2, column=0, columnspan=2, sticky="w", pady=(2, 0))
        ttk.Button(self_frame, text="Select All", width=11,
                   command=lambda: self._set_all_included(True)).pack(
            side=tk.LEFT, padx=2)
        ttk.Button(self_frame, text="Deselect All", width=12,
                   command=lambda: self._set_all_included(False)).pack(
            side=tk.LEFT, padx=2)
        ttk.Button(self_frame, text="Toggle", width=8,
                   command=self._toggle_selected).pack(side=tk.LEFT, padx=2)

        # status line below the list
        self._img_status_var = tk.StringVar(value="")
        ttk.Label(top, textvariable=self._img_status_var,
                  font=("Courier", 9), anchor="w").grid(
            row=3, column=0, columnspan=2, sticky="ew", pady=(2, 0))

        # bottom: preview canvas
        bot = ttk.Frame(paned, padding=(4, 4))
        bot.columnconfigure(0, weight=1)
        bot.rowconfigure(1, weight=1)
        paned.add(bot, minsize=300)

        ttk.Label(bot, text="Preview  (circle region ± 10 % margin):").grid(
            row=0, column=0, sticky="w")

        self._canvas = tk.Canvas(bot, bg="#1a1a1a", highlightthickness=0)
        self._canvas.grid(row=1, column=0, sticky="nsew")
        self._canvas.bind("<Configure>", self._on_canvas_resize)
        self._photo_ref: Optional[ImageTk.PhotoImage] = None  # prevent GC

    # ── directory management ──────────────────────────────────────────────────

    def _add_directory(self) -> None:
        raw = filedialog.askdirectory(
            title="Select directory (or parent of session directories)",
            initialdir=self._last_dir)
        if not raw:
            return
        root = Path(raw)
        # Remember this location so the next browse starts here.
        self._last_dir = raw
        _save_last_dir(raw)

        def _jpeg_count(d: Path) -> int:
            try:
                return sum(1 for p in d.iterdir()
                           if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg"))
            except PermissionError:
                return 0

        # Build candidate list: root itself + immediate subdirs that have JPEGs
        candidates: list[tuple[str, Path]] = []
        root_n = _jpeg_count(root)
        if root_n:
            candidates.append((f".  ({root_n} images)", root))
        try:
            for sub in sorted(p for p in root.iterdir() if p.is_dir()):
                n = _jpeg_count(sub)
                if n:
                    candidates.append((f"{sub.name}/  ({n} images)", sub))
        except PermissionError:
            pass

        if not candidates:
            self._log(f"No JPEG images found in {root} or its subdirectories.")
            return

        if len(candidates) == 1:
            # Only one option — skip the dialog
            self._add_single_directory(candidates[0][1])
            return

        dlg = _MultiDirDialog(self.root, root, candidates)
        for d in dlg.result:
            self._add_single_directory(d)

    def _add_single_directory(self, d: Path) -> None:
        if d in self._directories:
            self._log(f"Already added: {d}")
            return
        self._directories.append(d)
        self._dir_lb.insert(tk.END, str(d))
        self._scan_directory(d)

    def _remove_directory(self) -> None:
        sel = self._dir_lb.curselection()
        if not sel:
            return
        idx = sel[0]
        d = self._directories.pop(idx)
        self._dir_lb.delete(idx)
        self._image_paths = [p for p in self._image_paths if p.parent != d]
        self._rebuild_img_listbox()

    def _scan_directory(self, d: Path) -> None:
        images = sorted(
            p for p in d.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg") and p.is_file()
        )
        if not images:
            self._log(f"No JPEG images found in {d}")
            return
        for p in images:
            if p not in self._image_paths:
                self._image_paths.append(p)
                self._results.setdefault(str(p), {
                    "status": "pending",
                    "included": True,
                    "cx": None, "cy": None,
                    "radius": None, "fringe_score": None,
                    "regions": None,
                })
        self._log(f"Added {len(images)} image(s) from {d.name}")
        self._rebuild_img_listbox()

    # ── image listbox helpers ─────────────────────────────────────────────────

    def _is_included(self, p: Path) -> bool:
        return self._results.get(str(p), {}).get("included", True)

    def _img_label(self, p: Path) -> str:
        info = self._results.get(str(p), {})
        sym = _SYM.get(info.get("status", "pending"), "[ ]")
        inc = "☑" if info.get("included", True) else "☐"
        name = f"{p.parent.name}/{p.name}" if len(self._directories) > 1 else p.name
        return f"{inc} {sym} {name}"

    def _rebuild_img_listbox(self) -> None:
        self._img_lb.delete(0, tk.END)
        for i, p in enumerate(self._image_paths):
            self._img_lb.insert(tk.END, self._img_label(p))
            if not self._is_included(p):
                self._img_lb.itemconfig(i, foreground="#999999")
        self._update_count()

    def _update_img_entry(self, p: Path) -> None:
        try:
            idx = self._image_paths.index(p)
        except ValueError:
            return
        self._img_lb.delete(idx)
        self._img_lb.insert(idx, self._img_label(p))
        if not self._is_included(p):
            self._img_lb.itemconfig(idx, foreground="#999999")

    def _update_count(self) -> None:
        total = len(self._image_paths)
        inc = sum(1 for p in self._image_paths if self._is_included(p))
        self._img_count_var.set(
            f"Images — {inc}/{total} selected  "
            f"(click = preview, Space / double-click = toggle):")

    # ── inclusion toggling ────────────────────────────────────────────────────

    def _toggle_selected(self, _event: Optional[tk.Event] = None) -> str:
        sel = self._img_lb.curselection()
        if sel and sel[0] < len(self._image_paths):
            idx = sel[0]
            p = self._image_paths[idx]
            info = self._results.get(str(p))
            if info is not None:
                info["included"] = not info.get("included", True)
                self._update_img_entry(p)
                # re-inserting the row clears selection — restore it so
                # repeated Space presses keep toggling the same image
                self._img_lb.selection_set(idx)
                self._img_lb.activate(idx)
                self._update_count()
        return "break"   # swallow the key so the Listbox default doesn't fire

    def _set_all_included(self, value: bool) -> None:
        for p in self._image_paths:
            info = self._results.get(str(p))
            if info is not None:
                info["included"] = value
        self._rebuild_img_listbox()

    # ── run / cancel ──────────────────────────────────────────────────────────

    def _run(self) -> None:
        if not self._image_paths:
            self._log("No images to process — add a directory first.")
            return
        included = [p for p in self._image_paths if self._is_included(p)]
        if not included:
            self._log("No images selected — tick at least one "
                      "(Space or double-click).")
            return

        self._running = True
        self._cancel_flag.clear()
        self._run_btn.config(state=tk.DISABLED)
        self._cancel_btn.config(state=tk.NORMAL)

        # snapshot the selection so toggling during a run can't change it
        self._run_included = {str(p) for p in included}

        # reset only the selected images to pending
        for p in included:
            self._results[str(p)]["status"] = "pending"
        self._rebuild_img_listbox()

        threading.Thread(target=self._run_thread, daemon=True).start()
        self.root.after(50, self._poll_queue)

    def _cancel(self) -> None:
        self._cancel_flag.set()
        self._log("Cancelling…")

    def _run_thread(self) -> None:
        force      = self._force_var.get()
        dry_run    = self._dryrun_var.get()
        compare    = self._compare_var.get()
        re_look    = self._relook_var.get()
        find_spots = not self._nospots_var.get()
        scale      = self._scale_var.get()
        threshold  = self._threshold_var.get()
        jump_warn  = self._jumpwarn_var.get()
        spot_thresh = self._spotthresh_var.get()

        q_stream  = _QueueStream(self._work_queue)
        old_stdout = sys.stdout
        sys.stdout = q_stream

        try:
            for directory in self._directories:
                if self._cancel_flag.is_set():
                    break

                self._work_queue.put(("log", f"\n─── {directory} ───"))
                images = [p for p in self._image_paths
                          if p.parent == directory
                          and str(p) in self._run_included]

                if compare:
                    images = [p for p in images if p.with_suffix(".oln").exists()]

                prev_circle: Optional[tuple[float, float, float]] = None
                prev_spots: list = []
                all_results: list[tuple] = []

                for img_path in images:
                    if self._cancel_flag.is_set():
                        break

                    self._work_queue.put(("status", (img_path, "running")))
                    oln_path = img_path.with_suffix(".oln")

                    # skip if .oln exists and not force/compare
                    if oln_path.exists() and not force and not compare:
                        existing = load_oln_circle(oln_path)
                        prev_spots = load_oln_regions(oln_path)
                        if existing:
                            cx, cy, r = existing
                            prev_circle = existing
                            self._work_queue.put(
                                ("result", (img_path, cx, cy, r, None, prev_spots, "skipped")))
                        else:
                            self._work_queue.put(("status", (img_path, "skipped")))
                        print(f"  {img_path.name}: skipped (already has .oln)")
                        continue

                    print(f"  {img_path.name}: detecting…", end="", flush=True)
                    result = detect_mirror(
                        img_path,
                        scale=scale,
                        threshold=threshold,
                        prev_circle=prev_circle,
                        jump_warn=jump_warn,
                        find_spots=find_spots,
                        spot_threshold=spot_thresh,
                        prev_spots=prev_spots,
                    )

                    if result is None:
                        print(" FAILED")
                        self._work_queue.put(("status", (img_path, "failed")))
                        continue

                    cx, cy, radius, fringe_score, regions = result
                    status = "low-fringe" if fringe_score < LOW_FRINGE_SCORE else "ok"
                    fringe_tag = " [low-fringe]" if status == "low-fringe" else ""
                    spot_tag = f" [{len(regions)} spot(s)]" if regions else ""
                    tag = " [dry-run]" if dry_run else ""
                    print(
                        f" cx={cx:.1f} cy={cy:.1f} r={radius:.1f}"
                        f"  fringe={fringe_score:.2f}{fringe_tag}{spot_tag}{tag}"
                    )

                    if not dry_run and not compare:
                        write_oln(oln_path, cx, cy, radius, regions)

                    prev_circle = (cx, cy, radius)
                    prev_spots = regions
                    all_results.append(
                        (img_path, cx, cy, radius, fringe_score, regions))
                    self._work_queue.put(
                        ("result", (img_path, cx, cy, radius, fringe_score, regions, status)))

                if re_look and all_results and not self._cancel_flag.is_set():
                    print("\n─── Re-look (second pass) ───")
                    updated = _second_pass(all_results, scale=scale, dry_run=dry_run)
                    for img_path, cx, cy, r, fs, regions in updated:
                        st = "low-fringe" if fs < LOW_FRINGE_SCORE else "ok"
                        self._work_queue.put(
                            ("result", (img_path, cx, cy, r, fs, regions, st)))

        finally:
            sys.stdout = old_stdout
            q_stream.flush()
            self._work_queue.put(("done", None))

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, data = self._work_queue.get_nowait()
                if kind == "log":
                    self._log(data)
                elif kind == "status":
                    img_path, status = data
                    self._results[str(img_path)]["status"] = status
                    self._update_img_entry(img_path)
                elif kind == "result":
                    img_path, cx, cy, radius, fringe_score, regions, status = data
                    self._results[str(img_path)].update({
                        "status": status,
                        "cx": cx, "cy": cy,
                        "radius": radius,
                        "fringe_score": fringe_score,
                        "regions": regions,
                    })
                    self._update_img_entry(img_path)
                    # refresh preview if this is the currently displayed image
                    if img_path == self._current_img_path:
                        self._load_and_render(img_path)
                elif kind == "done":
                    self._running = False
                    self._run_btn.config(state=tk.NORMAL)
                    self._cancel_btn.config(state=tk.DISABLED)
                    self._log("\nDone.")
                    return
        except queue.Empty:
            pass

        if self._running:
            self.root.after(50, self._poll_queue)

    # ── image preview ─────────────────────────────────────────────────────────

    def _on_image_select(self, event: tk.Event) -> None:
        sel = self._img_lb.curselection()
        if not sel:
            return
        idx = sel[0]
        if idx >= len(self._image_paths):
            return
        img_path = self._image_paths[idx]
        self._load_and_render(img_path)

        # update status label
        info = self._results.get(str(img_path), {})
        cx, cy, r = info.get("cx"), info.get("cy"), info.get("radius")
        fs = info.get("fringe_score")
        regions = info.get("regions")
        if cx is not None:
            fs_str = f"  fringe={fs:.3f}" if fs is not None else ""
            spot_str = f"  spots={len(regions)}" if regions else ""
            self._img_status_var.set(
                f"cx={cx:.1f}  cy={cy:.1f}  r={r:.1f}{fs_str}{spot_str}")
        else:
            self._img_status_var.set(info.get("status", "pending"))

    def _load_and_render(self, img_path: Path) -> None:
        """Load the image (and circle if available), cache the crop, display."""
        self._current_img_path = img_path

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            self._log(f"Could not read {img_path.name}")
            return

        info = self._results.get(str(img_path), {})
        cx     = info.get("cx")
        cy     = info.get("cy")
        radius = info.get("radius")
        regions = info.get("regions")

        if cx is not None and cy is not None and radius is not None:
            self._crop_bgr = self._make_crop(img_bgr, cx, cy, radius, regions)
        else:
            # No detection yet — show the full image thumbnail
            h, w = img_bgr.shape[:2]
            thumb_w = min(w, 1200)
            thumb_h = int(h * thumb_w / w)
            self._crop_bgr = cv2.resize(
                img_bgr, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)

        self._render_crop()

    @staticmethod
    def _make_crop(img_bgr: np.ndarray,
                   cx: float, cy: float, radius: float,
                   regions: Optional[list] = None) -> np.ndarray:
        """Draw circle (and any spot regions) on a copy and crop to circle
        region + 10 % margin."""
        out = img_bgr.copy()
        cv2.circle(out, (int(cx), int(cy)), int(radius), (0, 220, 0), 3)
        # small crosshair at centre
        arm = max(10, int(radius * 0.03))
        cv2.line(out, (int(cx) - arm, int(cy)), (int(cx) + arm, int(cy)),
                 (0, 220, 0), 2)
        cv2.line(out, (int(cx), int(cy) - arm), (int(cx), int(cy) + arm),
                 (0, 220, 0), 2)

        # bright internal-reflection spot regions, drawn in red
        for poly in regions or []:
            pts = np.array([[p["x"], p["y"]] for p in poly], dtype=np.int32)
            if len(pts) >= 2:
                cv2.polylines(out, [pts], isClosed=True, color=(0, 0, 255),
                              thickness=3)

        margin = radius * 0.10
        h, w = out.shape[:2]
        x1 = max(0, int(cx - radius - margin))
        y1 = max(0, int(cy - radius - margin))
        x2 = min(w, int(cx + radius + margin + 1))
        y2 = min(h, int(cy + radius + margin + 1))
        return out[y1:y2, x1:x2]

    def _render_crop(self) -> None:
        """Scale the cached crop to fit the canvas and display it."""
        if self._crop_bgr is None:
            return

        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        if cw < 10 or ch < 10:
            return

        ih, iw = self._crop_bgr.shape[:2]
        scale = min(cw / iw, ch / ih, 1.0)
        dw, dh = max(1, int(iw * scale)), max(1, int(ih * scale))

        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(self._crop_bgr, (dw, dh), interpolation=interp)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(pil_img)

        self._canvas.delete("all")
        self._canvas.create_image(cw // 2, ch // 2, anchor=tk.CENTER, image=photo)
        self._photo_ref = photo   # keep reference — tkinter doesn't own it

    def _on_canvas_resize(self, event: tk.Event) -> None:
        self._render_crop()

    # ── logging ───────────────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        self._log_box.config(state=tk.NORMAL)
        self._log_box.insert(tk.END, msg + "\n")
        self._log_box.see(tk.END)
        self._log_box.config(state=tk.DISABLED)


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    root = tk.Tk()
    IgramFinderGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()