import os
import threading
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import warnings

import torch
import whisper

# Suppress the noisy Triton warning from Whisper on systems without full CUDA/Triton.
# This does NOT affect the transcription result – it only keeps the console clean.
warnings.filterwarnings("ignore", message="Failed to launch Triton kernels*")

# ============================================================
# FFMPEG CONFIGURATION
# Set this to the directory where your ffmpeg.exe / ffprobe.exe lives.
# If this path exists, we append it to PATH so Whisper can call ffmpeg.
# ============================================================
FFMPEG_DIR = r"C:\ffmpeg\bin"

if os.path.isdir(FFMPEG_DIR):
    os.environ["PATH"] += os.pathsep + FFMPEG_DIR
# ============================================================


def ffmpeg_available() -> bool:
    """Return True if ffmpeg is available in PATH."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except Exception:
        return False


def get_media_duration(path: str) -> float:
    """
    Try to read media duration (in seconds) via ffprobe.
    Returns 0.0 if ffprobe is not available or the call fails.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


class WhisperGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Whisper Transcription")
        self.geometry("620x360")
        self.resizable(False, False)

        # background worker thread for transcription
        self.worker_thread = None

        # ETA helpers
        self.estimated_seconds = 0.0
        self.elapsed_ms = 0

        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 10, "pady": 5}

        # ------------------------------------------------
        # Input file
        # ------------------------------------------------
        file_frame = ttk.LabelFrame(self, text="Input file")
        file_frame.pack(fill="x", **pad)

        self.input_var = tk.StringVar()
        entry_file = ttk.Entry(file_frame, textvariable=self.input_var)
        entry_file.pack(side="left", fill="x", expand=True, padx=5, pady=5)

        btn_browse = ttk.Button(file_frame, text="Browse...", command=self.select_file)
        btn_browse.pack(side="right", padx=5, pady=5)

        # ------------------------------------------------
        # Options
        # ------------------------------------------------
        options_frame = ttk.LabelFrame(self, text="Options")
        options_frame.pack(fill="x", **pad)

        # Model selection
        ttk.Label(options_frame, text="Model:").grid(row=0, column=0, sticky="w", padx=5, pady=3)
        self.model_var = tk.StringVar(value="large")
        model_box = ttk.Combobox(
            options_frame,
            textvariable=self.model_var,
            values=["tiny", "base", "small", "medium", "large"],
            state="readonly",
            width=10,
        )
        model_box.grid(row=0, column=1, sticky="w", padx=5, pady=3)

        # Language selection (5 main + Auto)
        ttk.Label(options_frame, text="Language:").grid(row=0, column=2, sticky="w", padx=5, pady=3)
        self.lang_var = tk.StringVar(value="German")
        lang_box = ttk.Combobox(
            options_frame,
            textvariable=self.lang_var,
            values=["Auto", "English", "German", "Spanish", "French", "Italian"],
            state="readonly",
            width=10,
        )
        lang_box.grid(row=0, column=3, sticky="w", padx=5, pady=3)

        # Device (GPU default)
        ttk.Label(options_frame, text="Device:").grid(row=1, column=0, sticky="w", padx=5, pady=3)
        self.device_var = tk.StringVar(value="cuda")
        device_box = ttk.Combobox(
            options_frame,
            textvariable=self.device_var,
            values=["cuda", "cpu"],
            state="readonly",
            width=10,
        )
        device_box.grid(row=1, column=1, sticky="w", padx=5, pady=3)

        # Output file + chooser button
        ttk.Label(options_frame, text="Output:").grid(row=1, column=2, sticky="w", padx=5, pady=3)
        out_frame = ttk.Frame(options_frame)
        out_frame.grid(row=1, column=3, sticky="w", padx=5, pady=3)

        self.output_var = tk.StringVar(value="transcript_output.txt")
        out_entry = ttk.Entry(out_frame, textvariable=self.output_var, width=20)
        out_entry.pack(side="left")

        out_btn = ttk.Button(out_frame, text="...", width=3, command=self.select_output_file)
        out_btn.pack(side="left", padx=(5, 0))

        # ------------------------------------------------
        # Status / Progress
        # ------------------------------------------------
        progress_frame = ttk.LabelFrame(self, text="Status")
        progress_frame.pack(fill="x", **pad)

        # determinate because we simulate ETA
        self.progress = ttk.Progressbar(progress_frame, mode="determinate", maximum=100)
        self.progress.pack(fill="x", padx=5, pady=5)

        self.status_var = tk.StringVar(value="Ready.")
        self.status_label = ttk.Label(progress_frame, textvariable=self.status_var)
        self.status_label.pack(fill="x", padx=5, pady=(0, 5))

        # Start button
        btn_start = ttk.Button(self, text="Start transcription", command=self.start_transcription)
        btn_start.pack(pady=5)

    # ------------------------------------------------
    # GUI callbacks
    # ------------------------------------------------
    def select_file(self):
        filename = filedialog.askopenfilename(
            title="Select audio/video file",
            filetypes=(
                ("Video/Audio", "*.mp4 *.mp3 *.m4a *.wav *.mkv"),
                ("All files", "*.*"),
            ),
        )
        if filename:
            self.input_var.set(filename)

    def select_output_file(self):
        # use current output name as default
        initial = self.output_var.get().strip() or "transcript_output.txt"
        filename = filedialog.asksaveasfilename(
            title="Save transcript as...",
            defaultextension=".txt",
            initialfile=initial,
            filetypes=(("Text file", "*.txt"), ("All files", "*.*")),
        )
        if filename:
            self.output_var.set(filename)

    def start_transcription(self):
        input_path = self.input_var.get().strip()
        if not input_path or not os.path.exists(input_path):
            messagebox.showerror("Error", "Please select a valid input file.")
            return

        if not ffmpeg_available():
            messagebox.showerror(
                "ffmpeg not found",
                "ffmpeg could not be executed.\nPlease adjust FFMPEG_DIR at the top of the script.",
            )
            return

        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo("Info", "Transcription is already running.")
            return

        # estimate duration for pseudo-ETA
        duration_sec = get_media_duration(input_path)
        if duration_sec <= 0:
            duration_sec = 60  # fallback if ffprobe failed

        # Whisper often takes roughly media_length * factor; we pick 1.2 as a rough guess
        self.estimated_seconds = duration_sec * 1.2
        self.elapsed_ms = 0
        self.progress["value"] = 0

        self.status_var.set("Loading model and starting transcription ...")

        # start background worker
        self.worker_thread = threading.Thread(target=self._do_transcription, daemon=True)
        self.worker_thread.start()

        # start UI progress ticker
        self.after(200, self._progress_tick)

    def _progress_tick(self):
        """UI-side progress update based on our rough ETA."""
        if self.worker_thread and self.worker_thread.is_alive():
            self.elapsed_ms += 200
            pct = (self.elapsed_ms / (self.estimated_seconds * 1000)) * 100
            pct = min(pct, 98)  # keep a little headroom for the final jump
            self.progress["value"] = pct

            remaining = max(0, (self.estimated_seconds * 1000 - self.elapsed_ms) / 1000)
            self.status_var.set(f"Running ... estimated ~{int(remaining)} s remaining")
            self.after(200, self._progress_tick)
        else:
            # worker finished
            self.progress["value"] = 100

    # ------------------------------------------------
    # Worker thread
    # ------------------------------------------------
    def _do_transcription(self):
        try:
            # pick device, fall back to CPU if CUDA is not available
            device = self.device_var.get()
            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"
                self._set_status("CUDA not available, using CPU.")

            torch.set_num_threads(12)

            # load model
            model_name = self.model_var.get()
            self._set_status(f"Loading model '{model_name}' ...")
            model = whisper.load_model(model_name, device=device)

            # run transcription
            self._set_status("Transcribing ...")
            input_path = self.input_var.get().strip()
            language = self.lang_var.get()
            if language == "Auto":
                language = None  # let Whisper auto-detect

            result = model.transcribe(
                input_path,
                language=language,
                temperature=0.0,
                without_timestamps=False,
                fp16=(device == "cuda"),
                word_timestamps=True,
            )

            text = result["text"]
            output_path = self.output_var.get().strip()
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)

            self._set_status(f"Done. Transcript saved to: {output_path}")
        except Exception as e:
            self._set_status(f"Error: {e}")

    def _set_status(self, msg: str):
        """Thread-safe status update."""
        def _update():
            self.status_var.set(msg)

        self.after(0, _update)


if __name__ == "__main__":
    app = WhisperGUI()
    app.mainloop()
