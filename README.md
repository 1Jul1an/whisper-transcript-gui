# whisper-transcript-gui

Simple Python GUI for running **OpenAI Whisper** locally to transcribe audio/video files.

- pick input file
- choose Whisper model (tiny → large)
- choose language (Auto + 5 main languages)
- choose device (GPU preferred, CPU fallback)
- optional: choose output file
- pseudo progress bar / ETA while Whisper is running
- configurable `ffmpeg` path at the top of the script

Useful if you want to transcribe interviews (e.g. for a thesis) without uploading them anywhere.

---

## Features

- **Tkinter GUI** – single file, no fancy frameworks
- **Model selection** – `tiny`, `base`, `small`, `medium`, `large`
- **Language selection** – `Auto`, `English`, `German`, `Spanish`, `French`, `Italian`
- **Device selection** – defaults to `cuda`, switches to `cpu` if CUDA isn’t available
- **Custom output path** – save transcript wherever you want
- **ffmpeg configurable** – set your ffmpeg folder at the top of the script
- **Pseudo ETA** – GUI shows an estimated remaining time based on media length

---

## Requirements

- Python 3.10+
- `torch` (PyTorch) – with CUDA if you want GPU
- `whisper` (OpenAI Whisper)
- `ffmpeg` + `ffprobe` available in `PATH` (or set at the top of the script)
- Windows, macOS or Linux with Tkinter (Windows 100% fine)

Install basics:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121  # example for CUDA, check PyTorch site
pip install openai-whisper
````

(If you don’t have a GPU or don’t care → just `pip install torch openai-whisper` and select `cpu` in the GUI.)

---

## ffmpeg

Whisper calls `ffmpeg` to read audio/video formats.
At the top of `whisper_gui.py` there is this section:

```python
FFMPEG_DIR = r"C:\ffmpeg\bin"
if os.path.isdir(FFMPEG_DIR):
    os.environ["PATH"] += os.pathsep + FFMPEG_DIR
```

* put the folder where your `ffmpeg.exe` and `ffprobe.exe` live
* or install ffmpeg globally and remove that section
* the GUI checks on start if `ffmpeg` is available and shows an error if not

On Windows you can download ffmpeg as a ZIP, extract, and point `FFMPEG_DIR` to `...\ffmpeg\bin`.

---

## GPU vs CPU

* If you select **cuda** in the GUI but PyTorch can’t find a GPU, the app will tell you and continue on **cpu**.
* On **GPU** you can enable `fp16=True` (the script does that automatically).
* On **CPU** it will be slower, especially with big models.

You may see a console warning like:

> `Failed to launch Triton kernels, likely due to missing CUDA toolkit ...`

This is expected on machines without full CUDA/Triton. The script suppresses that warning so the GUI isn’t flooded.

---

## Model sizes (rough guidance)

Whisper models (small → better quality, slower):

* `tiny` – very fast, lower quality
* `base` – fast, okay for clear speech
* `small` – good quality, still okay speed
* `medium` – better, slower
* `large` – best quality, slowest, also needs the most VRAM/RAM

**Tip for interviews:**
If your audio is okay and German/English, `small` or `medium` is usually enough.
`large` is nice, but on CPU it will take a while.

---

## How it estimates progress

Whisper itself doesn’t give live progress.
This GUI…

1. reads the media duration using `ffprobe`
2. guesses that transcription ≈ `duration * 1.2`
3. animates the progress bar based on that
4. sets it to 100% when the background thread is actually done

So: it’s an **estimate**, not a real-time progress from Whisper.

---

## Usage

1. Run the script:

   ```bash
   python whisper_gui.py
   ```

2. Select your audio/video file.

3. Pick model, language, device.

4. (Optional) choose an output file.

5. Click **Start transcription**.

6. Wait for status: `Done. Transcript saved to: ...`

The transcript is saved as a UTF-8 `.txt`.

---

## Structure

Everything is in a single file:

* ffmpeg config at the top
* helper functions (check ffmpeg, get duration)
* `WhisperGUI` class
* `if __name__ == "__main__": app.mainloop()`

So you can drop it into another project if you want.

---

## Example: Bachelor thesis note

If you use this for a thesis, you can write something like:

> “The interviews were recorded digitally and transcribed locally using the open-source Whisper model (via a custom Python GUI, repository: <link>). The audio was slightly cleaned (noise reduction), the content was not altered.”

That makes the process transparent.

---

## Credits

* Uses the open-source [OpenAI Whisper](https://github.com/openai/whisper) model.
* Requires [ffmpeg](https://ffmpeg.org/).

---

## License

Licensed under the **MIT License**.
See `LICENSE` for details.
