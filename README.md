# Audio Transcription Service

This repository contains tooling to prepare a Windows machine for transcribing audio with [WhisperX](https://github.com/m-bain/whisperX).

## Pre-setup script

Use the `pre-setup.ps1` PowerShell script to install all required software. It installs Chocolatey, Python, Git, FFmpeg, PyTorch and WhisperX.

### Usage
1. Open PowerShell **as Administrator**.
2. Run the following commands:

```powershell
Set-ExecutionPolicy Bypass -Scope Process -Force
./pre-setup.ps1             # installs CPU version of PyTorch
./pre-setup.ps1 -UseGPU     # optional: installs CUDA version
```

After completion, WhisperX will be available for transcribing audio files.

## Install Python dependencies

If you already have Python available, install the required packages with:

```bash
python -m pip install -r requirements.txt
```

This installs the Rich console helper along with both Whisper backends (`faster-whisper` by default and `whisperx` as an optional alternative) so the CLI and GUI can import their dependencies successfully.

## Running the application

Once dependencies are installed, launch the transcription CLI with:

```bash
python run.py --help
```

The `run.py` helper adds the `src` directory to `PYTHONPATH`, allowing the app to be started with a double-click or from the command line without extra setup. On first run the application will prompt for the location of your recordings directory, store it in `appsettings.json`, list session folders within that directory ordered by recency, and let you pick one. Pressing enter without a choice selects the most recent session. After choosing a session you can select an existing game profile or create a new one, and the transcript is saved to `transcript/<CampaignName>Transcript.txt` inside the session folder.

On Windows you can also use the `start.ps1` script, which simply launches the application:

```powershell
./start.ps1
```

## Graphical user interface

A Tkinter-based GUI is available for users who prefer not to run the CLI. Launch it from the repository root after adding `src` to your `PYTHONPATH`:

```bash
export PYTHONPATH=src
python -m gui_app.app
```

The GUI lets you choose the input folder, output base name, model, language, and engine, and it reports progress for each file as transcription runs.
