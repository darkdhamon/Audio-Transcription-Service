# Audio Transcription Service

This repository contains tooling to prepare a Windows machine for transcribing audio with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and [WhisperX](https://github.com/m-bain/whisperX).

## Pre-setup script

Use the `pre-setup.ps1` PowerShell script to install all required software. It installs Chocolatey, Python, Git, FFmpeg, PyTorch, faster-whisper, WhisperX, and Rich.

### Usage
1. Open PowerShell **as Administrator**.
2. Run the following commands:

```powershell
Set-ExecutionPolicy Bypass -Scope Process -Force
./scripts/detect-hardware.ps1          # optional: see the detected hardware path
./scripts/pre-setup.ps1                # auto-detects CPU vs NVIDIA CUDA
./scripts/pre-setup.ps1 -Target CPU    # force the CPU path
./scripts/pre-setup.ps1 -Target CUDA   # force the NVIDIA CUDA path
./scripts/pre-setup.ps1 -UseGPU        # legacy alias for -Target CUDA
```

If the machine has an AMD GPU, the current application still uses the CPU inference path. The runtime does not expose a DirectML backend today, so AMD systems remain CPU-first for compatibility.

## Running the application

Once dependencies are installed, launch the transcription CLI with:

```bash
python run.py --help
```

The application resolves a hardware-aware runtime automatically:

- NVIDIA CUDA systems prefer GPU acceleration.
- Windows CPU and AMD/DirectML systems fall back to CPU inference.
- The default model is selected automatically when `--model auto` is used.

The `run.py` helper adds the `src` directory to `PYTHONPATH`, allowing the app to be started with a double-click or from the command line without extra setup. On first run the application will prompt for the location of your recordings directory, store it in `appsettings.json`, list session folders within that directory ordered by recency, and let you pick one. Pressing enter without a choice selects the most recent session. After choosing a session you can select an existing game profile or create a new one, and the transcript is saved to `transcript/<CampaignName>Transcript.txt` inside the session folder.

On Windows you can also use the `start.ps1` script, which simply launches the application:

```powershell
./start.ps1
```

To inspect what the app would choose on the current machine without starting a transcription run:

```bash
python run.py --show-hardware
```
