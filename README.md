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

## Running the application

Once dependencies are installed, launch the transcription CLI with:

```bash
python run.py --help
```

The `run.py` helper adds the `src` directory to `PYTHONPATH`, allowing the app to be started with a double-click or from the command line without extra setup. On first run the application will prompt for the location of your recordings directory, store it in `appsettings.json`, list session folders within that directory, and ask you to choose one for transcription.

On Windows you can also use the `start.ps1` script, which simply launches the application:

```powershell
./start.ps1
```
