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
