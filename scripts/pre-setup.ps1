param(
    [switch]$UseGPU,
    [ValidateSet("Auto", "CPU", "CUDA")]
    [string]$Target = "Auto"
)

function Get-GpuNames {
    try {
        return @(Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name)
    } catch {
        return @()
    }
}

function Test-NvidiaGpuPresent {
    param(
        [string[]]$GpuNames
    )

    return [bool]($GpuNames | Where-Object { $_ -match "NVIDIA" })
}

# Keep the original ``-UseGPU`` switch for backward compatibility.
if ($UseGPU) {
    $Target = "CUDA"
}

# Ensure script is running as administrator.
if (-not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Error "Please run this script from an elevated PowerShell session."
    exit 1
}

# Install Chocolatey if it is not already present.
if (-not (Get-Command choco.exe -ErrorAction SilentlyContinue)) {
    Write-Host "Installing Chocolatey..." -ForegroundColor Cyan
    Set-ExecutionPolicy Bypass -Scope Process -Force
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.SecurityProtocolType]::Tls12
    Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
}

# Install core tools using Chocolatey.
choco install -y git python ffmpeg

# Refresh PATH so the freshly installed Python is visible in this session.
$env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine")

$gpuNames = Get-GpuNames
$hasNvidia = Test-NvidiaGpuPresent -GpuNames $gpuNames
$resolvedTarget = $Target
if ($resolvedTarget -eq "Auto") {
    $resolvedTarget = if ($hasNvidia) { "CUDA" } else { "CPU" }
}

if ($resolvedTarget -eq "CUDA" -and -not $hasNvidia) {
    Write-Error "CUDA installation was requested, but no NVIDIA GPU was detected."
    exit 1
}

Write-Host "Detected GPUs: $($gpuNames -join ', ')" -ForegroundColor DarkCyan
Write-Host "Resolved install target: $resolvedTarget" -ForegroundColor Cyan

if ($resolvedTarget -eq "CPU" -and ($gpuNames | Where-Object { $_ -match "AMD|Radeon" })) {
    Write-Host "AMD GPU detected. The current app uses the CPU path on AMD/DirectML systems." -ForegroundColor Yellow
}

# Upgrade packaging tools before installing pinned dependencies.
python -m pip install --upgrade pip setuptools wheel

# Install PyTorch.
if ($resolvedTarget -eq "CUDA") {
    Write-Host "Installing CUDA-enabled PyTorch 2.8.0 (CUDA 12.6)..." -ForegroundColor Cyan
    python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
} else {
    Write-Host "Installing CPU-only PyTorch 2.8.0..." -ForegroundColor Cyan
    python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu
}

# Install the transcription backends used by the app. These versions update
# the stack while remaining compatible with the CPU and CUDA paths above.
python -m pip install `
    faster-whisper==1.2.1 `
    whisperx==3.8.5 `
    rich==15.0.0

Write-Host "Transcription dependencies installed successfully." -ForegroundColor Green
