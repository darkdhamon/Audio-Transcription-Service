param(
    [switch]$UseGPU
)

# Ensure script is running as administrator
if (-not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Error "Please run this script from an elevated PowerShell session.";
    exit 1
}

# Install Chocolatey if it is not already present
if (-not (Get-Command choco.exe -ErrorAction SilentlyContinue)) {
    Write-Host "Installing Chocolatey..." -ForegroundColor Cyan
    Set-ExecutionPolicy Bypass -Scope Process -Force
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.SecurityProtocolType]::Tls12
    Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
}

# Install core dependencies using Chocolatey
choco install -y git python ffmpeg

# Ensure Python and pip are on PATH
$env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine")

# Upgrade pip
python -m pip install --upgrade pip

# Install PyTorch (CPU by default, GPU optional)
if ($UseGPU) {
    Write-Host "Installing CUDA-enabled PyTorch" -ForegroundColor Cyan
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
} else {
    Write-Host "Installing CPU-only PyTorch" -ForegroundColor Cyan
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
}

# Install WhisperX
python -m pip install -U git+https://github.com/m-bain/whisperX.git

Write-Host "WhisperX installation complete" -ForegroundColor Green
