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

$gpuNames = Get-GpuNames
$hasNvidia = Test-NvidiaGpuPresent -GpuNames $gpuNames
$hasAmd = [bool]($gpuNames | Where-Object { $_ -match "AMD|Radeon" })
$recommendedTarget = if ($hasNvidia) { "CUDA" } else { "CPU" }

Write-Host "Operating system: $([System.Environment]::OSVersion.VersionString)" -ForegroundColor Cyan
Write-Host "Detected GPUs: $($gpuNames -join ', ')" -ForegroundColor Cyan
Write-Host "Recommended install target: $recommendedTarget" -ForegroundColor Green

if ($hasAmd -and -not $hasNvidia) {
    Write-Host "AMD graphics detected. The app currently uses the CPU inference path on AMD/DirectML systems." -ForegroundColor Yellow
}

if (-not $gpuNames) {
    Write-Host "No display adapters were reported by Win32_VideoController." -ForegroundColor Yellow
}
