[CmdletBinding()]
param()

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

python "$root/run.py"

