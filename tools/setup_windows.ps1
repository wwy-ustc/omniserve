param(
    [string]$Python = "python",
    [ValidateSet("cuda", "cpu")][string]$Device = "cuda",
    [ValidateSet("cu128", "cu132", "cu134")][string]$CudaWheel = "cu128"
)
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
if ($Python -eq "python" -and -not (Get-Command $Python -ErrorAction SilentlyContinue)) {
    $bundledPython = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
    if (Test-Path -LiteralPath $bundledPython) { $Python = $bundledPython }
}

function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed: $Executable $($Arguments -join ' ')" }
}

if (-not (Test-Path -LiteralPath $venvPython)) {
    Invoke-Checked -Executable $Python -Arguments @("-c", "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 'Python 3.11+ required; Python 3.12 recommended')")
    Invoke-Checked -Executable $Python -Arguments @("-m", "venv", (Join-Path $projectRoot ".venv"))
}
$index = if ($Device -eq "cuda") { $CudaWheel } else { "cpu" }
Invoke-Checked -Executable $venvPython -Arguments @("-m", "pip", "install", "torch==2.15.0", "--index-url", "https://download.pytorch.org/whl/$index")
Invoke-Checked -Executable $venvPython -Arguments @("-m", "pip", "install", "-r", (Join-Path $projectRoot "requirements-voice.txt"))
if ($Device -eq "cuda") {
    Invoke-Checked -Executable $venvPython -Arguments @((Join-Path $PSScriptRoot "validate_cuda.py"))
}
Write-Host "Ready. Start: $venvPython voice_server.py --mode continuous --device $Device"
