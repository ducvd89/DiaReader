# Build DiaReader into a Windows installer.
#
#   powershell -ExecutionPolicy Bypass -File packaging\build.ps1
#
# Produces build\installer\DiaReader-<version>-setup.exe.
#
# Run this from a machine whose venv is already the one you want to ship: PyInstaller
# freezes whatever is installed, so the CUDA build of torch in the venv is the CUDA
# build that ends up in the installer. Building from a CPU-only venv silently produces
# a CPU-only app that then refuses to start on the user's GPU.
[CmdletBinding()]
param(
    [switch]$SkipFreeze,     # reuse an existing build\dist\DiaReader
    [switch]$SkipInstaller   # stop after freezing
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "No venv at $python — see README section 2." }

# The version comes from the package rather than being repeated here or in the .iss.
$version = & $python -c "import diareader; print(diareader.__version__)"
if ($LASTEXITCODE -ne 0 -or -not $version) { throw "Could not read diareader.__version__" }
Write-Host "==> DiaReader $version" -ForegroundColor Cyan

# The build machine's PATH may predate the FFmpeg install; PyInstaller needs to be
# able to import torchcodec to collect it.
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" +
            [System.Environment]::GetEnvironmentVariable("Path","User")

if (-not $SkipFreeze) {
    Write-Host "==> Freezing (this takes 15-30 minutes; the payload is ~4.5 GB)" -ForegroundColor Cyan
    & $python -m PyInstaller --noconfirm --clean `
        --distpath build\dist --workpath build\work `
        packaging\DiaReader.spec
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed with exit code $LASTEXITCODE" }
}

$app = Join-Path $root "build\dist\DiaReader\DiaReader.exe"
if (-not (Test-Path $app)) { throw "Freeze produced no executable at $app" }
$size = (Get-ChildItem "build\dist\DiaReader" -Recurse -File | Measure-Object Length -Sum).Sum / 1GB
Write-Host ("==> Frozen app is {0:N2} GB" -f $size) -ForegroundColor Green

if ($SkipInstaller) { return }

$iscc = @(
    "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $iscc) { throw "Inno Setup not found. winget install JRSoftware.InnoSetup" }

Write-Host "==> Building the installer" -ForegroundColor Cyan
& $iscc "/DAppVersion=$version" "packaging\DiaReader.iss"
if ($LASTEXITCODE -ne 0) { throw "Inno Setup failed with exit code $LASTEXITCODE" }

Get-ChildItem "build\installer\*.exe" | ForEach-Object {
    Write-Host ("==> {0}  ({1:N2} GB)" -f $_.Name, ($_.Length / 1GB)) -ForegroundColor Green
}
