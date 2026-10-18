$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
& .\setup_head.bat
if ($LASTEXITCODE -ne 0) { throw 'Head environment setup failed' }
& .\.venv-head\Scripts\python.exe -m pip install modelscope==1.18.0
if ($LASTEXITCODE -ne 0) { throw 'ModelScope installation failed' }
if (-not (Test-Path -LiteralPath 'third_party/HRN/.git')) {
    New-Item -ItemType Directory -Force third_party | Out-Null
    git clone --depth 1 https://github.com/youngLBW/HRN.git third_party/HRN
    if ($LASTEXITCODE -ne 0) { throw 'HRN source download failed' }
}
& .\.venv-head\Scripts\python.exe tools/hrn_setup/download_assets.py
if ($LASTEXITCODE -ne 0) { throw 'HRN asset download failed' }
& .\.venv-head\Scripts\python.exe tools/hrn_setup/prepare_vendor.py
if ($LASTEXITCODE -ne 0) { throw 'HRN asset preparation failed' }
$taskRootWsl = (wsl -d Ubuntu-22.04 -- wslpath -a $PSScriptRoot).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Ubuntu-22.04 is required' }
wsl -d Ubuntu-22.04 -- bash "$taskRootWsl/tools/hrn_setup/bootstrap.sh"
if ($LASTEXITCODE -ne 0) { throw 'Linux HRN installation failed' }
& .\.venv-head\Scripts\python.exe -m tools.hrn_setup.validate_runtime
if ($LASTEXITCODE -ne 0) { throw 'HRN sample inference failed; dense API remains disabled' }
Write-Host 'Dense reconstruction is ready. Run start_head.bat or restart the main server.'
