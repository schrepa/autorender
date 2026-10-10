# Optional workspace-local Rust installation. This does not change the user PATH.
# Dot-source from PowerShell: . ./scripts/activate.ps1
# Worktrees use the main checkout's installation.
$engineRoot = Split-Path -Parent $PSScriptRoot
$toolsRoot = $engineRoot
if (-not (Test-Path -LiteralPath (Join-Path $toolsRoot '.tools\cargo\bin\cargo.exe'))) {
    $commonDir = git -C $engineRoot rev-parse --path-format=absolute --git-common-dir 2>$null
    if ($LASTEXITCODE -eq 0 -and $commonDir) { $toolsRoot = Split-Path -Parent $commonDir }
}
$engineCargo = Join-Path $toolsRoot '.tools\cargo'
if (-not (Test-Path -LiteralPath (Join-Path $engineCargo 'bin\cargo.exe'))) {
    throw 'No local Rust toolchain exists. Install Rust with rustup, or use the repository setup instructions.'
}
$env:RUSTUP_HOME = Join-Path $toolsRoot '.tools\rustup'
$env:CARGO_HOME = $engineCargo
$env:PATH = "$engineCargo\bin;$env:PATH"
