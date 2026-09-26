# Optional workspace-local Rust installation. This does not change the user PATH.
# Dot-source from PowerShell: . ./scripts/activate.ps1
$engineRoot = Split-Path -Parent $PSScriptRoot
$engineCargo = Join-Path $engineRoot '.tools\cargo'
if (-not (Test-Path -LiteralPath (Join-Path $engineCargo 'bin\cargo.exe'))) {
    throw 'No local Rust toolchain exists. Install Rust with rustup, or use the repository setup instructions.'
}
$env:RUSTUP_HOME = Join-Path $engineRoot '.tools\rustup'
$env:CARGO_HOME = $engineCargo
$env:PATH = "$engineCargo\bin;$env:PATH"
