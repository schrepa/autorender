$ErrorActionPreference = 'Stop'
$comparisonRoot = $PSScriptRoot
$repositoryRoot = (Resolve-Path (Join-Path $comparisonRoot '..\..')).Path
$env:CARGO_HOME = Join-Path $repositoryRoot '.tools\cargo'
$env:RUSTUP_HOME = Join-Path $repositoryRoot '.tools\rustup'
$env:PATH = "$(Join-Path $env:CARGO_HOME 'bin');$env:PATH"
$cargo = Join-Path $env:CARGO_HOME 'bin\cargo.exe'
$bindgen = Join-Path $repositoryRoot '.tools\wasm-bindgen-0.2.127-x86_64-pc-windows-msvc\wasm-bindgen.exe'
if (-not (Test-Path -LiteralPath $cargo)) { throw "cargo missing at $cargo" }
if (-not (Test-Path -LiteralPath $bindgen)) { throw "wasm-bindgen 0.2.127 missing at $bindgen" }

Push-Location $comparisonRoot
try {
    & $cargo build --locked --release --target wasm32-unknown-unknown
    if ($LASTEXITCODE -ne 0) { throw 'Bevy Wasm build failed' }
    & $bindgen --target web --out-dir public/pkg --out-name bevy_stress_comparison target/wasm32-unknown-unknown/release/bevy-stress-comparison.wasm
    if ($LASTEXITCODE -ne 0) { throw 'wasm-bindgen failed' }
    Write-Host 'Built Bevy comparison at public/index.html'
} finally {
    Pop-Location
}
