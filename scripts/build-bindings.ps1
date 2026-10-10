# Builds the browser clients' release (web/pkg) and development (web/pkg-dev)
# Wasm bindings as `cargo xtask lint` does, then type-checks the web host.
# The clients are web/src/generated/clients.json's.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'activate.ps1')
$clients = Get-Content -Raw (Join-Path $root 'web/src/generated/clients.json') | ConvertFrom-Json
$packages = $clients | ForEach-Object { '-p'; $_.package }
# Clients declare no features of their own: development builds enable the engine's.
$development = 'engine/dev'
Push-Location $root
try {
    foreach ($variant in @(@{ out = 'web/pkg'; features = @() }, @{ out = 'web/pkg-dev'; features = @('--features', $development) })) {
        $arguments = @('build') + $packages + @('--target', 'wasm32-unknown-unknown', '--release', '--locked') + $variant.features
        cargo @arguments
        if ($LASTEXITCODE) { throw 'cargo build failed' }
        foreach ($client in $clients) {
            wasm-bindgen "target/wasm32-unknown-unknown/release/$($client.wasm).wasm" --target web --out-dir $variant.out --out-name $client.wasm
            if ($LASTEXITCODE) { throw "wasm-bindgen $($client.wasm) failed" }
        }
    }
    npm run check --prefix web
    if ($LASTEXITCODE) { throw 'web check failed' }
    'bindings built and checked'
} finally { Pop-Location }
