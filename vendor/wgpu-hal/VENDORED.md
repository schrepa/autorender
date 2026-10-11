# Vendored wgpu-hal 30.0.1

Upstream: wgpu-hal 30.0.1 from crates.io (MIT OR Apache-2.0, license files
kept), with its `src/`, `build.rs` and `README.md`. Examples, their manifest
entries and development dependencies are omitted; the workspace patches
crates.io's `wgpu-hal` with this copy (`[patch.crates-io]` in the root
`Cargo.toml`).

The one change: on D3D12, DXC's output is also kept on disk, so a later
process skips compiling the shaders an earlier one compiled (about 12–17 s of
an 18–21 s Metro start on the native host, every launch). Upstream keeps it
in memory only, for the device's lifetime.

- `src/dx12/disk_cache.rs` (new): the cache. Each file holds its whole key
  (DXC's version and commit, this crate's version, the debug and 16-bit
  arguments, the source name, entry point, target and HLSL source) before the
  DXIL, and is used only when the key matches byte for byte. Files are
  renamed into place whole, so processes share the folder; the least
  recently used go beyond 512 MiB. `ENGINE_DX12_SHADER_CACHE` names the
  folder or turns it `off`; the default is `wgpu-dx12-shader-cache` in the
  system's temporary folder.
- `src/dx12/device.rs`: between the in-memory cache and DXC, a disk hit
  hands back the DXIL (`CompiledShader::Precompiled`), and a compile is
  stored.
- `src/dx12/shader_compilation.rs`: a dynamically loaded DXC records its
  version and commit (`IDxcVersionInfo2`); without a commit, and with FXC or
  static DXC, nothing is cached.
- `src/dx12/mod.rs`: the module. `Cargo.toml`: `dx12` adds the `windows`
  feature `Win32_System_Com` (to free DXC's commit string).

To upgrade: copy the new release's `src`, `build.rs`, license files,
`README.md` and `Cargo.toml` (dropping `[[example]]` entries and
development dependencies), reapply the four changes above, and check that
the key still names every argument `compile_dxc` passes DXC. A version change
alone invalidates every file (the key holds the crate's version).
