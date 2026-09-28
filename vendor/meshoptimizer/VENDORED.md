# Vendored meshoptimizer 1.3

Upstream: meshoptimizer v1.3 (commit 9e1f07b159d3cb777f1c67ed31fc11fd117986f4,
MIT, `LICENSE.md` kept) from <https://github.com/zeux/meshoptimizer>, with its
`src/` and the `demo/clusterlod.h` single-header cluster LOD library. Nothing is
changed.

Only native tools compile it: `tools/vgbuild` builds virtual geometry
(docs/adr-044-virtual-geometry.md) offline with `clusterlod`'s cluster DAG and
bounding volume hierarchy and `meshopt_computeClusterBounds`. The browser never
runs it; games and the engine read the files it writes.

To upgrade: replace `src/`, `demo/clusterlod.h` and `LICENSE.md` with a new
release's, update the version above, and run `cargo xtask test`: the
`engine_vgbuild` tests build hierarchies and check their selection invariants.
Review `clodConfig`'s fields against the `#[repr(C)]` mirror in
`tools/vgbuild/src/ffi.rs`.
