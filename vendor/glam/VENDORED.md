# Vendored glam 0.30.9

Upstream: glam 0.30.9 from crates.io (MIT OR Apache-2.0, license files kept),
with its `src/`, `README.md` and `ATTRIBUTION.md`. Tests, benchmarks, templates
and development dependencies are omitted; the workspace patches crates.io's
`glam` with this copy (`[patch.crates-io]` in the root `Cargo.toml`).

The only source change is in `src/f32/math.rs`, module `std_math`: `sin` and
`sin_cos` call `engine_math::sin` and `engine_math::sin_cos` instead of the
standard library, and `Cargo.toml` depends on `engine_math`. Every `f32`
rotation constructor (`from_rotation_x/y/z`, `from_axis_angle`, `from_euler`,
`Vec2::from_angle`, the matrix and affine equivalents) and `Quat::slerp` then
use the engine's branch-free, deterministic trigonometry (`docs/math.md`);
the SSE2 and NEON `Quat::slerp` keep glam's own SIMD sine (`m128_sin`),
while the scalar and wasm32 ones call the patched `sin`.

To upgrade: copy the new release's `src`, license files, `README.md` and
`Cargo.toml` (dropping `[[test]]`, `[[bench]]` and dev-dependencies), reapply the
two-function change and the dependency, and run `cargo xtask test`: the
`engine_math` and `engine_core::math` tests check that glam uses the engine
routine. Review the engine's numerical proofs as for any glam upgrade.
