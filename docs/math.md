# Engine math

## Trigonometry

Use `math::sin`, `math::cos`, `math::sin_cos` and `math::sin_cos4` (in
`engine::prelude`, implemented by `crates/engine_math` and `engine_core::math`)
for `f32` angles. glam's `f32` rotation constructors (`Quat::from_rotation_x/y/z`,
`from_axis_angle`, `from_scaled_axis`, `from_euler`, `Vec2::from_angle`, and the
`Mat2`/`Mat3`/`Mat4`/affine equivalents) and `Quat::slerp` use the same routine:
the workspace patches glam 0.30.9 with `vendor/glam`, which changes only its
`sin`/`sin_cos` (see `vendor/glam/VENDORED.md`). `clippy.toml` disallows
`f32::sin`, `f32::cos` and `f32::sin_cos` in workspace code.

```rust
let wave = math::sin(angle);
let (s, c) = math::sin_cos(angle); // one reduction for both
let rotation = Quat::from_rotation_y(angle); // same routine through glam
let (sines, cosines) = math::sin_cos4(Vec4::new(a, b, c, d)); // four lanes
```

### Why

The standard `sinf`/`cosf` in Wasm is musl's (through `compiler_builtins`). It
branches on each angle's magnitude and quadrant, and when angles differ per object
those branches mispredict: on the Stress Lab's per-object angles a call cost
14.5 ns in Firefox and 18.4 ns in Chrome, against 3.1 and 3.8 ns for the same
angles sorted. The engine routine reduces with a round-to-nearest shifter and a
three-part Cody-Waite split of pi/2, evaluates Cephes' degree-7 sine and degree-8
cosine polynomials, and applies the quadrant with integer bit operations: 2.8 ns
(Firefox) and 3.1 ns (Chrome) per call, independent of the angles. Browsers
compile a Wasm `select` between floats as a branch, so data-dependent choices in
hot loops need integer bit patterns or SIMD blends to stay branch-free.

The Stress Lab's animation (five trig evaluations and two quaternion products per
object) measured 35.9-45.7 ns per object in Firefox with the standard functions
and 16.6-17.2 ns with the engine routine; four lanes at a time reached about
10 ns. Details, other options and sources:
`artifacts/trig-research-2026-09-26.md`.

### Accuracy

Tested exhaustively over every `f32` in each range, against the exact value
rounded to `f32`:

| Range | Error |
| --- | --- |
| `\|x\| <= pi` | within 2 ulp (within 1 ulp to `pi/4` on sampled inputs) |
| `\|x\| <= math::REDUCTION_LIMIT` (12,866) | absolute error below `1.2e-7` |
| larger, infinite or NaN | musl's `sincosf` (within 1 ulp; NaN for infinities and NaN) |

Sine is odd and cosine even bit for bit (`sin(-0.0)` is `-0.0`). `sin_cos4`
returns the scalar function's bits in every lane.

### Determinism

The routine uses only IEEE `f32` operations without fused multiply-add, so native
test builds and every browser produce the same bits. The standard functions do
not: native targets use the platform's C runtime, Wasm uses musl. Do not introduce
relaxed-SIMD fused multiply-add in these paths; its rounding depends on hardware.

### Patterns for large populations

- Compute sine and cosine together with `sin_cos` when both are needed.
- For bulk work, batch four angles into a `Vec4` for `sin_cos4`.
- Periodic motion at a fixed step can avoid trigonometry per tick: keep
  `(sin, cos)` per object, rotate it by a precomputed step each tick
  (four multiplies, two adds) and renormalize.
- Objects sharing a frequency can share its `(sin, cos)` per tick and apply their
  own phase with the angle-addition formula.
- Purely visual motion can run in shaders; WGSL only guarantees `2^-11` absolute
  error for `sin`/`cos` on `[-pi, pi]`, so keep simulation on the CPU routine.

### Measuring

With the standard functions, trigonometry grows more expensive as angles grow
(the Stress Lab's gameplay cost about 20% more after its first minute). The engine
routine does not depend on magnitude below `REDUCTION_LIMIT`, but compare
gameplay-heavy workloads at matched scene time anyway.
