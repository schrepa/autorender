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
- For bulk work, batch four angles into a `Vec4` for `sin_cos4`, most easily
  with `math::for_each4` (below).
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

## Minimum, maximum and clamping

Use `math::min`, `math::max` and `math::clamp` for `f32`. `clippy.toml` disallows
`f32::min`, `f32::max` and `f32::clamp` in workspace code. Rust lowers them (and a
plain `if b < a { b } else { a }`) to a float `select`, and V8 and SpiderMonkey
compile a float `select` to a data-dependent branch, which mispredicts when values
differ per object. The engine versions compile to one SIMD instruction in Wasm
(`f32x4.pmin`/`pmax` on a splat) and to the same comparison natively, so every
target gives the same bits:

| Function | Result |
| --- | --- |
| `min(a, b)` | `b` when `b < a`, otherwise `a` |
| `max(a, b)` | `b` when `a < b`, otherwise `a` |
| `clamp(x, low, high)` | `min(max(x, low), high)` |

NaN in either argument and equal values such as `0.0` and `-0.0` return `a`,
unlike `f32::min`, which prefers the non-NaN argument. `clamp` keeps NaN `x` and
does not panic when `low > high`: the result is then `high`. glam's `Vec3A` and
`Vec4` `min`, `max` and `clamp` already use `pmin`/`pmax`; its scalar-field `Vec3`
methods compare lane by lane and branch, so prefer `Vec3A` in hot loops.

## Four lanes at a time

`math::for_each4(items, kernel)` calls `kernel` with up to four items at a time
(`Lanes4`): load one `f32` per lane with `Lanes4::load` (absent lanes read zero),
compute on `Vec4` lanes, and write each item back with `Lanes4::for_each`. It
works on any iterator, including `Query::iter_mut`:

```rust
math::for_each4(query.iter_mut(), |lanes| {
    let angle = lanes.load(|(spin, _)| spin.speed) * t;
    let (sin, cos) = math::sin_cos4(angle * 0.5);
    lanes.for_each(|lane, (_, transform)| {
        transform.rotation = Quat::from_xyzw(0.0, sin[lane], 0.0, cos[lane]);
    });
});
```

`sin_cos4` gives every lane the scalar function's bits, and glam's rotation
constructors take `sin_cos` of half the angle, so a kernel written with the same
operations reproduces the one-at-a-time results exactly. The Stress Lab animates
this way (`move_lanes`, tested bit for bit against the one-at-a-time version).
In-page A/B in Firefox at 90,000 moving objects, three loads: gameplay per tick
3.14-3.27 ms one at a time, 2.16-2.36 ms four at a time (about 11 ns less per
object).
