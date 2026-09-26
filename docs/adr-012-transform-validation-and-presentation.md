# Decision 012: bounded transform validation and dense presentation

Status: adopted. Extends decisions 006, 008 and 011. Adds an explicit small-angle
rotation approximation, superseding decision 011's unchanged interpolation
arithmetic. Authoritative transforms, frame boundaries and exact agent captures
retain their contracts. Browser measurements must establish the timing benefit;
this decision makes no frame-rate claim.

[Decision 014](adr-014-gpu-presentation-and-culling.md) evaluates ordinary
interpolated browser frames on the GPU with this rotation policy. Its shader
admits only the common path below and displays the current pose outside it;
tests keep the shader's constants equal to these bounds.

## Decision

Use a sufficient acceptance bound for ordinary composed matrices, normalized
linear rotation interpolation for small angles, one dense parent-first
presentation pass, and compact private matrix storage. Keep the existing
general validation path for values outside the proven bound.

### Matrix validation

Every candidate must first be finite and exactly affine. Let `S` be the largest
absolute coefficient of its linear basis and `d` the existing computed raw
determinant. The common path requires:

- `2^-10 <= S <= 2^20`;
- every translation coefficient has magnitude at most `2^20`;
- `abs(d) >= 2^-9 * fl(S * S * S)`;
- the configured normal-conditioning limit is at least the proven bound, 6200.

These are sufficient conditions for the existing predicate to accept, not new
admission tolerances. With `epsilon = f32::EPSILON`, the raw determinant error is
bounded by `32 * epsilon * S^3`. Rounding the scale cube and normalizing the basis
leave the old computed normalized determinant above `2^-10`. Normalized
cofactors are bounded by 2 and absolute column sums by 3, giving the old normal
inverse-norm expression a rounded bound below 6200, within its limit of 10,000.

The lower scale bound keeps the accepted determinant magnitude and cubed scale
well above underflow. In pinned glam's general inverse, the affine determinant contains
only degree-three basis terms; translation affects inverse numerators. Under
the common gate its magnitude exceeds `2^-10 * S^3 >= 2^-40`. Its reciprocal
and the bounded cofactors therefore cannot overflow. The weaker inverse-only
shortcut retains its `2^-12` margin: its determinant exceeds `2^-14 * S^3`,
and inverse coefficients remain below `2^109`. Detailed error bounds live beside
the implementation in `hierarchy/validation.rs`.

The common path reads the basis as SIMD `Vec3A` columns instead of extracting
a scalar `Mat3`. Pinned glam evaluates `Vec3A` cross products and three-term
dot products with the scalar `Vec3` expressions and summation order on both its
Wasm SIMD and SSE2 backends, and absolute maxima are exact, so `S` and `d` are
bit-identical; tests compare them directly.

Traversals inline a vector gate instead of per-lane tests: every coefficient
within `2^20` (which also proves finiteness; NaN fails), the exact affine row by
four vector equalities, then the same raw determinant, scale and margin. A pass
is exactly a common-path acceptance. Anything else runs the original predicate
out of line — its finite/affine test, the common path, then the conditioned
predicate below as a separate cold function — so the hot loops carry no
cofactor or inverse code and decisions are unchanged. Local matrices come from
`Transform::matrix`, which performs glam's `from_scale_rotation_translation`
operations lane by lane without extracting and reinserting scalar lanes; tests
compare its bits (NaN payloads aside) with glam's. The change scan evaluates a
changed leaf whose parent is not awaiting evaluation immediately, while its pose
is hot, instead of in the parent-first pass (`hierarchy.scan_evaluated_nodes`). Before composition, a local pose
needs only its rotation-norm check: a non-finite scale or translation makes a
whole column of its matrix non-finite (`0 * inf` is NaN), which every parent
product keeps non-finite, so the composed-matrix check rejects it exactly as an
early local check would. Unchanged-output tests compare finite matrices bitwise
with four vector equalities plus sign masks (equal non-NaN floats with equal
signs are bit-identical, which also separates the two zeros).

Outside these gates, preserve the original cofactor, general-inverse finiteness
and normalized-conditioning checks. This includes tiny/large coefficients,
near-singular shear and cases where an affine-specific inverse would produce a
different result. Valid endpoints do not prove a valid interpolation midpoint;
every newly composed display matrix still passes the shared validator.

### Small-angle rotation policy

Normalize both quaternion endpoints. When their absolute dot product is at
least `0.9995f32`, use shortest-arc normalized linear interpolation (NLERP).
For larger rotations, retain glam's spherical interpolation and final
normalization. Alpha zero still returns the exact previous pose, and alpha one
uses exact-current presentation. Translation and scale arithmetic is unchanged.

This avoids the trigonometric path for small fixed-tick rotations. Unity
Mathematics uses the same dot threshold in its `slerp` implementation.
[Unity source](https://github.com/Unity-Technologies/Unity.Mathematics/blob/master/src/Unity.Mathematics/quaternion.cs).
NLERP follows the same shortest arc as SLERP but changes its angular speed.
[Kapoulkine's derivation](https://zeux.io/2015/07/23/approximating-slerp/).

For ideal unit endpoints, let `theta = acos(abs(dot))`. The physical angular
deviation is `2 * abs(atan2(t*sin(theta), 1-t+t*cos(theta)) - t*theta)`.
At the rounded f32 threshold, numerical evaluation of its extrema gives about
`1.015e-6` radians, at fractions near 0.2113 and 0.7887. This is the geometric
approximation error, excluding floating-point normalization, dot-product and
arithmetic errors. `1e-5` radians is the local-rotation regression target against
a stable f64 reference, not a proved total bound or a world-space error guarantee.

Fractional presentation is intentionally no longer bit-equivalent to the prior
SLERP path. Small differences can affect a pose near the existing numerical
fallback boundary. The shared validator and branch fallback still apply to the
actual composed matrix; authoritative simulation and default exact captures do
not use the approximation.

### Presentation and storage

Dense interpolation previously traversed histories to stage display inputs,
then traversed the hierarchy to compose them. The parent-first pass now reads
the existing source-to-history hint, accepts it when the record's own node index
names the same node (a pair every topology rebuild rewrites), and otherwise
resolves the full generational entity through the membership index. It resets each display
slot immediately before use, avoiding a separate dense reset/scatter pass and
touched-index list. Sparse branch traversal remains available; changing from
dense to sparse clears the former dense state before reuse.

Reset flags still propagate through unmarked ancestors. Invalid ancestry,
numerical fallback, inherited parent motion and direct presented-component
repair retain their control flow; the rotation policy above determines the
fractional pose supplied to composition.

Private hierarchy and display storage keep `Mat4` payloads with a separate
validity flag instead of `Option<Mat4>`. Invalid payloads remain initialized but
are never exposed or consumed as valid matrices. Matrices, revisions, local
poses, traversal links/flags and history intervals are separate arrays in one
node index space. At 90,000 moving nodes the earlier combined records made
presentation stream about 500 bytes per node, mostly fields it never read;
native timing showed the pass bound by that traffic (removing all of its
arithmetic did not shorten it). The split layout reads about 220 bytes. Public matrix getters still
return `Option<Mat4>`; snapshots remain owned, and extraction remains the only
operation that updates the last presented pose. Reported scratch bytes use
actual retained capacities and record sizes.

## Alternatives and evidence

glam's `Affine3A` represents affine transforms, including shear, but its SIMD
layout is still 64 bytes. A type substitution would not reduce matrix storage;
its different arithmetic also needs separate numerical review.
[Pinned layout test](https://github.com/bitshifter/glam-rs/blob/0.30.9/tests/affine3.rs),
[affine implementation](https://github.com/bitshifter/glam-rs/blob/0.30.9/src/f32/affine3a.rs).

Ozz uses separate component arrays for local animation and matrices for model
space. That supports choosing layouts for their processing patterns, but does
not establish a benefit from repacking this engine's public Bevy components
each frame. This change keeps those components and removes redundant private
work first. [Ozz runtime layout](https://guillaumeblanc.github.io/ozz-animation/documentation/animation_runtime/).

The validation proof depends on pinned glam 0.30.9's arithmetic. Revisit it on
dependency or floating-point compilation changes.
[Scalar inverse](https://github.com/bitshifter/glam-rs/blob/0.30.9/src/f32/scalar/mat4.rs),
[Wasm SIMD inverse](https://github.com/bitshifter/glam-rs/blob/0.30.9/src/f32/wasm32/mat4.rs).

Validation should compare the optimized predicate with the previous predicate
at bound edges and under seeded affine/shear inputs, and preserve dense/sparse
interpolation, reset and lifecycle regressions. Rotation checks should cover
endpoint orientations, both quaternion signs, threshold edges and accepted
near-unit inputs against a stable f64 angular reference. Compare matched browser captures
with identical workload metadata before reporting a speedup.
