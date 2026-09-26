# Decision 016: Deferred leaf globals

Status: adopted. Refines decisions 012 (transform validation) and 013 (canonical
poses). Public reads, validity decisions and matrix bits are unchanged; the
boundary no longer composes world matrices that only a later read would use.

## Context

With GPU presentation ([decision 014](adr-014-gpu-presentation-and-culling.md))
the GPU composes displayed poses from local poses, and in the 90,000-object
all-moving Stress Lab nothing reads the leaves' authoritative world matrices
each tick except collision (1,000 colliders) and a few gameplay queries. Yet
every boundary composed, validated and stored all 90,000 of them: in Firefox 156
about 1.6 ms of the 4.1 ms propagation per tick (validation 0.65–0.7 ms, the
matrix product about 0.3 ms, the compare-and-store commit about 0.6 ms, measured
by knocking each out in page). Validity itself must stay eager: GPU inputs,
invalid counts and interpolation observe it every tick.

## Decision

A changed node without children whose parent's global is stored (or a root) is
*certified* instead of composed when its parent's `Certificate` admits its local
pose. The certificate proves, from bounds computed once per stored parent and a
few scalar operations per child, that `parent * local.matrix()` passes the common
admission gate of decision 012. Such a leaf is published as valid with a new
output revision, and its matrix is *deferred*: `GlobalTransforms::get`,
`get_indexed_with_revision`, `iter` and the engine's internal reads compose the
parent's stored global with the cached local on demand, the same operations and
therefore the same bits eager evaluation would have stored. Anything the
certificate does not prove (nodes with children, parents without a stored
global, non-unit rotations, extreme or nearly singular scales, translations near
the coefficient limit, non-finite values) takes the unchanged eager path.

The certificate of a stored parent P holds four f32 bounds (see
`hierarchy/validation.rs`): the largest row 2-norm of P's linear part scaled by
the rotation's possible stretch, a lower bound on |det| of the composed linear
part per unit scale product, the gate's determinant threshold per unit cube of the
largest scale, and the largest child translation that keeps composed
coefficients within 2^20. glam's rotation matrix of a quaternion with
|‖q‖² − 1| ≤ 10⁻³ is exactly n·R + (1 − n)·I (n = ‖q‖²), so its singular values and
determinant are bounded; entry, product and determinant rounding are bounded
with the standard error analysis, and every bound is widened by at least 2^-12.
Tests compose certified random and adversarial children (anisotropic and
mirrored parents, near-threshold flat boxes, stretched quaternions at the 10⁻³
tolerance, far translations) and require every one to pass the gate itself; the
certified cases come within 5% of the gate's determinant bound.

## Observable behavior

- Matrices and validity of every read equal eager evaluation bit for bit.
- A deferred leaf takes a new output revision whenever it is re-evaluated (its
  local or an ancestor changed). Revisions may therefore change without a bit
  change, which the revision contract already allowed for rebuilds; consumers
  that cache by revision redo work at most once per such change.
- Reading a deferred leaf costs one matrix product (a few nanoseconds natively).
  Code that reads every global every frame pays that per read; the CPU
  presentation path does so only for nodes it does not blend.
- `hierarchy.deferred_globals` counts certified leaves per boundary;
  `hierarchy.evaluated_nodes` still counts every evaluated node.

## Measurements

Firefox 156 at 120 Hz, the 90,000-object all-moving configuration, in-page A/B
(both paths in one page load, alternating captures): propagation per tick fell
from 3.69 to 2.56 ms and from 4.94 to 3.60 ms in two loads, and main-thread time
per tick by 1.5–1.9 ms. Combined with gathering GPU inputs in the frame that
completes a tick ([decision 015](adr-015-pipelined-fixed-ticks.md)), the Stress
Lab held 119–120 fps at 90,000 and 120,000 objects in page loads where 5% and 0%
of frames exceeded 8.33 ms (frame p95 8.2 and 6.2 ms). See
`artifacts/firefox-120hz-2026-09-25.md`.

## Consequences

Leaf-heavy scenes stop paying for world matrices nobody reads, and the saving
grows with the number of moving leaves. Parents stay eager because their
children need their stored matrices and certificates. A future read-heavy
consumer can still read every global; if that becomes common, an explicit
materialization pass can store deferred matrices without changing this contract.
