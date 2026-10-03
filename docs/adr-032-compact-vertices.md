# Decision 032: Compact vertices and vertex outputs

Status: adopted. Mesh vertices take 24 bytes instead of 32, mesh vertex
shaders write four output slots instead of six, and normals transform with
fewer operations. Rendering is unchanged.

## Context

Dense, varied scenes are bound by geometry: the Stress Lab city averages
about 61 vertices per object against the tower field's 24, and at 90,000
objects roughly half of its opaque pass did not scale with canvas pixels.
Every vertex fetched 32 bytes, rebuilt a rescaled normal matrix and wrote
six interpolated output slots (position, color, normal, world position,
texture coordinates and the flat surface), although color and surface are
constant per object.

## Decision

- **Octahedral normals.** A vertex is its position (three f32), its normal
  as two 16-bit signed fractions on the octahedron |x| + |y| + |z| = 1 with
  the lower half folded over the upper (`gpu_data::octahedral`, decoded by
  `octahedral_normal` in `meshes.wgsl`, now `meshes/vertex.wgsl`), and its texture coordinates (two
  f32): 24 bytes. Directions keep about 0.003 degrees (tested within 0.01
  degrees over 20,000 directions). Positions and texture coordinates keep
  full precision, so large meshes and tiled coordinates are unaffected.
  Shadow passes read positions from the same buffers.
- **Four vertex output slots.** Texture coordinates ride in the w lanes of
  the world normal and world position, and color, opacity and surface pass
  as one flat `vec3<u32>` exactly as slots store them (four 16-bit color
  fractions and the packed surface); fragments unpack them. Public instance
  records read their packed color as `Uint32x2`.
- **Cheaper normal transform.** Normals still transform by the cofactor
  matrix of the basis rescaled to a largest component of 1, but without the
  second rescale: those cofactors stay within [-2, 2], and the final
  normalization already divides by the largest component. The Rust mirror
  of the shader math keeps its extreme-scale and conditioning tests.

## Observable behavior

- Captures are unchanged apart from sub-pixel normal quantization.
- Stress Lab city, 90,000 objects, 10% moving, discrete GPU (headless
  Chrome, ABBA over two loads each, 1508x640 canvas): opaque pass median
  0.687 ms before, 0.541 ms after (-21%); in a second session 0.738 ms
  before and 0.604 ms after (-18%), with shadow, presentation and bloom
  passes unchanged. The integrated GPU was not measured: another page
  occupied it.

## Consequences

Custom vertex layouts must follow the 24-byte format. A second texture
coordinate set or vertex colors would add output lanes; keep additions
packed and flat where values are per object.
