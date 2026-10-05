// Impostors' frames (decisions 048 and 060), which the meshes' and the
// shadow maps' shaders share: the grid of frames, each one's direction over
// the sphere or the upper hemisphere, and the records of baked impostors.
const IMPOSTOR_FRAMES: u32 = 8u;
// The flag of records whose frames cover only the upper hemisphere, above
// their texels across a frame.
const IMPOSTOR_HEMISPHERE: u32 = 0x10000u;

struct ImpostorRecord {
    // Mesh-space center and radius of the sphere every frame frames.
    sphere: vec4<f32>,
    // Mesh-space half extents of the mesh's bounds, and the layer.
    extent: vec3<f32>,
    layer: u32,
    // The material's emissive color (glTF), linear, and the flags.
    emissive: vec3<f32>,
    flags: u32,
}

// Octahedral coordinates in [-1, 1] of unit direction `d` about +y: the
// upper hemisphere fills the central diamond, the lower folds into the
// corners.
fn impostor_octahedral(d: vec3<f32>) -> vec2<f32> {
    let n = d / (abs(d.x) + abs(d.y) + abs(d.z));
    let sign = select(vec2<f32>(-1.0), vec2<f32>(1.0), n.xz >= vec2<f32>(0.0));
    return select(n.xz, (vec2<f32>(1.0) - abs(n.zx)) * sign, n.y < 0.0);
}

// The unit direction of octahedral coordinates `p`.
fn impostor_direction(p: vec2<f32>) -> vec3<f32> {
    var d = vec3<f32>(p.x, 1.0 - abs(p.x) - abs(p.y), p.y);
    let fold = max(-d.y, 0.0);
    d.x += select(fold, -fold, d.x >= 0.0);
    d.z += select(fold, -fold, d.z >= 0.0);
    return normalize(d);
}

// Hemi-octahedral coordinates in [-1, 1] of unit direction `d` about +y:
// the upper hemisphere's diamond turned to fill the square, its edges the
// horizon, from which views from below look.
fn impostor_hemi_octahedral(d: vec3<f32>) -> vec2<f32> {
    let above = vec3<f32>(d.x, max(d.y, 0.0), d.z);
    let n = above / max(abs(above.x) + above.y + abs(above.z), 1.0e-6);
    return vec2<f32>(n.x + n.z, n.x - n.z);
}

// The unit direction of hemi-octahedral coordinates `p`.
fn impostor_hemi_direction(p: vec2<f32>) -> vec3<f32> {
    let x = 0.5 * (p.x + p.y);
    let z = 0.5 * (p.x - p.y);
    return normalize(vec3<f32>(x, 1.0 - abs(x) - abs(z), z));
}

// The grid coordinates in [-1, 1] of direction `d`, over the hemisphere
// or the sphere.
fn impostor_grid(d: vec3<f32>, hemisphere: bool) -> vec2<f32> {
    if hemisphere {
        return impostor_hemi_octahedral(d);
    }
    return impostor_octahedral(d);
}

// Frame `frame`'s direction toward its viewer: the frames sit on the grid
// points of the octahedral map, or the hemisphere's, edges and corners
// included.
fn impostor_frame_direction(frame: u32, hemisphere: bool) -> vec3<f32> {
    let cell = vec2<f32>(f32(frame % IMPOSTOR_FRAMES), f32(frame / IMPOSTOR_FRAMES));
    let p = cell / f32(IMPOSTOR_FRAMES - 1u) * 2.0 - 1.0;
    if hemisphere {
        return impostor_hemi_direction(p);
    }
    return impostor_direction(p);
}

// Right and up of an image seen from direction `d`: up follows +y unless `d`
// nearly does.
struct ImpostorBasis {
    right: vec3<f32>,
    up: vec3<f32>,
}

fn impostor_basis(d: vec3<f32>) -> ImpostorBasis {
    var hint = vec3<f32>(0.0, 1.0, 0.0);
    if abs(d.y) > 0.999 {
        hint = vec3<f32>(0.0, 0.0, -sign(d.y));
    }
    let right = normalize(cross(hint, d));
    return ImpostorBasis(right, cross(d, right));
}
