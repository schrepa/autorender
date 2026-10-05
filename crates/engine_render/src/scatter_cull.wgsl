// Placed objects (decision 057): each frame, every object's bounding sphere
// against each view's frustum (the camera's, then those of the shadow maps
// drawn this frame), and the level of detail it draws at, chosen from the
// camera in every view so shadows match what casts them, by the same rule
// as GPU presentation's (`presentation.wgsl`): the coarsest level whose
// geometric error, projected at the object's distance, stays within the
// pixel error. Visible objects append their pose index to their draw
// group's list for that view and level and count in its indirect draw.
// Where the camera sees an object's impostor within the pixel error (on
// frames that draw impostors, decision 048), it appends its pose and
// impostor instead, drawn as a card; shadow maps leave it out.

struct Frame {
    // The camera's eye (world) and pixels per world unit at unit distance
    // over the pixel error; 0 keeps every object at its finest level.
    eye: vec4<f32>,
    // Objects, draw groups, levels each group lists, and views.
    counts: vec4<u32>,
    // The IDs and draws each view's lists take, and whether the camera's
    // view draws impostors.
    strides: vec4<u32>,
}

// A view: its frustum planes (world, inside positive; a shadow map's depth
// planes pass everything), the radius objects need to draw in it (x), and
// the geometric error its texels hide (y, world units; 0 for the camera).
struct View {
    planes: array<vec4<f32>, 6>,
    limits: vec4<f32>,
}

// One object: its bounding sphere's center (world) and radius, its draw
// group and scale, and the terrain it stands on and how far its base sinks
// (`scatter_snap.wgsl`).
struct Record {
    center: vec3<f32>,
    radius: f32,
    group: u32,
    scale: f32,
    terrain: u32,
    sink: f32,
}

// A draw group: where its objects' lists start, how many objects it has,
// its levels of detail, how many of them the camera draws (coarser ones
// serve shadow maps alone), their errors (levels 1 to 4, mesh units), and
// its impostor's error (mesh units; 0 without one) and record.
struct Group {
    ids: u32,
    count: u32,
    levels: u32,
    camera: u32,
    errors: vec4<f32>,
    impostor: f32,
    record: u32,
    pad: vec2<u32>,
}

// An indexed indirect draw, counting its instances.
struct DrawArgs {
    index_count: u32,
    instance_count: atomic<u32>,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

// A non-indexed indirect draw, counting its instances.
struct CardArgs {
    vertex_count: u32,
    instance_count: atomic<u32>,
    first_vertex: u32,
    first_instance: u32,
}

// The draw group of slots no object holds.
const DEAD: u32 = 0xffffffffu;

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> records: array<Record>;
@group(0) @binding(2) var<storage, read> groups: array<Group>;
@group(0) @binding(3) var<storage, read_write> ids: array<u32>;
@group(0) @binding(4) var<storage, read_write> draws: array<DrawArgs>;
@group(0) @binding(5) var<storage, read> views: array<View>;
// The camera's impostor cards: each one's pose and impostor record.
@group(0) @binding(6) var<storage, read_write> cards: array<vec2<u32>>;
@group(0) @binding(7) var<storage, read_write> card_args: CardArgs;

// One object (x) in one view (y).
@compute @workgroup_size(64)
fn cull_scatter(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= frame.counts.x {
        return;
    }
    let record = records[id.x];
    if record.group == DEAD {
        return;
    }
    let view = &views[id.y];
    if record.radius < (*view).limits.x {
        return;
    }
    for (var plane = 0u; plane < 6u; plane++) {
        let bound = (*view).planes[plane];
        if dot(bound.xyz, record.center) + bound.w < -record.radius {
            return;
        }
    }
    let group = groups[record.group];
    var level = 0u;
    if frame.eye.w > 0.0 {
        let away = distance(record.center, frame.eye.xyz);
        // The camera draws an impostor its texels show within the error,
        // as a card, which casts nothing.
        let impostor = group.impostor * record.scale;
        if frame.strides.z != 0u && impostor > 0.0 && impostor * frame.eye.w + record.radius <= away {
            if id.y == 0u {
                let card = atomicAdd(&card_args.instance_count, 1u);
                cards[card] = vec2<u32>(id.x, group.record);
            }
            return;
        }
        // The camera's level among those it draws, or a coarser one whose
        // error the view's texels hide.
        for (var l = 1u; l < group.levels; l++) {
            let error = group.errors[l - 1u] * record.scale;
            let seen = l < group.camera && error * frame.eye.w + record.radius <= away;
            if seen || error <= (*view).limits.y {
                level = l;
            }
        }
        // Shadow maps draw authored levels (all of which the camera draws)
        // one coarser than the camera: a shadow shows less than its caster.
        if id.y != 0u && group.camera == group.levels {
            level = min(level + 1u, group.levels - 1u);
        }
    }
    let draw = id.y * frame.strides.y + record.group * frame.counts.z + level;
    let slot = atomicAdd(&draws[draw].instance_count, 1u);
    ids[id.y * frame.strides.x + group.ids + level * group.count + slot] = id.x;
}
