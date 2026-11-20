struct Run {
    count: u32,
    gap: f32,
}

struct Node {
    low: array<f32, 3>,
    high: array<f32, 3>,
    start: u32,
    count: u32,
}

struct Triangle {
    corner: array<f32, 3>,
    toward_b: array<f32, 3>,
    toward_c: array<f32, 3>,
    two_sided: u32,
}

struct Point {
    xyz: array<f32, 3>,
}

struct Item {
    eye: u32,
    start: u32,
    count: u32,
}

const NONE: u32 = 0xffffffffu;

@group(0) @binding(0) var<uniform> run: Run;
@group(0) @binding(1) var<storage, read> nodes: array<Node>;
@group(0) @binding(2) var<storage, read> triangles: array<Triangle>;
@group(0) @binding(3) var<storage, read> targets: array<Point>;
@group(0) @binding(4) var<storage, read> eyes: array<Point>;
@group(0) @binding(5) var<storage, read> items: array<Item>;
@group(0) @binding(6) var<storage, read_write> found: array<u32>;

fn vector(held: array<f32, 3>) -> vec3<f32> {
    return vec3<f32>(held[0], held[1], held[2]);
}

fn crossed(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

fn dotted(a: vec3<f32>, b: vec3<f32>) -> f32 {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

fn hit(index: u32, origin: vec3<f32>, ray: vec3<f32>, reach: f32) -> bool {
    let held = triangles[index];
    let toward_b = vector(held.toward_b);
    let toward_c = vector(held.toward_c);
    let p = crossed(ray, toward_c);
    let det = dotted(toward_b, p);
    if det == 0.0 || (held.two_sided == 0u && det < 0.0) {
        return false;
    }
    let inverse = 1.0 / det;
    let s = origin - vector(held.corner);
    let u = dotted(s, p) * inverse;
    if !(u >= 0.0 && u <= 1.0) {
        return false;
    }
    let q = crossed(s, toward_b);
    let v = dotted(ray, q) * inverse;
    if v < 0.0 || u + v > 1.0 {
        return false;
    }
    let t = dotted(toward_c, q) * inverse;
    return t > 0.0 && t < reach;
}

fn enters(index: u32, origin: vec3<f32>, inverse: vec3<f32>, reach: f32) -> bool {
    var near = 0.0;
    var far = reach;
    for (var axis = 0u; axis < 3u; axis += 1u) {
        var enter = (nodes[index].low[axis] - origin[axis]) * inverse[axis];
        var leave = (nodes[index].high[axis] - origin[axis]) * inverse[axis];
        if enter > leave {
            let swapped = enter;
            enter = leave;
            leave = swapped;
        }
        if enter > near {
            near = enter;
        }
        if leave < far {
            far = leave;
        }
    }
    return near <= far;
}

fn blocked(origin: vec3<f32>, target_at: vec3<f32>, reach: f32) -> bool {
    let ray = target_at - origin;
    let inverse = vec3<f32>(1.0) / ray;
    var stack: array<u32, 64>;
    var depth = 1u;
    while depth > 0u {
        depth -= 1u;
        let index = stack[depth];
        if !enters(index, origin, inverse, reach) {
            continue;
        }
        let start = nodes[index].start;
        let count = nodes[index].count;
        if count == 0u {
            stack[depth] = start;
            stack[depth + 1u] = start + 1u;
            depth += 2u;
            continue;
        }
        for (var held = start; held < start + count; held += 1u) {
            if hit(held, origin, ray, reach) {
                return true;
            }
        }
    }
    return false;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= run.count {
        return;
    }
    let item = items[id.x];
    let eye = vector(eyes[item.eye].xyz);
    var answer = NONE;
    for (var offset = 0u; offset < item.count; offset += 1u) {
        let aim = vector(targets[item.start + offset].xyz);
        let toward = aim - eye;
        let span = sqrt(toward.x * toward.x + toward.y * toward.y + toward.z * toward.z);
        if span <= run.gap || !blocked(eye, aim, 1.0 - run.gap / span) {
            answer = offset;
            break;
        }
    }
    found[id.x] = answer;
}
