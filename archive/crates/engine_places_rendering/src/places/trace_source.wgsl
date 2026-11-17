struct TraceHit {
    met: bool,
    s: f32,
    light: vec3<f32>,
}

fn trace_holds(front: vec3<f32>, back: vec3<f32>) -> bool {
    return false;
}

fn trace_world(p: vec3<f32>, direction: vec3<f32>, nearest: f32, farthest: f32,
        spread: vec2<f32>) -> TraceHit {
    var hit: TraceHit;
    return hit;
}
