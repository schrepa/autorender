struct ShownVaryings {
    @invariant @builtin(position) clip_position: vec4<f32>,
    @location(0) normal_u: vec4<f32>,
    @location(1) world_v: vec4<f32>,
    @location(2) @interpolate(flat) material: vec4<u32>,
    @location(6) @interpolate(flat) object: vec4<u32>,
}

struct ShownExtraVaryings {
    @invariant @builtin(position) clip_position: vec4<f32>,
    @location(0) normal_u: vec4<f32>,
    @location(1) world_v: vec4<f32>,
    @location(2) @interpolate(flat) material: vec4<u32>,
    @location(4) color: vec4<f32>,
    @location(5) uv1: vec2<f32>,
    @location(6) @interpolate(flat) object: vec4<u32>,
}


struct ObjectDepthVaryings {
    @invariant @builtin(position) clip_position: vec4<f32>,
    @location(0) normal_u: vec4<f32>,
    @location(1) world_v: vec4<f32>,
    @location(2) @interpolate(flat) material: vec4<u32>,
    @location(6) @interpolate(flat) object: vec4<u32>,
}

struct ObjectDepthExtraVaryings {
    @invariant @builtin(position) clip_position: vec4<f32>,
    @location(0) normal_u: vec4<f32>,
    @location(1) world_v: vec4<f32>,
    @location(2) @interpolate(flat) material: vec4<u32>,
    @location(4) color: vec4<f32>,
    @location(5) uv1: vec2<f32>,
    @location(6) @interpolate(flat) object: vec4<u32>,
}


struct KindPoint {
    point: SurfacePoint,
    pixel: vec2<f32>,
}


fn object_fragment(values: KindFragment, object: vec4<u32>) -> KindFragment {
    var placed = values;
    placed.kind = object.w;
    placed.object = object.y;
    return placed;
}

fn shown_object(input: ShownVaryings) -> KindFragment {
    return object_fragment(fragment_values(input.clip_position, input.normal_u, input.world_v,
        input.material), input.object);
}

fn shown_extra_object(input: ShownExtraVaryings) -> KindFragment {
    return object_fragment(extra_values(input.clip_position, input.normal_u, input.world_v,
        input.material, input.color, input.uv1), input.object);
}



@fragment
fn fs_kind_object_masked_depth(input: ObjectDepthVaryings) {
    masked_depth(object_fragment(fragment_values(input.clip_position, input.normal_u,
        input.world_v, input.material), input.object));
}

@fragment
fn fs_kind_object_masked_depth_extra(input: ObjectDepthExtraVaryings) {
    masked_depth(object_fragment(extra_values(input.clip_position, input.normal_u,
        input.world_v, input.material, input.color, input.uv1), input.object));
}

@fragment
fn fs_kind_object_lit(input: ShownVaryings, @builtin(front_facing) front: bool) -> Outputs {
    let shaded = shade(shown_object(input), front);
    return outputs(shaded.point, shaded.shaded);
}

@fragment
fn fs_kind_object_lit_extra(input: ShownExtraVaryings, @builtin(front_facing) front: bool)
        -> Outputs {
    let shaded = shade(shown_extra_object(input), front);
    return outputs(shaded.point, shaded.shaded);
}

@fragment
fn fs_kind_object_blended(input: ShownVaryings, @builtin(front_facing) front: bool) -> Outputs {
    let shaded = shade(shown_object(input), front);
    return blended_outputs(shaded.point, shaded.shaded);
}

@fragment
fn fs_kind_object_blended_extra(input: ShownExtraVaryings, @builtin(front_facing) front: bool)
        -> Outputs {
    let shaded = shade(shown_extra_object(input), front);
    return blended_outputs(shaded.point, shaded.shaded);
}
