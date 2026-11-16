import package::generated::reflections::PYRAMID_GROUP;

@group(1) @binding(0) var far_cells: texture_2d<u32>;
@group(1) @binding(1) var light_level: texture_2d<f32>;
@group(1) @binding(2) var reach_level: texture_2d<f32>;
@group(1) @binding(3) var light_target: texture_storage_2d<rgba16float, write>;
@group(1) @binding(4) var reach_target: texture_storage_2d<rgba16float, write>;

const FAR_ACROSS: i32 = 4;

fn far_blurred(source: texture_2d<f32>, base: vec2<i32>) -> vec4<f32> {
    let size = vec2<i32>(textureDimensions(source));
    var sum = vec4<f32>(0.0);
    for (var at = 0; at < FAR_ACROSS * FAR_ACROSS; at++) {
        let aside = vec2<i32>(at % FAR_ACROSS, at / FAR_ACROSS);
        let cell = base + aside - 1;
        if any(cell < vec2<i32>(0)) || any(cell >= size) {
            continue;
        }
        let weight = 4.0 - 2.0 * abs(vec2<f32>(aside) - 1.5);
        sum += textureLoad(source, cell, 0) * (weight.x * weight.y);
    }
    return sum / 64.0;
}

@compute @workgroup_size(PYRAMID_GROUP, PYRAMID_GROUP)
fn cs_far_first(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= textureDimensions(light_target)) {
        return;
    }
    var light = vec4<f32>(0.0);
    var reach = vec4<f32>(0.0);
    if all(id.xy < textureDimensions(far_cells)) {
        let words = textureLoad(far_cells, id.xy, 0);
        let known = select(0.0, 1.0, bitcast<f32>(words.w) > 0.0);
        let met = select(0.0, known, bitcast<f32>(words.z) > 0.0);
        let blue = unpack2x16float(words.y);
        light = vec4<f32>(vec3<f32>(unpack2x16float(words.x), blue.x) * met, met);
        reach = vec4<f32>(blue.y * met, known, 0.0, 0.0);
    }
    textureStore(light_target, id.xy, light);
    textureStore(reach_target, id.xy, reach);
}

@compute @workgroup_size(PYRAMID_GROUP, PYRAMID_GROUP)
fn cs_far_down(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= textureDimensions(light_target)) {
        return;
    }
    let base = vec2<i32>(id.xy) * 2;
    textureStore(light_target, id.xy, far_blurred(light_level, base));
    textureStore(reach_target, id.xy, far_blurred(reach_level, base));
}
