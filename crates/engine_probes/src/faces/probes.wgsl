import package::face::sky::environment_sample;
import package::probes::sample::{PROBES, PROBE_ENTRY_WORDS, box_projected, probe_words};

struct EnvironmentLight {
    reflected: vec3<f32>,
    interior: vec2<f32>,
}

fn environment_light(direction: vec3<f32>, level: f32, world: vec3<f32>,
        reflected_sky: vec3<f32>) -> EnvironmentLight {
    var light = vec3<f32>(0.0);
    var covered = 0.0;
    var shares = vec2<f32>(1.0);
    let header = probe_words[0];
    if PROBES && header.w != 0u {
        let grid = probe_words[1];
        let cell = vec2<i32>(floor((world.xz - bitcast<vec2<f32>>(header.xy))
            / bitcast<f32>(header.z)));
        if all(cell >= vec2<i32>(0)) && all(vec2<u32>(cell) < grid.xy) {
            let list = probe_words[grid.z + u32(cell.y) * grid.x + u32(cell.x)];
            let interiors = list.z != 0u;
            for (var n = 0u; n < list.y && (covered < 0.999 || interiors); n++) {
                let entry = grid.w + (list.x + n) * PROBE_ENTRY_WORDS;
                let interior = probe_words[entry + 2u];
                let reflects = interior.w != 0u && covered < 0.999;
                let shades = interiors && interior.z != 0u;
                if !reflects && !shades {
                    continue;
                }
                let first = probe_words[entry];
                let second = probe_words[entry + 1u];
                let center = bitcast<vec3<f32>>(first.xyz);
                let blend = bitcast<f32>(first.w);
                let extent = bitcast<vec3<f32>>(second.xyz);
                let inside = extent - abs(world - center);
                let depth = min(inside.x, min(inside.y, inside.z));
                if depth <= 0.0 {
                    continue;
                }
                let fade = select(1.0, saturate(depth / blend), blend > 0.0);
                if reflects {
                    let weight = fade * (1.0 - covered);
                    let along = box_projected(world, direction, center, extent);
                    light += weight * environment_sample(along, second.w, level);
                    covered += weight;
                }
                if shades {
                    shares = min(shares, mix(vec2<f32>(1.0), bitcast<vec2<f32>>(interior.xy),
                        fade));
                }
            }
        }
    }
    if covered < 0.999 {
        light += (1.0 - covered) * reflected_sky;
    }
    return EnvironmentLight(light, shares);
}

fn in_interior(world: vec3<f32>) -> bool {
    let header = probe_words[0];
    if !PROBES || header.w == 0u {
        return false;
    }
    let grid = probe_words[1];
    let cell = vec2<i32>(floor((world.xz - bitcast<vec2<f32>>(header.xy))
        / bitcast<f32>(header.z)));
    if any(cell < vec2<i32>(0)) || any(vec2<u32>(cell) >= grid.xy) {
        return false;
    }
    let list = probe_words[grid.z + u32(cell.y) * grid.x + u32(cell.x)];
    let count = select(0u, list.y, list.z != 0u);
    for (var n = 0u; n < count; n++) {
        let entry = grid.w + (list.x + n) * PROBE_ENTRY_WORDS;
        if probe_words[entry + 2u].z == 0u {
            continue;
        }
        let center = bitcast<vec3<f32>>(probe_words[entry].xyz);
        let extent = bitcast<vec3<f32>>(probe_words[entry + 1u].xyz);
        if all(abs(world - center) < extent) {
            return true;
        }
    }
    return false;
}
