import package::face::sky::{environment_levels, environment_sample};
import package::generated::probes::PROBE_ENTRY_WORDS;
import package::place::ambient::{EnvironmentAt, EnvironmentLight};
import package::probes::sample::{box_projected, probe_words};

fn environment(light: EnvironmentLight, at: EnvironmentAt) -> EnvironmentLight {
    let world = at.world;
    let direction = at.lobe;
    let level = at.roughness * f32(environment_levels() - 1u);
    var found = vec3<f32>(0.0);
    var covered = 0.0;
    var shares = vec2<f32>(1.0);
    let header = probe_words[0];
    if header.w != 0u {
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
                    found += weight * environment_sample(along, second.w, level);
                    covered += weight;
                }
                if shades {
                    shares = min(shares, mix(vec2<f32>(1.0), bitcast<vec2<f32>>(interior.xy),
                        fade));
                }
            }
        }
    }
    var result = light;
    result.specular = found;
    result.covered = covered;
    result.inside = shares;
    return result;
}
