import package::meshes::debug::{DEBUG_KIND_SKY, DebugTargets};
import package::sky::draw::SkyOutput;

@fragment
fn fs_debug_sky(input: SkyOutput) -> DebugTargets {
    var targets: DebugTargets;
    targets.ids = vec4<u32>(DEBUG_KIND_SKY, 0u, 0u, 0u);
    targets.count = 1.0;
    targets.terrain = vec2<u32>(0u);
    return targets;
}
