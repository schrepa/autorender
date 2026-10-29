use crate::linker::ShaderModule;

macro_rules! modules {
    ($($path:literal in $brick:literal = $file:literal;)*) => {
        pub const MODULES: &[ShaderModule] = &[$(ShaderModule {
            path: $path,
            brick: $brick,
            source: include_str!($file),
        }),*];
    };
}

modules! {
    "engine::color" in "core" = "color.wgsl";
    "engine::math" in "core" = "math.wgsl";
    "engine::fullscreen" in "core" = "fullscreen.wgsl";
    "engine::wind" in "core" = "wind.wgsl";
    "engine::screen_points" in "core" = "screen_points.wgsl";
    "engine::mipmaps" in "core" = "mipmaps.wgsl";
    "frame::view" in "core" = "frame/view.wgsl";
    "frame::encode" in "core" = "frame/encode.wgsl";
    "meshes::declarations" in "core" = "meshes/declarations.wgsl";
    "meshes::vertex" in "core" = "meshes/vertex.wgsl";
    "meshes::surface" in "core" = "meshes/surface.wgsl";
    "meshes::forward" in "core" = "meshes/forward.wgsl";
    "meshes::debug" in "core" = "meshes/debug.wgsl";
    "meshes::impostors" in "core" = "meshes/impostors.wgsl";
    "meshes::impostor_frames" in "core" = "impostor_frames.wgsl";
    "transparent::weighted" in "core" = "transparent/weighted.wgsl";
    "transparent::composite" in "core" = "transparent.wgsl";
    "presentation::cull" in "core" = "presentation.wgsl";
    "presentation::screen_cull" in "core" = "screen_cull.wgsl";
    "culling::pyramid" in "culling" = "depth_pyramid.wgsl";
    "lighting::sun" in "lighting_model" = "lighting/sun.wgsl";
    "lighting::brdf" in "lighting_model" = "lighting/brdf.wgsl";
    "lighting::surface" in "lighting_model" = "lighting/surface.wgsl";
    "shadows::sample" in "shadows" = "shadows/sample.wgsl";
    "shadows::maps" in "shadows" = "shadows.wgsl";
    "shadows::marks" in "shadows" = "shadow_marks.wgsl";
    "lights::shade" in "lights" = "lights/shade.wgsl";
    "lights::binning" in "lights" = "lights.wgsl";
    "sky::common" in "sky" = "sky_common.wgsl";
    "sky::light" in "sky" = "sky_light.wgsl";
    "sky::air" in "sky" = "sky/air.wgsl";
    "sky::environment" in "sky" = "sky/environment.wgsl";
    "sky::tables" in "sky" = "sky.wgsl";
    "sky::clouds" in "sky" = "clouds.wgsl";
    "sky::environment_maps" in "sky" = "environment.wgsl";
    "sky::draw" in "sky" = "meshes/sky.wgsl";
    "probes::sample" in "reflection_probes" = "probes/sample.wgsl";
    "terrain::horizons" in "terrain" = "terrain/horizons.wgsl";
    "terrain::occupancy" in "terrain" = "terrain/occupancy.wgsl";
    "terrain::bounce" in "terrain" = "terrain/bounce.wgsl";
    "terrain::variety" in "terrain" = "terrain_variety.wgsl";
    "terrain::ground" in "terrain" = "terrain_ground.wgsl";
    "terrain::tile" in "terrain" = "terrain_tile.wgsl";
    "terrain::normals" in "terrain" = "terrain_normals.wgsl";
    "terrain::control" in "terrain" = "terrain_control.wgsl";
    "terrain::bounce_maps" in "terrain" = "terrain_bounce.wgsl";
    "terrain::shadow" in "terrain" = "terrain_shadow.wgsl";
    "terrain::occupancy_maps" in "terrain" = "occupancy.wgsl";
    "terrain::vertex" in "terrain" = "meshes/terrain_vertex.wgsl";
    "terrain::surface" in "terrain" = "meshes/terrain_surface.wgsl";
    "terrain::draw_shared" in "terrain" = "meshes/terrain_draw_shared.wgsl";
    "terrain::draw" in "terrain" = "meshes/terrain.wgsl";
    "terrain::subdivision" in "terrain" = "meshes/terrain_subdivision.wgsl";
    "terrain::sky" in "terrain" = "meshes/terrain_sky.wgsl";
    "terrain::debug" in "terrain" = "meshes/terrain_debug.wgsl";
    "scatter::cull" in "terrain" = "scatter_cull.wgsl";
    "scatter::snap" in "terrain" = "scatter_snap.wgsl";
    "cover::draw" in "terrain" = "meshes/cover.wgsl";
    "cover::grow" in "terrain" = "cover_grow.wgsl";
    "water::refraction" in "terrain" = "water/refraction.wgsl";
    "water::surface" in "terrain" = "meshes/water.wgsl";
    "water::footprint" in "terrain" = "meshes/water_footprint.wgsl";
    "water::flow" in "terrain" = "meshes/water_flow.wgsl";
    "water::setup" in "terrain" = "meshes/water_setup.wgsl";
    "water::falls" in "terrain" = "meshes/water_falls.wgsl";
    "water::foam" in "terrain" = "meshes/water_foam.wgsl";
    "water::particles" in "terrain" = "meshes/water_particles.wgsl";
    "water::bursts" in "terrain" = "water_bursts.wgsl";
    "water::sim_shared" in "terrain" = "water_sim_shared.wgsl";
    "water::sim" in "terrain" = "water_sim.wgsl";
    "post::pixel" in "post_effects" = "post/pixel.wgsl";
    "post::frame" in "post_effects" = "effects.wgsl";
    "post::effects" in "post_effects" = "post.wgsl";
    "post::resolve_plain" in "post_effects" = "post/resolve_plain.wgsl";
    "post::haze" in "post_effects" = "post/haze.wgsl";
    "post::bloom" in "post_effects" = "post/bloom.wgsl";
    "post::dither" in "post_effects" = "post/dither.wgsl";
    "post::taa" in "post_effects" = "temporal.wgsl";
    "post::haze_march" in "post_effects" = "haze.wgsl";
    "post::exposure" in "post_effects" = "exposure.wgsl";
    "post::reflections" in "post_effects" = "reflections.wgsl";
    "particles::draw" in "particles" = "particles.wgsl";
    "debug::view" in "debug_view" = "debug_view.wgsl";
}

pub const RESOLVES: &[&str] = &["post::resolve_plain", "post::taa"];

pub fn composites() -> Vec<(String, &'static str, String)> {
    let effects: Vec<crate::composite::PixelEffect> = crate::parts::standard()
        .iter()
        .flat_map(|entry| entry.decl.effects.iter().copied())
        .collect();
    RESOLVES
        .iter()
        .filter_map(|&resolve| {
            let source = crate::composite::composite_source(resolve, &effects, None).ok()?;
            Some((crate::composite::composite_path(resolve), "post_effects", source))
        })
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Vertex,
    Fragment,
    Compute,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Entry {
    pub module: &'static str,
    pub name: &'static str,
    pub stage: Stage,
}

macro_rules! stage {
    (vertex) => {
        Stage::Vertex
    };
    (fragment) => {
        Stage::Fragment
    };
    (compute) => {
        Stage::Compute
    };
}

macro_rules! entry_points {
    ($($rust:ident {
        $($stage:ident: [$($name:ident @ $module:literal),* $(,)?]),* $(,)?
    })*) => {
        pub mod entry {
            $(
                #[expect(non_upper_case_globals, reason = "named as their WGSL functions")]
                pub mod $rust {
                    use super::super::{Entry, Stage};
                    $($(
                        pub const $name: Entry = Entry {
                            module: $module,
                            name: stringify!($name),
                            stage: stage!($stage),
                        };
                    )*)*
                }
            )*
        }

        pub const ENTRIES: &[Entry] = &[$($($(entry::$rust::$name,)*)*)*];
    };
}

entry_points! {
    meshes {
        vertex: [
            vs_affine @ "meshes::vertex",
            vs_affine_temporal @ "meshes::vertex",
            vs_affine_extended @ "meshes::vertex",
            vs_affine_temporal_extended @ "meshes::vertex",
            vs_sky @ "sky::draw",
            vs_impostor_bake @ "meshes::impostors",
            vs_impostor_bake_extended @ "meshes::impostors",
            vs_impostor @ "meshes::impostors",
            vs_impostor_temporal @ "meshes::impostors",
            vs_terrain @ "terrain::draw",
            vs_terrain_temporal @ "terrain::draw",
            vs_fullscreen @ "engine::fullscreen",
            vs_cover @ "cover::draw",
            vs_cover_temporal @ "cover::draw",
            vs_water @ "water::surface",
            vs_foam @ "water::particles",
            vs_fall @ "water::falls",
            vs_fall_temporal @ "water::falls",
            vs_debug_affine @ "meshes::debug",
            vs_debug_affine_extended @ "meshes::debug",
            vs_debug_impostor @ "meshes::debug",
        ],
        fragment: [
            fs_main @ "meshes::forward",
            fs_main_lit @ "meshes::forward",
            fs_post @ "meshes::forward",
            fs_post_lit @ "meshes::forward",
            fs_temporal @ "meshes::forward",
            fs_temporal_lit @ "meshes::forward",
            fs_probe @ "meshes::forward",
            fs_probe_lit @ "meshes::forward",
            fs_main_extended @ "meshes::forward",
            fs_main_lit_extended @ "meshes::forward",
            fs_probe_extended @ "meshes::forward",
            fs_probe_lit_extended @ "meshes::forward",
            fs_post_extended @ "meshes::forward",
            fs_post_lit_extended @ "meshes::forward",
            fs_temporal_extended @ "meshes::forward",
            fs_temporal_lit_extended @ "meshes::forward",
            fs_masked_depth @ "meshes::forward",
            fs_masked_depth_extended @ "meshes::forward",
            fs_masked_depth_temporal_extended @ "meshes::forward",
            fs_sky @ "sky::draw",
            fs_probe_sky @ "sky::draw",
            fs_view_clouds @ "sky::draw",
            fs_view_clouds_post @ "sky::draw",
            fs_sky_post @ "sky::draw",
            fs_sky_temporal @ "sky::draw",
            fs_impostor_shade @ "meshes::impostors",
            fs_impostor_shade_extended @ "meshes::impostors",
            fs_impostor_bake @ "meshes::impostors",
            fs_impostor_bake_extended @ "meshes::impostors",
            fs_impostor @ "meshes::impostors",
            fs_impostor_lit @ "meshes::impostors",
            fs_impostor_post @ "meshes::impostors",
            fs_impostor_post_lit @ "meshes::impostors",
            fs_impostor_temporal @ "meshes::impostors",
            fs_impostor_temporal_lit @ "meshes::impostors",
            fs_terrain_ids @ "terrain::draw",
            fs_terrain_resolve_post @ "terrain::draw",
            fs_terrain_resolve_post_lit @ "terrain::draw",
            fs_terrain_resolve_temporal @ "terrain::draw",
            fs_terrain_resolve_temporal_lit @ "terrain::draw",
            fs_terrain @ "terrain::draw",
            fs_terrain_lit @ "terrain::draw",
            fs_terrain_post @ "terrain::draw",
            fs_terrain_post_lit @ "terrain::draw",
            fs_terrain_temporal @ "terrain::draw",
            fs_terrain_temporal_lit @ "terrain::draw",
            fs_terrain_probe @ "terrain::draw",
            fs_terrain_probe_lit @ "terrain::draw",
            fs_cover @ "cover::draw",
            fs_cover_lit @ "cover::draw",
            fs_cover_post @ "cover::draw",
            fs_cover_post_lit @ "cover::draw",
            fs_cover_temporal @ "cover::draw",
            fs_cover_temporal_lit @ "cover::draw",
            fs_water @ "water::surface",
            fs_water_post @ "water::surface",
            fs_water_temporal @ "water::surface",
            fs_foam @ "water::particles",
            fs_fall @ "water::falls",
            fs_fall_post @ "water::falls",
            fs_fall_temporal @ "water::falls",
            fs_debug_mesh @ "meshes::debug",
            fs_debug_impostor @ "meshes::debug",
            fs_debug_terrain @ "terrain::debug",
            fs_debug_cover @ "meshes::debug",
            fs_debug_water @ "meshes::debug",
            fs_debug_fall @ "meshes::debug",
            fs_debug_sky @ "meshes::debug",
        ],
        compute: [
            setup_blocks @ "water::setup",
            setup_ring @ "water::setup",
            setup_raise @ "water::setup",
            subdivide_begin @ "terrain::subdivision",
            subdivide_decide @ "terrain::subdivision",
            subdivide_apply @ "terrain::subdivision",
            particles_begin @ "water::particles",
            particles_gather @ "water::particles",
            particles_currents @ "water::particles",
            particles_age @ "water::particles",
            particles_births @ "water::particles",
            particles_sum @ "water::particles",
            particles_scan @ "water::particles",
            particles_place @ "water::particles",
            particles_splat @ "water::particles",
            particles_look @ "water::particles",
            particles_down @ "water::particles",
        ],
    }
    terrain_normals {
        compute: [tile_normals @ "terrain::normals"],
    }
    terrain_control {
        compute: [tile_control @ "terrain::control"],
    }
    terrain_bounce {
        compute: [
            bounce_ground @ "terrain::bounce_maps",
            bounce_down @ "terrain::bounce_maps",
            bounce_gather @ "terrain::bounce_maps",
        ],
    }
    cover_grow {
        compute: [grow_cover @ "cover::grow", finish_cover @ "cover::grow"],
    }
    water_sim {
        compute: [
            sim_exchange @ "water::sim",
            sim_waves @ "water::sim",
            sim_seconds @ "water::sim",
            sim_drain @ "water::sim",
            sim_update @ "water::sim",
            sim_source @ "water::sim",
            sim_touch @ "water::sim",
            sim_burst @ "water::sim",
            sim_push @ "water::sim",
            sim_ended @ "water::sim",
            sim_tally @ "water::sim",
            sim_activity @ "water::sim",
            sim_look @ "water::sim",
            sim_drawn @ "water::sim",
        ],
    }
    clouds {
        compute: [
            shadow_clouds @ "sky::clouds",
            light_clouds @ "sky::clouds",
            march_clouds @ "sky::clouds",
            march_view @ "sky::clouds",
        ],
    }
    sky {
        compute: [build_sky_view @ "sky::tables", derive_light @ "sky::tables"],
    }
    scatter_cull {
        compute: [cull_scatter @ "scatter::cull"],
    }
    scatter_snap {
        compute: [snap_scatter @ "scatter::snap"],
    }
    occupancy {
        vertex: [
            vs_splat @ "terrain::occupancy_maps",
            vs_clear @ "terrain::occupancy_maps",
            vs_means @ "terrain::occupancy_maps",
        ],
        fragment: [
            fs_splat @ "terrain::occupancy_maps",
            fs_clear @ "terrain::occupancy_maps",
            fs_means @ "terrain::occupancy_maps",
        ],
        compute: [select_covers @ "terrain::occupancy_maps"],
    }
    terrain_shadow {
        vertex: [vs_terrain_shadow @ "terrain::shadow"],
    }
    presentation {
        compute: [
            shift_previous @ "presentation::cull",
            compose_parents @ "presentation::cull",
            cull @ "presentation::cull",
            fade_levels @ "presentation::cull",
            occlusion_test @ "presentation::cull",
            count_still @ "presentation::cull",
            probe_cull @ "presentation::cull",
            sum_groups @ "presentation::cull",
            scan_groups @ "presentation::cull",
            batch_args @ "presentation::cull",
            scatter @ "presentation::cull",
        ],
    }
    depth_pyramid {
        compute: [
            pyramid_first @ "culling::pyramid",
            pyramid_down @ "culling::pyramid",
            pyramid_first_nearest @ "culling::pyramid",
            pyramid_down_nearest @ "culling::pyramid",
        ],
    }
    temporal {
        vertex: [vs_fullscreen @ "engine::fullscreen"],
    }
    mipmaps {
        vertex: [vs_mip @ "engine::mipmaps"],
        fragment: [fs_mip @ "engine::mipmaps"],
    }
    lights {
        compute: [
            count_lights @ "lights::binning",
            bin_lights @ "lights::binning",
            scan_clusters @ "lights::binning",
        ],
    }
    particles {
        vertex: [vs_particle @ "particles::draw"],
        fragment: [
            fs_particle @ "particles::draw",
            fs_particle_mist @ "particles::draw",
            fs_particle_add @ "particles::draw",
            fs_particle_add_post @ "particles::draw",
        ],
    }
    transparent {
        vertex: [vs_fullscreen @ "engine::fullscreen"],
        fragment: [fs_composite @ "transparent::composite"],
    }
    haze {
        vertex: [vs_fullscreen @ "engine::fullscreen"],
        fragment: [fs_haze @ "post::haze_march"],
    }
    shadow {
        vertex: [
            vs_affine @ "shadows::maps",
            vs_affine_masked @ "shadows::maps",
            vs_impostor_card @ "shadows::maps",
            vs_compose @ "shadows::maps",
            vs_moving @ "shadows::maps",
            vs_moving_masked @ "shadows::maps",
            vs_clear_moving @ "shadows::maps",
        ],
        fragment: [
            fs_impostor_card @ "shadows::maps",
            fs_masked @ "shadows::maps",
            fs_compose @ "shadows::maps",
            fs_moving @ "shadows::maps",
            fs_moving_masked @ "shadows::maps",
        ],
        compute: [select_casters @ "shadows::maps", count_casters @ "shadows::maps"],
    }
    shadow_marks {
        compute: [mark_shadow_tiles @ "shadows::marks"],
    }
    post {
        vertex: [vs_fullscreen @ "engine::fullscreen"],
        fragment: [
            fs_occlusion @ "post::effects",
            fs_effects_filter @ "post::effects",
            fs_haze_filter @ "post::effects",
            fs_bloom_first @ "post::effects",
            fs_bloom_down @ "post::effects",
            fs_levels_first @ "post::effects",
            fs_levels_down @ "post::effects",
            fs_bloom_up @ "post::effects",
        ],
    }
    reflections {
        vertex: [vs_fullscreen @ "engine::fullscreen"],
        fragment: [fs_reflections @ "post::reflections"],
    }
    exposure {
        compute: [meter_exposure @ "post::exposure", adapt_exposure @ "post::exposure"],
    }
    debug_view {
        vertex: [vs_fullscreen @ "engine::fullscreen"],
        fragment: [fs_debug_view @ "debug::view"],
    }
    environment {
        vertex: [vs_environment @ "sky::environment_maps"],
        fragment: [
            fs_environment_sky @ "sky::environment_maps",
            fs_environment_down @ "sky::environment_maps",
            fs_environment_prefilter @ "sky::environment_maps",
        ],
    }
}
