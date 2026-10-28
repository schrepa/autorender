macro_rules! shaders {
    ($($(#[$doc:meta])* $name:ident = [$($file:literal),+ $(,)?];)*) => {
        $($(#[$doc])* pub const $name: &str = concat!($(include_str!($file)),+);)*

        pub const MODULES: &[(&str, &str)] = &[$((stringify!($name), $name)),*];

        pub const FILES: &[(&str, &str)] = &[$($(($file, include_str!($file))),+),*];
    };
}

shaders! {
    SHADER = [
        "scene.wgsl",
        "color.wgsl",
        "math.wgsl",
        "wind.wgsl",
        "meshes/declarations.wgsl",
        "meshes/vertex.wgsl",
        "meshes/shading.wgsl",
        "meshes/forward.wgsl",
        "meshes/resolve.wgsl",
        "sky_common.wgsl",
        "sky_light.wgsl",
        "meshes/sky.wgsl",
        "meshes/decals.wgsl",
        "impostor_frames.wgsl",
        "meshes/impostors.wgsl",
        "meshes/terrain_vertex.wgsl",
        "meshes/terrain_surface.wgsl",
        "terrain_variety.wgsl",
        "meshes/terrain.wgsl",
        "meshes/terrain_subdivision.wgsl",
        "meshes/terrain_sky.wgsl",
        "fullscreen.wgsl",
        "meshes/cover.wgsl",
        "meshes/water.wgsl",
        "meshes/water_footprint.wgsl",
        "meshes/water_flow.wgsl",
        "meshes/water_setup.wgsl",
        "meshes/water_falls.wgsl",
        "meshes/ocean.wgsl",
        "meshes/water_foam.wgsl",
        "water_bursts.wgsl",
        "meshes/water_particles.wgsl",
        "meshes/debug.wgsl",
        "meshes/terrain_debug.wgsl",
    ];
    TERRAIN_NORMALS_SHADER = [
        "terrain_tile.wgsl",
        "terrain_normals.wgsl",
    ];
    TERRAIN_CONTROL_SHADER = [
        "terrain_tile.wgsl",
        "terrain_control.wgsl",
    ];
    BOUNCE_SHADER = [
        "scene.wgsl",
        "color.wgsl",
        "math.wgsl",
        "terrain_variety.wgsl",
        "terrain_bounce.wgsl",
    ];
    COVER_GROW_SHADER = [
        "math.wgsl",
        "terrain_variety.wgsl",
        "terrain_ground.wgsl",
        "cover_grow.wgsl",
    ];
    DEFORMATION_SHADER = [
        "deformation.wgsl",
    ];
    WATER_SIM_SHADER = [
        "water_bursts.wgsl",
        "water_sim.wgsl",
    ];
    CLOUDS_SHADER = [
        "sky_common.wgsl",
        "clouds.wgsl",
    ];
    SKY_SHADER = [
        "color.wgsl",
        "sky_common.wgsl",
        "sky.wgsl",
    ];
    SCATTER_CULL_SHADER = [
        "terrain_variety.wgsl",
        "scatter_cull.wgsl",
    ];
    SCATTER_SNAP_SHADER = [
        "terrain_variety.wgsl",
        "terrain_ground.wgsl",
        "scatter_snap.wgsl",
    ];
    OCCUPANCY_SHADER = [
        "occupancy.wgsl",
    ];
    TERRAIN_SHADOW_SHADER = [
        "meshes/terrain_vertex.wgsl",
        "terrain_shadow.wgsl",
    ];
    PRESENTATION_SHADER = [
        "math.wgsl",
        "screen_cull.wgsl",
        "presentation.wgsl",
    ];
    DEPTH_PYRAMID_SHADER = [
        "depth_pyramid.wgsl",
    ];
    VIRTUAL_SHADER = [
        "screen_cull.wgsl",
        "virtual_geometry.wgsl",
    ];
    VISIBILITY_SHADER = [
        "visibility.wgsl",
    ];
    TEMPORAL_SHADER = [
        "color.wgsl",
        "fullscreen.wgsl",
        "effects.wgsl",
        "temporal.wgsl",
    ];
    MIPMAP_SHADER = [
        "mipmaps.wgsl",
    ];
    LIGHT_SHADER = [
        "lights.wgsl",
    ];
    PARTICLE_SHADER = [
        "scene.wgsl",
        "color.wgsl",
        "math.wgsl",
        "particles.wgsl",
    ];
    TRANSPARENT_SHADER = [
        "color.wgsl",
        "fullscreen.wgsl",
        "transparent.wgsl",
    ];
    HAZE_SHADER = [
        "scene.wgsl",
        "color.wgsl",
        "math.wgsl",
        "fullscreen.wgsl",
        "effects.wgsl",
        "haze.wgsl",
    ];
    SHADOW_SHADER = [
        "wind.wgsl",
        "shadows.wgsl",
        "impostor_frames.wgsl",
    ];
    SHADOW_MARKS_SHADER = [
        "scene.wgsl",
        "color.wgsl",
        "math.wgsl",
        "shadow_marks.wgsl",
    ];
    POST_SHADER = [
        "color.wgsl",
        "fullscreen.wgsl",
        "effects.wgsl",
        "screen_points.wgsl",
        "post.wgsl",
    ];
    REFLECTIONS_SHADER = [
        "scene.wgsl",
        "color.wgsl",
        "math.wgsl",
        "fullscreen.wgsl",
        "effects.wgsl",
        "terrain_variety.wgsl",
        "terrain_ground.wgsl",
        "screen_points.wgsl",
        "impostor_frames.wgsl",
        "reflections.wgsl",
    ];
    EXPOSURE_SHADER = [
        "color.wgsl",
        "sky_common.wgsl",
        "exposure.wgsl",
    ];
    DEBUG_VIEW_SHADER = [
        "fullscreen.wgsl",
        "debug_view.wgsl",
    ];
    ENVIRONMENT_SHADER = [
        "scene.wgsl",
        "color.wgsl",
        "math.wgsl",
        "sky_common.wgsl",
        "sky_light.wgsl",
        "environment.wgsl",
    ];
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
    ($($rust:ident = $module:ident {
        $($stage:ident: [$($name:ident),* $(,)?]),* $(,)?
    })*) => {
        pub mod entry {
            $(
                #[expect(non_upper_case_globals, reason = "named as their WGSL functions")]
                pub mod $rust {
                    use super::super::{Entry, Stage};
                    $($(
                        pub const $name: Entry = Entry {
                            module: stringify!($module),
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
    meshes = SHADER {
        vertex: [
            vs_virtual, vs_virtual_temporal, vs_visibility, vs_affine, vs_affine_temporal,
            vs_affine_extended, vs_affine_temporal_extended, vs_affine_skinned,
            vs_affine_temporal_skinned, vs_resolve, vs_sky, vs_decal, vs_impostor_bake,
            vs_impostor_bake_extended, vs_impostor, vs_impostor_temporal, vs_terrain,
            vs_terrain_temporal, vs_fullscreen, vs_cover,
            vs_cover_temporal, vs_water, vs_foam, vs_ocean,
            vs_ocean_temporal, vs_fall, vs_fall_temporal,
            vs_debug_affine, vs_debug_affine_extended, vs_debug_affine_skinned, vs_debug_virtual,
            vs_debug_impostor
        ],
        fragment: [
            fs_visibility, fs_main, fs_main_lit, fs_post, fs_post_lit, fs_temporal, fs_temporal_lit,
            fs_transparent, fs_transparent_lit, fs_probe, fs_probe_lit, fs_main_extended,
            fs_main_lit_extended, fs_probe_extended, fs_probe_lit_extended, fs_post_extended,
            fs_post_lit_extended, fs_temporal_extended, fs_temporal_lit_extended,
            fs_transparent_extended, fs_transparent_lit_extended, fs_masked_depth,
            fs_masked_depth_extended, fs_masked_depth_temporal_extended, fs_resolve, fs_resolve_lit,
            fs_resolve_post, fs_resolve_post_lit, fs_resolve_temporal, fs_resolve_temporal_lit,
            fs_sky, fs_probe_sky, fs_view_clouds, fs_view_clouds_post, fs_sky_post, fs_sky_temporal,
            fs_decal, fs_decal_lit, fs_impostor_shade, fs_impostor_shade_extended, fs_impostor_bake,
            fs_impostor_bake_extended, fs_impostor, fs_impostor_lit, fs_impostor_post,
            fs_impostor_post_lit, fs_impostor_temporal, fs_impostor_temporal_lit,
            fs_terrain_ids, fs_terrain_resolve_post,
            fs_terrain_resolve_post_lit, fs_terrain_resolve_temporal,
            fs_terrain_resolve_temporal_lit, fs_terrain, fs_terrain_lit, fs_terrain_post,
            fs_terrain_post_lit, fs_terrain_temporal, fs_terrain_temporal_lit, fs_terrain_probe,
            fs_terrain_probe_lit, fs_cover, fs_cover_lit, fs_cover_post, fs_cover_post_lit,
            fs_cover_temporal, fs_cover_temporal_lit, fs_water, fs_water_post, fs_water_temporal,
            fs_foam,
            fs_ocean, fs_ocean_post, fs_ocean_temporal, fs_fall, fs_fall_post, fs_fall_temporal,
            fs_debug_mesh, fs_debug_impostor, fs_debug_terrain, fs_debug_cover, fs_debug_water,
            fs_debug_ocean, fs_debug_fall, fs_debug_sky
        ],
        compute: [
            setup_blocks,
            setup_ring,
            setup_raise,
            subdivide_begin,
            subdivide_decide,
            subdivide_apply,
            particles_begin,
            particles_gather,
            particles_currents,
            particles_age,
            particles_births,
            particles_sum,
            particles_scan,
            particles_place,
            particles_splat,
            particles_look,
            particles_down,
        ],
    }
    terrain_normals = TERRAIN_NORMALS_SHADER {
        compute: [tile_normals],
    }
    terrain_control = TERRAIN_CONTROL_SHADER {
        compute: [tile_control],
    }
    terrain_bounce = BOUNCE_SHADER {
        compute: [bounce_ground, bounce_down, bounce_gather],
    }
    cover_grow = COVER_GROW_SHADER {
        compute: [grow_cover, finish_cover],
    }
    deformation = DEFORMATION_SHADER {
        compute: [press, slopes, scroll, wipe, rebase],
    }
    water_sim = WATER_SIM_SHADER {
        compute: [
            sim_exchange, sim_waves, sim_seconds, sim_drain, sim_update, sim_source, sim_touch,
            sim_burst, sim_push, sim_ended, sim_tally, sim_activity, sim_look, sim_drawn
        ],
    }
    clouds = CLOUDS_SHADER {
        compute: [shadow_clouds, light_clouds, march_clouds, march_view],
    }
    sky = SKY_SHADER {
        compute: [build_sky_view, derive_light],
    }
    scatter_cull = SCATTER_CULL_SHADER {
        compute: [cull_scatter],
    }
    scatter_snap = SCATTER_SNAP_SHADER {
        compute: [snap_scatter],
    }
    occupancy = OCCUPANCY_SHADER {
        vertex: [vs_splat, vs_clear, vs_means],
        fragment: [fs_splat, fs_clear, fs_means],
        compute: [select_covers],
    }
    terrain_shadow = TERRAIN_SHADOW_SHADER {
        vertex: [vs_terrain_shadow],
    }
    presentation = PRESENTATION_SHADER {
        compute: [
            shift_previous, compose_parents, cull, fade_levels, occlusion_test, count_still,
            probe_cull, skin_palette, skin_bounds_pass, sum_groups, scan_groups, batch_args, scatter
        ],
    }
    depth_pyramid = DEPTH_PYRAMID_SHADER {
        compute: [pyramid_first, pyramid_down, pyramid_first_nearest, pyramid_down_nearest],
    }
    virtual_geometry = VIRTUAL_SHADER {
        vertex: [vs_shadow],
        compute: [
            seed, level_args_even, level_args_odd, traverse_even, traverse_odd, work_args,
            select_clusters, retest_args, retest_clusters, bin_args
        ],
    }
    visibility = VISIBILITY_SHADER {
        compute: [classify_tiles, resolve_args],
    }
    temporal = TEMPORAL_SHADER {
        vertex: [vs_fullscreen],
        fragment: [fs_resolve],
    }
    mipmaps = MIPMAP_SHADER {
        vertex: [vs_mip],
        fragment: [fs_mip],
    }
    lights = LIGHT_SHADER {
        compute: [count_lights, bin_lights, scan_clusters],
    }
    particles = PARTICLE_SHADER {
        vertex: [vs_particle],
        fragment: [fs_particle, fs_particle_mist, fs_particle_add, fs_particle_add_post],
    }
    transparent = TRANSPARENT_SHADER {
        vertex: [vs_fullscreen],
        fragment: [fs_composite],
    }
    haze = HAZE_SHADER {
        vertex: [vs_fullscreen],
        fragment: [fs_haze],
    }
    shadow = SHADOW_SHADER {
        vertex: [
            vs_affine, vs_affine_masked, vs_affine_skinned, vs_affine_masked_skinned,
            vs_impostor_card, vs_compose, vs_moving, vs_moving_masked, vs_moving_skinned,
            vs_moving_masked_skinned, vs_clear_moving
        ],
        fragment: [fs_impostor_card, fs_masked, fs_compose, fs_moving, fs_moving_masked],
        compute: [select_skinned, select_casters, count_casters],
    }
    shadow_marks = SHADOW_MARKS_SHADER {
        compute: [mark_shadow_tiles],
    }
    post = POST_SHADER {
        vertex: [vs_fullscreen],
        fragment: [
            fs_occlusion, fs_effects_filter, fs_haze_filter, fs_bloom_first, fs_bloom_down,
            fs_levels_first, fs_levels_down, fs_bloom_up, fs_composite
        ],
    }
    reflections = REFLECTIONS_SHADER {
        vertex: [vs_fullscreen],
        fragment: [fs_reflections],
    }
    exposure = EXPOSURE_SHADER {
        compute: [meter_exposure, adapt_exposure],
    }
    debug_view = DEBUG_VIEW_SHADER {
        vertex: [vs_fullscreen],
        fragment: [fs_debug_view],
    }
    environment = ENVIRONMENT_SHADER {
        vertex: [vs_environment],
        fragment: [fs_environment_sky, fs_environment_down, fs_environment_prefilter],
    }
}
