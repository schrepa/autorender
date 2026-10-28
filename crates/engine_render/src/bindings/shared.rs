//! The groups several features share (`SHARED_GROUPS`): each feature's
//! range of binding numbers in them, which `cargo xtask lint` holds every
//! table serving one to.

/// The bindings one feature takes in a group features share: its numbers
/// (a range of them) and its bindings' names.
#[derive(Debug)]
pub struct FeatureRange {
    pub feature: &'static str,
    pub numbers: core::ops::Range<u32>,
    pub names: &'static [&'static str],
}

/// The groups several features share and each feature's range in them:
/// group `group` of module `module`, every table serving it holding each
/// binding within its feature's range.
#[derive(Debug)]
pub struct SharedGroup {
    pub module: &'static str,
    pub group: u32,
    pub ranges: &'static [FeatureRange],
}

/// Declares [`SHARED_GROUPS`]: `MODULE @ group { feature: start..end [names], .. }`.
macro_rules! shared_groups {
    ($($module:ident @ $group:literal {
        $($feature:ident: $start:literal..$end:literal [$($name:ident),* $(,)?]),* $(,)?
    })*) => {
        /// The mesh module's groups, which every surface pipeline shares,
        /// each feature in a range of its own: `cargo xtask lint` holds every
        /// table serving one to the ranges, so a feature adding a binding
        /// takes a number in its range, and features never collide.
        pub const SHARED_GROUPS: &[SharedGroup] = &[$(SharedGroup {
            module: stringify!($module),
            group: $group,
            ranges: &[$(FeatureRange {
                feature: stringify!($feature),
                numbers: $start..$end,
                names: &[$(stringify!($name)),*],
            }),*],
        }),*];
    };
}

shared_groups! {
    SHADER @ 0 {
        frame: 0..2 [camera, lighting],
        shadows: 2..9 [shadow, shadow_atlas, shadow_tiles, shadow_sampler, shadow_texels],
        refraction: 9..11 [refraction_source, refraction_sampler],
        environment: 11..13 [environment_maps, environment_sampler],
        probes: 13..14 [probe_words],
        horizons: 14..17 [terrain_horizons, terrain_horizon_records, terrain_horizon_sampler],
        occupancy: 17..19 [occupancy_coverage, occupancy_heights],
        clouds: 19..20 [cloud_shadows],
        sky: 23..30 [
            sky_air, sky_view_table, sky_table_sampler, sky_clouds, sky_clouds_old,
            sky_cloud_sampler, sky_domes
        ],
    }
    SHADER @ 1 {
        material: 0..11 [
            base_color_texture, base_color_sampler, metallic_roughness_texture,
            metallic_roughness_sampler, normal_texture, normal_sampler, occlusion_texture,
            occlusion_sampler, emissive_texture, emissive_sampler, material_maps
        ],
        impostors: 11..17 [
            impostor_records, impostor_albedo, impostor_emissive, impostor_geometry,
            impostor_surface, impostor_sampler
        ],
        terrain: 17..33 [
            terrain_heights, terrain_view, terrain, terrain_holes, terrain_normals,
            terrain_sky_sampler, terrain_directory, terrain_control, terrain_palette,
            terrain_albedo, terrain_material_data, terrain_material_sampler, terrain_sky,
            terrain_deformation, terrain_control_sampler
        ],
        cover: 33..35 [cover_blades, cover_frame],
        macro_colors: 35..36 [terrain_macro],
        surface: 36..37 [terrain_surface_leaves],
        sky: 42..43 [sky_transmittance_table],
        view_clouds: 46..47 [view_clouds],
        water: 50..62 [water_surface, water_info, water, water_live, water_tree],
        errors: 62..64 [terrain_errors, water_errors],
    }
    SHADER @ 2 {
        poses: 0..6 [poses, colors, previous_poses, palette, previous_palette, grounds],
        virtual_geometry: 6..9 [virtual_meshes, virtual_clusters, virtual_pages],
        visibility: 9..13 [visibility_draw, visible_clusters, visibility_ids, resolve_tiles],
        decals: 13..15 [decal_records, decal_depth],
        impostor_bakes: 15..17 [impostor_bake, impostor_shade],
        terrain: 18..19 [terrain_ids_texture],
        water: 22..27 [
            water_depth, water_scene, water_particles, water_particles_sampler,
            water_particles_spread
        ],
        subdivision: 27..32 [
            subdivision, subdivision_tree, subdivision_nodes, subdivision_leaves, subdivision_args
        ],
        particles: 32..41 [
            particle_frame, particle_lattice, particle_pool, particle_counts, particle_splat,
            particle_source, particle_target, particle_spread_source, particle_spread_target
        ],
        setup: 41..47 [
            setup_frame, setup_list, setup_ground, setup_cells, setup_looks, setup_ring
        ],
    }
    SHADER @ 3 {
        lights: 0..4 [clusters, lights, cluster_ranges, cluster_lights],
    }
}
