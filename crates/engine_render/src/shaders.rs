//! Every shader module the renderer creates, as the WGSL files each
//! concatenates. Native builds read them too: `cargo xtask lint` validates
//! each module with naga against the GPU layouts and binding tables.

/// Builds each module's source from its files, and [`MODULES`] from the
/// same table.
macro_rules! shaders {
    ($($(#[$doc:meta])* $name:ident = [$($file:literal),+ $(,)?];)*) => {
        $($(#[$doc])* pub const $name: &str = concat!($(include_str!($file)),+);)*

        /// Every module by name, with its source.
        pub const MODULES: &[(&str, &str)] = &[$((stringify!($name), $name)),*];
    };
}

shaders! {
    /// Meshes after the shared scene declarations.
    SHADER = [
        "scene.wgsl",
        "meshes/declarations.wgsl",
        "meshes/vertex.wgsl",
        "meshes/shading.wgsl",
        "meshes/forward.wgsl",
        "meshes/resolve.wgsl",
        "sky_common.wgsl",
        "meshes/sky.wgsl",
        "meshes/decals.wgsl",
        "impostor_frames.wgsl",
        "meshes/impostors.wgsl",
        "meshes/terrain_vertex.wgsl",
        "terrain_variety.wgsl",
        "meshes/terrain.wgsl",
        "meshes/cover.wgsl",
        "meshes/water.wgsl",
        "meshes/ocean.wgsl",
    ];
    /// Terrain normal maps.
    TERRAIN_NORMALS_SHADER = [
        "terrain_normals.wgsl",
    ];
    /// Terrain control values from the surface's rules.
    TERRAIN_CONTROL_SHADER = [
        "terrain_control.wgsl",
    ];
    /// Terrain horizon map builds.
    TERRAIN_HORIZON_SHADER = [
        "terrain_horizon.wgsl",
    ];
    /// Terrain sky visibility map builds.
    TERRAIN_SKY_SHADER = [
        "terrain_sky.wgsl",
    ];
    /// Ground cover's growth.
    COVER_GROW_SHADER = [
        "terrain_variety.wgsl",
        "cover_grow.wgsl",
    ];
    DEFORMATION_SHADER = [
        "deformation.wgsl",
    ];
    CLOUDS_SHADER = [
        "clouds.wgsl",
    ];
    /// The physically based sky's tables and light.
    SKY_SHADER = [
        "sky_common.wgsl",
        "sky.wgsl",
    ];
    /// Placed objects' culling.
    SCATTER_CULL_SHADER = [
        "scatter_cull.wgsl",
    ];
    SCATTER_SNAP_SHADER = [
        "scatter_snap.wgsl",
    ];
    OCCUPANCY_SHADER = [
        "occupancy.wgsl",
    ];
    /// Terrain's depth drawing into shadow maps.
    TERRAIN_SHADOW_SHADER = [
        "meshes/terrain_vertex.wgsl",
        "terrain_shadow.wgsl",
    ];
    /// Compute passes of GPU presentation.
    PRESENTATION_SHADER = [
        "presentation.wgsl",
    ];
    /// The depth pyramid of occlusion culling.
    DEPTH_PYRAMID_SHADER = [
        "depth_pyramid.wgsl",
    ];
    /// Cluster selection of virtual geometry.
    VIRTUAL_SHADER = [
        "virtual_geometry.wgsl",
    ];
    /// Tile classification of the visibility buffer.
    VISIBILITY_SHADER = [
        "visibility.wgsl",
    ];
    /// Temporal antialiasing resolve, after the shared effects.
    TEMPORAL_SHADER = [
        "effects.wgsl",
        "temporal.wgsl",
    ];
    /// Mip generation of base color textures.
    MIPMAP_SHADER = [
        "mipmaps.wgsl",
    ];
    /// Local light binning into view clusters.
    LIGHT_SHADER = [
        "lights.wgsl",
    ];
    /// Stateless particles after the shared scene declarations.
    PARTICLE_SHADER = [
        "scene.wgsl",
        "particles.wgsl",
    ];
    /// The composite of weighted blended transparency.
    TRANSPARENT_SHADER = [
        "transparent.wgsl",
    ];
    /// Sun shafts after the scene declarations and effects.
    SHAFTS_SHADER = [
        "scene.wgsl",
        "effects.wgsl",
        "shafts.wgsl",
    ];
    /// Shadow caster selection and depth drawing.
    SHADOW_SHADER = [
        "shadows.wgsl",
        "impostor_frames.wgsl",
    ];
    /// Ambient occlusion, bloom and their composite.
    POST_SHADER = [
        "effects.wgsl",
        "post.wgsl",
    ];
    /// Environment maps after the scene declarations.
    ENVIRONMENT_SHADER = [
        "scene.wgsl",
        "sky_common.wgsl",
        "environment.wgsl",
    ];
}
