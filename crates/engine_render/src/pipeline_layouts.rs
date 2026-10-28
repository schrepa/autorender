//! Every pipeline layout the renderer builds, as the bind group tables of
//! its groups in order (`crate::bindings`). Pipelines build their layouts
//! from these alone (`crate::pipelines::pipeline_layout`), and `cargo xtask
//! lint` sums each stage's bindings over every layout against
//! [`REQUIRED_LIMITS`], which every device the renderer runs on grants, or
//! against the larger limits a layout names, whose users check the device
//! first; and it holds their number to [`LAYOUT_BUDGET`]: each distinct
//! pipeline layout costs Firefox's D3D12 backend a root signature, about
//! 150 ms of startup.

use crate::bindings::{
    BOUNCE, CAMERA, CAMERA_TERRAIN, CAMERA_WATER, CLOUD_COMPOSITE, CLOUD_MARCH, CLOUD_VIEW,
    COVER_DRAW, COVER_GROW, DEBUG_VIEW, DECALS, DEFORMATION, EMPTY, ENVIRONMENT_DOWN,
    ENVIRONMENT_PREFILTER, ENVIRONMENT_SKY, EXPOSURE, HAZE, IMPOSTOR_BAKE, IMPOSTOR_DRAW,
    LIGHT_BINNING, LIGHT_SHADING, MATERIAL, MIPMAPS, OCCUPANCY, PARTICLES, POST,
    PRESENTATION_COMPACT, PRESENTATION_COMPOSE, PRESENTATION_CULL, PRESENTATION_FADES,
    PRESENTATION_LEVEL, PRESENTATION_SKIN_BOUNDS, PRESENTATION_TESTS, PYRAMID_DOWN, PYRAMID_FIRST,
    REFLECTIONS, RETAINED_POSES, SCATTER_CULL, SCATTER_SNAP, SHADOW_MARKS, SHADOW_POSES,
    SHADOW_SELECT, SHADOW_SKINNED, SHADOW_VIEW, SKY_TABLES, SKY_TRANSMITTANCE, TAA_RESOLVE,
    TEMPORAL_POSES, TERRAIN, TERRAIN_CONTROL, TERRAIN_NORMALS, TERRAIN_RESOLVE,
    TERRAIN_SUBDIVISION, TERRAIN_SUBDIVISION_CAMERA, TRANSPARENT_COMPOSITE, Table, VIRTUAL_BIN,
    VIRTUAL_POSES, VIRTUAL_SEED, VIRTUAL_SELECT, VIRTUAL_SHADOW, VIRTUAL_TRAVERSE,
    VISIBILITY_CLASSIFY, VISIBILITY_PASS, VISIBILITY_RESOLVE, WATER, WATER_BEHIND, WATER_PARTICLES,
    WATER_SETUP, WATER_SIM,
};

/// The bytes one uniform buffer binding holds on every device (wgpu's
/// downlevel defaults), which the renderer requests: blocks larger than
/// this bind as read-only storage (`cargo xtask gpu` checks every shader's
/// uniforms).
pub const UNIFORM_BINDING_BYTES: u64 = 16_384;

/// What a pipeline layout keeps within, for each stage (and, for dynamic
/// buffers, over the whole layout).
#[derive(Clone, Copy, Debug)]
pub struct StageLimits {
    pub bind_groups: u32,
    pub uniform_buffers: u32,
    pub storage_buffers: u32,
    pub sampled_textures: u32,
    pub samplers: u32,
    pub storage_textures: u32,
    pub dynamic_uniform_buffers: u32,
    pub dynamic_storage_buffers: u32,
}

/// The limits every device grants, WebGPU's defaults, which the renderer
/// requests: no pipeline layout exceeds them unless it names its own.
pub const REQUIRED_LIMITS: StageLimits = StageLimits {
    bind_groups: 4,
    uniform_buffers: 12,
    storage_buffers: 8,
    sampled_textures: 16,
    samplers: 16,
    storage_textures: 4,
    dynamic_uniform_buffers: 8,
    dynamic_storage_buffers: 4,
};

/// A pipeline layout: its label, its groups' tables in order and, where
/// devices granting more build it alone, the limits it needs.
#[derive(Debug)]
pub struct PipelineLayout {
    pub label: &'static str,
    pub groups: &'static [&'static Table],
    pub limits: StageLimits,
}

/// Declares the pipeline layouts and [`PIPELINE_LAYOUTS`] from one list:
/// `NAME = [TABLE, ..];`, or `NAME = [TABLE, ..] needing { field: n, .. };`
/// for a layout only devices granting more build.
macro_rules! pipeline_layouts {
    ($(
        $(#[$doc:meta])*
        $name:ident = [$($table:ident),+ $(,)?] $(needing { $($field:ident: $value:literal),+ $(,)? })?;
    )*) => {
        $(
            $(#[$doc])*
            pub const $name: PipelineLayout = PipelineLayout {
                label: concat!("engine/", stringify!($name)),
                groups: &[$(&$table),+],
                limits: StageLimits {
                    $($($field: $value,)+)?
                    ..REQUIRED_LIMITS
                },
            };
        )*

        /// Every pipeline layout by name.
        pub const PIPELINE_LAYOUTS: &[(&str, &PipelineLayout)] = &[$((stringify!($name), &$name)),*];
    };
}

pipeline_layouts! {
    /// Retained meshes: the camera, a material, the slots' poses; lit ones
    /// add the local lights.
    MESH_PIPELINES = [CAMERA, MATERIAL, RETAINED_POSES];
    LIT_MESH_PIPELINES = [CAMERA, MATERIAL, RETAINED_POSES, LIGHT_SHADING];
    /// Temporally antialiased meshes, posed as this frame and the last.
    TEMPORAL_MESH_PIPELINES = [CAMERA, MATERIAL, TEMPORAL_POSES];
    LIT_TEMPORAL_MESH_PIPELINES = [CAMERA, MATERIAL, TEMPORAL_POSES, LIGHT_SHADING];
    /// Virtual geometry's clusters.
    VIRTUAL_MESH_PIPELINES = [CAMERA, MATERIAL, VIRTUAL_POSES];
    LIT_VIRTUAL_MESH_PIPELINES = [CAMERA, MATERIAL, VIRTUAL_POSES, LIGHT_SHADING];
    /// The visibility buffer's pass and resolves; resolves bind nine storage
    /// buffers in fragments, lit ones twelve, which only devices granting
    /// them build (`Renderer::prepare_visibility` checks).
    VISIBILITY_PIPELINES = [CAMERA, EMPTY, VISIBILITY_PASS];
    RESOLVE_PIPELINES = [CAMERA, MATERIAL, VISIBILITY_RESOLVE] needing {
        storage_buffers: 9,
    };
    LIT_RESOLVE_PIPELINES = [CAMERA, MATERIAL, VISIBILITY_RESOLVE, LIGHT_SHADING] needing {
        storage_buffers: 12,
    };
    /// Impostor cards, retained or temporally antialiased, and their bakes.
    IMPOSTOR_PIPELINES = [CAMERA, IMPOSTOR_DRAW, RETAINED_POSES];
    LIT_IMPOSTOR_PIPELINES = [CAMERA, IMPOSTOR_DRAW, RETAINED_POSES, LIGHT_SHADING];
    TEMPORAL_IMPOSTOR_PIPELINES = [CAMERA, IMPOSTOR_DRAW, TEMPORAL_POSES];
    LIT_TEMPORAL_IMPOSTOR_PIPELINES = [CAMERA, IMPOSTOR_DRAW, TEMPORAL_POSES, LIGHT_SHADING];
    IMPOSTOR_BAKE_PIPELINES = [EMPTY, MATERIAL, IMPOSTOR_BAKE];
    /// Decals over the opaque surfaces.
    DECAL_PIPELINES = [CAMERA, MATERIAL, DECALS];
    LIT_DECAL_PIPELINES = [CAMERA, MATERIAL, DECALS, LIGHT_SHADING];
    /// The sky and the view's clouds over the scene.
    SKY_PIPELINES = [CAMERA, SKY_TRANSMITTANCE];
    CLOUD_COMPOSITE_PIPELINES = [CAMERA, CLOUD_COMPOSITE];
    /// Terrain's forward passes and visibility, and its resolves, over its
    /// own first group (`bindings::CAMERA_TERRAIN`).
    TERRAIN_PIPELINES = [CAMERA_TERRAIN, TERRAIN];
    LIT_TERRAIN_PIPELINES = [CAMERA_TERRAIN, TERRAIN, EMPTY, LIGHT_SHADING];
    TERRAIN_RESOLVE_PIPELINES = [CAMERA_TERRAIN, TERRAIN, TERRAIN_RESOLVE];
    LIT_TERRAIN_RESOLVE_PIPELINES = [CAMERA_TERRAIN, TERRAIN, TERRAIN_RESOLVE, LIGHT_SHADING];
    /// Terrain's subdivision: the camera, the terrain group and each
    /// terrain's tree, leaves and arguments.
    TERRAIN_SUBDIVISION_PIPELINES = [TERRAIN_SUBDIVISION_CAMERA, TERRAIN, TERRAIN_SUBDIVISION];
    /// Ground cover's blades.
    COVER_PIPELINES = [CAMERA, COVER_DRAW];
    LIT_COVER_PIPELINES = [CAMERA, COVER_DRAW, EMPTY, LIGHT_SHADING];
    /// Water over the opaque scene, over its own first group
    /// (`bindings::CAMERA_WATER`); particles and the haze's march.
    WATER_PIPELINES = [CAMERA_WATER, WATER, WATER_BEHIND];
    PARTICLE_PIPELINES = [CAMERA, PARTICLES];
    HAZE_PIPELINES = [CAMERA, HAZE];
    /// Shadow casters: plain, masked by their material, impostor cards;
    /// terrain's; their selection; and the marks of the tiles pixels read.
    CASTER_PIPELINES = [SHADOW_VIEW, SHADOW_POSES];
    MASKED_CASTER_PIPELINES = [SHADOW_VIEW, SHADOW_POSES, MATERIAL];
    IMPOSTOR_CASTER_PIPELINES = [SHADOW_VIEW, SHADOW_POSES, IMPOSTOR_DRAW];
    TERRAIN_CASTER_PIPELINES = [SHADOW_VIEW, TERRAIN];
    SHADOW_SELECT_PIPELINES = [SHADOW_SELECT];
    SHADOW_SKINNED_PIPELINES = [SHADOW_SKINNED];
    SHADOW_MARKS_PIPELINES = [SHADOW_MARKS];
    /// The screen effects, temporal antialiasing's resolve and
    /// transparency's composite; reflections over the main passes' first
    /// group as terrain binds it and their own, sixteen sampled textures and
    /// seven storage buffers in fragments.
    POST_PIPELINES = [POST];
    REFLECTION_PIPELINES = [CAMERA_TERRAIN, REFLECTIONS];
    TAA_RESOLVE_PIPELINES = [TAA_RESOLVE];
    TRANSPARENT_COMPOSITE_PIPELINES = [TRANSPARENT_COMPOSITE];
    /// The debug views' pass, built only while a debug view shows.
    DEBUG_VIEW_PIPELINES = [DEBUG_VIEW];
    /// Auto exposure's meter and adaptation, built only while it is on.
    EXPOSURE_PIPELINES = [EXPOSURE];
    /// Compute and screen passes of one group each.
    LIGHT_BINNING_PIPELINES = [LIGHT_BINNING];
    MIPMAP_PIPELINES = [MIPMAPS];
    PYRAMID_FIRST_PIPELINES = [PYRAMID_FIRST];
    PYRAMID_DOWN_PIPELINES = [PYRAMID_DOWN];
    ENVIRONMENT_SKY_PIPELINES = [ENVIRONMENT_SKY];
    ENVIRONMENT_DOWN_PIPELINES = [ENVIRONMENT_DOWN];
    ENVIRONMENT_PREFILTER_PIPELINES = [ENVIRONMENT_PREFILTER];
    CLOUD_MARCH_PIPELINES = [CLOUD_MARCH];
    CLOUD_VIEW_PIPELINES = [CLOUD_VIEW];
    SKY_TABLE_PIPELINES = [SKY_TABLES];
    DEFORMATION_PIPELINES = [DEFORMATION];
    /// Water's simulation: faces, fluxes, cells, the water rivers bring and
    /// the cells past its edges.
    WATER_SIM_PIPELINES = [WATER_SIM];
    /// The water's beds' and the live water's setup: an empty group, the
    /// terrain group, through which it reads the ground, its materials as
    /// their mesh band does and the tiles' water as water's shading does,
    /// and its own, with the live cells of the settled water's records.
    WATER_SETUP_PIPELINES = [EMPTY, TERRAIN, WATER_SETUP];
    /// Water's particles: water's first group, whose camera measures the
    /// pixel their rafts are born no smaller than, the water's group, read
    /// as its shading reads it, and the particles' own.
    WATER_PARTICLE_PIPELINES = [CAMERA_WATER, WATER, WATER_PARTICLES];
    COVER_GROW_PIPELINES = [COVER_GROW];
    OCCUPANCY_PIPELINES = [OCCUPANCY];
    SCATTER_SNAP_PIPELINES = [SCATTER_SNAP];
    SCATTER_CULL_PIPELINES = [SCATTER_CULL];
    TERRAIN_NORMAL_PIPELINES = [TERRAIN_NORMALS];
    TERRAIN_CONTROL_PIPELINES = [TERRAIN_CONTROL];
    /// Terrain bounce light: the ground's light, its levels and the
    /// probes' gathers, built once a terrain's bounce transfer arrives.
    BOUNCE_PIPELINES = [BOUNCE];
    VISIBILITY_CLASSIFY_PIPELINES = [VISIBILITY_CLASSIFY];
    VIRTUAL_SHADOW_PIPELINES = [VIRTUAL_SHADOW];
    VIRTUAL_SEED_PIPELINES = [VIRTUAL_SEED];
    VIRTUAL_TRAVERSE_PIPELINES = [VIRTUAL_TRAVERSE];
    VIRTUAL_SELECT_PIPELINES = [VIRTUAL_SELECT];
    VIRTUAL_BIN_PIPELINES = [VIRTUAL_BIN];
    /// GPU presentation's passes; parents' composition adds each level's
    /// group.
    PRESENTATION_COMPOSE_PIPELINES = [PRESENTATION_COMPOSE, PRESENTATION_LEVEL];
    PRESENTATION_CULL_PIPELINES = [PRESENTATION_CULL];
    PRESENTATION_TEST_PIPELINES = [PRESENTATION_TESTS];
    PRESENTATION_COMPACT_PIPELINES = [PRESENTATION_COMPACT];
    PRESENTATION_FADE_PIPELINES = [PRESENTATION_FADES];
    PRESENTATION_SKIN_BOUND_PIPELINES = [PRESENTATION_SKIN_BOUNDS];
}

/// Distinct pipeline layouts the renderer may build: one more costs every
/// Firefox start about 150 ms, so raising it is a decision, not a side
/// effect. Water's simulation builds its layout only once water is
/// simulated, and terrain bounce light its own once a terrain's transfer
/// arrives; the debug views' and auto exposure's layouts count though no
/// start builds them unless they are on. Terrain's subdivision, which
/// replaced the camera's patches selected on the CPU, takes one, and the
/// shadow clipmap's marks one of their own: the one pass reading the depth
/// buffer with the camera, which tells which tiles get memory. Reflections
/// take one: their world trace lights what it meets as the main passes
/// light it, through their first group, and walks the terrain's and placed
/// objects' data beside the screen's, more sampled textures than the screen
/// effects' shared layout leaves. Water's particles take one, built once
/// terrains' water draws: their passes read the water through the water's
/// group, as its shading reads it, and write the textures that shading
/// samples, which no group a water draw binds may write. The live water's
/// bed takes one, built once water runs live: it reads the ground's
/// materials through the terrain group, which the simulation's own layout,
/// at a compute stage's eight storage buffers, has no room for.
pub const LAYOUT_BUDGET: usize = 74;
