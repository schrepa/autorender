use std::collections::BTreeMap;

use serde::{Serialize, Serializer};

use crate::Cut;

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct FrameStats {
    pub frame: u64,
    pub tick: u64,
    pub simulation_ms: f64,
    pub extraction_ms: f64,
    pub render_ms: f64,
    pub invalid_transforms: u32,
    pub render_candidates: u32,
    pub visible_instances: u32,
    pub culled_instances: u32,
    pub dropped_instances: u32,
    pub missing_assets: u32,
    pub culling_enabled: bool,
    pub fixed_steps: u32,
    pub dropped_seconds: f64,
    pub pipelines_compiling: bool,
    pub pipelines_listed: u32,
    pub pipelines_unlisted: u32,
    pub loading: bool,
    pub resolution: ResolutionStats,
    pub dynamic_resolution: DynamicResolutionStats,
    pub jobs: crate::JobStats,
    pub disabled_features: Names,
    #[serde(flatten)]
    pub render: RenderStats,
    pub bricks_cost: Option<crate::BrickCosts>,
    #[serde(flatten)]
    #[cfg_attr(feature = "schema", schemars(skip))]
    pub bricks: BTreeMap<&'static str, serde_json::Value>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct RenderStats {
    pub draw_calls: u32,
    pub upload_bytes: u64,
    pub instances: u32,
    #[serde(skip)]
    #[cfg_attr(feature = "schema", schemars(skip))]
    pub gpu_visible: Option<u32>,
    pub triangles: Option<u32>,
    pub read_back_age: Option<u32>,
    pub resident_meshes: u32,
    pub mesh_upload_bytes: u64,
    pub resident_mesh_bytes: u64,
    pub resident_textures: u32,
    pub texture_upload_bytes: u64,
    pub resident_texture_bytes: u64,
    #[serde(skip)]
    #[cfg_attr(feature = "schema", schemars(skip))]
    pub submitted: bool,
    pub pipelines: u32,
    pub pipelines_created: u32,
    pub render_passes: Names,
    pub shadows: ShadowStats,
    pub ambient_occlusion: Applied,
    pub bloom: Applied,
    pub local_lights: LightStats,
    pub haze: Applied,
    pub reflections: Applied,
    pub particles: ParticleStats,
    pub reflection_probes: ProbeStats,
    pub level_of_detail: DetailStats,
    pub terrain: TerrainRenderStats,
    pub ground_cover: GroundCoverStats,
    pub occupancy: OccupancyStats,
    pub water: WaterStats,
    pub clouds: CloudStats,
    pub sky: SkyStats,
    pub occlusion_culling: CullingStats,
    pub subpixel_culling: CullingStats,
    pub temporal_antialiasing: Applied,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Names {
    pub bits: u64,
    pub names: &'static [&'static str],
}

impl Serialize for Names {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(
            self.names
                .iter()
                .enumerate()
                .filter(|(index, _)| self.bits & (1 << index) != 0)
                .map(|(_, name)| name),
        )
    }
}

#[cfg(feature = "schema")]
impl schemars::JsonSchema for Names {
    fn inline_schema() -> bool {
        true
    }

    fn schema_name() -> std::borrow::Cow<'static, str> {
        "Names".into()
    }

    fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        <Vec<String>>::json_schema(generator)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Applied {
    pub active: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ResolutionStats {
    pub display_width: u32,
    pub display_height: u32,
    pub render_width: u32,
    pub render_height: u32,
    pub dropped_pixels: u64,
    pub gpu_ms: Option<f32>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DynamicResolutionStats {
    pub active: bool,
    pub scale: f32,
    pub budget_ms: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ParticleStats {
    pub drawn: u32,
    pub invalid: u32,
    pub particle_cut: Cut,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct CullingStats {
    pub active: bool,
    pub instances: Option<u32>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ShadowStats {
    pub active: bool,
    pub draws: u32,
    pub tiles_marked: u32,
    pub tiles_resident: u32,
    pub tiles_drawn: u32,
    pub tiles_reclaimed: u32,
    pub moving_tiles: u32,
    pub casters: Option<u32>,
    pub moving_casters: Option<u32>,
    pub sky_maps: u32,
    pub sky_drawn: u32,
    pub tile_cut: Cut,
    pub moving_cut: Cut,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct LightStats {
    pub shaded: u32,
    pub invalid: u32,
    pub visible: Option<u32>,
    pub light_cut: Cut,
    pub cluster_cut: Cut,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ProbeStats {
    pub ready: u32,
    pub faces: u32,
    pub probe_cut: Cut,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DetailStats {
    pub lod0_instances: Option<u32>,
    pub lod1_instances: Option<u32>,
    pub lod2_instances: Option<u32>,
    pub impostor_instances: Option<u32>,
    pub impostors: u32,
    pub meshes_without_levels: u32,
    pub impostor_cut: Cut,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct TerrainBounceStats {
    pub relit: u32,
    pub probes: u32,
    pub levels: u32,
    pub dropped: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct TerrainRenderStats {
    pub leaves: u32,
    pub triangles: u32,
    pub shallow_subdivisions: u32,
    pub shadow_patches: u32,
    pub tiles: u32,
    pub loading_tiles: u32,
    pub gpu_controls: u32,
    pub dropped_tiles: u32,
    pub fallback_tiles: u32,
    pub horizons: u32,
    pub sky_maps: u32,
    pub lighting_uploads: u32,
    pub bounce: TerrainBounceStats,
    pub macro_maps: u32,
    pub dropped_materials: u32,
    pub placed_objects: u32,
    pub placed_far_objects: u32,
    pub placed_impostors: u32,
    pub dropped_placed_objects: u32,
    pub white_water: u32,
    pub tile_cut: Cut,
    pub leaf_cut: Cut,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct GroundCoverStats {
    pub blades: u32,
    pub reach: u32,
    pub tiers: u32,
    pub blade_cut: Cut,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct OccupancyStats {
    pub redraws: u32,
    pub averages: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct WaterStats {
    pub leaves: u32,
    pub blocks: u32,
    pub dropped_blocks: u32,
    pub live_cells: u32,
    pub live_cut: u32,
    pub live_steps: u32,
    pub live_behind: u32,
    pub live_setup: u32,
    pub live_waiting: u32,
    pub touches: u32,
    pub dropped_touches: u32,
    pub particle_texels: u32,
    pub foam_points: u32,
    pub foam_reach: u32,
    pub foam_particles: u32,
    pub wave_particles: u32,
    pub particles_born: u32,
    pub dropped_particles: u32,
    pub cell_cut: Cut,
    pub foam_cut: Cut,
    pub leaf_cut: Cut,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct CloudStats {
    pub rows: u32,
    pub view_pixels: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct SkyStats {
    pub tables: u32,
    pub exposure: ExposureStats,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ExposureStats {
    pub active: bool,
    pub stops: i32,
    pub metered: i32,
    pub view: i32,
}
