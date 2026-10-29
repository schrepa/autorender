use engine_core::{List, Slot};

pub struct LightingSlot;

impl Slot for LightingSlot {
    const NAME: &'static str = "lighting";
}

pub struct MeshesSlot;

impl Slot for MeshesSlot {
    const NAME: &'static str = "meshes";
}

pub struct SunSlot;

impl Slot for SunSlot {
    const NAME: &'static str = "sun";
}

pub struct ShadowsSlot;

impl Slot for ShadowsSlot {
    const NAME: &'static str = "shadows";
}

pub struct LightsSlot;

impl Slot for LightsSlot {
    const NAME: &'static str = "lights";
}

pub struct SkySlot;

impl Slot for SkySlot {
    const NAME: &'static str = "sky";
}

pub struct AntialiasingSlot;

impl Slot for AntialiasingSlot {
    const NAME: &'static str = "antialiasing";
}

pub struct ExposureSlot;

impl Slot for ExposureSlot {
    const NAME: &'static str = "exposure";
}

pub struct TonemapSlot;

impl Slot for TonemapSlot {
    const NAME: &'static str = "tonemap";
}

pub struct WaterList;

impl List for WaterList {
    const NAME: &'static str = "water";
}

pub struct GroundCoverList;

impl List for GroundCoverList {
    const NAME: &'static str = "ground_cover";
}

pub struct ScatterList;

impl List for ScatterList {
    const NAME: &'static str = "scatter";
}

pub struct TerrainShadingList;

impl List for TerrainShadingList {
    const NAME: &'static str = "terrain_shading";
}

pub struct ProbesList;

impl List for ProbesList {
    const NAME: &'static str = "probes";
}

pub struct ParticlesList;

impl List for ParticlesList {
    const NAME: &'static str = "particles";
}

pub struct DebugViewList;

impl List for DebugViewList {
    const NAME: &'static str = "debug_view";
}

pub struct PostEffectsList;

impl List for PostEffectsList {
    const NAME: &'static str = "post_effects";
}
