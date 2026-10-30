use engine_core::smoke::SmokeEntry;
use engine_core::{List, Slot};

pub struct LightingSlot;

impl Slot for LightingSlot {
    const NAME: &'static str = "lighting";
    const SMOKES: &'static [SmokeEntry] = &[SmokeEntry {
        name: "slot_lighting",
        brick: Self::NAME,
        gpu: true,
        run: crate::smokes::lighting,
    }];
}

pub struct MeshesSlot;

impl Slot for MeshesSlot {
    const NAME: &'static str = "meshes";
    const SMOKES: &'static [SmokeEntry] = &[SmokeEntry {
        name: "slot_meshes",
        brick: Self::NAME,
        gpu: true,
        run: crate::smokes::meshes,
    }];
}

pub struct SunSlot;

impl Slot for SunSlot {
    const NAME: &'static str = "sun";
    const SMOKES: &'static [SmokeEntry] = &[SmokeEntry {
        name: "slot_sun",
        brick: Self::NAME,
        gpu: true,
        run: crate::smokes::sun,
    }];
}

pub struct ShadowsSlot;

impl Slot for ShadowsSlot {
    const NAME: &'static str = "shadows";
    const SMOKES: &'static [SmokeEntry] = &[SmokeEntry {
        name: "slot_shadows",
        brick: Self::NAME,
        gpu: true,
        run: crate::smokes::shadows,
    }];
}

pub struct LightsSlot;

impl Slot for LightsSlot {
    const NAME: &'static str = "lights";
    const SMOKES: &'static [SmokeEntry] = &[SmokeEntry {
        name: "slot_lights",
        brick: Self::NAME,
        gpu: true,
        run: crate::smokes::lights,
    }];
}

pub struct SkySlot;

impl Slot for SkySlot {
    const NAME: &'static str = "sky";
    const SMOKES: &'static [SmokeEntry] = &[SmokeEntry {
        name: "slot_sky",
        brick: Self::NAME,
        gpu: true,
        run: crate::smokes::sky,
    }];
}

pub struct AntialiasingSlot;

impl Slot for AntialiasingSlot {
    const NAME: &'static str = "antialiasing";
    const SMOKES: &'static [SmokeEntry] = &[SmokeEntry {
        name: "slot_antialiasing",
        brick: Self::NAME,
        gpu: true,
        run: crate::smokes::antialiasing,
    }];
}

pub struct ExposureSlot;

impl Slot for ExposureSlot {
    const NAME: &'static str = "exposure";
    const SMOKES: &'static [SmokeEntry] = &[SmokeEntry {
        name: "slot_exposure",
        brick: Self::NAME,
        gpu: true,
        run: crate::smokes::exposure,
    }];
}

pub struct TonemapSlot;

impl Slot for TonemapSlot {
    const NAME: &'static str = "tonemap";
    const SMOKES: &'static [SmokeEntry] = &[SmokeEntry {
        name: "slot_tonemap",
        brick: Self::NAME,
        gpu: true,
        run: crate::smokes::tonemap,
    }];
}

pub struct WaterList;

impl List for WaterList {
    const NAME: &'static str = "water";
    const SMOKES: &'static [SmokeEntry] = &[SmokeEntry {
        name: "list_water",
        brick: Self::NAME,
        gpu: true,
        run: crate::smokes::water,
    }];
}

pub struct GroundCoverList;

impl List for GroundCoverList {
    const NAME: &'static str = "ground_cover";
    const SMOKES: &'static [SmokeEntry] = &[SmokeEntry {
        name: "list_ground_cover",
        brick: Self::NAME,
        gpu: true,
        run: crate::smokes::ground_cover,
    }];
}

pub struct ScatterList;

impl List for ScatterList {
    const NAME: &'static str = "scatter";
    const SMOKES: &'static [SmokeEntry] = &[SmokeEntry {
        name: "list_scatter",
        brick: Self::NAME,
        gpu: true,
        run: crate::smokes::scatter,
    }];
}

pub struct TerrainShadingList;

impl List for TerrainShadingList {
    const NAME: &'static str = "terrain_shading";
    const SMOKES: &'static [SmokeEntry] = &[SmokeEntry {
        name: "list_terrain_shading",
        brick: Self::NAME,
        gpu: true,
        run: crate::smokes::terrain_shading,
    }];
}

pub struct ProbesList;

impl List for ProbesList {
    const NAME: &'static str = "probes";
    const SMOKES: &'static [SmokeEntry] = &[SmokeEntry {
        name: "list_probes",
        brick: Self::NAME,
        gpu: true,
        run: crate::smokes::probes,
    }];
}

pub struct ParticlesList;

impl List for ParticlesList {
    const NAME: &'static str = "particles";
    const SMOKES: &'static [SmokeEntry] = &[SmokeEntry {
        name: "list_particles",
        brick: Self::NAME,
        gpu: true,
        run: crate::smokes::particles,
    }];
}

pub struct DebugViewList;

impl List for DebugViewList {
    const NAME: &'static str = "debug_view";
    const SMOKES: &'static [SmokeEntry] = &[SmokeEntry {
        name: "list_debug_view",
        brick: Self::NAME,
        gpu: true,
        run: crate::smokes::debug_view,
    }];
}

pub struct PostEffectsList;

impl List for PostEffectsList {
    const NAME: &'static str = "post_effects";
    const SMOKES: &'static [SmokeEntry] = &[SmokeEntry {
        name: "list_post_effects",
        brick: Self::NAME,
        gpu: true,
        run: crate::smokes::post_effects,
    }];
}
