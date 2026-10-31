use engine_core::smoke::SmokeEntry;
use engine_core::{Absent, Face, FaceType, List, Place, Slot};

use crate::display::MeteredExposure;
use crate::environment::EnvironmentFace;
use crate::heightfield::HeightfieldFace;
use crate::lights::LightListFace;
use crate::occupancy::OccupancyFace;
use crate::placed::PlacedFace;
use crate::shadows::ShadowCastersFace;
use crate::sun::SunFace;

const fn contract(wgsl: &'static str, absent: Absent, gpu: Option<FaceType>) -> Face {
    Face {
        cpu: None,
        gpu,
        wgsl: Some(wgsl),
        absent,
    }
}

pub struct LightingSlot;

impl Slot for LightingSlot {
    const NAME: &'static str = "lighting";
    const FACE: Face = contract(include_str!("places/lighting.wgsl"), Absent::Required, None);
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
    const FACE: Face = contract(
        include_str!("places/sun.wgsl"),
        Absent::Nobody,
        Some(FaceType::of::<dyn SunFace>()),
    );
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
    const FACE: Face = contract(
        include_str!("places/shadows.wgsl"),
        Absent::Nobody,
        Some(FaceType::of::<dyn ShadowCastersFace>()),
    );
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
    const FACE: Face = contract(
        include_str!("places/lights.wgsl"),
        Absent::Nobody,
        Some(FaceType::of::<dyn LightListFace>()),
    );
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
    const FACE: Face = contract(
        include_str!("places/sky.wgsl"),
        Absent::Nobody,
        Some(FaceType::of::<dyn EnvironmentFace>()),
    );
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
    const FACE: Face = Face {
        cpu: None,
        gpu: Some(FaceType::of::<dyn MeteredExposure>()),
        wgsl: None,
        absent: Absent::Nobody,
    };
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
    const FACE: Face = contract(include_str!("places/tonemap.wgsl"), Absent::Nobody, None);
    const SMOKES: &'static [SmokeEntry] = &[SmokeEntry {
        name: "slot_tonemap",
        brick: Self::NAME,
        gpu: true,
        run: crate::smokes::tonemap,
    }];
}

pub struct BloomSlot;

impl Slot for BloomSlot {
    const NAME: &'static str = "bloom";
    const FACE: Face = contract(include_str!("places/bloom.wgsl"), Absent::Nobody, None);
    const SMOKES: &'static [SmokeEntry] = &[SmokeEntry {
        name: "slot_bloom",
        brick: Self::NAME,
        gpu: true,
        run: crate::smokes::bloom,
    }];
}

pub struct DitherSlot;

impl Slot for DitherSlot {
    const NAME: &'static str = "dither";
    const FACE: Face = contract(include_str!("places/dither.wgsl"), Absent::Nobody, None);
    const SMOKES: &'static [SmokeEntry] = &[SmokeEntry {
        name: "slot_dither",
        brick: Self::NAME,
        gpu: true,
        run: crate::smokes::dither,
    }];
}

pub struct ReflectionsSlot;

impl Slot for ReflectionsSlot {
    const NAME: &'static str = "reflections";
    const FACE: Face = contract(
        include_str!("places/reflections.wgsl"),
        Absent::Nobody,
        None,
    );
    const SMOKES: &'static [SmokeEntry] = &[SmokeEntry {
        name: "slot_reflections",
        brick: Self::NAME,
        gpu: true,
        run: crate::smokes::reflections,
    }];
}

pub struct ProbeLightSlot;

impl Slot for ProbeLightSlot {
    const NAME: &'static str = "probe_light";
    const FACE: Face = contract(
        include_str!("places/probe_light.wgsl"),
        Absent::Nobody,
        None,
    );
    const SMOKES: &'static [SmokeEntry] = &[SmokeEntry {
        name: "slot_probe_light",
        brick: Self::NAME,
        gpu: true,
        run: crate::smokes::probe_light,
    }];
}

pub struct OccupancySlot;

impl Slot for OccupancySlot {
    const NAME: &'static str = "occupancy";
    const FACE: Face = contract(
        include_str!("places/occupancy.wgsl"),
        Absent::Nobody,
        Some(FaceType::of::<dyn OccupancyFace>()),
    );
    const SMOKES: &'static [SmokeEntry] = &[SmokeEntry {
        name: "slot_occupancy",
        brick: Self::NAME,
        gpu: true,
        run: crate::smokes::occupancy,
    }];
}

pub struct PlacedSlot;

impl Slot for PlacedSlot {
    const NAME: &'static str = "placed";
    const FACE: Face = contract(
        include_str!("places/placed.wgsl"),
        Absent::Nobody,
        Some(FaceType::of::<dyn PlacedFace>()),
    );
    const SMOKES: &'static [SmokeEntry] = &[SmokeEntry {
        name: "slot_placed",
        brick: Self::NAME,
        gpu: true,
        run: crate::smokes::placed,
    }];
}

pub struct CoverSlot;

impl Slot for CoverSlot {
    const NAME: &'static str = "cover";
    const FACE: Face = contract(include_str!("places/cover.wgsl"), Absent::Nobody, None);
    const SMOKES: &'static [SmokeEntry] = &[SmokeEntry {
        name: "slot_cover",
        brick: Self::NAME,
        gpu: true,
        run: crate::smokes::cover,
    }];
}

pub struct HorizonsSlot;

impl Slot for HorizonsSlot {
    const NAME: &'static str = "horizons";
    const FACE: Face = contract(include_str!("places/horizons.wgsl"), Absent::Nobody, None);
    const SMOKES: &'static [SmokeEntry] = &[SmokeEntry {
        name: "slot_horizons",
        brick: Self::NAME,
        gpu: true,
        run: crate::smokes::horizons,
    }];
}

pub struct HeightfieldSlot;

impl Slot for HeightfieldSlot {
    const NAME: &'static str = "heightfield";
    const FACE: Face = contract(
        include_str!("places/heightfield.wgsl"),
        Absent::Required,
        Some(FaceType::of::<dyn HeightfieldFace>()),
    );
    const SMOKES: &'static [SmokeEntry] = &[SmokeEntry {
        name: "slot_heightfield",
        brick: Self::NAME,
        gpu: true,
        run: crate::smokes::heightfield,
    }];
}

pub struct GroundTilesSlot;

impl Slot for GroundTilesSlot {
    const NAME: &'static str = "ground_tiles";
    const FACE: Face = contract(
        include_str!("places/ground_tiles.wgsl"),
        Absent::Required,
        None,
    );
    const SMOKES: &'static [SmokeEntry] = &[SmokeEntry {
        name: "slot_ground_tiles",
        brick: Self::NAME,
        gpu: true,
        run: crate::smokes::ground_tiles,
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

pub static PLACES: &[Place] = &[
    Place::slot::<LightingSlot>(),
    Place::slot::<MeshesSlot>(),
    Place::slot::<SunSlot>(),
    Place::slot::<ShadowsSlot>(),
    Place::slot::<LightsSlot>(),
    Place::slot::<SkySlot>(),
    Place::slot::<AntialiasingSlot>(),
    Place::slot::<ExposureSlot>(),
    Place::slot::<TonemapSlot>(),
    Place::slot::<BloomSlot>(),
    Place::slot::<DitherSlot>(),
    Place::slot::<ReflectionsSlot>(),
    Place::slot::<ProbeLightSlot>(),
    Place::slot::<OccupancySlot>(),
    Place::slot::<PlacedSlot>(),
    Place::slot::<CoverSlot>(),
    Place::slot::<HorizonsSlot>(),
    Place::slot::<HeightfieldSlot>(),
    Place::slot::<GroundTilesSlot>(),
    Place::list::<WaterList>(),
    Place::list::<GroundCoverList>(),
    Place::list::<ScatterList>(),
    Place::list::<TerrainShadingList>(),
    Place::list::<ProbesList>(),
    Place::list::<ParticlesList>(),
    Place::list::<DebugViewList>(),
    Place::list::<PostEffectsList>(),
];
