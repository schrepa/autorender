use engine_core::{Absent, Face, FaceType, Place, Slot};

use crate::display::MeteredExposure;
use crate::environment::EnvironmentFace;
use crate::heightfield::HeightfieldFace;
use crate::lights::LightListFace;
use crate::occupancy::OccupancyFace;
use crate::placed::PlacedFace;
use crate::shadows::ShadowCastersFace;
use crate::slots::{
    AntialiasingSlot, DebugViewList, ExposureSlot, GroundCoverList, LightingSlot, LightsSlot,
    MeshesSlot, ParticlesList, PostEffectsList, ProbesList, ScatterList, ShadowsSlot, SkySlot,
    SunSlot, TerrainShadingList, TonemapSlot, WaterList,
};
use crate::sun::SunFace;

const fn contract(wgsl: &'static str, absent: Absent, gpu: Option<FaceType>) -> Face {
    Face {
        cpu: None,
        gpu,
        wgsl: Some(wgsl),
        absent,
    }
}

const fn faced(place: Place, face: Face) -> Place {
    Place { face, ..place }
}

pub const LIGHTING: Face = contract(include_str!("places/lighting.wgsl"), Absent::Required, None);

pub const SUN: Face = contract(
    include_str!("places/sun.wgsl"),
    Absent::Nobody,
    Some(FaceType::of::<dyn SunFace>()),
);

pub const SHADOWS: Face = contract(
    include_str!("places/shadows.wgsl"),
    Absent::Nobody,
    Some(FaceType::of::<dyn ShadowCastersFace>()),
);

pub const LIGHTS: Face = contract(
    include_str!("places/lights.wgsl"),
    Absent::Nobody,
    Some(FaceType::of::<dyn LightListFace>()),
);

pub const SKY: Face = contract(
    include_str!("places/sky.wgsl"),
    Absent::Nobody,
    Some(FaceType::of::<dyn EnvironmentFace>()),
);

pub const EXPOSURE: Face = Face {
    cpu: None,
    gpu: Some(FaceType::of::<dyn MeteredExposure>()),
    wgsl: None,
    absent: Absent::Nobody,
};

pub const TONEMAP: Face = contract(include_str!("places/tonemap.wgsl"), Absent::Nobody, None);

pub const BLOOM: Face = contract(include_str!("places/bloom.wgsl"), Absent::Nobody, None);

pub const DITHER: Face = contract(include_str!("places/dither.wgsl"), Absent::Nobody, None);

pub const REFLECTIONS: Face = contract(
    include_str!("places/reflections.wgsl"),
    Absent::Nobody,
    None,
);

pub const PROBE_LIGHT: Face = contract(
    include_str!("places/probe_light.wgsl"),
    Absent::Nobody,
    None,
);

pub const OCCUPANCY: Face = contract(
    include_str!("places/occupancy.wgsl"),
    Absent::Nobody,
    Some(FaceType::of::<dyn OccupancyFace>()),
);

pub const PLACED: Face = contract(
    include_str!("places/placed.wgsl"),
    Absent::Nobody,
    Some(FaceType::of::<dyn PlacedFace>()),
);

pub const COVER: Face = contract(include_str!("places/cover.wgsl"), Absent::Nobody, None);

pub const HORIZONS: Face = contract(include_str!("places/horizons.wgsl"), Absent::Nobody, None);

pub const HEIGHTFIELD: Face = contract(
    include_str!("places/heightfield.wgsl"),
    Absent::Required,
    Some(FaceType::of::<dyn HeightfieldFace>()),
);

pub const GROUND_TILES: Face = contract(
    include_str!("places/ground_tiles.wgsl"),
    Absent::Required,
    None,
);

pub struct BloomSlot;

impl Slot for BloomSlot {
    const NAME: &'static str = "bloom";
    const FACE: Face = BLOOM;
}

pub struct DitherSlot;

impl Slot for DitherSlot {
    const NAME: &'static str = "dither";
    const FACE: Face = DITHER;
}

pub struct ReflectionsSlot;

impl Slot for ReflectionsSlot {
    const NAME: &'static str = "reflections";
    const FACE: Face = REFLECTIONS;
}

pub struct ProbeLightSlot;

impl Slot for ProbeLightSlot {
    const NAME: &'static str = "probe_light";
    const FACE: Face = PROBE_LIGHT;
}

pub struct OccupancySlot;

impl Slot for OccupancySlot {
    const NAME: &'static str = "occupancy";
    const FACE: Face = OCCUPANCY;
}

pub struct PlacedSlot;

impl Slot for PlacedSlot {
    const NAME: &'static str = "placed";
    const FACE: Face = PLACED;
}

pub struct CoverSlot;

impl Slot for CoverSlot {
    const NAME: &'static str = "cover";
    const FACE: Face = COVER;
}

pub struct HorizonsSlot;

impl Slot for HorizonsSlot {
    const NAME: &'static str = "horizons";
    const FACE: Face = HORIZONS;
}

pub struct HeightfieldSlot;

impl Slot for HeightfieldSlot {
    const NAME: &'static str = "heightfield";
    const FACE: Face = HEIGHTFIELD;
}

pub struct GroundTilesSlot;

impl Slot for GroundTilesSlot {
    const NAME: &'static str = "ground_tiles";
    const FACE: Face = GROUND_TILES;
}

pub static PLACES: &[Place] = &[
    faced(Place::slot::<LightingSlot>(), LIGHTING),
    Place::slot::<MeshesSlot>(),
    faced(Place::slot::<SunSlot>(), SUN),
    faced(Place::slot::<ShadowsSlot>(), SHADOWS),
    faced(Place::slot::<LightsSlot>(), LIGHTS),
    faced(Place::slot::<SkySlot>(), SKY),
    Place::slot::<AntialiasingSlot>(),
    faced(Place::slot::<ExposureSlot>(), EXPOSURE),
    faced(Place::slot::<TonemapSlot>(), TONEMAP),
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
