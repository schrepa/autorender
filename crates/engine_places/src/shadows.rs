use bevy_ecs::prelude::{Component, Resource};
use engine_place::{DrawPoint, Slot};
use glam::{Mat4, Vec3};

use crate::ground::Pyramid;

pub const GROUND_TEXELS: f32 = 2.0;
pub const SHADOW_REGIONS: u32 = 7;
pub const DEBUG_SUN_NONE: u32 = 0;
pub const DEBUG_SUN_LEVEL: u32 = 1;
pub const DEBUG_SUN_HORIZON: u32 = 21;
pub const DEBUG_SUN_BEYOND: u32 = DEBUG_SUN_HORIZON + 1;
pub const DEBUG_SHADOWED: u32 = 0x80;

pub fn ground_level(pyramid: &Pyramid, texel: f32) -> u32 {
    let coarsest = pyramid.levels().saturating_sub(1);
    (0..coarsest)
        .find(|&level| pyramid.spacing(level) >= GROUND_TEXELS * texel)
        .unwrap_or(coarsest)
}

#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct ShadowMaps {
    pub on: bool,
    pub writes: bool,
    pub sky_darkens: bool,
    pub levels: u32,
    pub regions: Vec<Region>,
}

impl Slot for ShadowMaps {
    const NAME: &'static str = "shadows";
    const WGSL: Option<&'static str> = Some(contract!(
        "places/shadows.wgsl",
        DEBUG_SUN_NONE,
        DEBUG_SUN_LEVEL,
        DEBUG_SUN_HORIZON,
        DEBUG_SUN_BEYOND,
        DEBUG_SHADOWED;
        GROUND_TEXELS,
    ));
    const SMOKES: &'static [engine_place::Smoke] = &[crate::smokes::SHADOWS];
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Region {
    pub view: Mat4,
    pub map: CasterMap,
    pub refresh: Refresh,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CasterMap {
    Region,
    Moving,
    Sky,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KeptRefresh {
    pub planes: [[f32; 4]; 6],
    pub texel: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Refresh {
    Kept(KeptRefresh),
    Redraw,
}

#[derive(Resource, Default)]
pub struct ShadowCasters;

impl Slot for ShadowCasters {
    const NAME: &'static str = "shadow_casters";
}

impl DrawPoint for ShadowCasters {
    type Layer = ();
    type Args = CasterView;
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CasterView {
    pub region: u32,
    pub map: CasterMap,
    pub refresh: Refresh,
}

#[derive(Resource, Default)]
pub struct CasterRoutes;

impl Slot for CasterRoutes {
    const NAME: &'static str = "caster_routes";
    const WGSL: Option<&'static str> = Some(include_str!("places/caster_routes.wgsl"));
}

#[derive(Component, Clone, Debug, Default, PartialEq)]
pub struct ShadowChanges {
    pub boxes: Vec<(Vec3, Vec3, f32)>,
    pub all: bool,
    pub casters: bool,
}
