use bevy_ecs::prelude::Resource;
use engine_place::{Cursor, Feed, Since, Slot};
use glam::{Mat4, Vec3};

pub const GROUND_TEXELS: f32 = 2.0;
pub const SHADOW_REGIONS: u32 = 7;
pub const SHADOW_CHANGES: usize = 1024;
pub const SHADOW_PASS: &str = "shadow";
pub const DEBUG_SUN_NONE: u32 = 0;
pub const DEBUG_SUN_HORIZON: u32 = 1;
pub const DEBUG_SUN_BEYOND: u32 = 2;
pub const DEBUG_SUN_LEVEL: u32 = 3;

#[engine_place::gpu_bits(crate = engine_place)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SunShown {
    #[bits(7)]
    pub source: u32,
    pub shadowed: bool,
}

pub fn ground_level(levels: u32, spacing: impl Fn(u32) -> f32, texel: f32) -> u32 {
    let coarsest = levels.saturating_sub(1);
    (0..coarsest)
        .find(|&level| spacing(level) >= GROUND_TEXELS * texel)
        .unwrap_or(coarsest)
}

#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct ShadowMaps {
    pub on: bool,
    pub writes: bool,
    pub levels: u32,
    pub regions: Vec<Region>,
}

impl ShadowMaps {
    pub fn views(&self) -> impl Iterator<Item = CasterView> + '_ {
        self.regions
            .iter()
            .enumerate()
            .map(|(index, region)| CasterView {
                region: index as u32,
                map: region.map,
                refresh: region.refresh,
            })
    }

    pub fn view(&self, view: u32) -> Option<CasterView> {
        self.views().nth(view as usize)
    }
}

impl Slot for ShadowMaps {
    const NAME: &'static str = "shadows";
    const WGSL: Option<&'static str> = Some(engine_places::contract!(
        "places/shadows.wgsl",
        DEBUG_SUN_NONE,
        DEBUG_SUN_LEVEL,
        DEBUG_SUN_HORIZON,
        DEBUG_SUN_BEYOND;
        GROUND_TEXELS,
    ));
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CasterView {
    pub region: u32,
    pub map: CasterMap,
    pub refresh: Refresh,
}

impl CasterView {
    pub const fn view(&self) -> u32 {
        self.region
    }
}

#[derive(Resource, Default)]
pub struct CasterRoutes;

impl Slot for CasterRoutes {
    const NAME: &'static str = "caster_routes";
    const WGSL: Option<&'static str> = Some(include_str!("places/caster_routes.wgsl"));
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ShadowChange {
    Box { min: Vec3, max: Vec3, spacing: f32 },
    All,
    Casters,
}

#[derive(Resource, Clone, Debug)]
pub struct ShadowChanges {
    feed: Feed<ShadowChange>,
}

impl Default for ShadowChanges {
    fn default() -> Self {
        Self {
            feed: Feed::new(SHADOW_CHANGES),
        }
    }
}

impl ShadowChanges {
    pub fn push(&mut self, change: ShadowChange) {
        self.feed.push(change);
    }

    pub fn since<'a>(&'a self, cursor: &mut Cursor) -> Since<'a, ShadowChange> {
        self.feed.since(cursor)
    }
}
