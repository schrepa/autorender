engine::brick!(effects);

mod assemble;
mod foes;
mod gather;
mod gpu;
mod models;
mod numbers;
mod previews;

use std::ops::Range;

use engine::prelude::*;
use engine_core::{Attributes, DrawKind, KindDecl};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::names::COLOR_PASS;
use crate::play::effects::EffectsFeed;

pub const EFFECTS: DrawKind = DrawKind("effects");

pub const DEVICE: DrawKind = DrawKind("device");

pub const EFFECTS_PASS: &str = "effects";

const GENERATED: &[&str] = &[<EffectsFrame as GpuStruct>::WGSL];

gpu_struct! {
    pub struct EffectsFrame {
        pub clock: [f32; 4],
        pub flash: [f32; 4],
        pub damage: [f32; 4],
        pub sun: [f32; 4],
        pub sun_color: [f32; 4],
        pub sky_color: [f32; 4],
        pub field: [f32; 4],
        pub field_shape: [f32; 4],
        pub rind_skin: [f32; 4],
        pub rind_stripe: [f32; 4],
        pub rind_pith: [f32; 4],
        pub rind_flesh: [f32; 4],
        pub rind_seed: [f32; 4],
        pub curtain_peel: [f32; 4],
        pub curtain_pulp: [f32; 4],
        pub curtain_foam: [f32; 4],
        pub curtain_shape: [f32; 4],
        pub panel: [f32; 4],
        pub zones: [f32; 4],
        pub barrier: [f32; 4],
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct Item {
    pub rows: [[f32; 4]; 8],
}

#[derive(Resource, Default)]
pub(crate) struct EffectsDrawing {
    pub items: Vec<Item>,
    pub glow: Range<u32>,
    pub solid: Range<u32>,
    pub slab: Range<u32>,
    pub sheet: Range<u32>,
    pub device: Range<u32>,
    pub smoke: Range<u32>,
    pub marks: Range<u32>,
    pub barrier: Range<u32>,
    pub glow_vertices: u32,
    pub frame: EffectsFrame,
    pub screen: bool,
    pub field: Option<engine_core::TextureRef>,
    pub covered: bool,
}

impl EffectsDrawing {
    pub fn own_pass(&self) -> bool {
        self.screen || !self.smoke.is_empty() || !self.marks.is_empty()
    }
}

#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Device {}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Show {
    #[default]
    Off,
    All,
    Shots,
    Impacts,
    Juice,
    Smoke,
    Walls,
    Zones,
    Traps,
    Strike,
    Pulse,
    Device,
    Flash,
    NearSight,
    Damage,
    InsideSmoke,
    Barrier,
    Orbs,
    Shatter,
    Wire,
    Models,
}

settings! {
    #[section("effects_show")]
    #[serde(deny_unknown_fields)]
    pub struct EffectsShow {
        pub show: Show = Show::Off => any,
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ScreenStats {
    pub flash: f32,
    pub damage: f32,
    pub near_sight: f32,
}

stats! {
    #[section("effects")]
    pub struct EffectsStats {
        pub refusal: Option<String>,
        pub tracers: u32,
        pub flashes: u32,
        pub sprays: u32,
        pub decals: u32,
        pub rings: u32,
        pub previews: u32,
        pub smokes: u32,
        pub zones: u32,
        pub panels: u32,
        pub slabs: u32,
        pub glowing: u32,
        pub blobs: u32,
        pub trails: u32,
        pub barrier_faces: u32,
        pub barrier_drop: f32,
        pub orbs: u32,
        pub orb_beams: u32,
        pub orb_bursts: u32,
        pub shatters: u32,
        pub faded_wires: u32,
        pub covered: bool,
        pub lobby: bool,
        pub shadows: u32,
        pub puffs: u32,
        pub meshed: u32,
        pub showcase_model: Option<String>,
        pub lamp: f32,
        pub device: bool,
        pub screen: ScreenStats,
        pub draws: u32,
        pub instances: u32,
        pub bytes: u64,
    }
}

pub struct EffectsBrick;

impl Brick for EffectsBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .join::<SceneComponents>(
                ComponentKind::of::<Device>("device").schema(engine_core::schema_of::<Device>),
            )
            .settings::<EffectsShow>()
            .stats::<EffectsStats>()
            .resource::<EffectsFeed>()
            .resource::<gather::Setup>()
            .resource::<gather::Spawns>()
            .resource::<EffectsDrawing>()
            .resource::<models::ShowcaseModels>()
            .moves::<Device>(Pace::Tick)
            .moves::<models::ShowcaseModel>(Pace::Tick)
            .draw_kind(KindDecl {
                kind: DEVICE,
                needs: Attributes::NONE,
                streams: &[],
                passes: &[COLOR_PASS],
            })
            .draw_kind(KindDecl {
                kind: EFFECTS,
                needs: Attributes::NONE,
                streams: &[],
                passes: &[COLOR_PASS],
            })
            .generated(GENERATED)
            .draws::<gpu::EffectsGpu>(DEVICE, gpu::EffectsGpu::draw_device)
            .draws::<gpu::EffectsGpu>(EFFECTS, gpu::EffectsGpu::draw_effects)
            .systems(Schedule::FixedSimulation, gather::setup)
            .systems(Schedule::FixedSimulation, gather::gather)
            .systems(Schedule::FixedSimulation, gather::place_device)
            .systems(Schedule::FixedSimulation, models::showcase_models)
            .systems(Schedule::Presentation, assemble::assemble)
            .systems(Schedule::Presentation, foes::mark_figures);
    }
}
