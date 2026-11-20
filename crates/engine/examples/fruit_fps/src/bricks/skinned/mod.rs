engine::brick!(skinned);

mod clock;
mod play;
mod pose;
mod rig;

use engine_core::{
    AssetKind, AssetKinds, Brick, Bricks, ComponentKind, Pace, SceneComponents, Schedule, Source,
    stats,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::play::skinned::{Figure, FigurePalette, FigureSight, Held, StepJoints};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct FigureHead {
    pub entity: String,
    pub time: f32,
    pub head: Option<[f32; 3]>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct FigureRefused {
    pub entity: String,
    pub part: String,
    pub reason: String,
}

stats! {
    #[section("skinned")]
    pub struct SkinnedStats {
        pub sampled: u32,
        pub ticked: u32,
        pub unposed: u32,
        pub joints: u32,
        pub palette_bytes: u64,
        pub skeletons: u32,
        pub clips: u32,
        pub heads: Vec<FigureHead>,
        pub refused: Vec<FigureRefused>,
    }
}

pub struct SkinnedBrick;

impl Brick for SkinnedBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .join::<SceneComponents>(
                ComponentKind::of::<Figure>("figure")
                    .schema(engine_core::schema_of::<Figure>),
            )
            .join::<AssetKinds>(AssetKind::of("skeleton", rig::skeleton))
            .join::<AssetKinds>(AssetKind::of("clip", rig::clip))
            .resource::<rig::FigureFiles>()
            .resource::<clock::FigureClocks>()
            .resource::<FigurePalette>()
            .resource::<FigureSight>()
            .resource::<StepJoints>()
            .moves::<Held>(Pace::Frame)
            .systems(Schedule::FixedSimulation, play::tick)
            .systems(Schedule::Presentation, play::pose)
            .dev(|dev| {
                dev.stats::<SkinnedStats>()
                    .systems(Schedule::Presentation, play::count);
            });
    }
}
