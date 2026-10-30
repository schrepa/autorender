#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

mod layouts;
mod part;
mod scene;
mod shaders;

use engine_core::bevy_ecs::{
    self,
    prelude::{Component, World},
};
use engine_core::{Brick, Bricks, Ground, RenderSnapshot, WaterFace, WaterKind, WaterSample};
use engine_faces::slots::WaterList;
use engine_render::render_bricks::AddRenderParts;
use engine_scene::AddSceneComponents;
use glam::{Vec2, Vec3};
use serde::{Deserialize, Serialize};

pub use scene::POND;

pub struct PondBrick;

impl Brick for PondBrick {
    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .list::<WaterList>()
            .component::<Pond>()
            .component_inspector::<Pond, Pond>(|_, _, pond| *pond)
            .scene_components(&[&POND])
            .water_body(PondWater)
            .extract(extract_ponds)
            .render_part(&part::PART)
            .shaders(shaders::MODULES)
            .face_defaults(engine_faces::defaults::MODULES)
            .pipeline_layouts(layouts::PIPELINE_LAYOUTS)
            .mirrors(layouts::LAYOUTS, layouts::DECLARED);
    }
}

engine_settings::settings! {
    #[derive(Component, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(deny_unknown_fields)]
    pub struct Pond {
        pub center: [f32; 2] = [0.0; 2] => finite,
        pub size: [f32; 2] = [1.0; 2] => above(0.0),
        pub surface: f32 = 0.0 => finite,
    }
}

impl Pond {
    pub fn low(&self) -> Vec2 {
        Vec2::from(self.center) - 0.5 * Vec2::from(self.size)
    }

    pub fn high(&self) -> Vec2 {
        Vec2::from(self.center) + 0.5 * Vec2::from(self.size)
    }

    pub fn holds(&self, x: f32, z: f32) -> bool {
        let (low, high) = (self.low(), self.high());
        x >= low.x && x <= high.x && z >= low.y && z <= high.y
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RenderPonds {
    pub ponds: Vec<Pond>,
}

impl RenderPonds {
    pub fn of(snapshot: &RenderSnapshot) -> &[Pond] {
        snapshot
            .extracted
            .get::<Self>()
            .map(|extracted| extracted.ponds.as_slice())
            .unwrap_or_default()
    }
}

fn extract_ponds(world: &World, snapshot: &mut RenderSnapshot) {
    let ponds = world
        .try_query::<&Pond>()
        .map_or_else(Vec::new, |mut ponds| ponds.iter(world).copied().collect());
    snapshot.extracted.insert(RenderPonds { ponds });
}

struct PondWater;

impl WaterFace for PondWater {
    fn water(&self, world: &World, point: Vec3) -> Option<WaterSample> {
        let mut ponds = world.try_query::<&Pond>()?;
        let surface = ponds
            .iter(world)
            .filter(|pond| pond.holds(point.x, point.z))
            .map(|pond| pond.surface)
            .reduce(engine_core::math::max)?;
        let ground = Ground::sample_in(world, point.x, point.z)?;
        let depth = surface - ground.height;
        (depth > 0.0).then_some(WaterSample {
            kind: WaterKind::Lake,
            surface,
            depth,
            velocity: [0.0; 3],
            foam: 0.0,
        })
    }
}
