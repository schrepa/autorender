//! Terrains' water as the world's water reads it ([`crate::WaterSources`]):
//! the water their tiles hold, lakes and rivers alike, as deep as its surface
//! stands over the ground the same tiles hold, and their waterfalls' jets as
//! their bake solved them.

use bevy_ecs::prelude::World;
use glam::{Vec2, Vec3};

use super::Terrains;
use crate::math::{max, min};
use crate::terrain_data::hydraulics::GRAVITY;
use crate::terrain_data::{Fall, Layers};
use crate::{WaterKind, WaterSample};

/// Meters thick a fall's sheet of water stands about its jet's path.
const SHEET: f32 = 1.0;

/// The world's water at `point` that terrains hold: the source the terrain
/// plugin registers.
pub(crate) fn terrain_water(world: &World, point: Vec3) -> Option<WaterSample> {
    world.get_resource::<Terrains>()?.water_at(point)
}

impl Terrains {
    /// Each terrain whose hydrology bake is done: its entity, origin and
    /// baked layers (whose hydrology holds its rivers, lakes and falls).
    pub fn waters(&self) -> impl Iterator<Item = (bevy_ecs::entity::Entity, Vec3, &Layers)> {
        self.entries
            .iter()
            .filter(|state| state.around.is_none())
            .filter_map(|state| {
                let layers = state.baked.as_deref()?;
                layers
                    .hydrology()
                    .map(|_| (state.entity, state.terrain.origin, layers))
            })
    }

    /// The flow of terrain `entity`'s rivers once a solid stood in them:
    /// what stands in them now and the pools behind it.
    pub fn river_flow(
        &self,
        entity: bevy_ecs::entity::Entity,
    ) -> Option<&crate::terrain_data::RiverFlow> {
        self.entries
            .iter()
            .find(|state| state.entity == entity)?
            .hydrology
            .flow
            .as_ref()
    }

    /// The water terrains hold over world `point`: what their tiles hold
    /// there, lake or river, or a waterfall's jet, whichever stands highest;
    /// `None` where none does or no tile holding the point is resident.
    pub fn water_at(&self, point: Vec3) -> Option<WaterSample> {
        self.entries
            .iter()
            .filter(|state| state.around.is_none())
            .find_map(|state| {
                let origin = state.terrain.origin;
                let local = point - origin;
                let held = state
                    .water(Vec2::new(local.x, local.z))
                    .map(|water| WaterSample {
                        kind: if water.lake {
                            WaterKind::Lake
                        } else {
                            WaterKind::River
                        },
                        surface: water.surface,
                        depth: water.depth,
                        velocity: [water.flow.x, 0.0, water.flow.y],
                        foam: water.white,
                    });
                [held, fall(&state.falls, local)]
                    .into_iter()
                    .flatten()
                    .max_by(|a, b| a.surface.total_cmp(&b.surface))
                    .map(|sample| WaterSample {
                        surface: sample.surface + origin.y,
                        ..sample
                    })
            })
    }
}

/// The jet of a waterfall at terrain-local `point`: within half the river's
/// width of a fall's path and [`SHEET`] of its height there, falling along
/// it at the speed its drop has given it.
fn fall(falls: &[Fall], point: Vec3) -> Option<WaterSample> {
    let at = Vec2::new(point.x, point.z);
    falls.iter().find_map(|fall| {
        let top = Vec3::from_array(fall.top);
        fall.path.windows(2).find_map(|pair| {
            let (a, b) = (Vec3::from_array(pair[0]), Vec3::from_array(pair[1]));
            let run = Vec2::new(b.x - a.x, b.z - a.z);
            let length = run.length();
            let t = if length > 0.0 {
                min(
                    max((at - Vec2::new(a.x, a.z)).dot(run) / (length * length), 0.0),
                    1.0,
                )
            } else {
                0.0
            };
            let on = a.lerp(b, t);
            let height = on.y;
            if at.distance(Vec2::new(on.x, on.z)) >= 0.5 * fall.width
                || (point.y - height).abs() > SHEET
            {
                return None;
            }
            // The jet keeps the lip's speed across and gains its drop's.
            let across = Vec2::new(fall.bottom[0] - fall.top[0], fall.bottom[2] - fall.top[2]);
            let drop = max(top.y - fall.bottom[1], 0.1);
            let horizontal = across.length() * (GRAVITY / (2.0 * drop)).sqrt();
            let falling = (2.0 * GRAVITY * max(top.y - height, 0.0)).sqrt();
            let heading = across.normalize_or(Vec2::X) * horizontal;
            Some(WaterSample {
                kind: WaterKind::Fall,
                surface: height,
                depth: SHEET,
                velocity: [heading.x, -falling, heading.y],
                foam: 1.0,
            })
        })
    })
}
