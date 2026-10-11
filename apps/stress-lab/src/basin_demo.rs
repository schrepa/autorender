//! The on-only Highlands demonstration, expressed as ordinary terrain
//! edits over the original completed stack. The numeric overview is kept.

use engine::prelude::*;
use engine::terrain::*;
use serde::Serialize;
use std::sync::Arc;

/// The fixed demonstration's preparation and authored geometry.
#[derive(Resource, Clone, Debug, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct BasinDemoView {
    pub ready: bool,
    pub eye: [f32; 3],
    pub target: [f32; 3],
    pub centers: [[f32; 2]; 2],
    pub heights: [f32; 2],
    pub terrain: u64,
    pub original_edits: usize,
    pub final_edits: usize,
    /// Authored layer records, followed by the original edits.
    #[cfg_attr(feature = "schema", schemars(with = "Vec<serde_json::Value>"))]
    pub original_layers: Vec<engine::scene::SceneTerrainLayer>,
    /// The same records with the four enabled-only basin edits appended.
    #[cfg_attr(feature = "schema", schemars(with = "Vec<serde_json::Value>"))]
    pub final_layers: Vec<engine::scene::SceneTerrainLayer>,
}

pub(crate) struct BasinDemo {
    original: Vec<(Entity, Arc<[TerrainEdit]>)>,
    view: BasinDemoView,
}

impl BasinDemo {
    pub(crate) fn new(engine: &mut Engine) -> Self {
        let world = engine.app.world_mut();
        let config = world.resource::<crate::StressConfig>();
        let (eye, target) = crate::highlands::overview(world, config);
        let offset = Vec2::new(eye.x - target.x, eye.z - target.z);
        let distance = offset.length();
        let direction = offset / distance;
        let right = Vec2::new(-direction.y, direction.x);
        let center = Vec2::new(target.x, target.z) + direction * (0.55 * distance);
        let centers = [
            center + right * (0.22 * distance),
            center - right * (0.22 * distance),
        ]
        .map(|p| p.to_array());
        let original = world
            .query::<(Entity, &Terrain)>()
            .iter(world)
            .map(|(id, t)| (id, t.edits.clone()))
            .collect();
        let view = BasinDemoView {
            ready: false,
            eye: eye.to_array(),
            target: target.to_array(),
            centers,
            heights: [0.0; 2],
            terrain: 0,
            original_edits: 0,
            final_edits: 0,
            original_layers: Vec::new(),
            final_layers: Vec::new(),
        };
        world.insert_resource(view.clone());
        Self { original, view }
    }

    pub(crate) fn restore(self, engine: &mut Engine) {
        let world = engine.app.world_mut();
        for (id, edits) in self.original {
            if let Some(mut terrain) = world.get_mut::<Terrain>(id) {
                terrain.edits = edits;
            }
        }
        world.remove_resource::<BasinDemoView>();
    }

    pub(crate) fn prepare(&mut self, engine: &mut Engine) {
        if self.view.ready {
            return;
        }
        let world = engine.app.world_mut();
        let water = world.get_resource::<Water>().is_some_and(|w| w.enabled);
        let sea = world
            .get_resource::<Ocean>()
            .filter(|o| o.enabled)
            .map(|o| o.level);
        for (entity, original) in &self.original {
            let Some(terrain) = world.get::<Terrain>(*entity).cloned() else {
                continue;
            };
            let layers = world
                .get_resource::<Terrains>()
                .and_then(|t| t.layers(*entity));
            let Some(layers) = layers.filter(|l| l.unbaked().is_none()) else {
                return;
            };
            let centers = self
                .view
                .centers
                .map(|p| Vec2::from(p) - Vec2::new(terrain.origin.x, terrain.origin.z));
            let extent = terrain.pyramid.extent();
            if centers.iter().any(|p| {
                p.x < 192.0 || p.y < 192.0 || p.x + 192.0 > extent.x || p.y + 192.0 > extent.y
            }) {
                continue;
            }
            let level = 4.min(terrain.pyramid.levels() - 1);
            let spacing = terrain.pyramid.spacing(level);
            let mut edits = original.to_vec();
            for (which, center) in centers.iter().enumerate() {
                let low = ((*center - Vec2::splat(192.0)) / spacing)
                    .floor()
                    .as_ivec2();
                let high = ((*center + Vec2::splat(192.0)) / spacing).ceil().as_ivec2();
                let mut height = f32::NEG_INFINITY;
                for z in low.y..=high.y {
                    for x in low.x..=high.x {
                        let point = Vec2::new(x as f32, z as f32) * spacing;
                        if (point - *center).length_squared() > 192.0 * 192.0 {
                            continue;
                        }
                        let mut floor = layers.sample_height(
                            &terrain.pyramid,
                            level,
                            [i64::from(x), i64::from(z)],
                        );
                        if water
                            && let Some((surface, _)) = layers
                                .hydrology()
                                .and_then(|h| h.water_at(point.x, point.y))
                        {
                            floor = math::max(floor, surface);
                        }
                        if let Some(sea) = sea {
                            floor = math::max(floor, sea - terrain.origin.y);
                        }
                        height = math::max(height, floor);
                    }
                }
                height += 8.0;
                self.view.heights[which] = height + terrain.origin.y;
                for (suffix, radius, height) in
                    [("rim", 160.0, height + 32.0), ("floor", 112.0, height)]
                {
                    edits.push(TerrainEdit {
                        id: format!("basin-fog-{which}-{suffix}").into(),
                        stamp: Stamp {
                            center: center.to_array(),
                            radius,
                            falloff: 32.0,
                            op: StampOp::Flatten(height),
                        },
                    });
                }
            }
            self.view.terrain = entity.to_bits();
            self.view.original_edits = original.len();
            self.view.final_edits = edits.len();
            let layers =
                engine::scene::SceneTerrain::from_runtime(&terrain, None, |_| None, |_| None)
                    .layers;
            self.view.original_layers = layers
                .iter()
                .cloned()
                .chain(
                    original
                        .iter()
                        .map(|edit| engine::scene::SceneTerrainLayer::Stamp(edit.stamp)),
                )
                .collect();
            self.view.final_layers = layers
                .into_iter()
                .chain(
                    edits
                        .iter()
                        .map(|edit| engine::scene::SceneTerrainLayer::Stamp(edit.stamp)),
                )
                .collect();
            world.get_mut::<Terrain>(*entity).expect("present").edits = edits.into();
            self.view.ready = true;
            world.insert_resource(self.view.clone());
            return;
        }
    }
}
