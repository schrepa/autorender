use std::sync::Arc;

use engine_core::Pyramid;
use engine_terrain::Layers;
use engine_terrain::layer::{LayerBake, TerrainLayer};
use glam::Vec2;

use crate::TerrainPath;
use crate::profile::{PathLayer, PathProfile, RIVER_DEPTH};

pub(crate) struct PathWork {
    layer: Arc<PathLayer>,
    profile: Vec<Vec2>,
    step: f32,
    ground: Vec<f32>,
    water: Vec<Option<f32>>,
}

impl PathWork {
    pub(crate) fn start(layer: &Arc<PathLayer>) -> Self {
        let (profile, step) = layer.profile_points();
        Self {
            layer: layer.clone(),
            ground: Vec::with_capacity(profile.len()),
            water: Vec::with_capacity(profile.len()),
            profile,
            step,
        }
    }
}

impl LayerBake for PathWork {
    fn advance(
        &mut self,
        below: &Arc<Layers>,
        pyramid: &Pyramid,
        budget: &mut u64,
    ) -> Option<Arc<dyn TerrainLayer>> {
        let spacing = pyramid.spacing(0);
        let cost = below.height_cost();
        while let Some(&point) = self.profile.get(self.ground.len()) {
            if *budget == 0 {
                return None;
            }
            let sample = [
                (point.x / spacing).round() as i64,
                (point.y / spacing).round() as i64,
            ];
            let ground = below.sample_height(pyramid, 0, sample);
            let water = below.hydrology().and_then(|hydrology| {
                hydrology
                    .water_at(point.x, point.y)
                    .filter(|&(_, depth)| depth > RIVER_DEPTH)
                    .map(|(surface, _)| surface)
            });
            self.ground.push(ground);
            self.water.push(water);
            *budget = budget.saturating_sub(cost);
        }
        Some(Arc::new(TerrainPath {
            settings: self.layer.clone(),
            profile: Some(Arc::new(PathProfile::new(
                &self.layer,
                self.step,
                &self.ground,
                &self.water,
            ))),
        }))
    }
}
