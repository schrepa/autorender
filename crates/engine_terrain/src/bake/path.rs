use std::sync::Arc;

use glam::Vec2;

use super::Bake;
use crate::Pyramid;
use crate::layers::Layers;
use crate::path::{PathLayer, PathProfile, RIVER_DEPTH};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PathBake;

pub struct PathWork {
    layer: Arc<PathLayer>,
    profile: Vec<Vec2>,
    step: f32,
    ground: Vec<f32>,
    water: Vec<Option<f32>>,
}

impl Bake for PathBake {
    type Layer = Arc<PathLayer>;
    type Found = PathProfile;
    type Work = PathWork;

    fn start(layer: &Arc<PathLayer>, _pyramid: &Pyramid) -> PathWork {
        let (profile, step) = layer.profile_points();
        PathWork {
            layer: layer.clone(),
            ground: Vec::with_capacity(profile.len()),
            water: Vec::with_capacity(profile.len()),
            profile,
            step,
        }
    }

    fn advance(
        work: &mut PathWork,
        below: &Arc<Layers>,
        pyramid: &Pyramid,
        budget: &mut u64,
    ) -> Option<PathProfile> {
        let spacing = pyramid.spacing(0);
        let cost = below.height_cost();
        while let Some(&point) = work.profile.get(work.ground.len()) {
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
            work.ground.push(ground);
            work.water.push(water);
            *budget = budget.saturating_sub(cost);
        }
        Some(PathProfile::new(
            &work.layer,
            work.step,
            &work.ground,
            &work.water,
        ))
    }
}
