//! The path bake: the ground along a road's line, and any water standing
//! on it, read a point at a time.

use std::sync::Arc;

use glam::Vec2;

use super::Bake;
use crate::Pyramid;
use crate::layers::Layers;
use crate::path::{PathLayer, PathProfile, RIVER_DEPTH};

/// A road leveling the layers before it to its profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PathBake;

/// A path bake in progress: its layer, the points the ground is read at
/// and the meters between them, and the ground and any water standing on
/// it read so far.
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
        // The ground at each profile point's nearest level-0 sample: the
        // profile averages it anyway.
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
            // Rivers and lakes the hydrology bake found
            // there, on its own grid: streams narrower than its samples
            // can run below the finest ground's detail.
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
