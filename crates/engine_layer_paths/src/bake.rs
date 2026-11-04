use std::sync::Arc;

use engine_heightfield::layer::TerrainLayer;
use engine_heightfield::{Layers, Pyramid};

use crate::TerrainPath;
use crate::profile::{PathLayer, PathProfile, RIVER_DEPTH};

pub(crate) fn path_stage(
    layer: &Arc<PathLayer>,
    below: &Layers,
    pyramid: &Pyramid,
) -> Arc<dyn TerrainLayer> {
    let (profile, step) = layer.profile_points();
    let spacing = pyramid.spacing(0);
    let mut ground = Vec::with_capacity(profile.len());
    let mut water = Vec::with_capacity(profile.len());
    for point in profile {
        let sample = [
            (point.x / spacing).round() as i64,
            (point.y / spacing).round() as i64,
        ];
        ground.push(below.sample_height(pyramid, 0, sample));
        water.push(below.hydrology().and_then(|hydrology| {
            hydrology
                .water_at(point.x, point.y)
                .filter(|&(_, depth)| depth > RIVER_DEPTH)
                .map(|(surface, _)| surface)
        }));
    }
    Arc::new(TerrainPath {
        settings: layer.clone(),
        profile: Some(Arc::new(PathProfile::new(layer, step, &ground, &water))),
    })
}
