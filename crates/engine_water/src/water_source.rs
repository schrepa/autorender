use engine_core::math::{GRAVITY, max, min};
use engine_heightfield::Fall;
use engine_places::water::{WaterKind, WaterSample};
use glam::{Vec2, Vec3};

use crate::WaterPoint;
use crate::state::SurfaceTerrain;

const SHEET: f32 = 1.0;

pub(crate) fn water_at(terrains: &[SurfaceTerrain], point: Vec3) -> Option<WaterSample> {
    terrains.iter().find_map(|terrain| {
        let origin = terrain.origin;
        let local = point - origin;
        let held = water_point(terrain, Vec2::new(local.x, local.z)).map(|water| WaterSample {
            kind: if water.lake {
                WaterKind::Lake
            } else {
                WaterKind::River
            },
            surface: water.surface,
            depth: water.depth,
            velocity: [water.flow.x, 0.0, water.flow.y],
            foam: water.state.breaking,
        });
        [held, fall(&terrain.falls, local)]
            .into_iter()
            .flatten()
            .max_by(|a, b| a.surface.total_cmp(&b.surface))
            .map(|sample| WaterSample {
                surface: sample.surface + origin.y,
                ..sample
            })
    })
}

fn water_point(terrain: &SurfaceTerrain, local: Vec2) -> Option<WaterPoint> {
    let pyramid = &terrain.pyramid;
    (0..pyramid.levels())
        .find_map(|level| {
            let key = pyramid.tile_at(level, local)?;
            let (face, water) = terrain.held.iter().find(|(face, _)| face.key == key)?;
            let local = local - pyramid.tile_origin(key);
            crate::tile_water::point(face, water.as_deref(), local)
        })
        .flatten()
}

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
