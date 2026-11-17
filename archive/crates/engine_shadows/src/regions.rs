use std::ops::Range;

use engine_core::glam::{Mat4, Vec3, Vec4};
use engine_core::{Clear, ViewLayer};
use engine_places_rendering::shadows::{
    CasterMap, KeptRefresh, Refresh, Region as MapView, SHADOW_REGIONS,
};

use crate::gpu_layouts::{ComposeData, ShadowLevel};
use crate::levels::{Level, Levels, Rect};
use crate::sky::{RESOLUTION, SkyDraw};

const NORMAL_OFFSET_TEXELS: f32 = 0.5;
const SLOPE_LIMIT_TEXELS: f32 = 16.0;
const REGION_WIDEST: u32 = 8192;

fn region_cells(side: u32) -> u32 {
    if side.saturating_mul(2) <= REGION_WIDEST {
        2
    } else {
        1
    }
}

pub(crate) fn region_shape(side: u32) -> (u32, u32) {
    let cells = region_cells(side);
    (side * cells, SHADOW_REGIONS.div_ceil(cells * cells))
}

fn region_at(index: u32, side: u32) -> (u32, u32, u32, bool) {
    let cells = region_cells(side);
    let cell = index % (cells * cells);
    (
        index / (cells * cells),
        cell % cells * side,
        cell / cells * side,
        cell == 0,
    )
}

pub(crate) struct Placed {
    pub(crate) clip: Mat4,
    pub(crate) target: ViewLayer,
    pub(crate) clear: Clear,
}

fn planes(rows: [Vec3; 3], rect: [f32; 4], depth: f32) -> [[f32; 4]; 6] {
    let [x0, y0, x1, y1] = rect;
    let [right, up, away] = rows;
    let plane = |normal: Vec3, distance: f32| normal.extend(distance).to_array();
    [
        plane(right, -x0),
        plane(-right, x1),
        plane(up, -y0),
        plane(-up, y1),
        plane(-away, depth),
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn level_view(level: &Level, rect: Rect, map: CasterMap, sample_texels: f32) -> MapView {
    MapView {
        view: level.clip(rect),
        map,
        refresh: Refresh::Kept(KeptRefresh {
            planes: planes(level.basis.rows, level.light_rect(rect), level.band()[1]),
            texel: level.texel,
            finest_spacing: sample_texels * level.texel,
        }),
    }
}

pub(crate) fn views(
    levels: &Levels,
    (moving, side, sample_texels): (u32, u32, f32),
    sky: &[SkyDraw],
    (views, placed): (&mut Vec<MapView>, &mut Vec<Placed>),
) {
    views.clear();
    placed.clear();
    for (index, region) in (0_u32..).zip(&levels.regions) {
        let level = levels.level(region.level);
        let [x0, y0, x1, y1] = region.rect;
        let (layer, x, y, first) = region_at(index, side);
        views.push(level_view(level, region.rect, CasterMap::Region, sample_texels));
        placed.push(Placed {
            clip: level.clip(region.rect),
            target: ViewLayer {
                layer,
                rect: [x, y, (x1 - x0) as u32, (y1 - y0) as u32],
            },
            clear: if first {
                Clear::DepthOnly
            } else {
                Clear::Nothing
            },
        });
    }
    for (layer, level) in (levels.len() as u32..).zip(levels.iter().take(moving as usize)) {
        views.push(level_view(level, level.wanted, CasterMap::Moving, sample_texels));
        placed.push(Placed {
            clip: level.clip(level.wanted),
            target: ViewLayer {
                layer,
                rect: [0, 0, side, side],
            },
            clear: Clear::DepthOnly,
        });
    }
    for draw in sky {
        let rows = draw.rows.map(|row| Vec4::from_array(row).truncate());
        views.push(MapView {
            view: draw.clip,
            map: CasterMap::Sky,
            refresh: Refresh::Kept(KeptRefresh {
                planes: planes(rows, draw.rect, draw.depth),
                texel: 0.0,
                finest_spacing: 0.0,
            }),
        });
        placed.push(Placed {
            clip: draw.clip,
            target: ViewLayer {
                layer: draw.layer as u32,
                rect: [0, 0, RESOLUTION, RESOLUTION],
            },
            clear: Clear::DepthOnly,
        });
    }
}

fn pieces(from: i32, to: i32, side: i32) -> impl Iterator<Item = (i32, i32, i32)> {
    let start = from.rem_euclid(side);
    let first = (to - from).min(side - start);
    [(start, 0, first), (0, first, to - from - first)]
        .into_iter()
        .filter(|&(_, _, length)| length > 0)
}

pub(crate) fn composes(
    levels: &Levels,
    side: u32,
    (out, ranges): (&mut Vec<ComposeData>, &mut Vec<(u32, Range<u32>)>),
) {
    let span = side as i32;
    let ndc = |at: i32| at as f32 / side as f32 * 2.0 - 1.0;
    let mut listed: Vec<(u32, ComposeData)> = Vec::new();
    for (index, region) in (0_u32..).zip(&levels.regions) {
        let [x0, y0, x1, y1] = region.rect;
        let (layer, x, y, _) = region_at(index, side);
        let (layer, x, y) = (layer as i32, x as i32, y as i32);
        for (dx, sx, wide) in pieces(x0, x1, span) {
            for (dy, sy, high) in pieces(y0, y1, span) {
                listed.push((
                    region.level as u32,
                    ComposeData {
                        place: [ndc(dx), -ndc(dy), ndc(dx + wide), -ndc(dy + high)],
                        source: [layer, x + sx - dx, y + sy - dy, 0],
                    },
                ));
            }
        }
    }
    listed.sort_by_key(|&(level, _)| level);
    out.clear();
    ranges.clear();
    for (at, (level, compose)) in (0_u32..).zip(listed) {
        match ranges.last_mut() {
            Some((last, range)) if *last == level => range.end = at + 1,
            Some(_) | None => ranges.push((level, at..at + 1)),
        }
        out.push(compose);
    }
}

pub(crate) fn level_data(levels: &Levels, moving: u32, out: &mut Vec<ShadowLevel>) {
    out.clear();
    for (index, level) in (0_u32..).zip(levels.iter()) {
        let depth = level.depth_span();
        out.push(ShadowLevel {
            rows: level.rows(),
            window: level.window.map(|at| at as f32),
            band: [
                level.texel,
                1.0 / depth,
                level.texel * NORMAL_OFFSET_TEXELS,
                SLOPE_LIMIT_TEXELS * level.texel / depth,
            ],
            moving: [
                level.wanted[0] as f32,
                level.wanted[1] as f32,
                if index < moving { 1.0 } else { 0.0 },
                0.0,
            ],
        });
    }
}
