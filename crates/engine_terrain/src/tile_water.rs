//! Water as a terrain tile holds it, the one place the world's water is:
//! per sample, in the layout of the tile's heights, the surface water stands
//! at wherever any reaches (NaN where none does), and beside it its flow, its
//! whitewater and whether a lake holds it. Depth is never held: it is the
//! surface less the tile's ground at the same point, on the triangles every
//! consumer splits cells into ([`crate::Tile::water_at`]), so drawing,
//! physics, queries, placement and navigation read one water. Tile
//! generation writes it from its hydrology bake's lakes and rivers
//! ([`water_rows`]), each river through the spans that cut its channel; a
//! resident tile's water is written again the same way where its rivers
//! run otherwise since, as what stands in them holds them back
//! ([`Tile::rewatered`]). However it was written, every dry sample beside
//! water holds a surface its water meets the ground by ([`skirt`]), so
//! shores fall where the ground rises through the water, between samples,
//! never at a sample.

use std::ops::Range;

use glam::Vec2;

use crate::hydrology::WaterGrid;
use crate::rivers::{RiverSpan, RiverWater, Rivers};
use crate::{Layers, PATCH_QUADS, PATCHES_PER_TILE, Pyramid, TILE_SAMPLES, TILE_STRIDE, Tile};

/// Meters a second either way the packed flow spans.
pub const WATER_FLOW_RANGE: f32 = 8.0;
/// Meters of half width the packed width of a river's water spans: wider
/// water packs as this wide, as lakes do.
pub const WATER_WIDTH_RANGE: f32 = 63.0;
/// Meters below a river's bed ground may lie and hold its water: the bed's
/// own samples, rounded.
const BED_SLACK: f32 = 0.05;

/// The water at a point of a tile: the surface it stands at (terrain-local
/// meters), its depth over the tile's ground there, its flow along x and z
/// (meters a second), its whitewater (0 to 1) and whether a lake holds it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterPoint {
    pub surface: f32,
    pub depth: f32,
    pub flow: Vec2,
    pub white: f32,
    pub lake: bool,
}

/// A tile's water: each sample's surface (NaN where no water reaches) and
/// packed flow, whitewater and kind ([`pack`]), in the layout of its heights,
/// and which of its patches hold water over their ground.
#[derive(Clone, Debug, PartialEq)]
pub struct TileWater {
    surface: Box<[f32]>,
    info: Box<[u32]>,
    wet: [u64; 4],
}

impl TileWater {
    /// The water of a tile whose samples' heights are `heights`, from its
    /// samples' `surface` and packed `info`, its shores skirted
    /// ([`skirt`]): `None` when the sizes do not match or no sample's
    /// surface stands over its ground.
    pub(crate) fn new(
        mut surface: Box<[f32]>,
        mut info: Box<[u32]>,
        heights: &[f32],
    ) -> Option<Self> {
        let stride = TILE_STRIDE as usize;
        if surface.len() != stride * stride
            || info.len() != stride * stride
            || heights.len() != stride * stride
        {
            return None;
        }
        skirt(&mut surface, &mut info, heights);
        let mut wet = [0_u64; 4];
        let (patch, patches) = (PATCH_QUADS as usize, PATCHES_PER_TILE as usize);
        for pz in 0..patches {
            for px in 0..patches {
                let holds = (pz * patch..=(pz + 1) * patch).any(|z| {
                    (px * patch..=(px + 1) * patch).any(|x| {
                        let at = (z + 1) * stride + x + 1;
                        surface[at] > heights[at]
                    })
                });
                if holds {
                    let index = pz * patches + px;
                    wet[index / 64] |= 1 << (index % 64);
                }
            }
        }
        wet.iter()
            .any(|&bits| bits != 0)
            .then_some(Self { surface, info, wet })
    }

    /// Every sample's surface, the border included, row by row: the
    /// layout GPU tiles upload.
    pub fn surfaces(&self) -> &[f32] {
        &self.surface
    }

    /// Every sample's packed flow, whitewater and kind ([`pack`]), in the
    /// layout of [`Self::surfaces`].
    pub fn info(&self) -> &[u32] {
        &self.info
    }

    /// Whether any sample of patch `(x, z)` stands under its water.
    pub fn patch_holds_water(&self, x: u32, z: u32) -> bool {
        let index = (z * PATCHES_PER_TILE + x) as usize;
        self.wet[index / 64] & (1 << (index % 64)) != 0
    }

    /// The surface of sample `(x, z)`, -1 to 257 along each axis.
    #[inline]
    pub(crate) fn surface(&self, x: i32, z: i32) -> f32 {
        let stride = TILE_STRIDE as i32;
        self.surface[((z + 1) * stride + x + 1) as usize]
    }

    /// The flow, whitewater, kind and half width of sample `(x, z)`, -1 to
    /// 257 along each axis ([`unpack`]).
    #[inline]
    pub(crate) fn sample_info(&self, x: i32, z: i32) -> (Vec2, f32, bool, f32) {
        let stride = TILE_STRIDE as i32;
        unpack(self.info[((z + 1) * stride + x + 1) as usize])
    }
}

/// Gives the dry samples beside water (any of their eight neighbors' surface
/// standing over its ground) a surface for the water to meet their ground
/// by, unless they hold one already, with their wet neighbors' flow,
/// whitewater, kind and width, averaged: those neighbors' surface, averaged, where
/// their ground stands at or above it, so the shore falls on the triangles
/// between them where the ground rises through the water. Samples whose
/// ground lies below it had their water held back (a levee's outer side, a
/// road's ditch): they stand dry, as far under their surface as their wet
/// neighbors stand under theirs, so the shore falls halfway to them. Dry
/// samples beside no water lose any surface, so none lingers where water was
/// held back before.
fn skirt(surface: &mut [f32], info: &mut [u32], heights: &[f32]) {
    let stride = TILE_STRIDE as usize;
    let wet = |at: usize| surface[at] > heights[at];
    let mut changes = Vec::new();
    for z in 0..stride {
        for x in 0..stride {
            let at = z * stride + x;
            if wet(at) {
                continue;
            }
            let (mut level, mut depth, mut flow, mut white, mut lakes, mut width, mut count) =
                (0.0, 0.0, Vec2::ZERO, 0.0, 0, 0.0, 0);
            for nz in z.saturating_sub(1)..(z + 2).min(stride) {
                for nx in x.saturating_sub(1)..(x + 2).min(stride) {
                    let near = nz * stride + nx;
                    if wet(near) {
                        let (near_flow, near_white, lake, near_width) = unpack(info[near]);
                        level += surface[near];
                        depth += surface[near] - heights[near];
                        flow += near_flow;
                        white += near_white;
                        lakes += usize::from(lake);
                        width += near_width;
                        count += 1;
                    }
                }
            }
            if count == 0 {
                if surface[at].is_finite() {
                    changes.push((at, f32::NAN, 0));
                }
                continue;
            }
            if surface[at].is_finite() {
                continue;
            }
            let share = 1.0 / count as f32;
            let level = level * share;
            let standing = if heights[at] >= level {
                level
            } else {
                heights[at] - depth * share
            };
            changes.push((
                at,
                standing,
                pack(
                    flow * share,
                    white * share,
                    lakes * 2 > count,
                    width * share,
                ),
            ));
        }
    }
    for (at, level, packed) in changes {
        surface[at] = level;
        info[at] = packed;
    }
}

/// One sample's flow (meters a second along x and z), whitewater (0 to 1),
/// kind and the half width of a river's water there (meters) packed as the
/// four signed bytes of an RGBA8 snorm texel, low first: the flow as a
/// share of [`WATER_FLOW_RANGE`] either way, the whitewater, and 127 where a
/// lake holds the water, else the width as 126ths of [`WATER_WIDTH_RANGE`].
/// Still water packs to zero.
pub fn pack(flow: Vec2, white: f32, lake: bool, width: f32) -> u32 {
    let byte =
        |value: f32| u32::from((engine_math::clamp(value, -1.0, 1.0) * 127.0).round() as i8 as u8);
    let kind = if lake {
        127
    } else {
        (engine_math::clamp(width / WATER_WIDTH_RANGE, 0.0, 1.0) * 126.0).round() as u32
    };
    byte(flow.x / WATER_FLOW_RANGE)
        | byte(flow.y / WATER_FLOW_RANGE) << 8
        | byte(white) << 16
        | kind << 24
}

/// [`pack`] undone: flow, whitewater, whether a lake holds the water and its
/// half width (a lake's [`WATER_WIDTH_RANGE`]).
pub fn unpack(info: u32) -> (Vec2, f32, bool, f32) {
    let share = |shift: u32| engine_math::max(f32::from((info >> shift) as u8 as i8) / 127.0, -1.0);
    let kind = engine_math::max(f32::from((info >> 24) as u8 as i8), 0.0);
    let lake = kind > 126.5;
    (
        Vec2::new(share(0), share(8)) * WATER_FLOW_RANGE,
        share(16),
        lake,
        if lake {
            WATER_WIDTH_RANGE
        } else {
            kind / 126.0 * WATER_WIDTH_RANGE
        },
    )
}

/// Writes the water of a tile's stored rows `rows` (0 to 258, samples -1 to
/// 257), at the stored columns `columns` gives each, into `surface` and
/// `info`, row by row from the first of `rows`: its samples at x positions
/// `xs` and its rows at z positions `zs`, `spacing` meters apart, over its
/// ground `heights` (every stored row). Each sample
/// takes the river of `rivers` it lies deepest within, out past the
/// river's edge so its shore falls where the ground rises through its
/// surface ([`crate::Rivers::water_row`], through the tile's spans `span`),
/// but none past the edge where the ground lies below the water's surface
/// (a levee's outer side, a road's ditch): the water ends at the edge; nor
/// anywhere it would stand over ground lower than the channel's bed (down a
/// drop below a pool, beside a channel on a slope): a river's water stands
/// deepest over its bed, at its middle. A
/// lake of `lakes` holding a sample in its water or shore takes it where it
/// stands higher.
/// Surroundings keep water only where they show their bakes whole
/// (`shown`): where the bakes fade, so does the channel they cut.
pub(crate) fn water_rows(
    (lakes, rivers): (&WaterGrid, &Rivers),
    span: Option<&RiverSpan>,
    (rows, columns): (Range<usize>, &dyn Fn(usize) -> Range<usize>),
    (xs, zs, heights): (&[f32], &[f32], &[f32]),
    spacing: f32,
    shown: &dyn Fn(f32, f32) -> f32,
    (surface, info): (&mut [f32], &mut [u32]),
) {
    let stride = TILE_STRIDE as usize;
    let mut best = vec![RiverWater::NONE; stride];
    let first = rows.start;
    for row in rows {
        let (z, columns) = (zs[row], columns(row));
        if columns.is_empty() {
            continue;
        }
        best.fill(RiverWater::NONE);
        if let Some(span) = span {
            rivers.water_row(span, row, (xs, z, columns.clone()), spacing, &mut best);
        }
        for column in columns {
            let (river, x) = (&best[column], xs[column]);
            if shown(x, z) < 1.0 {
                continue;
            }
            let lake = lakes.lake_reach(x, z);
            let flowing = river.surface.is_finite();
            let at = (row - first) * stride + column;
            match lake {
                // A river within its own water keeps it where it runs
                // above the lake beside it.
                Some(level) if !(flowing && river.inside < 0.0 && river.surface > level) => {
                    surface[at] = level;
                    info[at] = pack(Vec2::ZERO, 0.0, true, 0.0);
                }
                _ if flowing => {
                    let ground = heights[row * stride + column];
                    if ground < river.bed - BED_SLACK
                        || (river.inside > 0.0 && ground < river.surface)
                    {
                        continue;
                    }
                    surface[at] = river.surface;
                    info[at] = pack(river.flow, river.white, false, river.width);
                }
                _ => {}
            }
        }
    }
}

impl Tile {
    /// This tile of `pyramid`, generated from `layers`, with the water its
    /// hydrology bake's lakes and `rivers` hold over it: the bake's own
    /// rivers as what stands in them holds them back now. Only its samples
    /// within `regions` (terrain-local bounds, where `rivers` run otherwise
    /// than the rivers its water was made from, out as far as their water
    /// reaches at its level) are made again; `None` where the regions miss
    /// it or nothing holds water.
    #[must_use]
    pub fn rewatered(
        &self,
        layers: &Layers,
        pyramid: &Pyramid,
        rivers: &Rivers,
        regions: &[(Vec2, Vec2)],
    ) -> Option<Self> {
        let (_, hydrology) = layers.hydrology_layer()?;
        let spacing = self.spacing();
        let [x, z] = pyramid.first_sample(self.key());
        let [dx, dz] = layers.offset_samples(spacing);
        let positions = |first: i64| -> Vec<f32> {
            (-1..=i64::from(TILE_SAMPLES))
                .map(|index| (first + index) as f32 * spacing)
                .collect()
        };
        let (xs, zs) = (positions(x + dx), positions(z + dz));
        // A sample past the regions' bounds skirts the water within them.
        let within = |values: &[f32], low: f32, high: f32| {
            values.partition_point(|&value| value < low - spacing)
                ..values.partition_point(|&value| value <= high + spacing)
        };
        // Each row's columns the regions reach, from the first to the last.
        let stride = TILE_STRIDE as usize;
        let mut columns: Vec<Range<usize>> = vec![0..0; stride];
        for (low, high) in regions {
            let reached = within(&xs, low.x, high.x);
            if reached.is_empty() {
                continue;
            }
            for row in within(&zs, low.y, high.y) {
                let held = &mut columns[row];
                *held = if Range::is_empty(held) {
                    reached.clone()
                } else {
                    held.start.min(reached.start)..held.end.max(reached.end)
                };
            }
        }
        let (Some(first), Some(last)) = (
            columns.iter().position(|row| !row.is_empty()),
            columns.iter().rposition(|row| !row.is_empty()),
        ) else {
            return None;
        };
        let (mut surface, mut info) = self.water().map_or_else(
            || (vec![f32::NAN; stride * stride], vec![0; stride * stride]),
            |water| (water.surfaces().to_vec(), water.info().to_vec()),
        );
        for (row, reached) in columns.iter().enumerate() {
            let at = row * stride + reached.start..row * stride + reached.end;
            surface[at.clone()].fill(f32::NAN);
            info[at].fill(0);
        }
        let span = rivers.span(&xs, &zs, spacing);
        let at = first * stride..(last + 1) * stride;
        water_rows(
            (&hydrology.water, rivers),
            Some(&span),
            (first..last + 1, &|row| columns[row].clone()),
            (&xs, &zs, self.heights()),
            spacing,
            &|x, z| layers.shown(x, z),
            (&mut surface[at.clone()], &mut info[at]),
        );
        Some(
            self.clone()
                .with_water(surface.into_boxed_slice(), info.into_boxed_slice()),
        )
    }
}
