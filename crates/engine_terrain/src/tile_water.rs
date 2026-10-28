//! Water as a terrain tile holds it, the one place the world's water is:
//! per sample, in the layout of the tile's heights, the surface water stands
//! at wherever any reaches (NaN where none does), and beside it two words
//! ([`TileWater::info`]): its flow, whether a fall's sheet runs over it, how
//! far across its water body is (its span: a river's width, a lake's
//! equal-area diameter, the fetch the wind has over it) and whether a lake
//! holds it; and its state, how turbulent it runs, how white it breaks, its
//! shear and the reach of the stream it runs in. Beside them it holds the solids standing
//! in it (objects placed in its water, [`TileWater::solids`]). Depth is
//! never held: it is the surface less the tile's ground at the same point,
//! on the triangles every consumer splits cells into
//! ([`crate::Tile::water_at`]), so drawing, physics, queries,
//! placement and navigation read one water. Tile generation writes it from
//! its hydrology bake's lakes and rivers ([`water_rows`]), each river's from
//! the raster that cut its channel; a resident tile's water is written
//! again the same way where its rivers run otherwise since, as what stands
//! in them holds them back ([`Tile::rewatered`]), and as its water settles
//! ([`Tile::steadied`], [`Tile::flowing`], a job's record that the tile
//! then holds: [`Tile::holding`]): rivers' own water runs in no state.
//! However it was written, the
//! dry samples within [`WATER_REACH`] rings of water hold its surface
//! continued under the banks, and no others any ([`extend`]), so shores fall
//! where the ground rises through the water, between samples, never at a
//! sample, and a mesh standing on the surface reaches past them.

use std::ops::Range;

use glam::Vec2;

use crate::hydraulics::GRAVITY;
use crate::hydrology::WaterGrid;
use crate::noise::smooth;
use crate::rivers::{RiverRaster, RiverWater, Rivers};
use crate::shallow_water::DENSITY;
use crate::{
    Footprint, Layers, PATCH_QUADS, PATCHES_PER_TILE, Pyramid, Reach, TILE_QUADS, TILE_SAMPLES,
    TILE_STRIDE, Tile,
};

/// Meters a second either way the packed flow spans.
pub const WATER_FLOW_RANGE: f32 = 8.0;
/// The narrowest and the widest span the packed span tells apart (meters
/// across a water body: a river's width, a lake's equal-area diameter), a
/// quarter meter to a lake about 13 km² in area; wider water packs as the
/// widest. Its code is a signed byte's magnitude, 1 to 127 on a logarithmic
/// scale between them, each of its 126 steps 2^(1/9), about 8%, wider than
/// the one before.
pub const WATER_SPAN_LEAST: f32 = 0.25;
pub const WATER_SPAN_MOST: f32 = 4096.0;
/// Steps of the packed span's code from [`WATER_SPAN_LEAST`] to
/// [`WATER_SPAN_MOST`]: a signed byte's magnitude, from 1 to 127.
const SPAN_STEPS: f32 = 126.0;
/// Per second of shear the packed shear spans: water's velocity changing by
/// eight meters a second over a meter, past what torrents shear.
pub const WATER_SHEAR_RANGE: f32 = 8.0;
/// The least speed (meters a second) turbulence measures water's swings of
/// speed against: slow water's small swings stay calm.
pub const TURBULENT_SPEED: f32 = 0.5;
/// Per second the water's vorticity swings by where it runs wholly
/// turbulent: eddies half a meter across turning their rims at a meter a
/// second.
pub const TURBULENT_SWIRL: f32 = 4.0;
/// The turbulent Froude number of running water (the speed of eddies as
/// large as it is deep over √(g·h), the speed gravity smooths its surface
/// at) from which its turbulence breaks its surface white (Brocchini and
/// Peregrine 2001), and from which it breaks white all over. In uniform flow
/// its cube is the slope times the Froude number: a 10% run at Froude 1.5
/// gives 0.53 and stays clear but rough; a 20% chute at Froude 2.9 gives
/// about 0.84 and shows white in patches; only water plunging down a step
/// or a chute over 30% at Froude 5 and more (1.2) breaks white all over, as
/// mountain streams run clear between their white drops.
pub const BREAK_ONSET: f32 = 0.6;
pub const BREAK_FULL: f32 = 1.2;
/// The meters deep thinner water breaks as, so films divide by nothing:
/// about where surface tension, not gravity, holds a surface flat against
/// eddies (waves shorter than 1.7 cm are capillary).
pub const BREAK_THINNEST: f32 = 0.02;
/// What a sample nearest no seed ([`nearest`]) holds as its seed.
pub(crate) const NONE: u32 = u32::MAX;
/// Rings of dry samples beside water, each the dry samples among the eight
/// neighbors of the ring inside it, that hold the water's surface continued
/// under its banks: the samples past the shore where the ground drawn meets
/// the water at its level.
pub const WATER_REACH: u32 = 4;
/// A sample's ring as [`extend`] counts them, and as [`drawn_surfaces`]
/// finds them again: none yet, waiting to be worked out in the ring being
/// made, and a ring the rest stop at.
const NO_RING: u8 = u8::MAX;
const QUEUED: u8 = u8::MAX - 1;
const STOPPED: u8 = u8::MAX - 2;

/// The water at a point of a tile: the surface it stands at (terrain-local
/// meters), its depth over the tile's ground there, its flow along x and z
/// (meters a second), how far across its water body is (its span, meters),
/// whether a lake holds it and its state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterPoint {
    pub surface: f32,
    pub depth: f32,
    pub flow: Vec2,
    pub span: f32,
    pub lake: bool,
    pub state: WaterState,
}

/// What water does where it runs, beyond its flow: how turbulent it runs (0
/// calm to 1), how strongly it breaks white there (0 to 1: where
/// short-lived white water is born), its shear (the magnitude of its
/// velocity's gradient, per second) and the reach of the stream it runs
/// in, if any.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WaterState {
    pub turbulence: f32,
    pub breaking: f32,
    pub shear: f32,
    pub reach: Option<Reach>,
}

/// A tile's water: each sample's surface (NaN where no water reaches) and
/// words ([`Self::info`]), in the layout of its heights, which of its
/// patches hold a surface, wet or continued under the banks, and the solids
/// standing in it ([`Self::solids`]).
#[derive(Clone, Debug, PartialEq)]
pub struct TileWater {
    surface: Box<[f32]>,
    info: Box<[[u32; 2]]>,
    held: [u64; 4],
    solids: Box<[Footprint]>,
}

/// A tile's water as its mesh draws it ([`crate::Tile::water_mesh`]), in
/// the layout of its heights: which samples hold the surface water's pass
/// draws through them, a bit each, and the level the water's mesh stands at
/// over each sample, made with the tile's water.
#[derive(Clone, Debug, PartialEq)]
pub struct WaterMesh {
    holding: Box<[u64]>,
    levels: Box<[f32]>,
}

impl WaterMesh {
    /// Whether sample `at` of the tile's layout holds the surface water's
    /// pass draws: wet, or dry with the surface continued under it where
    /// the rings continuing it did not stop, short of the crest of a bank
    /// they stop past.
    pub fn holds(&self, at: usize) -> bool {
        self.holding[at / 64] & (1 << (at % 64)) != 0
    }

    /// The level the water's mesh stands at over sample `at` of the tile's
    /// layout (meters above the terrain's origin): the surface drawn where
    /// it holds one, else the surface drawn of the sample holding one
    /// nearest it, or its ground where a patch beside it holds no surface
    /// ([`TileWater::patch_holds_water`]) or no sample holds one.
    pub fn level(&self, at: usize) -> f32 {
        self.levels[at]
    }

    /// Every sample's level, row by row.
    pub(crate) fn levels(&self) -> &[f32] {
        &self.levels
    }
}

impl TileWater {
    /// The water of a tile whose samples, `spacing` meters apart from its
    /// first at terrain-local `origin`, stand at `heights`, from its
    /// samples' `surface` and words `info`: its surface continued under the
    /// banks ([`extend`]), and those of `solids` standing in it
    /// ([`standing`]); `None` when the sizes do not match or no sample's
    /// surface stands over its ground.
    pub(crate) fn new(
        mut surface: Box<[f32]>,
        mut info: Box<[[u32; 2]]>,
        (heights, spacing, origin): (&[f32], f32, Vec2),
        solids: &[Footprint],
    ) -> Option<Self> {
        let stride = TILE_STRIDE as usize;
        if surface.len() != stride * stride
            || info.len() != stride * stride
            || heights.len() != stride * stride
            || !surface.iter().zip(heights).any(|(s, h)| s > h)
        {
            return None;
        }
        extend(&mut surface, &mut info, heights);
        let mut held = [0_u64; 4];
        let (patch, patches) = (PATCH_QUADS as usize, PATCHES_PER_TILE as usize);
        for pz in 0..patches {
            for px in 0..patches {
                let holds = (pz * patch..=(pz + 1) * patch).any(|z| {
                    (px * patch..=(px + 1) * patch).any(|x| {
                        let at = (z + 1) * stride + x + 1;
                        surface[at].is_finite()
                    })
                });
                if holds {
                    let index = pz * patches + px;
                    held[index / 64] |= 1 << (index % 64);
                }
            }
        }
        let wet: Vec<bool> = surface
            .iter()
            .zip(heights)
            .map(|(&surface, &height)| surface > height)
            .collect();
        let solids = standing(&wet, (spacing, origin, solids));
        Some(Self {
            surface,
            info,
            held,
            solids,
        })
    }

    /// Every sample's surface, the border included, row by row: the
    /// layout GPU tiles upload. Wet samples' stand over their ground; those
    /// of the dry samples within [`WATER_REACH`] rings of them continue it
    /// under the banks, at or below their ground; the rest are NaN.
    pub fn surfaces(&self) -> &[f32] {
        &self.surface
    }

    /// Every sample's words, in the layout of [`Self::surfaces`], as GPU
    /// tiles upload them, each four bytes, low first: its flow, whether a
    /// fall's sheet runs over it, its span and kind ([`pack`]); and its state ([`WaterState`]: its
    /// turbulence, how strongly it breaks white and its shear as a share of
    /// [`WATER_SHEAR_RANGE`], in 255ths, and 0 for no reach, else its reach
    /// plus one).
    pub fn info(&self) -> &[[u32; 2]] {
        &self.info
    }

    /// The solids standing in it, terrain-local: objects placed in its
    /// water ([`crate::Placed::footprint`], [`standing`]).
    pub fn solids(&self) -> &[Footprint] {
        &self.solids
    }

    /// Whether any sample of patch `(x, z)`, its edges included, holds a
    /// surface of its water: wet, or dry with the surface continued under it
    /// ([`Self::surfaces`]). Whether water stands at a point is its depth's
    /// to say ([`crate::Tile::water_at`]).
    pub fn patch_holds_water(&self, x: u32, z: u32) -> bool {
        let index = (z * PATCHES_PER_TILE + x) as usize;
        self.held[index / 64] & (1 << (index % 64)) != 0
    }

    /// The water as its mesh draws it over the tile whose ground stands at
    /// `heights` (its layout): the samples holding the surface water's pass
    /// draws ([`drawn_surfaces`]), and the level its mesh stands at over
    /// each, so the mesh reaches at the water's level past wherever its
    /// surface ends and each pixel alone decides where water shows: a
    /// sample's surface drawn where it holds one; its ground where a patch
    /// whose cells it is a corner of holds no surface
    /// ([`Self::patch_holds_water`]), as water's pass, which takes no block
    /// of samples for that patch, reads it there; and else the surface drawn
    /// of the sample holding one nearest it ([`nearest`]), or its ground
    /// where none does.
    pub(crate) fn mesh(&self, heights: &[f32]) -> WaterMesh {
        let stride = TILE_STRIDE as usize;
        let drawn = drawn_surfaces(&self.surface, heights);
        let holding: Vec<bool> = drawn.iter().map(|level| level.is_finite()).collect();
        let nearest = nearest(&holding);
        // The patches along one axis whose cells the samples at `place` of
        // the layout along it are a corner of.
        let (patch, quads) = (PATCH_QUADS as usize, TILE_QUADS as usize);
        let patches = |place: usize| {
            let sample = place.saturating_sub(1).min(quads);
            (sample.saturating_sub(1) / patch) as u32..=(sample.min(quads - 1) / patch) as u32
        };
        let unblocked = |at: usize| {
            let (x, z) = (at % stride, at / stride);
            patches(z).any(|pz| patches(x).any(|px| !self.patch_holds_water(px, pz)))
        };
        let levels = (0..drawn.len())
            .map(|at| {
                if holding[at] {
                    drawn[at]
                } else if nearest[at] == NONE || unblocked(at) {
                    heights[at]
                } else {
                    drawn[nearest[at] as usize]
                }
            })
            .collect();
        let mut bits = vec![0_u64; holding.len().div_ceil(64)];
        for (at, _) in holding.iter().enumerate().filter(|&(_, &holds)| holds) {
            bits[at / 64] |= 1 << (at % 64);
        }
        WaterMesh {
            holding: bits.into_boxed_slice(),
            levels,
        }
    }

    /// The surface of sample `(x, z)`, -1 to 257 along each axis.
    #[inline]
    pub(crate) fn surface(&self, x: i32, z: i32) -> f32 {
        let stride = TILE_STRIDE as i32;
        self.surface[((z + 1) * stride + x + 1) as usize]
    }

    /// The flow, kind and span of sample `(x, z)`, -1 to 257 along each
    /// axis ([`unpack`]).
    #[inline]
    pub(crate) fn sample_info(&self, x: i32, z: i32) -> (Vec2, bool, f32) {
        let stride = TILE_STRIDE as i32;
        unpack(self.info[((z + 1) * stride + x + 1) as usize][0])
    }

    /// The state of sample `(x, z)`, -1 to 257 along each axis
    /// ([`unpack_state`]).
    #[inline]
    pub(crate) fn sample_state(&self, x: i32, z: i32) -> WaterState {
        let stride = TILE_STRIDE as i32;
        unpack_state(self.info[((z + 1) * stride + x + 1) as usize][1])
    }

    /// What the bed holds under sample `(x, z)`, -1 to 257 along each axis,
    /// as its words hold it: its reach's code ([`reach_code`]) in the low
    /// byte, beside its span's code as [`pack`] packs it (negative where a
    /// lake holds it) in the byte above.
    pub(crate) fn bed_word(&self, x: i32, z: i32) -> u32 {
        let stride = TILE_STRIDE as i32;
        let [packed, state] = self.info[((z + 1) * stride + x + 1) as usize];
        state >> 24 | (packed >> 24) << 8
    }
}

/// A tile's water as its record holds it, apart from its tile: its samples'
/// surfaces at their nearest steps of the height grid
/// ([`crate::codec::HEIGHT_STEP`]), their words, which patches hold a
/// surface and the solids standing in it, read back as they were written,
/// the surface under the banks, states and all.
impl crate::codec::Codec for TileWater {
    fn write(&self, out: &mut crate::codec::Writer) {
        crate::codec::put_grid(out, &self.surface, TILE_STRIDE as usize);
        out.put(&self.info).put(&self.held).put(&self.solids);
    }

    fn read(input: &mut crate::codec::Reader<'_>) -> Result<Self, crate::codec::CodecError> {
        let stride = TILE_STRIDE as usize;
        let surface = crate::codec::get_grid(input, stride)?;
        let info: Box<[[u32; 2]]> = input.get()?;
        let held: [u64; 4] = input.get()?;
        let solids: Box<[Footprint]> = input.get()?;
        if surface.len() != stride * stride
            || info.len() != stride * stride
            || !solids.iter().all(Footprint::valid)
        {
            return Err(crate::codec::CodecError("invalid tile water"));
        }
        Ok(Self {
            surface,
            info,
            held,
            solids,
        })
    }
}

/// Continues the water's surface under its banks: gives the dry samples
/// within [`WATER_REACH`] rings of water (each ring the dry samples among
/// the eight neighbors of the ring inside it, the wet samples the first's)
/// a surface for the water to meet their ground by and for meshes standing
/// on it to reach past the shore, and every other dry sample none, so none
/// lingers where water was held back before. Each ring is made from the
/// rings inside it alone, each of its samples from its neighbors in the
/// ring just inside ([`continued`]), so the order samples are visited in
/// changes nothing; the rings stop at a sample whose ground lies below the
/// surface it would take, so no surface crosses it.
fn extend(surface: &mut [f32], info: &mut [[u32; 2]], heights: &[f32]) {
    let stride = TILE_STRIDE as usize;
    let mut rings: Vec<u8> = surface
        .iter()
        .zip(heights)
        .map(|(level, ground)| if level > ground { 0 } else { NO_RING })
        .collect();
    for (at, &ring) in rings.iter().enumerate() {
        if ring == NO_RING {
            surface[at] = f32::NAN;
            info[at] = [0; 2];
        }
    }
    let mut inside: Vec<usize> = (0..rings.len()).filter(|&at| rings[at] == 0).collect();
    for ring in 1..=WATER_REACH as u8 {
        // The dry samples beside the ring inside, each once.
        let mut queued = Vec::new();
        for &from in &inside {
            let (x, z) = (from % stride, from / stride);
            for nz in z.saturating_sub(1)..(z + 2).min(stride) {
                for nx in x.saturating_sub(1)..(x + 2).min(stride) {
                    let near = nz * stride + nx;
                    if rings[near] == NO_RING {
                        rings[near] = QUEUED;
                        queued.push(near);
                    }
                }
            }
        }
        let made: Vec<(f32, [u32; 2], bool)> = queued
            .iter()
            .map(|&at| continued(at, ring - 1, (&*surface, &*info, heights), &rings))
            .collect();
        inside.clear();
        for (&at, &(level, words, stops)) in queued.iter().zip(&made) {
            surface[at] = level;
            info[at] = words;
            rings[at] = if stops { STOPPED } else { ring };
            if !stops {
                inside.push(at);
            }
        }
    }
}

/// What dry sample `at` takes from its neighbors in ring `inner` of the
/// water standing at `surface` with words `info` over `heights`, its rings
/// as [`extend`] counts them (`rings`): their surface, averaged, where its
/// ground stands at or above it, so the shore falls on the triangles
/// between them where the ground rises through the water, with their flow,
/// kind, span and state, averaged, and the reach most of them
/// run in (a reach is never averaged); and whether the rings stop at
/// it. A sample whose ground lies below that surface lies past where the
/// water ends (beside a channel whose ground lies lower than its banks'
/// top, a road's ditch, the land behind a levee): it stands dry, as far
/// under its surface as they stand from theirs, so the shore falls halfway
/// to it from wet ones, and the rings stop there.
fn continued(
    at: usize,
    inner: u8,
    (surface, info, heights): (&[f32], &[[u32; 2]], &[f32]),
    rings: &[u8],
) -> (f32, [u32; 2], bool) {
    let stride = TILE_STRIDE as usize;
    let (x, z) = (at % stride, at / stride);
    let (mut level, mut apart, mut flow, mut lakes, mut span, mut count) =
        (0.0, 0.0, Vec2::ZERO, 0, 0.0, 0);
    let (mut state, mut reaches) = (WaterState::default(), [0_u32; 6]);
    for nz in z.saturating_sub(1)..(z + 2).min(stride) {
        for nx in x.saturating_sub(1)..(x + 2).min(stride) {
            let near = nz * stride + nx;
            if rings[near] != inner {
                continue;
            }
            let [packed, stirred] = info[near];
            let (near_flow, lake, near_span) = unpack(packed);
            let near_state = unpack_state(stirred);
            level += surface[near];
            apart += (surface[near] - heights[near]).abs();
            flow += near_flow;
            lakes += usize::from(lake);
            span += near_span;
            state.turbulence += near_state.turbulence;
            state.breaking += near_state.breaking;
            state.shear += near_state.shear;
            reaches[reach_code(near_state.reach) as usize] += 1;
            count += 1;
        }
    }
    let share = 1.0 / count as f32;
    let level = level * share;
    let stops = heights[at] < level;
    let standing = if stops {
        heights[at] - apart * share
    } else {
        level
    };
    // The reach most of them run in, the first of those as many do.
    let most = (0..reaches.len()).fold(0, |most, code| {
        if reaches[code] > reaches[most] {
            code
        } else {
            most
        }
    });
    let words = [
        pack(flow * share, false, lakes * 2 > count, span * share),
        pack_state(WaterState {
            turbulence: state.turbulence * share,
            breaking: state.breaking * share,
            shear: state.shear * share,
            reach: reach_of(most as u32),
        }),
    ];
    (standing, words, stops)
}

/// The surface water's pass draws through a tile's water standing at
/// `surfaces` ([`TileWater::surfaces`]) at each of its samples, whose ground
/// stands at `heights` (both the tile's layout, its border included): where
/// the water stands over the ground, and where its surface continues under
/// the banks at the water's level, so the smooth surface through them stays
/// level; and none (NaN) where the rings continuing it stop ([`extend`]: the
/// ground of a ditch, of the land behind a levee or of a bank falling away
/// below the water, its surface pushed under that ground), nor where a ring
/// past the first would have continued it into one, the bank's crest the
/// water stands against there. Water's pass ends the water halfway from the
/// samples holding a surface to those holding none (`meshes/water.wgsl`'s
/// `WATER_HELD`), so it ends halfway to where the rings stop beside the
/// water, and short of a crest it stands against. The rings are found again
/// from the surfaces as their making left them, each from the ring inside it
/// alone: a sample of a ring whose surface stands under the mean of its
/// neighbors' in the ring inside, the level it would have taken, by more
/// than the height grid's step ([`crate::codec::HEIGHT_STEP`]) is one they
/// stop at.
fn drawn_surfaces(surfaces: &[f32], heights: &[f32]) -> Vec<f32> {
    let stride = TILE_STRIDE as usize;
    let mut drawn = surfaces.to_vec();
    let mut rings: Vec<u8> = surfaces
        .iter()
        .zip(heights)
        .map(|(level, ground)| if level > ground { 0 } else { NO_RING })
        .collect();
    let mut inside: Vec<usize> = (0..rings.len()).filter(|&at| rings[at] == 0).collect();
    // The samples around sample `at` in the layout, itself among them, row
    // by row.
    let around = move |at: usize| {
        let (x, z) = (at % stride, at / stride);
        (z.saturating_sub(1)..(z + 2).min(stride)).flat_map(move |nz| {
            (x.saturating_sub(1)..(x + 2).min(stride)).map(move |nx| nz * stride + nx)
        })
    };
    for ring in 1..=WATER_REACH as u8 {
        let inner = ring - 1;
        // The samples beside the ring inside no ring holds yet, each once.
        let mut queued = Vec::new();
        for &from in &inside {
            for near in around(from) {
                if rings[near] == NO_RING {
                    rings[near] = QUEUED;
                    queued.push(near);
                }
            }
        }
        // Whether the rings stop at each, from the ring inside alone, its
        // neighbors' surfaces summed as the rings were made.
        let stops: Vec<bool> = queued
            .iter()
            .map(|&at| {
                let (mut level, mut count) = (0.0_f32, 0_u32);
                for near in around(at) {
                    if rings[near] == inner {
                        level += surfaces[near];
                        count += 1;
                    }
                }
                let level = level * (1.0 / count as f32);
                surfaces[at].is_nan() || surfaces[at] < level - crate::codec::HEIGHT_STEP
            })
            .collect();
        inside.clear();
        for (&at, &stopped) in queued.iter().zip(&stops) {
            if !stopped {
                rings[at] = ring;
                inside.push(at);
                continue;
            }
            rings[at] = STOPPED;
            drawn[at] = f32::NAN;
            // Past the first ring, the crest it would have continued from.
            if inner > 0 && surfaces[at].is_finite() {
                for near in around(at) {
                    if rings[near] == inner {
                        drawn[near] = f32::NAN;
                    }
                }
            }
        }
    }
    drawn
}

/// One sample's flow (meters a second along x and z), whether a fall's
/// sheet runs over it, whether a lake holds it and how far across its water
/// body is (its span, meters: a river's full width, a lake's equal-area
/// diameter) packed as the four signed bytes of a word, low first, which
/// shaders read as snorm shares: the flow as a share of
/// [`WATER_FLOW_RANGE`] either way, a step below zero where a sheet runs
/// over it (else none), and the span's code (its nearest step on a
/// logarithmic scale from [`WATER_SPAN_LEAST`], 1, to [`WATER_SPAN_MOST`],
/// 127; 0 where none is known), negative where a lake holds the water.
/// Still water packs to zero.
pub fn pack(flow: Vec2, sheeted: bool, lake: bool, span: f32) -> u32 {
    let byte =
        |value: f32| u32::from((engine_math::clamp(value, -1.0, 1.0) * 127.0).round() as i8 as u8);
    let sheet = if sheeted { -1.0 / 127.0 } else { 0.0 };
    // A lake's code is a step at least, so its sign holds.
    let kind = if lake {
        -span_code(span).max(1)
    } else {
        span_code(span)
    };
    byte(flow.x / WATER_FLOW_RANGE)
        | byte(flow.y / WATER_FLOW_RANGE) << 8
        | byte(sheet) << 16
        | u32::from(kind as u8) << 24
}

/// [`pack`] undone: flow, whether a lake holds the water and its span
/// (meters; none where none is known).
pub fn unpack(info: u32) -> (Vec2, bool, f32) {
    let share = |shift: u32| engine_math::max(f32::from((info >> shift) as u8 as i8) / 127.0, -1.0);
    let kind = (info >> 24) as u8 as i8;
    (
        Vec2::new(share(0), share(8)) * WATER_FLOW_RANGE,
        kind < 0,
        span_of(kind),
    )
}

/// The code a span of `span` meters packs as: none (0) where none is known,
/// else the nearest of its steps on a logarithmic scale from
/// [`WATER_SPAN_LEAST`] (1) to [`WATER_SPAN_MOST`] (127), libm's logarithms
/// making it the same on every host.
fn span_code(span: f32) -> i8 {
    if span.is_nan() || span <= 0.0 {
        return 0;
    }
    let octaves = libm::log2f(WATER_SPAN_MOST / WATER_SPAN_LEAST);
    let steps = libm::log2f(span / WATER_SPAN_LEAST) / octaves * SPAN_STEPS;
    engine_math::clamp(steps.round() + 1.0, 1.0, SPAN_STEPS + 1.0) as i8
}

/// The span (meters) the code `code` packs ([`span_code`]), whatever its
/// sign: none for none.
fn span_of(code: i8) -> f32 {
    let steps = engine_math::min(f32::from(code).abs(), SPAN_STEPS + 1.0);
    if steps < 1.0 {
        return 0.0;
    }
    WATER_SPAN_LEAST
        * libm::powf(
            WATER_SPAN_MOST / WATER_SPAN_LEAST,
            (steps - 1.0) / SPAN_STEPS,
        )
}

/// One sample's state packed as the four bytes of a word, low first: its
/// turbulence, how strongly it breaks white and its shear as a share of
/// [`WATER_SHEAR_RANGE`], each in 255ths, and its reach's code
/// ([`reach_code`]). Water in no state packs to zero.
pub(crate) fn pack_state(state: WaterState) -> u32 {
    unit(state.turbulence)
        | unit(state.breaking) << 8
        | unit(state.shear / WATER_SHEAR_RANGE) << 16
        | reach_code(state.reach) << 24
}

/// [`pack_state`] undone.
pub(crate) fn unpack_state(word: u32) -> WaterState {
    let share = |shift: u32| f32::from((word >> shift) as u8) / 255.0;
    WaterState {
        turbulence: share(0),
        breaking: share(8),
        shear: share(16) * WATER_SHEAR_RANGE,
        reach: reach_of(word >> 24),
    }
}

/// The code a word packs `reach` as: 0 for none, else the reach plus one.
fn reach_code(reach: Option<Reach>) -> u32 {
    reach.map_or(0, |reach| u32::from(reach as u8) + 1)
}

/// The reach code `code` names ([`reach_code`]).
fn reach_of(code: u32) -> Option<Reach> {
    match code {
        1 => Some(Reach::Pool),
        2 => Some(Reach::Plane),
        3 => Some(Reach::Steps),
        4 => Some(Reach::Cascade),
        5 => Some(Reach::Falls),
        _ => None,
    }
}

/// `value` from 0 to 1 in 255ths.
fn unit(value: f32) -> u32 {
    (engine_math::clamp(value, 0.0, 1.0) * 255.0).round() as u32
}

/// How turbulent water runs (0 calm to 1): the swings of its speed (`rush`,
/// meters a second) against its mean speed `speed`, at least
/// [`TURBULENT_SPEED`], and those of its vorticity (`swirl`, per second) as
/// a share of [`TURBULENT_SWIRL`].
pub(crate) fn turbulence(rush: f32, speed: f32, swirl: f32) -> f32 {
    engine_math::clamp(
        rush / engine_math::max(speed, TURBULENT_SPEED) + swirl / TURBULENT_SWIRL,
        0.0,
        1.0,
    )
}

/// How strongly water `depth` meters deep (at least [`BREAK_THINNEST`])
/// that loses `energy` watts a square meter of bed breaks white (0 to 1:
/// where short-lived white water is born): where the turbulence its losses
/// stir breaks its surface against gravity, eddies as large as it is deep
/// turning at (ε·h)^(1/3) (Kolmogorov; ε the watts it loses a kilogram)
/// from [`BREAK_ONSET`] to [`BREAK_FULL`] of √(g·h), smoothly; and where it
/// leaves the bed (`lip`, the share of the time it does), plunging.
pub(crate) fn breaking(energy: f32, depth: f32, lip: f32) -> f32 {
    let depth = engine_math::max(depth, BREAK_THINNEST);
    let losing = energy / (DENSITY * depth);
    let eddies = libm::cbrtf(losing * depth);
    let breaks = eddies / (GRAVITY * depth).sqrt();
    let born = smooth((breaks - BREAK_ONSET) / (BREAK_FULL - BREAK_ONSET));
    engine_math::clamp(born + lip, 0.0, 1.0)
}

/// The shear of a tile's water at each of its samples `spacing` meters
/// apart (per second): the magnitude of the gradient of the velocity
/// `flows` holds at each (meters a second), along x and along z between
/// its neighbors either side where both hold water (`wet`), between it and
/// the one that does where one does, and none where neither does; none
/// where it holds none.
pub(crate) fn shears(flows: &[Vec2], wet: &dyn Fn(usize) -> bool, spacing: f32) -> Vec<f32> {
    let stride = TILE_STRIDE as usize;
    let mut shears = vec![0.0; flows.len()];
    for z in 0..stride {
        for x in 0..stride {
            let at = z * stride + x;
            if !wet(at) {
                continue;
            }
            // The change of the velocity a meter between the samples
            // either side.
            let change = |before: Option<usize>, after: Option<usize>| {
                let before = before.filter(|&near| wet(near));
                let after = after.filter(|&near| wet(near));
                match (before, after) {
                    (Some(before), Some(after)) => (flows[after] - flows[before]) / (2.0 * spacing),
                    (Some(before), None) => (flows[at] - flows[before]) / spacing,
                    (None, Some(after)) => (flows[after] - flows[at]) / spacing,
                    (None, None) => Vec2::ZERO,
                }
            };
            let along_x = change((x > 0).then(|| at - 1), (x + 1 < stride).then(|| at + 1));
            let along_z = change(
                (z > 0).then(|| at - stride),
                (z + 1 < stride).then(|| at + stride),
            );
            shears[at] = (along_x.length_squared() + along_z.length_squared()).sqrt();
        }
    }
    shears
}

/// The solids of `solids` (terrain-local) standing in a tile's water,
/// whose samples hold water where `wet` says, `spacing` meters apart from
/// its first at terrain-local `origin`: those any sample within a sample of
/// their shadow from above holds water at, in their order.
fn standing(
    wet: &[bool],
    (spacing, origin, solids): (f32, Vec2, &[Footprint]),
) -> Box<[Footprint]> {
    let stride = i64::from(TILE_STRIDE);
    // The layout's samples from `low` to `high` (terrain-local) along an
    // axis, a sample more either way, within the layout.
    let samples = |low: f32, high: f32, axis: usize| {
        let first = ((low - origin[axis]) / spacing).floor() as i64;
        let last = ((high - origin[axis]) / spacing).ceil() as i64 + 2;
        first.max(0)..=last.min(stride - 1)
    };
    solids
        .iter()
        .filter(|solid| {
            let (middle, reach) = (solid.middle(), solid.reach());
            let (low, high) = (middle - reach, middle + reach);
            samples(low.y, high.y, 1)
                .any(|z| samples(low.x, high.x, 0).any(|x| wet[(z * stride + x) as usize]))
        })
        .copied()
        .collect()
}

/// Each sample's nearest of the samples `seeds` marks in a tile's layout,
/// as the eight-neighbor sequential transform finds it (Danielsson 1980):
/// two passes over the rows, down and then up, each sweeping a row forward
/// and back, every sample taking the nearest seed of a neighbor it has
/// swept where that seed lies nearer to it than its own; the seed's index,
/// [`NONE`] where no sample is marked.
pub(crate) fn nearest(seeds: &[bool]) -> Vec<u32> {
    let stride = TILE_STRIDE as i32;
    let mut nearest: Vec<u32> = (0..seeds.len())
        .map(|at| if seeds[at] { at as u32 } else { NONE })
        .collect();
    let mut distance: Vec<i32> = seeds
        .iter()
        .map(|&seed| if seed { 0 } else { i32::MAX })
        .collect();
    // Sample (x, z) takes the nearest seed of its neighbor `(dx, dz)` away
    // where that seed lies nearer to it than its own.
    let mut offer = |x: i32, z: i32, (dx, dz): (i32, i32)| {
        let (nx, nz) = (x + dx, z + dz);
        if nx < 0 || nz < 0 || nx >= stride || nz >= stride {
            return;
        }
        let seed = nearest[(nz * stride + nx) as usize];
        if seed == NONE {
            return;
        }
        let (sx, sz) = (seed as i32 % stride - x, seed as i32 / stride - z);
        let squared = sx * sx + sz * sz;
        let at = (z * stride + x) as usize;
        if squared < distance[at] {
            distance[at] = squared;
            nearest[at] = seed;
        }
    };
    for z in 0..stride {
        for x in 0..stride {
            for step in [(-1, 0), (0, -1), (-1, -1), (1, -1)] {
                offer(x, z, step);
            }
        }
        for x in (0..stride).rev() {
            offer(x, z, (1, 0));
        }
    }
    for z in (0..stride).rev() {
        for x in (0..stride).rev() {
            for step in [(1, 0), (0, 1), (1, 1), (-1, 1)] {
                offer(x, z, step);
            }
        }
        for x in 0..stride {
            offer(x, z, (-1, 0));
        }
    }
    nearest
}

/// Writes the water of a tile's stored rows `rows` (0 to 258, samples -1 to
/// 257), at the stored columns `columns` gives each, into `surface` and
/// `info` (each sample's words, [`TileWater::info`]), row by row from the
/// first of `rows`: its samples at x positions `xs` and its rows at z
/// positions `zs`. Each sample takes the water of the river whose channel
/// holds it as the tile's raster says (`river`, by stored row and column),
/// flat across the river, its span the channel's width there, in no state; a lake of `lakes` holding a sample in its
/// water or shore takes it where it stands higher, its span the lake's
/// equal-area diameter. Shores fall where the ground rises through the
/// water, between the channel's samples and the dry ones beside them
/// ([`extend`]).
/// Surroundings keep water only where they show their bakes whole
/// (`shown`): where the bakes fade, so does the channel they cut.
pub(crate) fn water_rows(
    lakes: &WaterGrid,
    river: &dyn Fn(usize, usize) -> Option<RiverWater>,
    (rows, columns): (Range<usize>, &dyn Fn(usize) -> Range<usize>),
    (xs, zs): (&[f32], &[f32]),
    shown: &dyn Fn(f32, f32) -> f32,
    (surface, info): (&mut [f32], &mut [[u32; 2]]),
) {
    let stride = TILE_STRIDE as usize;
    let first = rows.start;
    for row in rows {
        let z = zs[row];
        for column in columns(row) {
            let x = xs[column];
            if shown(x, z) < 1.0 {
                continue;
            }
            let water = river(row, column);
            let at = (row - first) * stride + column;
            match lakes.lake_reach(x, z) {
                // A river keeps its own water where it runs above the lake
                // beside it.
                Some((level, span)) if !water.is_some_and(|water| water.surface > level) => {
                    surface[at] = level;
                    info[at] = [pack(Vec2::ZERO, false, true, span), 0];
                }
                _ => {
                    if let Some(water) = water {
                        surface[at] = water.surface;
                        info[at] = [pack(water.flow, water.sheeted, false, water.span), 0];
                    }
                }
            }
        }
    }
}

impl Tile {
    /// This tile of `pyramid`, generated from `layers`, with the water its
    /// hydrology bake's lakes and `rivers` hold over it: the bake's own
    /// rivers as what stands in them holds them back now, rasterized over
    /// its samples as a generated tile's are, over the ground the bake's
    /// rivers cut when it was made, which stays. Only its samples
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
        // The regions' samples alone: the surface under the banks is made
        // again from the wet samples wherever they lie (`extend`).
        let within = |values: &[f32], low: f32, high: f32| {
            values.partition_point(|&at| at < low)..values.partition_point(|&at| at <= high)
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
            || {
                (
                    vec![f32::NAN; stride * stride],
                    vec![[0; 2]; stride * stride],
                )
            },
            |water| (water.surfaces().to_vec(), water.info().to_vec()),
        );
        for (row, reached) in columns.iter().enumerate() {
            let at = row * stride + reached.start..row * stride + reached.end;
            surface[at.clone()].fill(f32::NAN);
            info[at].fill([0; 2]);
        }
        // The rivers over the rows the regions reach.
        let raster = RiverRaster::whole(rivers, &xs, &zs[first..=last], spacing);
        let at = first * stride..(last + 1) * stride;
        water_rows(
            &hydrology.water,
            &|row, column| raster.as_ref()?.water(rivers, row - first, column),
            (first..last + 1, &|row| columns[row].clone()),
            (&xs, &zs),
            &|x, z| layers.shown(x, z),
            (&mut surface[at.clone()], &mut info[at]),
        );
        let solids = self.water().map_or(&[][..], TileWater::solids);
        Some(
            self.clone()
                .with_water(surface.into_boxed_slice(), info.into_boxed_slice(), solids),
        )
    }

    /// Its first sample's terrain-local point.
    pub(crate) fn origin(&self) -> Vec2 {
        let key = self.key();
        Vec2::new(f32::from(key.x), f32::from(key.z)) * (TILE_QUADS as f32 * self.spacing())
    }
}
