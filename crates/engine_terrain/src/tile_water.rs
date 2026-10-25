//! Water as a terrain tile holds it, the one place the world's water is:
//! per sample, in the layout of the tile's heights, the surface water stands
//! at wherever any reaches (NaN where none does), and beside it three words
//! ([`TileWater::info`]): its flow, its whitewater and whether a lake holds
//! it; its state, how turbulent it runs, how white it breaks, its shear and
//! the reach of the stream it runs in; and its nearest obstacle, rock
//! breaking its surface or a boulder under it, with whether a drop or a
//! jump lies just upstream. Depth is never held: it is the surface less the
//! tile's ground at the same point, on the triangles every consumer splits
//! cells into ([`crate::Tile::water_at`]), so drawing, physics, queries,
//! placement and navigation read one water. Tile generation writes it from
//! its hydrology bake's lakes and rivers ([`water_rows`]), each river's from
//! the raster that cut its channel; a resident tile's water is written
//! again the same way where its rivers run otherwise since, as what stands
//! in them holds them back ([`Tile::rewatered`]), and as its water settles
//! ([`Tile::steadied`], [`Tile::flowing`], a job's record that the tile
//! then holds: [`Tile::holding`]), its whitewater the foam the settled water
//! carries, a share of [`crate::shallow_water::FOAM_MOST`]: rivers' own
//! water carries none, nor runs in any state. However it was written, the
//! dry samples within [`WATER_REACH`] rings of water hold its surface
//! continued under the banks, and no others any ([`extend`]), so shores fall
//! where the ground rises through the water, between samples, never at a
//! sample, and a mesh standing on the surface reaches past them; and every
//! sample knows its nearest obstacle ([`obstacles`]).

use std::ops::Range;

use glam::Vec2;

use crate::hydraulics::GRAVITY;
use crate::hydrology::WaterGrid;
use crate::noise::smooth;
use crate::rivers::{RiverRaster, RiverWater, Rivers};
use crate::shallow_water::DENSITY;
use crate::{
    Layers, PATCH_QUADS, PATCHES_PER_TILE, Pyramid, Reach, TILE_SAMPLES, TILE_STRIDE, Tile,
};

/// Meters a second either way the packed flow spans.
pub const WATER_FLOW_RANGE: f32 = 8.0;
/// Meters of half width the packed width of a river's water spans: wider
/// water packs as this wide, as lakes do.
pub const WATER_WIDTH_RANGE: f32 = 63.0;
/// Per second of shear the packed shear spans: water's velocity changing by
/// eight meters a second over a meter, past what torrents shear.
pub const WATER_SHEAR_RANGE: f32 = 8.0;
/// Meters a step of the packed offset to a sample's nearest obstacle spans,
/// a signed byte reaching nearly 16 meters, and a step of the obstacle's
/// packed radius, a byte holding as much.
pub const WATER_OBSTACLE_STEP: f32 = 0.125;
pub const WATER_RADIUS_STEP: f32 = 0.0625;
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
/// its cube is the slope times the Froude number: a 3% riffle at Froude 0.8
/// gives 0.29 and stays clear; a 10% run at Froude 1.5 gives 0.53 and shows
/// flecks; a 20% chute at Froude 2.9, or water landing below a 30 cm step,
/// gives about 0.84 and breaks white.
pub const BREAK_ONSET: f32 = 0.45;
pub const BREAK_FULL: f32 = 0.85;
/// The meters deep thinner water breaks as, so films divide by nothing:
/// about where surface tension, not gravity, holds a surface flat against
/// eddies (waves shorter than 1.7 cm are capillary).
pub const BREAK_THINNEST: f32 = 0.02;
/// The share of the time the water leaves the bed (its lip) at which a cell
/// counts wholly as a lip water plunges from: most of the time.
const LIP_HIGH: f32 = 0.5;
/// Meters the surface falls into a sample along the flow, beyond what it
/// falls on past it, where water lands below a drop or jumps: falling that
/// far, water lands at about a meter a second (√(2·g·0.05)), the speed past
/// which a plunging jet starts to draw air in.
const DROP: f32 = 0.05;
/// Meters a submerged bump's ground stands over the median of its eight
/// neighbors' at least to stand in the water as an obstacle: the least of
/// the boulders a steep stream's bed raises, which throw pillows and wakes
/// up at the surface over them.
const BUMP: f32 = 0.3;
/// The widest rock breaking the water's surface stands (meters of radius):
/// half what the packed offsets reach, so the water out to as far again
/// from its rim finds its middle. Wider dry ground the water stands around
/// is land.
const MOST_RADIUS: f32 = 0.5 * 127.0 * WATER_OBSTACLE_STEP;
/// What a sample outside every part ([`parts`]) holds as its part, and one
/// nearest no seed ([`nearest`]) as its seed.
const NONE: u32 = u32::MAX;
/// Rings of dry samples beside water, each the dry samples among the eight
/// neighbors of the ring inside it, that hold the water's surface continued
/// under its banks: the samples a mesh standing on the surface may reach
/// past the shore, as a subdivision splits its pieces until they lie within
/// them where the shore is wider (an [`crate::ErrorPyramid`]'s
/// [`crate::WIDE_SHORE`]).
pub const WATER_REACH: u32 = 4;
/// A sample's ring as [`extend`] counts them: none yet, waiting to be
/// worked out in the ring being made, and a ring the rest stop at.
const NO_RING: u8 = u8::MAX;
const QUEUED: u8 = u8::MAX - 1;
const STOPPED: u8 = u8::MAX - 2;

/// The water at a point of a tile: the surface it stands at (terrain-local
/// meters), its depth over the tile's ground there, its flow along x and z
/// (meters a second), its whitewater (0 to 1), whether a lake holds it and
/// its state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterPoint {
    pub surface: f32,
    pub depth: f32,
    pub flow: Vec2,
    pub white: f32,
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
/// words ([`Self::info`]), in the layout of its heights, and which of its
/// patches hold a surface, wet or continued under the banks.
#[derive(Clone, Debug, PartialEq)]
pub struct TileWater {
    surface: Box<[f32]>,
    info: Box<[[u32; 3]]>,
    held: [u64; 4],
}

impl TileWater {
    /// The water of a tile whose samples, `spacing` meters apart, stand at
    /// `heights`, from its samples' `surface` and words `info` (the jump of
    /// each one's obstacle word as what made the water marks it): its
    /// surface continued under the banks ([`extend`]) and each sample's
    /// nearest obstacle found ([`obstacles`]); `None` when the sizes do not
    /// match or no sample's surface stands over its ground.
    pub(crate) fn new(
        mut surface: Box<[f32]>,
        mut info: Box<[[u32; 3]]>,
        heights: &[f32],
        spacing: f32,
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
        obstacles(&surface, &mut info, heights, spacing);
        Some(Self {
            surface,
            info,
            held,
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
    /// tiles upload them, each four bytes, low first: its flow, whitewater
    /// and kind ([`pack`]); its state ([`WaterState`]: its turbulence, how
    /// strongly it breaks white and its shear as a share of
    /// [`WATER_SHEAR_RANGE`], in 255ths, and 0 for no reach, else its reach
    /// plus one); and its nearest obstacle (the offset to the obstacle's
    /// middle along x and z in signed steps of [`WATER_OBSTACLE_STEP`], its
    /// radius in steps of [`WATER_RADIUS_STEP`], 0 for none, and how
    /// strongly a drop or a jump lies just upstream, in 255ths).
    pub fn info(&self) -> &[[u32; 3]] {
        &self.info
    }

    /// Whether any sample of patch `(x, z)`, its edges included, holds a
    /// surface of its water: wet, or dry with the surface continued under it
    /// ([`Self::surfaces`]). Whether water stands at a point is its depth's
    /// to say ([`crate::Tile::water_at`]).
    pub fn patch_holds_water(&self, x: u32, z: u32) -> bool {
        let index = (z * PATCHES_PER_TILE + x) as usize;
        self.held[index / 64] & (1 << (index % 64)) != 0
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
        unpack(self.info[((z + 1) * stride + x + 1) as usize][0])
    }

    /// The state of sample `(x, z)`, -1 to 257 along each axis
    /// ([`unpack_state`]).
    #[inline]
    pub(crate) fn sample_state(&self, x: i32, z: i32) -> WaterState {
        let stride = TILE_STRIDE as i32;
        unpack_state(self.info[((z + 1) * stride + x + 1) as usize][1])
    }

    /// What the bed holds under a point `offset` meters from sample `(x,
    /// z)`, -1 to 257 along each axis, as that sample's words hold it: its
    /// reach's code ([`reach_code`]), and its nearest obstacle packed as
    /// [`pack_obstacle`] packs it, offset from the point instead of the
    /// sample.
    pub(crate) fn bed_words(&self, x: i32, z: i32, offset: Vec2) -> [u32; 2] {
        let stride = TILE_STRIDE as i32;
        let [_, state, obstacle] = self.info[((z + 1) * stride + x + 1) as usize];
        let step = |shift: u32| f32::from((obstacle >> shift) as u8 as i8) * WATER_OBSTACLE_STEP;
        let radius = f32::from((obstacle >> 16) as u8) * WATER_RADIUS_STEP;
        let toward = Vec2::new(step(0), step(8)) - offset;
        [
            state >> 24,
            pack_obstacle(toward, radius, jump_of(obstacle)),
        ]
    }
}

/// A tile's water as its record holds it, apart from its tile: its samples'
/// surfaces at their nearest steps of the height grid
/// ([`crate::codec::HEIGHT_STEP`]), their words and which patches hold a
/// surface, read back as they were written, the surface under the banks,
/// states and obstacles and all.
impl crate::codec::Codec for TileWater {
    fn write(&self, out: &mut crate::codec::Writer) {
        crate::codec::put_grid(out, &self.surface, TILE_STRIDE as usize);
        out.put(&self.info).put(&self.held);
    }

    fn read(input: &mut crate::codec::Reader<'_>) -> Result<Self, crate::codec::CodecError> {
        let stride = TILE_STRIDE as usize;
        let surface = crate::codec::get_grid(input, stride)?;
        let info: Box<[[u32; 3]]> = input.get()?;
        let held: [u64; 4] = input.get()?;
        if surface.len() != stride * stride || info.len() != stride * stride {
            return Err(crate::codec::CodecError("invalid tile water"));
        }
        Ok(Self {
            surface,
            info,
            held,
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
fn extend(surface: &mut [f32], info: &mut [[u32; 3]], heights: &[f32]) {
    let stride = TILE_STRIDE as usize;
    let mut rings: Vec<u8> = surface
        .iter()
        .zip(heights)
        .map(|(level, ground)| if level > ground { 0 } else { NO_RING })
        .collect();
    for (at, &ring) in rings.iter().enumerate() {
        if ring == NO_RING {
            surface[at] = f32::NAN;
            info[at] = [0; 3];
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
        let made: Vec<(f32, [u32; 3], bool)> = queued
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
/// whitewater, kind, width, state and jump, averaged, and the reach most of
/// them run in (a reach is never averaged); and whether the rings stop at
/// it. A sample whose ground lies below that surface lies past where the
/// water ends (beside a channel whose ground lies lower than its banks'
/// top, a road's ditch, the land behind a levee): it stands dry, as far
/// under its surface as they stand from theirs, so the shore falls halfway
/// to it from wet ones, and the rings stop there.
fn continued(
    at: usize,
    inner: u8,
    (surface, info, heights): (&[f32], &[[u32; 3]], &[f32]),
    rings: &[u8],
) -> (f32, [u32; 3], bool) {
    let stride = TILE_STRIDE as usize;
    let (x, z) = (at % stride, at / stride);
    let (mut level, mut apart, mut flow, mut white, mut lakes, mut width, mut count) =
        (0.0, 0.0, Vec2::ZERO, 0.0, 0, 0.0, 0);
    let (mut state, mut reaches, mut jump) = (WaterState::default(), [0_u32; 6], 0.0);
    for nz in z.saturating_sub(1)..(z + 2).min(stride) {
        for nx in x.saturating_sub(1)..(x + 2).min(stride) {
            let near = nz * stride + nx;
            if rings[near] != inner {
                continue;
            }
            let [packed, stirred, obstacle] = info[near];
            let (near_flow, near_white, lake, near_width) = unpack(packed);
            let near_state = unpack_state(stirred);
            level += surface[near];
            apart += (surface[near] - heights[near]).abs();
            flow += near_flow;
            white += near_white;
            lakes += usize::from(lake);
            width += near_width;
            state.turbulence += near_state.turbulence;
            state.breaking += near_state.breaking;
            state.shear += near_state.shear;
            reaches[reach_code(near_state.reach) as usize] += 1;
            jump += jump_of(obstacle);
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
        pack(
            flow * share,
            white * share,
            false,
            lakes * 2 > count,
            width * share,
        ),
        pack_state(WaterState {
            turbulence: state.turbulence * share,
            breaking: state.breaking * share,
            shear: state.shear * share,
            reach: reach_of(most as u32),
        }),
        pack_obstacle(Vec2::ZERO, 0.0, jump * share),
    ];
    (standing, words, stops)
}

/// One sample's flow (meters a second along x and z), whitewater (0 to 1),
/// whether a fall's sheet runs over it, kind and the half width of a
/// river's water there (meters) packed as the four signed bytes of a word,
/// low first, which shaders read as snorm shares: the flow as a share of
/// [`WATER_FLOW_RANGE`] either way, the whitewater (negative, a step from
/// zero at least, where a sheet runs over it), and 127 where a lake holds
/// the water, else the width as 126ths of [`WATER_WIDTH_RANGE`]. Still
/// water packs to zero.
pub fn pack(flow: Vec2, white: f32, sheeted: bool, lake: bool, width: f32) -> u32 {
    let byte =
        |value: f32| u32::from((engine_math::clamp(value, -1.0, 1.0) * 127.0).round() as i8 as u8);
    let white = if sheeted {
        -engine_math::max(white, 1.0 / 127.0)
    } else {
        white
    };
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
        share(16).abs(),
        lake,
        if lake {
            WATER_WIDTH_RANGE
        } else {
            kind / 126.0 * WATER_WIDTH_RANGE
        },
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

/// One sample's nearest obstacle packed as the four bytes of a word, low
/// first: the offset from the sample to the obstacle's middle along x and z
/// (meters) in signed steps of [`WATER_OBSTACLE_STEP`], the obstacle's
/// radius (meters) in steps of [`WATER_RADIUS_STEP`], 0 for none and a step
/// at least for one, and how strongly a drop or a jump lies just upstream
/// of the sample (0 to 1) in 255ths.
pub(crate) fn pack_obstacle(offset: Vec2, radius: f32, jump: f32) -> u32 {
    let step = |value: f32| {
        let steps = engine_math::clamp((value / WATER_OBSTACLE_STEP).round(), -127.0, 127.0);
        u32::from(steps as i8 as u8)
    };
    let radius = if radius > 0.0 {
        engine_math::clamp((radius / WATER_RADIUS_STEP).round(), 1.0, 255.0) as u32
    } else {
        0
    };
    step(offset.x) | step(offset.y) << 8 | radius << 16 | unit(jump) << 24
}

/// How strongly a drop or a jump lies just upstream of a sample whose
/// obstacle word is `word` ([`pack_obstacle`]).
fn jump_of(word: u32) -> f32 {
    f32::from((word >> 24) as u8) / 255.0
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

/// How strongly a drop lies where water leaves the bed `lip` of the time (0
/// to 1): its lip as a share of [`LIP_HIGH`].
pub(crate) fn lipped(lip: f32) -> f32 {
    engine_math::clamp(lip / LIP_HIGH, 0.0, 1.0)
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

/// Where water lands below a drop or jumps, at each sample of a tile's
/// water standing at `surface` over `heights` and running along `flows`
/// (meters a second): where it, its neighbor upstream along the flow and
/// its neighbor downstream ([`toward`]) all hold water, and its surface
/// falls into it from upstream at least [`DROP`] more than it falls on
/// past it. Water running evenly down a slope falls as much on as in.
pub(crate) fn plunges(surface: &[f32], heights: &[f32], flows: &[Vec2]) -> Vec<bool> {
    let stride = TILE_STRIDE as i32;
    let wet = |x: i32, z: i32| {
        x >= 0 && z >= 0 && x < stride && z < stride && {
            let at = (z * stride + x) as usize;
            surface[at] > heights[at]
        }
    };
    let level = |x: i32, z: i32| surface[(z * stride + x) as usize];
    let mut plunges = vec![false; surface.len()];
    for z in 0..stride {
        for x in 0..stride {
            let at = (z * stride + x) as usize;
            let Some((dx, dz)) = toward(flows[at]) else {
                continue;
            };
            if !(wet(x, z) && wet(x - dx, z - dz) && wet(x + dx, z + dz)) {
                continue;
            }
            let here = level(x, z);
            let falling_in = level(x - dx, z - dz) - here;
            let falling_on = here - level(x + dx, z + dz);
            plunges[at] = falling_in - falling_on >= DROP;
        }
    }
    plunges
}

/// The step to the one of the eight samples around a sample lying nearest
/// the way `way` (along x and z) runs: along an axis where the way runs
/// within 22.5° of it, else along a diagonal; none for no way.
fn toward(way: Vec2) -> Option<(i32, i32)> {
    // The tangent of 22.5°.
    const ASIDE: f32 = std::f32::consts::SQRT_2 - 1.0;
    let (along, across) = (way.x.abs(), way.y.abs());
    let sign = |value: f32| if value < 0.0 { -1 } else { 1 };
    let x = if along > across * ASIDE {
        sign(way.x)
    } else {
        0
    };
    let z = if across > along * ASIDE {
        sign(way.y)
    } else {
        0
    };
    (x != 0 || z != 0).then_some((x, z))
}

/// Finds each sample's nearest obstacle and writes it into its obstacle
/// word ([`pack_obstacle`]), the word's jump kept: of a tile's water
/// standing at `surface` over `heights`, its samples `spacing` meters
/// apart. An obstacle is rock breaking the water's surface, a part of dry
/// ground the water stands all around ([`parts`]: clear of the tile's
/// border, its footprint no wider than [`MOST_RADIUS`]), or a boulder under
/// it, a wet sample whose ground stands [`BUMP`] or more over the median of
/// its eight neighbors'; samples marked either way touching one another
/// make one, as round as its footprint about the middle of its samples.
/// Each sample takes the obstacle holding the marked sample nearest it
/// ([`nearest`]) where the packed offset reaches that obstacle's middle, and
/// none elsewhere. Every host finds the same: the parts and the transform
/// run in a fixed order, on whole samples.
fn obstacles(surface: &[f32], info: &mut [[u32; 3]], heights: &[f32], spacing: f32) {
    let stride = TILE_STRIDE as usize;
    let wet: Vec<bool> = surface
        .iter()
        .zip(heights)
        .map(|(&surface, &height)| surface > height)
        .collect();
    let dry: Vec<bool> = wet.iter().map(|&wet| !wet).collect();
    let (grounds, found) = parts(&dry);
    let most = MOST_RADIUS / spacing;
    let widest = (std::f32::consts::PI * most * most) as u32;
    let mut marked: Vec<bool> = grounds
        .iter()
        .map(|&part| {
            part != NONE && {
                let ground = &found[part as usize];
                !ground.border && ground.count <= widest
            }
        })
        .collect();
    for z in 1..stride - 1 {
        for x in 1..stride - 1 {
            let at = z * stride + x;
            if !wet[at] {
                continue;
            }
            let mut around = [0.0_f32; 8];
            let mut next = 0;
            for nz in z - 1..=z + 1 {
                for nx in x - 1..=x + 1 {
                    if nx != x || nz != z {
                        around[next] = heights[nz * stride + nx];
                        next += 1;
                    }
                }
            }
            around.sort_unstable_by(f32::total_cmp);
            if heights[at] - 0.5 * (around[3] + around[4]) >= BUMP {
                marked[at] = true;
            }
        }
    }
    let (labels, standing) = parts(&marked);
    let nearest = if standing.is_empty() {
        vec![NONE; marked.len()]
    } else {
        nearest(&marked)
    };
    for (at, words) in info.iter_mut().enumerate() {
        let jump = jump_of(words[2]);
        let mut word = pack_obstacle(Vec2::ZERO, 0.0, jump);
        let seed = nearest[at];
        if seed != NONE {
            let obstacle = &standing[labels[seed as usize] as usize];
            let count = f64::from(obstacle.count);
            let middle = Vec2::new(
                (obstacle.columns as f64 / count) as f32,
                (obstacle.rows as f64 / count) as f32,
            );
            let offset = (middle - Vec2::new((at % stride) as f32, (at / stride) as f32)) * spacing;
            if (offset / WATER_OBSTACLE_STEP).round().abs().max_element() <= 127.0 {
                let radius = spacing * (obstacle.count as f32 / std::f32::consts::PI).sqrt();
                word = pack_obstacle(offset, radius, jump);
            }
        }
        words[2] = word;
    }
}

/// A part of a tile's samples ([`parts`]): how many samples it holds, the
/// sums of their columns and of their rows in the tile's layout, and
/// whether it reaches the layout's border.
struct Part {
    count: u32,
    columns: u64,
    rows: u64,
    border: bool,
}

/// The parts the samples `inside` marks in a tile's layout make, samples
/// touching along a side or at a corner in one part: each sample's part
/// ([`NONE`] outside them), and the parts in the order of their first
/// samples.
fn parts(inside: &[bool]) -> (Vec<u32>, Vec<Part>) {
    let stride = TILE_STRIDE as usize;
    let mut labels = vec![NONE; inside.len()];
    let mut found = Vec::new();
    let mut open = Vec::new();
    for first in 0..inside.len() {
        if !inside[first] || labels[first] != NONE {
            continue;
        }
        let label = found.len() as u32;
        let mut part = Part {
            count: 0,
            columns: 0,
            rows: 0,
            border: false,
        };
        labels[first] = label;
        open.push(first);
        while let Some(at) = open.pop() {
            let (x, z) = (at % stride, at / stride);
            part.count += 1;
            part.columns += x as u64;
            part.rows += z as u64;
            part.border |= x == 0 || z == 0 || x + 1 == stride || z + 1 == stride;
            for nz in z.saturating_sub(1)..(z + 2).min(stride) {
                for nx in x.saturating_sub(1)..(x + 2).min(stride) {
                    let near = nz * stride + nx;
                    if inside[near] && labels[near] == NONE {
                        labels[near] = label;
                        open.push(near);
                    }
                }
            }
        }
        found.push(part);
    }
    (labels, found)
}

/// Each sample's nearest of the samples `seeds` marks in a tile's layout,
/// as the eight-neighbor sequential transform finds it (Danielsson 1980):
/// two passes over the rows, down and then up, each sweeping a row forward
/// and back, every sample taking the nearest seed of a neighbor it has
/// swept where that seed lies nearer to it than its own; the seed's index,
/// [`NONE`] where no sample is marked.
fn nearest(seeds: &[bool]) -> Vec<u32> {
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
/// flat across the river, without whitewater and in no state; a lake of
/// `lakes` holding a sample in its water or shore takes it where it stands
/// higher. Shores fall where the ground rises through the water, between
/// the channel's samples and the dry ones beside them ([`extend`]).
/// Surroundings keep water only where they show their bakes whole
/// (`shown`): where the bakes fade, so does the channel they cut.
pub(crate) fn water_rows(
    lakes: &WaterGrid,
    river: &dyn Fn(usize, usize) -> Option<RiverWater>,
    (rows, columns): (Range<usize>, &dyn Fn(usize) -> Range<usize>),
    (xs, zs): (&[f32], &[f32]),
    shown: &dyn Fn(f32, f32) -> f32,
    (surface, info): (&mut [f32], &mut [[u32; 3]]),
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
                Some(level) if !water.is_some_and(|water| water.surface > level) => {
                    surface[at] = level;
                    info[at] = [pack(Vec2::ZERO, 0.0, false, true, 0.0), 0, 0];
                }
                _ => {
                    if let Some(water) = water {
                        surface[at] = water.surface;
                        info[at] = [
                            pack(water.flow, 0.0, water.sheeted, false, water.half),
                            0,
                            0,
                        ];
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
                    vec![[0; 3]; stride * stride],
                )
            },
            |water| (water.surfaces().to_vec(), water.info().to_vec()),
        );
        for (row, reached) in columns.iter().enumerate() {
            let at = row * stride + reached.start..row * stride + reached.end;
            surface[at.clone()].fill(f32::NAN);
            info[at].fill([0; 3]);
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
        Some(
            self.clone()
                .with_water(surface.into_boxed_slice(), info.into_boxed_slice()),
        )
    }
}
