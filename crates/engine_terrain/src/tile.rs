//! Tiles: their samples, bounds and holes, sampling within them, and their
//! generation from a layer stack, a step at a time.

use std::ops::Range;
use std::sync::Arc;

use glam::{Vec2, Vec3};

use engine_math::{max, min};

use crate::layers::{Layers, state_len};
use crate::surface::{BiomeRules, Control, RowInputs, RowScratch};
use crate::{
    HydrologySample, PATCH_QUADS, PATCHES_PER_TILE, PathSample, Placing, Pyramid, TILE_QUADS,
    TILE_SAMPLES, TILE_STRIDE, TileKey,
};

/// Cells with a hole bit: those starting at samples 0..=256, so tiles agree
/// on the cells along their shared edges.
const HOLE_CELLS: usize = TILE_SAMPLES as usize;
const HOLE_WORDS: usize = (HOLE_CELLS * HOLE_CELLS).div_ceil(64);

/// A tile's samples: heights in meters above the terrain's origin of
/// samples -1 to 257 along x and z (its samples and a border of one), row
/// by row along z, with their bounds, where a stamp cut them holes, and
/// with a surface each sample's control value (decision 054).
#[derive(Clone, Debug, PartialEq)]
pub struct Tile {
    key: TileKey,
    spacing: f32,
    heights: Box<[f32]>,
    holes: Option<Box<[u64]>>,
    control: Option<Box<[u32]>>,
    placed: Option<Arc<[crate::Placed]>>,
    bounds: [f32; 2],
    patch_bounds: Box<[[f32; 2]]>,
    patch_holes: [u64; 4],
}

/// The surface at a point: its height, the normal of its triangle and
/// whether its cell is a hole.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TileSample {
    pub height: f32,
    pub normal: Vec3,
    pub hole: bool,
}

impl Tile {
    /// A tile of `key` at sample spacing `spacing` from its
    /// [`TILE_STRIDE`]-squared `heights` and optional hole bits (one per
    /// cell starting at samples 0..=256, row by row). `None` when the sizes
    /// do not match or a height is not finite.
    pub fn new(
        key: TileKey,
        spacing: f32,
        heights: Box<[f32]>,
        holes: Option<Box<[u64]>>,
    ) -> Option<Self> {
        let stride = TILE_STRIDE as usize;
        if heights.len() != stride * stride
            || holes.as_ref().is_some_and(|bits| bits.len() != HOLE_WORDS)
            || !heights.iter().all(|height| height.is_finite())
        {
            return None;
        }
        let holes = holes.filter(|bits| bits.iter().any(|&word| word != 0));
        let mut tile = Self {
            key,
            spacing,
            heights,
            holes,
            control: None,
            placed: None,
            bounds: [0.0; 2],
            patch_bounds: vec![[0.0; 2]; (PATCHES_PER_TILE * PATCHES_PER_TILE) as usize]
                .into_boxed_slice(),
            patch_holes: [0; 4],
        };
        let mut low = f32::INFINITY;
        let mut high = f32::NEG_INFINITY;
        let patch = PATCH_QUADS as i32;
        for pz in 0..PATCHES_PER_TILE as i32 {
            for px in 0..PATCHES_PER_TILE as i32 {
                let mut patch_low = f32::INFINITY;
                let mut patch_high = f32::NEG_INFINITY;
                for z in pz * patch..=(pz + 1) * patch {
                    for x in px * patch..=(px + 1) * patch {
                        let height = tile.height(x, z);
                        patch_low = min(patch_low, height);
                        patch_high = max(patch_high, height);
                    }
                }
                let index = (pz * PATCHES_PER_TILE as i32 + px) as usize;
                tile.patch_bounds[index] = [patch_low, patch_high];
                low = min(low, patch_low);
                high = max(high, patch_high);
                let holed = (pz * patch..(pz + 1) * patch).any(|z| {
                    (px * patch..(px + 1) * patch).any(|x| tile.is_hole(x as u32, z as u32))
                });
                if holed {
                    tile.patch_holes[index / 64] |= 1 << (index % 64);
                }
            }
        }
        tile.bounds = [low, high];
        Some(tile)
    }

    /// The tile with `control`: one packed [`Control`] per sample, the
    /// layout of its heights. `None` when the size does not match.
    pub fn with_control(mut self, control: Box<[u32]>) -> Option<Self> {
        let stride = TILE_STRIDE as usize;
        (control.len() == stride * stride).then(|| {
            self.control = Some(control);
            self
        })
    }

    pub fn key(&self) -> TileKey {
        self.key
    }

    /// Meters between samples.
    pub fn spacing(&self) -> f32 {
        self.spacing
    }

    /// Every sample's packed control value, the border included, row by
    /// row: the layout GPU tiles upload. `None` without a surface.
    pub fn control(&self) -> Option<&[u32]> {
        self.control.as_deref()
    }

    /// The objects the surface's placement layers of this tile's level
    /// stand on it (decision 057).
    pub fn placed(&self) -> &[crate::Placed] {
        self.placed.as_deref().unwrap_or(&[])
    }

    /// The tile with `placed` standing on it.
    pub fn with_placed(mut self, placed: Vec<crate::Placed>) -> Self {
        self.placed = (!placed.is_empty()).then(|| Arc::from(placed));
        self
    }

    /// The control value of sample `(x, z)`, -1 to 257 along each axis.
    pub fn control_at(&self, x: i32, z: i32) -> Option<Control> {
        let stride = TILE_STRIDE as i32;
        let x = x.clamp(-1, stride - 2);
        let z = z.clamp(-1, stride - 2);
        self.control
            .as_ref()
            .map(|control| Control::unpack(control[((z + 1) * stride + x + 1) as usize]))
    }

    /// Height of sample `(x, z)`, -1 to 257 along each axis.
    #[inline]
    pub fn height(&self, x: i32, z: i32) -> f32 {
        let stride = TILE_STRIDE as i32;
        self.heights[((z + 1) * stride + x + 1) as usize]
    }

    /// The largest difference, in meters, between this tile's heights and
    /// those of `parent`, the tile of the next coarser level holding it,
    /// between its samples: what this level's patches move by as they
    /// morph into the parent level's. Every third sample along each axis
    /// is compared, alternately on and between the parent's.
    pub fn morph_error(&self, parent: &Tile) -> f32 {
        let offset = [
            i32::from(self.key.x % 2) * 128,
            i32::from(self.key.z % 2) * 128,
        ];
        let mut largest = 0.0_f32;
        for z in (0..=TILE_QUADS as i32).step_by(3) {
            for x in (0..=TILE_QUADS as i32).step_by(3) {
                // The parent's samples around this one, half as far apart.
                let (px, pz) = (offset[0] * 2 + x, offset[1] * 2 + z);
                let (x0, z0) = (px.div_euclid(2), pz.div_euclid(2));
                let (fx, fz) = ((px % 2) as f32 * 0.5, (pz % 2) as f32 * 0.5);
                let a = parent.height(x0, z0);
                let b = parent.height(x0 + 1, z0);
                let c = parent.height(x0, z0 + 1);
                let d = parent.height(x0 + 1, z0 + 1);
                let coarse = (a + (b - a) * fx) + ((c + (d - c) * fx) - (a + (b - a) * fx)) * fz;
                largest = max(largest, (self.height(x, z) - coarse).abs());
            }
        }
        largest
    }

    /// Every sample, the border included, row by row: the layout GPU tiles
    /// upload.
    pub fn heights(&self) -> &[f32] {
        &self.heights
    }

    /// Lowest and highest height of samples 0 to 256.
    pub fn bounds(&self) -> [f32; 2] {
        self.bounds
    }

    /// Lowest and highest height of patch `(x, z)`'s samples.
    pub fn patch_bounds(&self, x: u32, z: u32) -> [f32; 2] {
        self.patch_bounds[(z * PATCHES_PER_TILE + x) as usize]
    }

    /// Whether any cell of patch `(x, z)` is a hole.
    pub fn patch_has_holes(&self, x: u32, z: u32) -> bool {
        let index = (z * PATCHES_PER_TILE + x) as usize;
        self.patch_holes[index / 64] & (1 << (index % 64)) != 0
    }

    /// Hole bits of cells starting at samples 0..=256, row by row; `None`
    /// without holes.
    pub fn hole_bits(&self) -> Option<&[u64]> {
        self.holes.as_deref()
    }

    /// Whether the cell starting at sample `(x, z)` is a hole.
    pub fn is_hole(&self, x: u32, z: u32) -> bool {
        let Some(bits) = &self.holes else {
            return false;
        };
        if x as usize >= HOLE_CELLS || z as usize >= HOLE_CELLS {
            return false;
        }
        let index = z as usize * HOLE_CELLS + x as usize;
        bits[index / 64] & (1 << (index % 64)) != 0
    }

    /// The surface at `local`, meters from the tile's first corner, on the
    /// triangles every consumer splits cells into; `None` outside the tile.
    pub fn sample(&self, local: Vec2) -> Option<TileSample> {
        let size = TILE_QUADS as f32;
        let u = local.x / self.spacing;
        let v = local.y / self.spacing;
        if !(u.is_finite() && v.is_finite()) || u < 0.0 || v < 0.0 || u > size || v > size {
            return None;
        }
        let i = (u.floor() as i32).min(TILE_QUADS as i32 - 1);
        let j = (v.floor() as i32).min(TILE_QUADS as i32 - 1);
        let fu = u - i as f32;
        let fv = v - j as f32;
        let h00 = self.height(i, j);
        let h10 = self.height(i + 1, j);
        let h01 = self.height(i, j + 1);
        let h11 = self.height(i + 1, j + 1);
        // The diagonal runs from the cell's (x, z+1) corner to its (x+1, z)
        // corner.
        let (height, slope) = if fu + fv <= 1.0 {
            (
                h00 + fu * (h10 - h00) + fv * (h01 - h00),
                Vec2::new(h10 - h00, h01 - h00),
            )
        } else {
            (
                h11 + (1.0 - fu) * (h01 - h11) + (1.0 - fv) * (h10 - h11),
                Vec2::new(h11 - h01, h11 - h10),
            )
        };
        let normal = Vec3::new(-slope.x / self.spacing, 1.0, -slope.y / self.spacing).normalize();
        Some(TileSample {
            height,
            normal,
            hole: self.is_hole(i as u32, j as u32),
        })
    }
}

/// Coarse grid points along each axis: every fourth sample from sample -8
/// to 264, enough for Catmull-Rom interpolation at samples -1 to 257.
const GRID: usize = 69;
const GRID_FIRST: i64 = -2;
const GRID_STEP: i64 = 4;
/// Work units of a coarse grid point and of a noise layer's sample,
/// against one per sample of other layers: about their time.
const COARSE_COST: u64 = 24;
const NOISE_SAMPLE_COST: u64 = 4;

/// Four coarse states `taps` combined with Catmull-Rom `weights` into
/// `out`, value by value, in one order wherever states are interpolated.
#[inline(always)]
fn interpolate(out: &mut [f32], weights: [f32; 4], taps: [&[f32]; 4]) {
    let [a, b, c, d] = taps;
    for ((((value, a), b), c), d) in out.iter_mut().zip(a).zip(b).zip(c).zip(d) {
        *value = weights[0] * a + weights[1] * b + weights[2] * c + weights[3] * d;
    }
}

/// Catmull-Rom weights at quarter steps, exact in `f32`.
const QUARTER_WEIGHTS: [[f32; 4]; 4] = [
    [0.0, 1.0, 0.0, 0.0],
    [-0.070_312_5, 0.867_187_5, 0.226_562_5, -0.023_437_5],
    [-0.0625, 0.5625, 0.5625, -0.0625],
    [-0.023_437_5, 0.226_562_5, 0.867_187_5, -0.070_312_5],
];

#[derive(Clone, Copy, Debug)]
enum Stage {
    Coarse { layer: usize, row: usize },
    Samples { row: i32 },
    Holes { row: u32 },
    Biomes { row: usize },
    Done,
    Place { layer: usize, row: usize },
    Failed,
    // A band's rows are done (decision 117).
    Banded,
}

/// Rows of a tile a band of its generation made (decision 117): the tile's
/// key, the first stored row (0 to 258, samples -1 to 257), and those rows'
/// heights and control values, with the hole bits of the cells starting in
/// them; [`merge_bands`] joins a tile's bands.
#[derive(Clone, Debug, PartialEq)]
pub struct TileBand {
    pub key: TileKey,
    pub first: u32,
    pub heights: Vec<f32>,
    pub holes: Option<Vec<u64>>,
    pub control: Option<Vec<u32>>,
}

crate::codec::record!(TileBand {
    key,
    first,
    heights,
    holes,
    control,
});

/// Tile `key` at `spacing` joined from `bands` (decision 117), each row from
/// its band: the same tile generating it whole gives, without objects.
/// `None` unless the bands cover every row once, alike in holes and
/// control values, with valid heights and control values.
pub fn merge_bands(key: TileKey, spacing: f32, bands: &[TileBand]) -> Option<Tile> {
    let stride = TILE_STRIDE as usize;
    let mut heights = vec![0.0; stride * stride];
    let mut covered = vec![false; stride];
    let holed = bands.first()?.holes.is_some();
    let surfaced = bands.first()?.control.is_some();
    let mut holes = holed.then(|| vec![0_u64; HOLE_WORDS]);
    let mut control = surfaced.then(|| vec![0_u32; stride * stride]);
    for band in bands {
        let rows = band.heights.len() / stride;
        let first = band.first as usize;
        if band.key != key
            || band.heights.len() != rows * stride
            || first + rows > stride
            || band.holes.is_some() != holed
            || band.control.as_ref().map(Vec::len) != surfaced.then_some(rows * stride)
        {
            return None;
        }
        for row in &mut covered[first..first + rows] {
            if std::mem::replace(row, true) {
                return None;
            }
        }
        let at = first * stride..(first + rows) * stride;
        heights[at.clone()].copy_from_slice(&band.heights);
        if let (Some(out), Some(values)) = (&mut control, &band.control) {
            out[at].copy_from_slice(values);
        }
        if let (Some(out), Some(bits)) = (&mut holes, &band.holes) {
            if bits.len() != HOLE_WORDS {
                return None;
            }
            for (word, bits) in out.iter_mut().zip(bits) {
                *word |= bits;
            }
        }
    }
    if !covered.iter().all(|&row| row) {
        return None;
    }
    let tile = Tile::new(
        key,
        spacing,
        heights.into_boxed_slice(),
        holes.map(Vec::into_boxed_slice),
    )?;
    match control {
        Some(control) => tile.with_control(control.into_boxed_slice()),
        None => Some(tile),
    }
}

/// Generates one tile from a layer stack in steps, so streaming can spread
/// the work over frames. Every sample is a pure function of its position
/// and level: noise octaves at least 32 samples long are evaluated on a
/// grid of every fourth sample and interpolated, shorter ones down to four
/// samples at every sample, so neighboring tiles agree on shared samples.
pub struct Generator {
    layers: Arc<Layers>,
    key: TileKey,
    spacing: f32,
    first: [i64; 2],
    bands: Vec<Option<(usize, Range<usize>, usize)>>,
    // Each noise layer's coarse states, grid point by grid point.
    grids: Vec<Option<Box<[f32]>>>,
    stage: Stage,
    heights: Box<[f32]>,
    holes: Option<Box<[u64]>>,
    xs: Box<[f32]>,
    row: Box<[f32]>,
    values: Box<[f32]>,
    // A row's coarse states interpolated along z, grid column by column,
    // and one sample's.
    column: Box<[f32]>,
    state: Box<[f32]>,
    min: Vec2,
    max: Vec2,
    // The surface whose control values the tile gets, their samples, and
    // scratch space for one sample's material weights.
    surface: Option<Arc<BiomeRules>>,
    control: Option<Box<[u32]>>,
    weights: Vec<(u8, f32)>,
    // The finished tile while its objects are placed (decision 057), a row
    // of candidate squares at a time, the objects placed so far and which
    // of them it places (decision 112).
    tile: Option<Tile>,
    placed: Vec<crate::Placed>,
    placing: Placing,
    // Each hydrology layer's rivers near the tile, read a row at a time
    // (decision 113), once a stage needs them, and its stored rows' z; a
    // row's inputs to the surface's masks.
    spans: Vec<Option<Arc<crate::rivers::RiverSpan>>>,
    zs: Box<[f32]>,
    inputs: SurfaceInputs,
    places: crate::placement::PlaceScratch,
    // Whether it hands out the tile's shape before its surface (decision
    // 114), and the shape until taken.
    staged: bool,
    shape: Option<Tile>,
    // The stored rows a band generates alone (decision 117), and its rows
    // once done until taken.
    rows: Option<(usize, usize)>,
    banded: Option<TileBand>,
    // Whether it gives its tile control values; the GPU gives them where
    // not (decision 118).
    materials: bool,
    // Where a band of placing stops (decision 119): the layer and row it
    // leaves to the next band, none for the last, which also places what
    // authors stand on the tile.
    place_end: Option<(usize, usize)>,
}

/// A row's inputs to a surface's masks, reused row after row (decision
/// 113).
#[derive(Default)]
struct SurfaceInputs {
    zs: Vec<f32>,
    height: Vec<f32>,
    gradient: Vec<f32>,
    curvature: Vec<f32>,
    water: Vec<HydrologySample>,
    path: Vec<PathSample>,
    scratch: RowScratch,
}

impl SurfaceInputs {
    fn clear(&mut self, len: usize) {
        for row in [&mut self.height, &mut self.gradient, &mut self.curvature] {
            row.clear();
            row.reserve(len);
        }
        self.water.clear();
        self.path.clear();
    }
}

impl Generator {
    pub fn new(layers: Arc<Layers>, pyramid: &Pyramid, key: TileKey) -> Self {
        let level = u32::from(key.level);
        let spacing = pyramid.spacing(level);
        // Surroundings (decision 094) read their stack where their samples
        // lie in the ground it was drawn for.
        let [x, z] = pyramid.first_sample(key);
        let [dx, dz] = layers.offset_samples(spacing);
        let first = [x + dx, z + dz];
        let stride = TILE_STRIDE as usize;
        let bands: Vec<_> = (0..layers.layers().len())
            .map(|index| layers.noise(index).map(|tables| tables.bands(spacing)))
            .collect();
        let lens: Vec<_> = bands
            .iter()
            .map(|bands| bands.as_ref().map(|(coarse, _, _)| state_len(*coarse)))
            .collect();
        let grids = lens
            .iter()
            .map(|len| len.map(|len| vec![0.0; GRID * GRID * len].into_boxed_slice()))
            .collect();
        let widest = lens.iter().flatten().copied().max().unwrap_or(0);
        let xs = (-1..=TILE_SAMPLES as i64)
            .map(|i| (first[0] + i) as f32 * spacing)
            .collect();
        let min = Vec2::new(
            (first[0] - 1) as f32 * spacing,
            (first[1] - 1) as f32 * spacing,
        );
        let max = Vec2::new(
            (first[0] + i64::from(TILE_SAMPLES)) as f32 * spacing,
            (first[1] + i64::from(TILE_SAMPLES)) as f32 * spacing,
        );
        let holes = layers
            .holes_reach(min, max)
            .then(|| vec![0; HOLE_WORDS].into_boxed_slice());
        Self {
            layers,
            key,
            spacing,
            first,
            bands,
            grids,
            stage: Stage::Coarse { layer: 0, row: 0 },
            heights: vec![0.0; stride * stride].into_boxed_slice(),
            holes,
            xs,
            row: vec![0.0; stride].into_boxed_slice(),
            values: vec![0.0; stride].into_boxed_slice(),
            column: vec![0.0; GRID * widest].into_boxed_slice(),
            state: vec![0.0; widest].into_boxed_slice(),
            min,
            max,
            surface: None,
            control: None,
            weights: Vec::new(),
            tile: None,
            placed: Vec::new(),
            placing: Placing::All,
            spans: Vec::new(),
            zs: (-1..=TILE_SAMPLES as i64)
                .map(|i| (first[1] + i) as f32 * spacing)
                .collect(),
            inputs: SurfaceInputs::default(),
            places: crate::placement::PlaceScratch::default(),
            staged: false,
            shape: None,
            rows: None,
            banded: None,
            materials: true,
            place_end: None,
        }
    }

    /// The generator leaving its tile without control values (decision
    /// 118), which the GPU gives to rendering and the rules to gameplay at
    /// the samples it asks for; it still places objects.
    pub fn with_materials(mut self, materials: bool) -> Self {
        self.materials = materials;
        self
    }

    /// A generator of stored rows `rows` of tile `key` alone (decision
    /// 117; 0 to 259, samples -1 to 257): their heights, holes and control
    /// values as generating the whole tile gives them, for [`merge_bands`]
    /// to join with the tile's other bands ([`Self::take_band`]). `None`
    /// for rows outside the tile.
    pub fn band(
        layers: Arc<Layers>,
        pyramid: &Pyramid,
        key: TileKey,
        rows: Range<usize>,
    ) -> Option<Self> {
        if rows.start >= rows.end || rows.end > TILE_STRIDE as usize {
            return None;
        }
        let mut generator = Self::new(layers, pyramid, key);
        generator.rows = Some((rows.start, rows.end));
        generator.stage = Stage::Coarse {
            layer: 0,
            row: generator.grid_rows().0,
        };
        Some(generator)
    }

    /// A band's rows once done (decision 117), taken once.
    pub fn take_band(&mut self) -> Option<TileBand> {
        self.banded.take()
    }

    /// The sample rows it samples heights of: every row (-1 to 257), or a
    /// band's (decision 117) with the rows on either side its control
    /// values read.
    fn height_rows(&self) -> (i32, i32) {
        let end = TILE_SAMPLES as i32 + 1;
        match self.rows {
            None => (-1, end),
            Some((first, last)) => (
                std::cmp::max(first as i32 - 2, -1),
                std::cmp::min(last as i32, end),
            ),
        }
    }

    /// The coarse grid rows its height rows read: each sample row reads
    /// four from the one after its grid point's.
    fn grid_rows(&self) -> (usize, usize) {
        let (first, end) = self.height_rows();
        let low = i64::from(first).div_euclid(GRID_STEP) - GRID_FIRST - 1;
        let high = i64::from(end - 1).div_euclid(GRID_STEP) - GRID_FIRST + 3;
        (low as usize, std::cmp::min(high as usize, GRID))
    }

    /// The rows of hole cells it marks, each cell starting at its sample
    /// row (0 to 256).
    fn hole_rows(&self) -> (u32, u32) {
        match self.rows {
            None => (0, HOLE_CELLS as u32),
            Some((first, end)) => (
                first.saturating_sub(1) as u32,
                std::cmp::min(end.saturating_sub(1), HOLE_CELLS) as u32,
            ),
        }
    }

    /// The stored rows it gives control values.
    fn control_rows(&self) -> (usize, usize) {
        self.rows.unwrap_or((0, TILE_STRIDE as usize))
    }

    /// The generator giving its tile `surface`'s control values.
    pub fn with_surface(mut self, surface: Arc<BiomeRules>) -> Self {
        self.surface = Some(surface);
        self
    }

    /// The generator placing only `placing`'s objects (decision 112).
    pub fn with_placing(mut self, placing: Placing) -> Self {
        self.placing = placing;
        self
    }

    /// The generator handing out its tile's shape, its heights and holes,
    /// once they are done (decision 114): [`Self::take_shape`] it while the
    /// surface and objects follow.
    pub fn with_staged(mut self, staged: bool) -> Self {
        self.staged = staged;
        self
    }

    /// The tile's shape once its heights and holes are done, before its
    /// surface (decision 114), where staged: a tile without control values
    /// or objects, taken once.
    pub fn take_shape(&mut self) -> Option<Tile> {
        self.shape.take()
    }

    /// A generator placing on `tile` (generated from `layers`) the objects
    /// of `surface`'s layers standing far away at its level (decision
    /// 112), under the same budget as generation: the tile it returns
    /// holds those alone.
    pub fn far(tile: Tile, layers: Arc<Layers>, surface: Arc<BiomeRules>) -> Self {
        Self::placing(tile, layers, surface, Placing::Far)
    }

    /// A generator placing `placing`'s objects of `surface`'s layers on
    /// `tile`, generated from `layers` without them (decisions 112 and
    /// 117), under the same budget as generation: the tile it returns
    /// holds those alone, as generating it whole places them.
    pub fn placing(
        tile: Tile,
        layers: Arc<Layers>,
        surface: Arc<BiomeRules>,
        placing: Placing,
    ) -> Self {
        Self {
            layers,
            key: tile.key(),
            spacing: tile.spacing(),
            first: [0; 2],
            bands: Vec::new(),
            grids: Vec::new(),
            stage: Stage::Place { layer: 0, row: 0 },
            heights: Box::default(),
            holes: None,
            xs: Box::default(),
            row: Box::default(),
            values: Box::default(),
            column: Box::default(),
            state: Box::default(),
            min: Vec2::ZERO,
            max: Vec2::ZERO,
            surface: Some(surface),
            control: None,
            weights: Vec::new(),
            tile: Some(tile.with_placed(Vec::new())),
            placed: Vec::new(),
            placing,
            spans: Vec::new(),
            zs: Box::default(),
            inputs: SurfaceInputs::default(),
            places: crate::placement::PlaceScratch::default(),
            staged: false,
            shape: None,
            rows: None,
            banded: None,
            materials: true,
            place_end: None,
        }
    }

    /// A generator placing on `tile` as [`Self::placing`] does, from
    /// layer and row `start` up to `end` alone (decision 119), the rows
    /// [`crate::BiomeRules::place_rows`] counts: bands of a tile's placing
    /// on several workers, their objects joined in order giving the tile's.
    /// The band without an end also places what authors stand on it.
    pub fn placing_rows(
        tile: Tile,
        (layers, surface): (Arc<Layers>, Arc<BiomeRules>),
        placing: Placing,
        (start, end): ((usize, usize), Option<(usize, usize)>),
    ) -> Self {
        let mut generator = Self::placing(tile, layers, surface, placing);
        generator.stage = Stage::Place {
            layer: start.0,
            row: start.1,
        };
        generator.place_end = end;
        generator
    }

    /// Hydrology layer `index`'s rivers near the tile (decision 113), made
    /// the first time a stage reads them; `None` without rivers.
    fn span(&mut self, index: usize) -> Option<Arc<crate::rivers::RiverSpan>> {
        let rivers = self.layers.rivers_of(index)?;
        if self.spans.len() <= index {
            self.spans.resize_with(index + 1, || None);
        }
        let span = &mut self.spans[index];
        if span.is_none() {
            *span = Some(Arc::new(rivers.span(&self.xs, &self.zs, self.spacing)));
        }
        span.clone()
    }

    /// A generator giving `tile` (of `pyramid`, its heights read from a
    /// file) `surface`'s control values, under the same budget as
    /// generation.
    pub fn resurface(tile: &Tile, pyramid: &Pyramid, surface: Arc<BiomeRules>) -> Self {
        let layers = Arc::new(Layers::new(Vec::new()).expect("an empty stack is valid"));
        let mut generator = Self::new(layers, pyramid, tile.key()).with_surface(surface);
        generator.heights = tile.heights().into();
        generator.holes = tile.hole_bits().map(Box::from);
        generator.stage = Stage::Biomes { row: 0 };
        generator
    }

    pub fn key(&self) -> TileKey {
        self.key
    }

    /// Whether generation stopped without a tile: its heights or control
    /// values were invalid (decision 112).
    pub fn failed(&self) -> bool {
        matches!(self.stage, Stage::Failed)
    }

    /// The stage after the samples and holes: the surface's control values
    /// unless the GPU gives them (decision 118), else done; the tile's shape
    /// handed out first where staged and more follows (decision 114), its
    /// control values or its objects.
    fn after_holes(&mut self) -> Stage {
        let next = match &self.surface {
            Some(_) if self.materials => Stage::Biomes {
                row: self.control_rows().0,
            },
            _ => Stage::Done,
        };
        let more = matches!(next, Stage::Biomes { .. })
            || self
                .surface
                .as_ref()
                .is_some_and(|surface| surface.places(self.key.level, self.placing));
        if self.staged && self.rows.is_none() && more {
            self.shape = Tile::new(
                self.key,
                self.spacing,
                self.heights.clone(),
                self.holes.clone(),
            );
        }
        next
    }

    /// Control values of stored row `row` (0 to 258, samples -1 to 257),
    /// from the heights around each sample; border samples read their
    /// nearest neighbors inside.
    fn surface_row(&mut self, row: usize) {
        let stride = TILE_STRIDE as usize;
        let Some(surface) = self.surface.clone() else {
            return;
        };
        // Inputs the masks never read are left unread (decision 113); each
        // sample's meters to the nearest river's edge are read along the row
        // at once.
        let reads = surface.reads();
        let mut rivers = vec![crate::hydrology::RIVER_FAR; stride];
        if reads.water
            && let Some((index, _)) = self.layers.hydrology_layer()
            && let Some(span) = self.span(index)
            && let Some(found) = self.layers.rivers_of(index)
        {
            let z = self.zs[row];
            found.nearest_row(
                &span,
                row,
                (&self.xs, z),
                crate::hydrology::RIVER_FAR,
                &mut rivers,
            );
        }
        let control = self
            .control
            .get_or_insert_with(|| vec![0; stride * stride].into_boxed_slice());
        let spacing = self.spacing;
        let heights = &self.heights;
        let (up, down) = (row.saturating_sub(1), (row + 1).min(stride - 1));
        let z = (self.first[1] + row as i64 - 1) as f32 * spacing;
        // The row's inputs, then its control values all at once (decision
        // 113).
        let inputs = &mut self.inputs;
        inputs.clear(stride);
        for (column, &river) in rivers.iter().enumerate() {
            let (left, right) = (column.saturating_sub(1), (column + 1).min(stride - 1));
            let at = |r: usize, c: usize| heights[r * stride + c];
            let height = at(row, column);
            let (west, east, north, south) = (
                at(row, left),
                at(row, right),
                at(up, column),
                at(down, column),
            );
            let gx = (east - west) / (2.0 * spacing);
            let gz = (south - north) / (2.0 * spacing);
            let x = self.xs[column];
            // Surroundings (decision 094) lose the masks of the bakes and
            // roads they fade out.
            let shown = self.layers.shown(x, z);
            let water = match self.layers.hydrology() {
                Some(water) if shown > 0.0 && reads.water => water.masks_at(x, z, spacing, river),
                _ => HydrologySample::default(),
            };
            inputs.height.push(height);
            inputs.gradient.push((gx * gx + gz * gz).sqrt());
            inputs
                .curvature
                .push((west + east + north + south - 4.0 * height) / (spacing * spacing));
            inputs.water.push(HydrologySample {
                flow: water.flow * shown,
                wetness: water.wetness * shown,
                lake: water.lake * shown,
                carve: water.carve * shown,
                river: water.river,
            });
            inputs.path.push(if shown > 0.0 && reads.path {
                self.layers.path_at(x, z)
            } else {
                PathSample::FAR
            });
        }
        inputs.zs.clear();
        inputs.zs.resize(stride, z);
        let row_inputs = RowInputs {
            xs: &self.xs,
            zs: &inputs.zs,
            spacing,
            height: &inputs.height,
            gradient: &inputs.gradient,
            curvature: &inputs.curvature,
            water: &inputs.water,
            path: &inputs.path,
        };
        surface.control_row(
            &row_inputs,
            &mut control[row * stride..(row + 1) * stride],
            (&mut inputs.scratch, &mut self.weights),
        );
    }

    /// Generates until the tile is done or `budget` work units run out
    /// (about one per sample), returning the tile once done. A step that
    /// starts always finishes, so a budget can go slightly negative in
    /// effect; callers pass what is left.
    pub fn advance(&mut self, budget: &mut u64) -> Option<Tile> {
        loop {
            if *budget == 0 {
                return None;
            }
            match self.stage {
                Stage::Coarse { layer, row } => {
                    if layer >= self.grids.len() {
                        self.stage = Stage::Samples {
                            row: self.height_rows().0,
                        };
                        continue;
                    }
                    let Some(tables) = self.layers.noise(layer) else {
                        self.stage = Stage::Coarse {
                            layer: layer + 1,
                            row: self.grid_rows().0,
                        };
                        continue;
                    };
                    let (coarse, _, gullies) = self.bands[layer].clone().unwrap_or_default();
                    let z = (self.first[1] + (GRID_FIRST + row as i64) * GRID_STEP) as f32
                        * self.spacing;
                    let grid = self.grids[layer].as_mut().expect("noise layers have grids");
                    let len = state_len(coarse);
                    for column in 0..GRID {
                        let x = (self.first[0] + (GRID_FIRST + column as i64) * GRID_STEP) as f32
                            * self.spacing;
                        let at = (row * GRID + column) * len;
                        tables.coarse(x, z, coarse, gullies, &mut grid[at..at + len]);
                    }
                    *budget = budget.saturating_sub(GRID as u64 * COARSE_COST);
                    let (first, end) = self.grid_rows();
                    self.stage = if row + 1 < end {
                        Stage::Coarse {
                            layer,
                            row: row + 1,
                        }
                    } else {
                        Stage::Coarse {
                            layer: layer + 1,
                            row: first,
                        }
                    };
                }
                Stage::Samples { row } => {
                    self.sample_row(row);
                    let layers = self
                        .grids
                        .iter()
                        .map(|grid| if grid.is_some() { NOISE_SAMPLE_COST } else { 1 })
                        .sum::<u64>();
                    *budget = budget.saturating_sub(u64::from(TILE_STRIDE) * layers.max(1));
                    let (first, end) = self.hole_rows();
                    self.stage = if row + 1 < self.height_rows().1 {
                        Stage::Samples { row: row + 1 }
                    } else if self.holes.is_some() && first < end {
                        Stage::Holes { row: first }
                    } else {
                        self.after_holes()
                    };
                }
                Stage::Holes { row } => {
                    let z = ((self.first[1] + i64::from(row)) as f32 + 0.5) * self.spacing;
                    let bits = self.holes.as_mut().expect("holes stage has bits");
                    for column in 0..HOLE_CELLS as u32 {
                        let x = ((self.first[0] + i64::from(column)) as f32 + 0.5) * self.spacing;
                        if self.layers.hole_at(x, z) {
                            let index = row as usize * HOLE_CELLS + column as usize;
                            bits[index / 64] |= 1 << (index % 64);
                        }
                    }
                    *budget = budget.saturating_sub(HOLE_CELLS as u64);
                    self.stage = if row + 1 < self.hole_rows().1 {
                        Stage::Holes { row: row + 1 }
                    } else {
                        self.after_holes()
                    };
                }
                Stage::Biomes { row } => {
                    self.surface_row(row);
                    let cost = self.surface.as_ref().map_or(1, |surface| surface.cost());
                    *budget = budget.saturating_sub(u64::from(TILE_STRIDE) * cost);
                    self.stage = if row + 1 < self.control_rows().1 {
                        Stage::Biomes { row: row + 1 }
                    } else {
                        Stage::Done
                    };
                }
                // A band hands out its own rows (decision 117).
                Stage::Done if self.rows.is_some() => {
                    let (first, end) = self.control_rows();
                    let stride = TILE_STRIDE as usize;
                    let at = first * stride..end * stride;
                    self.banded = Some(TileBand {
                        key: self.key,
                        first: first as u32,
                        heights: self.heights[at.clone()].to_vec(),
                        holes: self.holes.as_deref().map(<[u64]>::to_vec),
                        control: self.control.as_deref().map(|control| control[at].to_vec()),
                    });
                    self.stage = Stage::Banded;
                    return None;
                }
                Stage::Banded => return None,
                Stage::Done => {
                    let heights = std::mem::take(&mut self.heights);
                    self.stage = Stage::Failed;
                    let tile = Tile::new(self.key, self.spacing, heights, self.holes.take())?;
                    let tile = match self.control.take() {
                        Some(control) => tile.with_control(control)?,
                        None => tile,
                    };
                    self.stage = Stage::Done;
                    // Objects stand on the tiles of their layers' levels
                    // (decision 057), placed in the stage after.
                    match &self.surface {
                        Some(surface) if surface.places(self.key.level, self.placing) => {
                            self.tile = Some(tile);
                            self.stage = Stage::Place { layer: 0, row: 0 };
                        }
                        _ => return Some(tile),
                    }
                }
                Stage::Failed => return None,
                Stage::Place { layer, row } => {
                    let (Some(surface), Some(tile)) = (&self.surface, &self.tile) else {
                        self.stage = Stage::Failed;
                        return None;
                    };
                    let (cost, next) = surface.place_row(
                        tile,
                        (layer, row),
                        &mut self.placed,
                        (&self.layers, self.placing, &mut self.places),
                    );
                    *budget = budget.saturating_sub(cost);
                    // A band stops where the next starts (decision 119).
                    let band_ends = next.is_some() && next == self.place_end;
                    match next {
                        Some((layer, row)) if !band_ends => {
                            self.stage = Stage::Place { layer, row }
                        }
                        _ => {
                            // Then what authors stand on it (decision 085),
                            // with its own level's objects.
                            if self.placing != Placing::Far && !band_ends {
                                *budget = budget
                                    .saturating_sub(surface.place_authored(tile, &mut self.placed));
                            }
                            let tile = self.tile.take()?;
                            return Some(tile.with_placed(std::mem::take(&mut self.placed)));
                        }
                    }
                }
            }
        }
    }

    /// Heights of sample row `row` (-1 to 257), every layer applied in
    /// order.
    fn sample_row(&mut self, row: i32) {
        let z = (self.first[1] + i64::from(row)) as f32 * self.spacing;
        self.row.fill(0.0);
        let coarse_row = i64::from(row).div_euclid(GRID_STEP);
        let vertical = QUARTER_WEIGHTS[i64::from(row).rem_euclid(GRID_STEP) as usize];
        for layer in 0..self.layers.layers().len() {
            let Some(tables) = self.layers.noise(layer) else {
                let at = (row + 1) as usize;
                let span = self.span(layer);
                self.layers.apply_local(
                    layer,
                    &mut self.row,
                    &self.xs,
                    z,
                    (self.min, self.max),
                    self.spacing,
                    span.as_deref().map(|span| (span, at)),
                );
                continue;
            };
            let (octaves, fine, _) = self.bands[layer].clone().unwrap_or_default();
            let len = state_len(octaves);
            let grid = self.grids[layer].as_ref().expect("noise layers have grids");
            // The coarse state along z first, for every grid column.
            let base = (coarse_row - 1 - GRID_FIRST) as usize;
            let tap = |row: usize, column: usize| {
                let at = ((base + row) * GRID + column) * len;
                &grid[at..at + len]
            };
            for (column, sum) in self.column.chunks_exact_mut(len).take(GRID).enumerate() {
                let taps = [0, 1, 2, 3].map(|row| tap(row, column));
                interpolate(sum, vertical, taps);
            }
            // Then along x for every sample, which sums every octave.
            let sum = &mut self.state[..len];
            for (index, value) in self.values.iter_mut().enumerate() {
                let sample = index as i64 - 1;
                let coarse = sample.div_euclid(GRID_STEP);
                let weights = QUARTER_WEIGHTS[sample.rem_euclid(GRID_STEP) as usize];
                let first = (coarse - 1 - GRID_FIRST) as usize;
                let taps = [0, 1, 2, 3].map(|tap| {
                    let at = (first + tap) * len;
                    &self.column[at..at + len]
                });
                interpolate(sum, weights, taps);
                *value = tables.finish(self.xs[index], z, sum, octaves, fine.clone());
            }
            self.layers.apply_noise(layer, &mut self.row, &self.values);
        }
        let stride = TILE_STRIDE as usize;
        let start = (row + 1) as usize * stride;
        self.heights[start..start + stride].copy_from_slice(&self.row);
    }
}

impl Layers {
    /// Work units of one [`Self::sample_height`], against one per sample of
    /// a layer: sixteen coarse grid points and a sample per noise layer.
    pub(crate) fn height_cost(&self) -> u64 {
        (0..self.layers().len())
            .map(|layer| {
                if self.noise(layer).is_some() {
                    16 * COARSE_COST + NOISE_SAMPLE_COST
                } else {
                    1
                }
            })
            .sum()
    }

    /// The height of global sample `sample` (x and z, counted from the
    /// terrain's first corner) at `level`, exactly as a generated tile holds
    /// it: the same coarse grid points, interpolation and layer order.
    pub fn sample_height(&self, pyramid: &Pyramid, level: u32, sample: [i64; 2]) -> f32 {
        let spacing = pyramid.spacing(level);
        let [dx, dz] = self.offset_samples(spacing);
        let sample = [sample[0] + dx, sample[1] + dz];
        let x = sample[0] as f32 * spacing;
        let z = sample[1] as f32 * spacing;
        let [cx, cz] = sample.map(|index| index.div_euclid(GRID_STEP));
        let horizontal = QUARTER_WEIGHTS[sample[0].rem_euclid(GRID_STEP) as usize];
        let vertical = QUARTER_WEIGHTS[sample[1].rem_euclid(GRID_STEP) as usize];
        let point = Vec2::new(x, z);
        let mut height = [0.0_f32];
        for layer in 0..self.layers().len() {
            let Some(tables) = self.noise(layer) else {
                self.apply_local(layer, &mut height, &[x], z, (point, point), spacing, None);
                continue;
            };
            let (octaves, fine, gullies) = tables.bands(spacing);
            // The coarse state along z first for each column, then along x,
            // as tiles interpolate it.
            let len = state_len(octaves);
            // The sixteen grid points' states, column by column.
            let mut points = vec![0.0_f32; 16 * len];
            for (index, point) in points.chunks_exact_mut(len).enumerate() {
                let column = (cx - 1 + (index / 4) as i64) * GRID_STEP;
                let row = (cz - 1 + (index % 4) as i64) * GRID_STEP;
                tables.coarse(
                    column as f32 * spacing,
                    row as f32 * spacing,
                    octaves,
                    gullies,
                    point,
                );
            }
            let mut columns = vec![0.0_f32; 4 * len];
            for (column, value) in columns.chunks_exact_mut(len).enumerate() {
                let taps = [0, 1, 2, 3].map(|row| {
                    let at = (column * 4 + row) * len;
                    &points[at..at + len]
                });
                interpolate(value, vertical, taps);
            }
            let mut sum = vec![0.0_f32; len];
            let taps = [0, 1, 2, 3].map(|column| &columns[column * len..(column + 1) * len]);
            interpolate(&mut sum, horizontal, taps);
            self.apply_noise(
                layer,
                &mut height,
                &[tables.finish(x, z, &sum, octaves, fine)],
            );
        }
        height[0]
    }

    /// The height at terrain-local `point` on level 0's triangles, as
    /// generated tiles, renderers and collision hold it: four samples of
    /// [`Self::sample_height`].
    pub fn height_at(&self, pyramid: &Pyramid, point: Vec2) -> f32 {
        let spacing = pyramid.spacing(0);
        let u = point.x / spacing;
        let v = point.y / spacing;
        let (i, j) = (u.floor(), v.floor());
        let (fu, fv) = (u - i, v - j);
        let (i, j) = (i as i64, j as i64);
        let h = |di: i64, dj: i64| self.sample_height(pyramid, 0, [i + di, j + dj]);
        if fu + fv <= 1.0 {
            let h00 = h(0, 0);
            h00 + fu * (h(1, 0) - h00) + fv * (h(0, 1) - h00)
        } else {
            let h11 = h(1, 1);
            h11 + (1.0 - fu) * (h(0, 1) - h11) + (1.0 - fv) * (h(1, 0) - h11)
        }
    }
}

/// A tile's key, spacing, samples, holes, control values and placed
/// objects; reading rebuilds its bounds as [`Tile::new`] does.
impl crate::codec::Codec for Tile {
    fn write(&self, out: &mut crate::codec::Writer) {
        out.put(&self.key)
            .put(&self.spacing)
            .put(&self.heights)
            .put(&self.holes)
            .put(&self.control)
            .put(&self.placed);
    }

    fn read(input: &mut crate::codec::Reader<'_>) -> Result<Self, crate::codec::CodecError> {
        let invalid = crate::codec::CodecError("invalid tile");
        let key = input.get()?;
        let spacing = input.get()?;
        let heights = input.get()?;
        let holes = input.get()?;
        let control: Option<Box<[u32]>> = input.get()?;
        let placed: Option<Arc<[crate::Placed]>> = input.get()?;
        let mut tile = Self::new(key, spacing, heights, holes).ok_or(invalid)?;
        if let Some(control) = control {
            tile = tile.with_control(control).ok_or(invalid)?;
        }
        tile.placed = placed;
        Ok(tile)
    }
}
