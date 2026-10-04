//! Tile pyramids: levels, tile keys and where tiles lie.

use glam::Vec2;

use crate::TILE_QUADS;

/// A tile of a pyramid: its level (0 is the finest) and its column and row
/// there, along x and z.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TileKey {
    pub level: u8,
    pub x: u16,
    pub z: u16,
}

impl TileKey {
    pub const fn new(level: u8, x: u16, z: u16) -> Self {
        Self { level, x, z }
    }
}

/// Why a pyramid's dimensions are refused.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PyramidError {
    /// Tiles along an axis must be 1 to [`Pyramid::MAX_TILES`].
    Tiles(u32),
    /// The spacing must be finite and in [`Pyramid::MIN_SPACING`] to
    /// [`Pyramid::MAX_SPACING`] meters.
    Spacing(f32),
}

impl std::fmt::Display for PyramidError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Tiles(tiles) => write!(
                formatter,
                "tiles must be 1 to {} along each axis (received {tiles})",
                Pyramid::MAX_TILES
            ),
            Self::Spacing(spacing) => write!(
                formatter,
                "spacing must be {} to {} meters (received {spacing})",
                Pyramid::MIN_SPACING,
                Pyramid::MAX_SPACING
            ),
        }
    }
}

impl std::error::Error for PyramidError {}

/// The levels of a terrain: level 0 holds `tiles` tiles of [`TILE_QUADS`]
/// quads `spacing` meters across; each coarser level halves the tiles along
/// each axis, rounding up, and doubles the spacing, up to the level of one
/// tile. Positions are terrain-local meters from the terrain's first corner.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pyramid {
    tiles: [u32; 2],
    spacing: f32,
    levels: u32,
}

impl Pyramid {
    /// The largest tile count along an axis: tile keys hold 16 bits.
    pub const MAX_TILES: u32 = 4096;
    pub const MIN_SPACING: f32 = 1.0 / 64.0;
    pub const MAX_SPACING: f32 = 1024.0;

    pub fn new(tiles: [u32; 2], spacing: f32) -> Result<Self, PyramidError> {
        for count in tiles {
            if !(1..=Self::MAX_TILES).contains(&count) {
                return Err(PyramidError::Tiles(count));
            }
        }
        if !(spacing.is_finite() && (Self::MIN_SPACING..=Self::MAX_SPACING).contains(&spacing)) {
            return Err(PyramidError::Spacing(spacing));
        }
        let widest = tiles[0].max(tiles[1]);
        let levels = 1 + (u32::BITS - (widest - 1).leading_zeros());
        Ok(Self {
            tiles,
            spacing,
            levels,
        })
    }

    /// Level-0 tiles along x and z.
    pub fn base_tiles(&self) -> [u32; 2] {
        self.tiles
    }

    pub fn levels(&self) -> u32 {
        self.levels
    }

    /// Meters between samples at `level`: the level-0 spacing doubled per
    /// level, exactly.
    pub fn spacing(&self, level: u32) -> f32 {
        self.spacing * (1u32 << level) as f32
    }

    /// Tiles along x and z at `level`.
    pub fn tiles(&self, level: u32) -> [u32; 2] {
        self.tiles.map(|count| count.div_ceil(1 << level))
    }

    /// Meters a tile of `level` spans along each axis.
    pub fn tile_size(&self, level: u32) -> f32 {
        TILE_QUADS as f32 * self.spacing(level)
    }

    /// The terrain's extent along x and z: its level-0 tiles.
    pub fn extent(&self) -> Vec2 {
        Vec2::new(self.tiles[0] as f32, self.tiles[1] as f32) * self.tile_size(0)
    }

    /// Whether `key` is a tile of this pyramid.
    pub fn contains(&self, key: TileKey) -> bool {
        let level = u32::from(key.level);
        if level >= self.levels {
            return false;
        }
        let [x, z] = self.tiles(level);
        u32::from(key.x) < x && u32::from(key.z) < z
    }

    /// The tile of the next coarser level covering `key`, if any.
    pub fn parent(&self, key: TileKey) -> Option<TileKey> {
        (u32::from(key.level) + 1 < self.levels)
            .then(|| TileKey::new(key.level + 1, key.x / 2, key.z / 2))
    }

    /// The first sample of `key` along x and z, counted at its level from
    /// the terrain's first corner.
    pub fn first_sample(&self, key: TileKey) -> [i64; 2] {
        [
            i64::from(key.x) * i64::from(TILE_QUADS),
            i64::from(key.z) * i64::from(TILE_QUADS),
        ]
    }

    /// The terrain-local position of `key`'s first corner.
    pub fn tile_origin(&self, key: TileKey) -> Vec2 {
        let size = self.tile_size(u32::from(key.level));
        Vec2::new(f32::from(key.x) * size, f32::from(key.z) * size)
    }

    /// The tile of `level` holding terrain-local `point`, if inside the
    /// terrain. Points on a shared edge belong to the later tile, except at
    /// the terrain's far edges.
    pub fn tile_at(&self, level: u32, point: Vec2) -> Option<TileKey> {
        let extent = self.extent();
        if level >= self.levels
            || !point.is_finite()
            || point.x < 0.0
            || point.y < 0.0
            || point.x > extent.x
            || point.y > extent.y
        {
            return None;
        }
        let size = self.tile_size(level);
        let [x, z] = self.tiles(level);
        let column = ((point.x / size) as u32).min(x - 1);
        let row = ((point.y / size) as u32).min(z - 1);
        Some(TileKey::new(level as u8, column as u16, row as u16))
    }

    /// Every tile of `level`, row by row.
    pub fn level_tiles(&self, level: u32) -> impl Iterator<Item = TileKey> {
        let [x, z] = self.tiles(level);
        (0..z).flat_map(move |row| {
            (0..x).map(move |column| TileKey::new(level as u8, column as u16, row as u16))
        })
    }

    /// Tiles at all levels together.
    pub fn tile_count(&self) -> u32 {
        (0..self.levels)
            .map(|level| {
                let [x, z] = self.tiles(level);
                x * z
            })
            .sum()
    }
}
