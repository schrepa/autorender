use engine_codec::{Error, ErrorCode};
use glam::Vec2;

pub const TILE_QUADS: u32 = 256;
pub const TILE_SAMPLES: u32 = TILE_QUADS + 1;
pub const TILE_STRIDE: u32 = TILE_SAMPLES + 2;
pub const MAX_LEVEL: u32 = 15;

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

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pyramid {
    tiles: [u32; 2],
    spacing: f32,
    levels: u32,
}

fn grid_error(path: &str) -> Error {
    Error::new(ErrorCode::InvalidValue, path)
}

impl Pyramid {
    pub const MAX_TILES: u32 = 4096;
    pub const MIN_SPACING: f32 = 1.0 / 64.0;
    pub const MAX_SPACING: f32 = 1024.0;

    pub fn new(tiles: [u32; 2], spacing: f32) -> Result<Self, Error> {
        for count in tiles {
            if !(1..=Self::MAX_TILES).contains(&count) {
                return Err(grid_error("$.tiles")
                    .range(1.0, f64::from(Self::MAX_TILES))
                    .got(count));
            }
        }
        if !(spacing.is_finite() && (Self::MIN_SPACING..=Self::MAX_SPACING).contains(&spacing)) {
            return Err(grid_error("$.spacing")
                .range(f64::from(Self::MIN_SPACING), f64::from(Self::MAX_SPACING))
                .got(spacing));
        }
        let widest = tiles[0].max(tiles[1]);
        let levels = 1 + (u32::BITS - (widest - 1).leading_zeros());
        Ok(Self {
            tiles,
            spacing,
            levels,
        })
    }

    pub fn base_tiles(&self) -> [u32; 2] {
        self.tiles
    }

    pub fn levels(&self) -> u32 {
        self.levels
    }

    pub fn spacing(&self, level: u32) -> f32 {
        self.spacing * (1u32 << level) as f32
    }

    pub fn tiles(&self, level: u32) -> [u32; 2] {
        self.tiles.map(|count| count.div_ceil(1 << level))
    }

    pub fn tile_size(&self, level: u32) -> f32 {
        TILE_QUADS as f32 * self.spacing(level)
    }

    pub fn extent(&self) -> Vec2 {
        Vec2::new(self.tiles[0] as f32, self.tiles[1] as f32) * self.tile_size(0)
    }

    pub fn contains(&self, key: TileKey) -> bool {
        let level = u32::from(key.level);
        if level >= self.levels {
            return false;
        }
        let [x, z] = self.tiles(level);
        u32::from(key.x) < x && u32::from(key.z) < z
    }

    pub fn parent(&self, key: TileKey) -> Option<TileKey> {
        (u32::from(key.level) + 1 < self.levels)
            .then(|| TileKey::new(key.level + 1, key.x / 2, key.z / 2))
    }

    pub fn first_sample(&self, key: TileKey) -> [i64; 2] {
        [
            i64::from(key.x) * i64::from(TILE_QUADS),
            i64::from(key.z) * i64::from(TILE_QUADS),
        ]
    }

    pub fn tile_origin(&self, key: TileKey) -> Vec2 {
        let size = self.tile_size(u32::from(key.level));
        Vec2::new(f32::from(key.x) * size, f32::from(key.z) * size)
    }

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

    pub fn level_tiles(&self, level: u32) -> impl Iterator<Item = TileKey> {
        let [x, z] = self.tiles(level);
        (0..z).flat_map(move |row| {
            (0..x).map(move |column| TileKey::new(level as u8, column as u16, row as u16))
        })
    }

    pub fn tile_count(&self) -> u32 {
        (0..self.levels)
            .map(|level| {
                let [x, z] = self.tiles(level);
                x * z
            })
            .sum()
    }
}

engine_codec::record!(TileKey { level, x, z });

impl engine_codec::Codec for Pyramid {
    fn write(&self, out: &mut engine_codec::Writer) {
        out.put(&self.base_tiles()).put(&self.spacing(0));
    }

    fn read(input: &mut engine_codec::Reader<'_>) -> Result<Self, Error> {
        let tiles = input.get()?;
        let spacing = input.get()?;
        Self::new(tiles, spacing)
    }
}
