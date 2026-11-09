use std::sync::Arc;

use bevy_ecs::prelude::{Component, Entity, Resource};
use engine_codec::{Error, ErrorCode};
use engine_place::{JobRef, Slot};
use glam::{Vec2, Vec3};
use serde::{Deserialize, Serialize};

pub const UNKNOWN_TERRAIN: ErrorCode = ErrorCode::brick("unknown_terrain");

pub const TILE_QUADS: u32 = 256;
pub const TILE_SAMPLES: u32 = TILE_QUADS + 1;
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroundPoint {
    pub height: f32,
    pub normal: Vec3,
    pub level: u32,
    pub hole: bool,
    pub material: Option<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GroundObstacle {
    Circle {
        center: Vec2,
        radius: f32,
    },
    Box {
        center: Vec2,
        half_extents: Vec2,
        unit_axis: Vec2,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GroundRead<T> {
    Ready(T),
    NotLoaded,
    Off,
}

impl<T> GroundRead<T> {
    pub fn ready(self) -> Option<T> {
        match self {
            Self::Ready(value) => Some(value),
            Self::NotLoaded | Self::Off => None,
        }
    }
}

impl GroundRead<GroundPoint> {
    pub fn solid(self) -> Option<GroundPoint> {
        self.ready().filter(|point| !point.hole)
    }
}

#[derive(Clone, PartialEq)]
pub struct GroundSource {
    pub ground: Entity,
    pub origin: Vec3,
    pub pyramid: Pyramid,
    pub identity: Arc<[u8]>,
}

impl std::fmt::Debug for GroundSource {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GroundSource")
            .field("ground", &self.ground)
            .field("origin", &self.origin)
            .field("pyramid", &self.pyramid)
            .field("identity", &self.identity.len())
            .finish()
    }
}

pub trait GroundQuery: Send + Sync + 'static {
    fn point(&self, at: Vec2) -> GroundRead<GroundPoint>;

    fn obstacles(&self, ground: Entity) -> Vec<GroundObstacle>;

    fn source(&self, from: Vec2, to: Vec2) -> GroundRead<GroundSource>;

    fn grid(&self) -> Option<JobRef>;

    fn revision(&self) -> u64;

    fn edits(&self) -> u64 {
        self.revision()
    }
}

struct NoGround;

impl GroundQuery for NoGround {
    fn point(&self, _: Vec2) -> GroundRead<GroundPoint> {
        GroundRead::Off
    }

    fn obstacles(&self, _: Entity) -> Vec<GroundObstacle> {
        Vec::new()
    }

    fn source(&self, _: Vec2, _: Vec2) -> GroundRead<GroundSource> {
        GroundRead::Off
    }

    fn grid(&self) -> Option<JobRef> {
        None
    }

    fn revision(&self) -> u64 {
        0
    }
}

#[derive(Resource, Clone)]
pub struct Ground(pub Arc<dyn GroundQuery>);

impl Ground {
    pub fn point(&self, at: Vec2) -> GroundRead<GroundPoint> {
        self.0.point(at)
    }

    pub fn revision(&self) -> u64 {
        self.0.revision()
    }

    pub fn edits(&self) -> u64 {
        self.0.edits()
    }

    pub fn stand(&self, at: Vec2, walked: Option<f32>, deck_above: f32) -> Option<Stand> {
        let point = self.point(at).solid();
        match (point, walked) {
            (Some(point), Some(walked)) if walked > point.height + deck_above => Some(Stand {
                height: walked,
                level: None,
            }),
            (Some(point), _) => Some(Stand {
                height: point.height,
                level: Some(point.level),
            }),
            (None, Some(walked)) => Some(Stand {
                height: walked,
                level: None,
            }),
            (None, None) => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stand {
    pub height: f32,
    pub level: Option<u32>,
}

impl Default for Ground {
    fn default() -> Self {
        Self(Arc::new(NoGround))
    }
}

impl Slot for Ground {
    const NAME: &'static str = "ground";
}

#[derive(Clone, Debug, PartialEq)]
pub struct GroundGrid {
    pub key: TileKey,
    pub spacing: f32,
    pub heights: Box<[f32]>,
    pub holes: Option<Box<[u64]>>,
    pub water: Option<Box<[f32]>>,
}

engine_codec::record!(GroundGrid {
    key,
    spacing,
    heights,
    holes,
    water,
});

#[derive(
    Component, Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize, schemars::JsonSchema,
)]
#[require(GroundedAt)]
#[serde(default, deny_unknown_fields)]
pub struct Grounded {
    pub offset: f32,
}

#[derive(Component, Clone, Copy, Debug, Default, PartialEq)]
pub enum GroundedAt {
    #[default]
    Unsettled,
    Stood {
        position: Vec2,
        offset: f32,
        level: Option<u32>,
        revision: u64,
        edits: u64,
    },
    Unsupported {
        position: Vec2,
        offset: f32,
        revision: u64,
    },
}

impl GroundedAt {
    pub fn level(self) -> Option<u32> {
        match self {
            Self::Stood { level, .. } => level,
            Self::Unsettled | Self::Unsupported { .. } => None,
        }
    }

    pub fn settle(
        self,
        ground: &Ground,
        (position, offset): (Vec2, f32),
        walked: Option<f32>,
        deck_above: f32,
    ) -> Option<(Self, Option<f32>)> {
        let (revision, edits) = (ground.revision(), ground.edits());
        let holds = match self {
            Self::Unsettled => false,
            Self::Stood {
                position: at,
                offset: held,
                level,
                revision: seen,
                edits: edited,
            } => {
                at == position
                    && held == offset
                    && edited == edits
                    && (level == Some(0) || seen == revision)
            }
            Self::Unsupported {
                position: at,
                offset: held,
                revision: seen,
            } => at == position && held == offset && seen == revision,
        };
        if holds {
            return None;
        }
        Some(match ground.stand(position, walked, deck_above) {
            Some(stand) => (
                Self::Stood {
                    position,
                    offset,
                    level: stand.level,
                    revision,
                    edits,
                },
                Some(stand.height + offset),
            ),
            None => (
                Self::Unsupported {
                    position,
                    offset,
                    revision,
                },
                None,
            ),
        })
    }
}
