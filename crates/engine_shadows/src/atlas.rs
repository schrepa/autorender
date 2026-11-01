use crate::constants::{LEVEL_WORDS, SHADOW_APRON};
use crate::shadow_fit::{LEVEL_TEXELS, region_tiles};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AtlasLayout {
    pub(crate) slot: u32,
    pub(crate) per_row: u32,
    pub(crate) rows: u32,
    pub(crate) still: u32,
    pub(crate) moving: u32,
}

impl AtlasLayout {
    pub(crate) fn new(tile_texels: u32, (still, moving): (u32, u32), limit: u32) -> Self {
        let slot = tile_texels + 2 * SHADOW_APRON;
        let most = (limit / slot).max(1);
        let wanted = (still + moving).max(1);
        let per_row = (f64::from(wanted).sqrt().ceil() as u32).clamp(1, most);
        let rows = wanted.div_ceil(per_row).min(most);
        let held = per_row * rows;
        let still = still.min(held);
        Self {
            slot,
            per_row,
            rows,
            still,
            moving: moving.min(held - still),
        }
    }

    pub(crate) fn size(&self) -> (u32, u32) {
        (self.per_row * self.slot, self.rows * self.slot)
    }

    pub(crate) fn texels(&self, slot: u32) -> [f32; 4] {
        let x = (slot % self.per_row * self.slot) as f32;
        let y = (slot / self.per_row * self.slot) as f32;
        [x, y, x + self.slot as f32, y + self.slot as f32]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MapsKey {
    pub(crate) atlas: AtlasLayout,
    pub(crate) tile_texels: u32,
    pub(crate) levels: u32,
    pub(crate) side: u32,
}

impl MapsKey {
    pub(crate) fn new(settings: &crate::ShadowSettings, limit: u32) -> Self {
        let side = (LEVEL_TEXELS / settings.tile_texels).max(1);
        Self {
            atlas: AtlasLayout::new(
                settings.tile_texels,
                (settings.atlas_slots, settings.moving_tiles),
                limit,
            ),
            tile_texels: settings.tile_texels,
            levels: settings.levels.min(limit / (side + 1)).max(1),
            side,
        }
    }

    pub(crate) fn table_size(&self) -> (u32, u32) {
        (self.side.max(LEVEL_WORDS), (self.side + 1) * self.levels)
    }

    pub(crate) fn region_side(&self) -> u32 {
        region_tiles(self.tile_texels) * self.tile_texels + 2 * SHADOW_APRON
    }
}
