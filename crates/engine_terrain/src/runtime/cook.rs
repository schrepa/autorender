use super::*;
use bevy_ecs::world::World;
use engine_core::{CookAsks, Detail, ViewDetail};

const COOK_VIEW_HEIGHT: u32 = 2160;

pub(crate) fn cook(world: &World, asks: &mut CookAsks<'_>) {
    let (Some(terrains), Some(settings), Some(detail)) = (
        world.get_resource::<Terrains>(),
        world.get_resource::<TerrainSettings>(),
        world.get_resource::<Detail>(),
    ) else {
        return;
    };
    for state in &terrains.entries {
        if state.terrain.validate().is_err() || (state.around.is_some() && !settings.surroundings) {
            continue;
        }
        if state.baking() {
            asks.left(1);
            continue;
        }
        let Some(recipe) = state.recipe.as_ref().map(|recipe| recipe.key) else {
            continue;
        };
        for key in cooked_tiles(state, settings, detail) {
            asks.ask(&TILE_BUILD, state.tile_job(recipe.hash(), key).identity());
        }
    }
}

fn cooked_tiles(state: &TerrainState, settings: &TerrainSettings, detail: &Detail) -> Vec<TileKey> {
    let pyramid = state.terrain.pyramid;
    let view = ViewDetail::perspective(
        Vec3::ZERO,
        Camera::default().clip(1.0),
        COOK_VIEW_HEIGHT,
        detail.chosen_error().unwrap_or(0.0),
    );
    let ranges = LodRanges::new(&pyramid, view.scaled(settings.pixel_error));
    let mut keys = Vec::new();
    for level in (0..pyramid.levels()).rev() {
        let [columns, rows] = pyramid.tiles(level);
        let size = pyramid.tile_size(level);
        for z in 0..rows {
            for x in 0..columns {
                let first = Vec2::new(x as f32, z as f32) * size;
                let last = first + Vec2::splat(size);
                let wanted = state.around.as_ref().is_none_or(|around| {
                    let (low, high) = around.cutout;
                    let inside = first.x >= low.x
                        && first.y >= low.y
                        && last.x <= high.x
                        && last.y <= high.y;
                    let gap = (low - last).max(first - high).max(Vec2::ZERO).length();
                    !inside && gap <= LOAD_REACH * ranges.range(level)
                });
                if wanted {
                    keys.push(TileKey::new(level as u8, x as u16, z as u16));
                }
            }
        }
    }
    keys
}
