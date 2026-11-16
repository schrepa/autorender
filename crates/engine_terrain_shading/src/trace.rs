use bevy_ecs::prelude::*;
use engine_core::{DeviceLimits, Error, Joined};
use engine_open_world_kit::ground::{GroundTiles, PlacedTiles, TerrainTiles};
use engine_places_rendering::reflections::TraceReaders;
use engine_render::{Bind, Buffer, GpuPart, Install, Lifetime, Prepare};

use crate::shaders::terrain_shading::faces::trace_source;
use crate::state::{PartStats, TRACE_STATS};
use crate::world_trace::{TraceFrame, WorldTrace};

pub(crate) struct TraceGpu {
    terrains: Buffer<Self>,
    placed: Buffer<Self>,
    world: WorldTrace,
    placed_changed: bool,
}

impl GpuPart for TraceGpu {
    type Reads = (
        Res<'static, DeviceLimits>,
        Res<'static, TerrainTiles>,
        Res<'static, GroundTiles>,
        Res<'static, PlacedTiles>,
        Joined<'static, TraceReaders>,
    );
    const NAME: &'static str = "terrain_trace";
    const LAZY: bool = true;
    const LABELS: &'static [&'static str] = &["engine/terrain-trace"];

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        let terrains = cx.buffer("world_terrains", Lifetime::Kept);
        let placed = cx.buffer("world_placed", Lifetime::Kept);
        cx.frame(trace_source::WORLD_TERRAINS, Bind::Buffer(terrains))?;
        cx.frame(trace_source::WORLD_PLACED, Bind::Buffer(placed))?;
        Ok(Self {
            terrains,
            placed,
            world: WorldTrace::default(),
            placed_changed: false,
        })
    }

    fn prepare(
        &mut self,
        (limits, tiles, ground, placed, readers): SystemParamItem<'_, '_, Self::Reads>,
        cx: &mut Prepare<'_, Self>,
    ) {
        self.placed_changed |= placed.is_changed();
        if readers.iter().next().is_none() {
            return;
        }
        let placed_changed = std::mem::take(&mut self.placed_changed);
        let pool = self.world.prepare(
            cx,
            (self.terrains, self.placed),
            &TraceFrame {
                tiles: &tiles,
                ground: &ground,
                placed: &placed,
                room: limits.storage_room / 16 * 16,
                placed_changed,
            },
        );
        let [turns, limit] = self.world.turns();
        cx.deliver::<PartStats>(
            TRACE_STATS,
            bytemuck::bytes_of(&[pool.asked, pool.kept, turns, limit]).to_vec(),
        );
    }
}
