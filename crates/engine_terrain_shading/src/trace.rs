use bevy_ecs::prelude::*;
use engine_core::{DeviceLimits, Error, Joined};
use engine_places_rendering::reflections::TraceReaders;
use engine_places_world::ground::{Cover, GroundTiles, PALETTE_ENTRIES, PlacedTiles, TerrainTiles};
use engine_render::{
    Bind, Buffer, ComputePipeline, Count, Encode, GpuPart, Install, Lifetime, Pass, Prepare,
};

use crate::gpu::CANOPY_GROUP;
use crate::shaders::terrain_shading::faces::trace_source;
use crate::shaders::terrain_shading::trace_canopy;
use crate::state::{PartStats, TRACE_STATS};
use crate::world_trace::{TraceFrame, WorldTrace};

pub(crate) struct TraceGpu {
    canopy: Pass<Self>,
    terrains: Buffer<Self>,
    placed: Buffer<Self>,
    canopies: ComputePipeline<Self>,
    world: WorldTrace,
    placed_changed: bool,
}

impl GpuPart for TraceGpu {
    type Reads = (
        Res<'static, DeviceLimits>,
        Res<'static, TerrainTiles>,
        Res<'static, GroundTiles>,
        Res<'static, PlacedTiles>,
        Res<'static, Cover>,
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
        let canopy = cx
            .pass("trace_canopy")
            .label("engine/terrain-trace")
            .span("terrain_trace")
            .id();
        let canopies = cx.compute_pipeline(
            canopy,
            trace_canopy::CS_CANOPY,
            &[
                (trace_canopy::WORLD_TERRAINS, Bind::Buffer(terrains)),
                (trace_canopy::WORLD_PLACED, Bind::Buffer(placed)),
            ],
        )?;
        Ok(Self {
            canopy,
            terrains,
            placed,
            canopies,
            world: WorldTrace::default(),
            placed_changed: false,
        })
    }

    fn prepare(
        &mut self,
        (limits, tiles, ground, placed, cover, readers): SystemParamItem<'_, '_, Self::Reads>,
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
                cover: &cover,
                room: limits.storage_room / 16 * 16,
                placed_changed,
            },
        );
        cx.deliver::<PartStats>(
            TRACE_STATS,
            bytemuck::bytes_of(&[pool.asked, pool.kept]).to_vec(),
        );
        if self.world.terrains() > 0 {
            cx.run(self.canopy);
        }
    }

    fn encode(&mut self, pass: Pass<Self>, cx: &mut Encode<'_, Self>) -> u32 {
        if pass == self.canopy {
            let groups = [PALETTE_ENTRIES.div_ceil(CANOPY_GROUP), self.world.terrains(), 1];
            cx.dispatch(&self.canopies, Count::Groups(groups));
        }
        0
    }
}
