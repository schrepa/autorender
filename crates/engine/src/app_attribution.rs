use bevy_ecs::change_detection::Mut;
use engine_core::ErrorCode;
use engine_registry::{BrickRecord, Registry};

use super::AppHost;
use crate::{Error, Result};

#[cfg(any(feature = "dev", feature = "profiling"))]
pub(super) use measured::{Attribution, core_scope};
#[cfg(not(any(feature = "dev", feature = "profiling")))]
pub(super) use unmeasured::core_scope;

impl AppHost {
    pub(super) fn claim_passes(&mut self) -> Result<()> {
        let world = self.engine.app.world_mut();
        if !world.contains_resource::<Registry>() {
            return Ok(());
        }
        let renderer = &mut self.renderer;
        let installed = world.resource_scope(|world, registry: Mut<Registry>| {
            let bricks: Vec<&BrickRecord> = registry.bricks().iter().collect();
            renderer.install(world, &bricks)
        });
        installed.map_err(|mut problems| match problems.len() {
            1 => problems.remove(0),
            _ => Error::new(ErrorCode::RenderFailed, "$.bricks").got(&problems),
        })
    }
}

#[cfg(not(any(feature = "dev", feature = "profiling")))]
mod unmeasured {
    use super::AppHost;

    pub(in crate::app_host) struct Core;

    #[inline(always)]
    pub(in crate::app_host) fn core_scope() -> Core {
        Core
    }

    impl AppHost {
        #[inline(always)]
        pub(in crate::app_host) fn begin_attribution(&mut self) {}

        #[inline(always)]
        pub(in crate::app_host) fn end_attribution(&mut self) {}
    }
}

#[cfg(any(feature = "dev", feature = "profiling"))]
mod measured {
    use engine_registry::{
        BrickCost, BrickCosts, CORE_ROW, GPU_IDLE_ROW, Registry, UNATTRIBUTED_ROW, WARMUP_FRAMES,
    };
    use engine_profile::MAX_BRICKS;

    use super::AppHost;
    use crate::app_host::now_ms;

    #[cfg(feature = "profiling")]
    const CAPTURED_FRAMES: usize = 4_096;

    pub(in crate::app_host) struct Attribution {
        started: Option<f64>,
        allocations: [u64; MAX_BRICKS],
        frames: u64,
        settled: bool,
        warm: bool,
        #[cfg(feature = "profiling")]
        pub(in crate::app_host) captured: Vec<BrickCosts>,
    }

    impl Default for Attribution {
        fn default() -> Self {
            Self {
                started: None,
                allocations: [0; MAX_BRICKS],
                frames: 0,
                settled: false,
                warm: false,
                #[cfg(feature = "profiling")]
                captured: Vec::new(),
            }
        }
    }

    pub(in crate::app_host) fn core_scope() -> engine_profile::BrickScope {
        engine_profile::enter(engine_profile::CORE)
    }

    impl AppHost {
        pub(in crate::app_host) fn begin_attribution(&mut self) {
            engine_profile::install_clock(now_ms);
            engine_profile::take_spent();
            engine_profile::take_scopes();
            self.attribution.started = Some(now_ms());
        }

        pub(in crate::app_host) fn end_attribution(&mut self) {
            let Some(started) = self.attribution.started.take() else {
                return;
            };
            let costs = self.brick_costs(now_ms() - started);
            self.warm_up();
            #[cfg(feature = "profiling")]
            if engine_profile::active() && self.attribution.captured.len() < CAPTURED_FRAMES {
                self.attribution.captured.push(costs.clone());
            }
            self.stats.bricks_cost = Some(costs.clone());
            self.engine.app.world_mut().insert_resource(costs);
        }

        fn warm_up(&mut self) {
            if self.attribution.warm {
                return;
            }
            self.attribution.frames += 1;
            self.attribution.settled |= !self.loading() && !self.renderer.pipelines_compiling();
            if self.attribution.settled && self.attribution.frames >= WARMUP_FRAMES {
                self.attribution.warm = true;
                self.renderer.end_gpu_warmup();
            }
        }

        fn brick_costs(&mut self, frame_ms: f64) -> BrickCosts {
            let begun = now_ms();
            let spent = engine_profile::take_spent();
            let scopes = engine_profile::take_scopes();
            let gpu = self.renderer.gpu_pass_times();
            let memory = self.renderer.take_gpu_memory();
            let world = self.engine.app.world();
            let frame = world.resource::<engine_core::FrameTime>().frame;
            let registry = world.get_resource::<Registry>();
            let renderer = &self.renderer;
            let owner = |pass: Option<&str>| {
                pass.and_then(|pass| renderer.pass_owner(pass))
                    .unwrap_or(CORE_ROW)
            };
            let mut names: Vec<&'static str> = registry
                .map(|registry| registry.names().collect())
                .unwrap_or_default();
            names.push(CORE_ROW);
            let mut rows: Vec<BrickCost> = Vec::with_capacity(names.len() + 2);
            for brick in names {
                let index = engine_profile::brick_index(brick);
                let slot = usize::from(index).min(MAX_BRICKS - 1);
                let heap = engine_profile::brick_heap(index);
                let owned = memory
                    .iter()
                    .filter(|memory| memory.brick.unwrap_or(CORE_ROW) == brick);
                rows.push(BrickCost {
                    brick,
                    cpu_ms: spent[slot],
                    gpu_ms: gpu.as_ref().map(|gpu| {
                        let preludes = gpu
                            .preludes
                            .iter()
                            .filter(|&&(_, noted, _)| noted == brick)
                            .map(|&(.., ms)| ms);
                        gpu.passes
                            .iter()
                            .filter(|&&(pass, _)| owner(Some(pass)) == brick)
                            .map(|&(_, ms)| ms)
                            .chain(preludes)
                            .sum()
                    }),
                    heap_live_bytes: heap.live_bytes,
                    heap_peak_bytes: heap.peak_bytes,
                    allocations: heap
                        .allocations
                        .saturating_sub(self.attribution.allocations[slot]),
                    gpu_bytes: owned.clone().map(|memory| memory.bytes).sum(),
                    created_after_warmup: owned
                        .clone()
                        .map(|memory| memory.created_after_warmup)
                        .sum(),
                    duplicate_upload_bytes: owned.map(|memory| memory.duplicate_upload_bytes).sum(),
                });
                self.attribution.allocations[slot] = heap.allocations;
            }
            let measured: f64 = rows.iter().map(|row| row.cpu_ms).sum();
            let drawn: Option<f64> = rows.iter().map(|row| row.gpu_ms).sum();
            let allocations: u64 = rows.iter().map(|row| row.allocations).sum();
            let gpu = gpu.zip(drawn);
            rows.push(BrickCost {
                brick: UNATTRIBUTED_ROW,
                cpu_ms: (frame_ms - measured).max(0.0),
                gpu_ms: gpu
                    .as_ref()
                    .map(|(gpu, drawn)| (gpu.frame_ms - gpu.idle_ms - drawn).max(0.0)),
                ..BrickCost::default()
            });
            if let Some((gpu, _)) = &gpu {
                rows.push(BrickCost {
                    brick: GPU_IDLE_ROW,
                    gpu_ms: Some(gpu.idle_ms),
                    ..BrickCost::default()
                });
            }
            let measuring_ms = scopes as f64 * engine_profile::scope_cost_ms()
                + allocations as f64 * engine_profile::allocation_cost_ms()
                + (now_ms() - begun);
            BrickCosts {
                frame,
                measuring_ms,
                rows,
            }
        }
    }
}
