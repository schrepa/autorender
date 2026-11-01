use bevy_ecs::prelude::Resource;

pub const CORE_ROW: &str = "core";
pub const UNATTRIBUTED_ROW: &str = "unattributed";
pub const GPU_IDLE_ROW: &str = "gpu_idle";
pub const WARMUP_FRAMES: u64 = 120;

#[derive(Resource, Clone, Debug, Default, PartialEq, serde::Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct BrickCosts {
    pub frame: u64,
    pub measuring_ms: f64,
    pub rows: Vec<BrickCost>,
}

#[derive(Clone, Debug, Default, PartialEq, serde::Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct BrickCost {
    pub brick: &'static str,
    pub cpu_ms: f64,
    pub gpu_ms: Option<f64>,
    pub heap_live_bytes: u64,
    pub heap_peak_bytes: u64,
    pub allocations: u64,
    pub gpu_bytes: u64,
    pub created_after_warmup: u64,
    pub duplicate_upload_bytes: u64,
}

#[derive(Clone, Debug, Default, PartialEq, serde::Serialize)]
pub struct BrickTable {
    pub frames: usize,
    pub cpu_ms: f64,
    pub gpu_ms: Option<f64>,
    pub measuring_ms: f64,
    pub rows: Vec<BrickRow>,
}

#[derive(Clone, Debug, Default, PartialEq, serde::Serialize)]
pub struct BrickRow {
    pub brick: &'static str,
    pub cpu_ms: f64,
    pub gpu_ms: Option<f64>,
    pub heap_live_bytes: u64,
    pub heap_peak_bytes: u64,
    pub allocations_per_frame: f64,
    pub gpu_bytes: u64,
    pub created_after_warmup: u64,
    pub duplicate_upload_bytes_per_frame: f64,
}

impl BrickCosts {
    pub fn cpu_ms(&self) -> f64 {
        self.rows.iter().map(|row| row.cpu_ms).sum()
    }

    pub fn gpu_ms(&self) -> Option<f64> {
        self.rows.iter().map(|row| row.gpu_ms).sum()
    }

    pub fn row(&self, brick: &str) -> Option<&BrickCost> {
        self.rows.iter().find(|row| row.brick == brick)
    }

    pub fn table(frames: &[Self]) -> BrickTable {
        let count = frames.len();
        let mut names: Vec<&'static str> = Vec::new();
        for row in frames.iter().flat_map(|frame| &frame.rows) {
            if !names.contains(&row.brick) {
                names.push(row.brick);
            }
        }
        let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len().max(1) as f64;
        let mut rows: Vec<BrickRow> = names
            .into_iter()
            .map(|brick| {
                let costs: Vec<&BrickCost> =
                    frames.iter().filter_map(|each| each.row(brick)).collect();
                let cpu: Vec<f64> = costs.iter().map(|cost| cost.cpu_ms).collect();
                let gpu: Vec<f64> = costs.iter().filter_map(|cost| cost.gpu_ms).collect();
                let allocations: Vec<f64> =
                    costs.iter().map(|cost| cost.allocations as f64).collect();
                let duplicates: Vec<f64> = costs
                    .iter()
                    .map(|cost| cost.duplicate_upload_bytes as f64)
                    .collect();
                let last = costs.last();
                BrickRow {
                    brick,
                    cpu_ms: mean(&cpu),
                    gpu_ms: (!gpu.is_empty()).then(|| mean(&gpu)),
                    heap_live_bytes: last.map_or(0, |cost| cost.heap_live_bytes),
                    heap_peak_bytes: costs
                        .iter()
                        .map(|cost| cost.heap_peak_bytes)
                        .max()
                        .unwrap_or(0),
                    allocations_per_frame: mean(&allocations),
                    gpu_bytes: last.map_or(0, |cost| cost.gpu_bytes),
                    created_after_warmup: last.map_or(0, |cost| cost.created_after_warmup),
                    duplicate_upload_bytes_per_frame: mean(&duplicates),
                }
            })
            .collect();
        rows.sort_by(|a, b| {
            let cost = |row: &BrickRow| row.cpu_ms + row.gpu_ms.unwrap_or(0.0);
            cost(b).total_cmp(&cost(a))
        });
        let totals: Vec<f64> = frames.iter().map(Self::cpu_ms).collect();
        let gpu: Vec<f64> = frames.iter().filter_map(Self::gpu_ms).collect();
        let measuring: Vec<f64> = frames.iter().map(|frame| frame.measuring_ms).collect();
        BrickTable {
            frames: count,
            cpu_ms: mean(&totals),
            gpu_ms: (!gpu.is_empty()).then(|| mean(&gpu)),
            measuring_ms: mean(&measuring),
            rows,
        }
    }
}

#[cfg(any(feature = "dev", feature = "profiling"))]
pub fn under<T>(brick: &'static str, run: impl FnOnce() -> T) -> T {
    let _span = engine_profile::span("brick", brick);
    let _scope = engine_profile::enter(engine_profile::brick_index(brick));
    run()
}

#[cfg(not(any(feature = "dev", feature = "profiling")))]
#[inline(always)]
pub fn under<T>(_brick: &'static str, run: impl FnOnce() -> T) -> T {
    run()
}

#[cfg(any(feature = "dev", feature = "profiling"))]
pub(super) use measured::{measure_systems, system_keys};

#[cfg(any(feature = "dev", feature = "profiling"))]
mod measured {
    use std::any::TypeId;

    use bevy_app::App;
    use bevy_ecs::schedule::{ApplyDeferred, InternedScheduleLabel, SystemKey, SystemWithAccess};
    use bevy_ecs::system::{
        Adapt, AdapterSystem, IntoSystem, RunSystemError, System, SystemIn, SystemInput,
    };

    struct Measured {
        brick: &'static str,
        index: u16,
    }

    impl Adapt<SystemWithAccess> for Measured {
        type In = ();
        type Out = ();

        fn adapt(
            &mut self,
            input: <Self::In as SystemInput>::Inner<'_>,
            run_system: impl FnOnce(
                SystemIn<'_, SystemWithAccess>,
            )
                -> Result<<SystemWithAccess as System>::Out, RunSystemError>,
        ) -> Result<Self::Out, RunSystemError> {
            let _span = engine_profile::span("brick", self.brick);
            let _scope = engine_profile::enter(self.index);
            run_system(input)
        }
    }

    fn idle() {}

    pub(in crate::bricks) fn system_keys(
        app: &App,
        schedule: InternedScheduleLabel,
    ) -> Vec<SystemKey> {
        app.get_schedule(schedule)
            .map(|schedule| {
                schedule
                    .graph()
                    .systems
                    .iter()
                    .map(|(key, _, _)| key)
                    .collect()
            })
            .unwrap_or_default()
    }

    pub(in crate::bricks) fn measure_systems(
        app: &mut App,
        schedule: InternedScheduleLabel,
        before: &[SystemKey],
        brick: &'static str,
    ) {
        let index = engine_profile::brick_index(brick);
        let Some(schedule) = app.get_schedule_mut(schedule) else {
            return;
        };
        let systems = &mut schedule.graph_mut().systems;
        let added: Vec<SystemKey> = systems
            .iter()
            .filter(|(key, system, _)| {
                !before.contains(key) && system.system_type() != TypeId::of::<ApplyDeferred>()
            })
            .map(|(key, _, _)| key)
            .collect();
        for key in added {
            let Some(slot) = systems.get_mut(key) else {
                continue;
            };
            let inner = std::mem::replace(
                slot,
                SystemWithAccess::new(Box::new(IntoSystem::<(), (), _>::into_system(idle))),
            );
            let name = inner.name();
            *slot = SystemWithAccess::new(Box::new(AdapterSystem::new(
                Measured { brick, index },
                inner,
                name,
            )));
        }
    }
}
