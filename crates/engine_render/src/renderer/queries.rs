use bevy_ecs::world::World;

use super::*;


impl Renderer {
    pub fn adapter(&self) -> &crate::AdapterInfo {
        &self.core.adapter
    }

    pub fn limits(&self) -> wgpu::Limits {
        self.core.device.limits()
    }

    pub fn features(&self) -> wgpu::Features {
        self.core.device.features()
    }

    pub fn post(&self, _world: &World) -> bool {
        let Some(run) = self.run.as_ref() else {
            return false;
        };
        let composite = <crate::shared::Composite as crate::GpuPart>::NAME;
        run.machine
            .declared
            .parts
            .iter()
            .position(|part| part.name == composite)
            .is_some_and(|part| run.drawable(part as u32))
    }

    pub fn motion(&self, world: &World) -> bool {
        world
            .get_resource::<engine_registry::Registry>()
            .zip(world.get_resource::<engine_registry::BrickStates>())
            .is_some_and(|(registry, states)| {
                registry.iter().any(|(brick, record)| {
                    states.brick_on(brick)
                        && record
                            .render_entries::<crate::TargetDecl>()
                            .any(|target| target.name == engine_core::TargetName::MOTION)
                })
            })
    }

    pub fn texture_sampling(&self) -> engine_core::TextureSampling {
        use engine_core::{TextureFamily, TextureSampling};
        let features = self.core.device.features();
        if features.contains(wgpu::Features::TEXTURE_COMPRESSION_BC) {
            TextureSampling::Family(TextureFamily::Bc7)
        } else if features.contains(wgpu::Features::TEXTURE_COMPRESSION_ASTC) {
            TextureSampling::Family(TextureFamily::Astc)
        } else {
            TextureSampling::Decompressed
        }
    }

    pub fn view_size(&self) -> Option<(u32, u32)> {
        (!self.core.display.suspended()).then(|| self.core.display.size().pixels())
    }

    pub fn render_size(&self) -> Option<(u32, u32)> {
        (!self.core.display.suspended()).then(|| self.core.display.region.render.pixels())
    }

    pub fn times_frames(&self) -> bool {
        self.core.display.times_frames()
    }

    pub fn take_frame_time(&self) -> Option<(f64, (u32, u32))> {
        self.core.display.take_frame_time()
    }

    #[cfg(feature = "profiling")]
    pub fn profile_begin(&mut self, capture_id: u64) {
        self.core.profiler.begin(&self.core.device, capture_id);
    }

    #[cfg(feature = "profiling")]
    pub fn profile_poll(&mut self) {
        self.collect();
    }

    #[cfg(feature = "profiling")]
    pub fn profile_status(&self) -> serde_json::Value {
        self.core.profiler.status()
    }

    #[cfg(feature = "profiling")]
    pub fn profile_report(&mut self) -> serde_json::Value {
        self.collect();
        self.core.profiler.report()
    }

    pub fn is_texture_resident(&self, handle: engine_core::TextureHandle) -> bool {
        self.held_assets()
            .is_some_and(|assets| assets.materials.is_texture_resident(handle))
    }

    pub fn pipelines_compiling(&self) -> u32 {
        crate::platform::poll(&self.core.device);
        self.core.ledger.compiling()
    }

    pub fn compiled_per_pass(&self) -> Vec<(&'static str, u32)> {
        self.core.ledger.per_pass().to_vec()
    }

    pub fn compile_ahead(&mut self, world: &World) -> u32 {
        let Some(run) = self.run.as_mut() else {
            return 0;
        };
        let before = run.compiled.len();
        run.warm_reached(world);
        (run.compiled.len() - before) as u32
    }

    pub fn pipeline_list(&self) -> Vec<u8> {
        let fixed = self.core.ledger.list();
        let mut record = (fixed.len() as u32).to_le_bytes().to_vec();
        record.extend(fixed);
        if let Some(run) = self.run.as_ref() {
            record.extend(run.listing());
        }
        record
    }

    pub fn create_listed_pipelines(&mut self, record: &[u8]) -> Result<u32, Error> {
        let broken = |expected: Expected| failed("$.pipelines").expected(expected);
        let (length, rest) = record
            .split_first_chunk::<4>()
            .ok_or_else(|| broken(Expected::Present))?;
        let length = u32::from_le_bytes(*length) as usize;
        let fixed = rest
            .get(..length)
            .ok_or_else(|| broken(Expected::Present))?;
        let machine = &rest[length..];
        let mut created = self.core.ledger.create_listed(fixed)?;
        if let Some(run) = self.run.as_mut() {
            created += run.create_listed(machine)?;
        }
        Ok(created)
    }

    pub fn pipelines_listed(&self) -> (u32, u32) {
        let (up_front, unlisted) = self.core.ledger.listed();
        let (made, missed) = self
            .run
            .as_ref()
            .map_or((0, 0), |run| (run.up_front, run.unlisted));
        (up_front + made, unlisted + missed)
    }

    pub fn disabled_features(&self, world: &World) -> engine_core::Names {
        let states = world.get_resource::<engine_registry::BrickStates>();
        let faulted = |gate: engine_registry::GateId| {
            states
                .and_then(|states| states.state(states.brick(gate)?))
                .is_some_and(|state| matches!(state, engine_registry::BrickState::Faulted(_)))
        };
        let bits = self.run.as_ref().map_or(0, |run| {
            run.gates
                .iter()
                .zip(&run.turned)
                .enumerate()
                .filter(|(_, (gate, turned))| match gate {
                    Some(gate) => faulted(*gate),
                    None => turned.is_none(),
                })
                .fold(0_u128, |bits, (at, _)| {
                    bits | 1_u128.checked_shl(at as u32).unwrap_or(0)
                })
        });
        engine_core::Names {
            bits,
            names: self.part_names,
        }
    }

    fn held_assets(&self) -> Option<&crate::encode::Assets> {
        self.run
            .as_ref()
            .map(|run| &run.assets)
            .or(self.assets.as_ref())
    }
}
