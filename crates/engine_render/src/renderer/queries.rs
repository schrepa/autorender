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

    pub fn take_frame_time(&self) -> Option<(f64, (u32, u32))> {
        self.core.display.take_frame_time()
    }

    #[cfg(any(feature = "dev", feature = "profiling"))]
    pub fn gpu_instrument(&mut self, name: &str, on: bool) -> Result<&'static str, &'static str> {
        if name == crate::linker::counting::INSTRUMENT {
            return match self.run.as_mut() {
                Some(run) => run.count_work(on),
                None => Err("not_installed"),
            };
        }
        self.core.profiler.switch(name, on)
    }

    #[cfg(any(feature = "dev", feature = "profiling"))]
    pub(crate) fn counter(&self) -> Option<&crate::linker::counting::Counter> {
        Some(&self.run.as_ref()?.pipelines.work.as_ref()?.counter)
    }

    #[cfg(any(feature = "dev", feature = "profiling"))]
    pub fn timer_names(world: &mut bevy_ecs::world::World) {
        use engine_registry::{Kind, NameEntry, register, unavailable};
        let names = [
            ("gpu.passes.time", Kind::Time, Some("ns")),
            ("gpu.passes.timed", Kind::Count, Some("frames")),
            ("gpu.passes.frames", Kind::Count, Some("frames")),
            ("gpu.passes.groups", Kind::State, None),
            ("gpu.draws.vertex_runs", Kind::Count, None),
            ("gpu.draws.fragment_runs", Kind::Count, None),
            ("gpu.draws.clipped", Kind::Count, Some("primitives")),
            ("gpu.draws.measured", Kind::Count, Some("draws")),
            ("gpu.draws.unmeasured", Kind::Count, Some("draws")),
            ("gpu.draws.measuring", Kind::State, None),
            ("gpu.pipelines", Kind::State, None),
            ("gpu.pipelines.counts", Kind::Count, Some("pipelines")),
        ];
        for (name, kind, unit) in names {
            let entry = NameEntry::read(name, kind, move |reach, _| {
                Ok(reach
                    .part::<Self>()
                    .map_or_else(|| unavailable("no_renderer"), |renderer| renderer.timer_read(name)))
            });
            register(
                world,
                match unit {
                    Some(unit) => entry.unit(unit),
                    None => entry,
                },
            );
        }
        crate::linker::counting::names(world);
    }

    #[cfg(any(feature = "dev", feature = "profiling"))]
    fn timer_read(&self, name: &str) -> serde_json::Value {
        let Some(run) = self.run.as_ref() else {
            return engine_registry::unavailable("not_installed");
        };
        let pipelines = &run.pipelines;
        match name {
            "gpu.pipelines" => pipelines.making.book.state(|program| {
                (run.machine.pipelines.iter())
                    .find(|built| built.program_key == program)
                    .map_or_else(
                        || engine_registry::unavailable("unknown_program"),
                        |built| crate::install::pipeline_facts(&built.decl),
                    )
            }),
            "gpu.pipelines.counts" => pipelines
                .making
                .book
                .counts((pipelines.up_front, pipelines.unlisted)),
            _ => self.core.profiler.read(name),
        }
    }

    pub fn is_texture_resident(&self, handle: engine_core::TextureHandle) -> bool {
        self.held_assets()
            .is_some_and(|assets| assets.textures.is_texture_resident(handle))
    }

    pub fn pipelines_compiling(&mut self) -> u32 {
        crate::platform::poll(&self.core.device);
        let pending = self.run.as_mut().map_or(0, Run::take_made);
        self.core.ledger.compiling().saturating_add(pending)
    }

    pub fn pipelines_stood_in(&self) -> u32 {
        self.run.as_ref().map_or(0, |run| run.pipelines.stood_in())
    }

    pub fn pipelines_warming(&self) -> u32 {
        self.run
            .as_ref()
            .map_or(0, |run| run.pipelines.warming_left())
    }

    pub fn pipelines_warmed(&self) -> u32 {
        self.run
            .as_ref()
            .map_or(0, |run| run.pipelines.warming.made())
    }

    pub fn settle_pipelines(&mut self) {
        if let Some(run) = self.run.as_mut() {
            run.pipelines.settled = true;
        }
    }

    pub fn keep_pipeline_cache(&self) {
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(kept) = &self.core.pipeline_cache {
            super::cache::save(kept);
        }
    }

    #[cfg(all(feature = "dev", not(target_arch = "wasm32")))]
    pub fn shader_errors(&self) -> &[Error] {
        self.watched.errors()
    }

    #[cfg(not(all(feature = "dev", not(target_arch = "wasm32"))))]
    pub fn shader_errors(&self) -> &[Error] {
        &[]
    }

    pub fn pipelines_per_pass(&self) -> Vec<(&'static str, [u32; 2])> {
        self.run
            .as_ref()
            .map(|run| run.pipelines.per_pass.clone())
            .unwrap_or_default()
    }

    pub fn compiled_per_pass(&self) -> Vec<(&'static str, u32)> {
        self.core.ledger.per_pass().to_vec()
    }

    pub fn pipeline_list(&self) -> Vec<u8> {
        self.run.as_ref().map(Run::listing).unwrap_or_default()
    }

    pub fn create_listed_pipelines(&mut self, record: &[u8]) -> Result<(), Error> {
        let Some(run) = self.run.as_mut() else {
            return Ok(());
        };
        let cause = engine_profile::open("launch", "pipeline_list");
        if run.reached.generation.is_some() {
            return engine_profile::caused(cause, || run.create_listed(record));
        }
        let scope = crate::gpu_calls::validation_scope(&self.core.device);
        let created = engine_profile::caused(cause, || run.create_listed(record));
        validate_also(&mut self.installing, Box::pin(scope.pop()));
        created
    }

    pub fn pipelines_listed(&self) -> (u32, u32) {
        self.run.as_ref().map_or((0, 0), |run| {
            (run.pipelines.up_front, run.pipelines.unlisted)
        })
    }

    fn held_assets(&self) -> Option<&crate::encode::Assets> {
        self.run
            .as_ref()
            .map(|run| &run.assets)
            .or(self.assets.as_ref())
    }
}
