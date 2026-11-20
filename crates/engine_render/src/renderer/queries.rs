use super::*;

impl Renderer {
    pub fn adapter(&self) -> &crate::AdapterInfo {
        &self.core.adapter
    }

    pub fn features(&self) -> wgpu::Features {
        self.core.device.features()
    }

    pub fn view_size(&self) -> Option<(u32, u32)> {
        (!self.core.display.suspended()).then(|| self.core.display.size().pixels())
    }

    pub fn take_frame_time(&self) -> Option<(f64, (u32, u32))> {
        self.core.display.take_frame_time()
    }

    pub fn time_frame(&self) {
        self.core.display.time_frame();
    }

    #[cfg(feature = "dev")]
    pub fn gpu_instrument(&mut self, name: &str, on: bool) -> Result<&'static str, &'static str> {
        if name == crate::linker::counting::INSTRUMENT {
            return match self.run.as_mut() {
                Some(run) => run.count_work(on),
                None => Err("not_installed"),
            };
        }
        self.core.profiler.switch(name, on)
    }

    #[cfg(feature = "dev")]
    pub(crate) fn counter(&self) -> Option<&crate::linker::counting::Counter> {
        Some(&self.run.as_ref()?.pipelines.work.as_ref()?.counter)
    }

    #[cfg(feature = "dev")]
    pub(crate) fn layouts(&self) -> Option<&crate::machine::Layouts> {
        Some(&self.run.as_ref()?.machine.layouts)
    }

    #[cfg(feature = "dev")]
    pub(super) fn timer_names(world: &mut bevy_ecs::world::World) {
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
            ("gpu.pipelines.cache", Kind::State, None),
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
        crate::machine::layout_names(world);
    }

    #[cfg(feature = "dev")]
    fn timer_read(&self, name: &str) -> serde_json::Value {
        if name == "gpu.pipelines.cache" {
            return self.cache_facts();
        }
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
            "gpu.pipelines.counts" => {
                let mut counts = pipelines.making.book.counts();
                if let Some(counts) = counts.as_object_mut() {
                    counts.insert("warming".to_owned(), pipelines.warming_left().into());
                    counts.insert("warmed".to_owned(), pipelines.warming.made().into());
                    counts.insert("stood_in".to_owned(), pipelines.warming.stood_in().into());
                }
                counts
            }
            _ => self.core.profiler.read(name),
        }
    }

    #[cfg(all(feature = "dev", not(target_arch = "wasm32")))]
    fn cache_facts(&self) -> serde_json::Value {
        self.core.pipeline_cache.as_ref().map_or_else(
            |&reason| engine_registry::unavailable(reason),
            super::cache::KeptCache::facts,
        )
    }

    #[cfg(all(feature = "dev", target_arch = "wasm32"))]
    fn cache_facts(&self) -> serde_json::Value {
        engine_registry::unavailable("wasm")
    }

    pub fn pipelines_compiling(&mut self) -> u32 {
        crate::platform::poll(&self.core.device);
        let pending = self.run.as_mut().map_or(0, Run::take_made);
        self.core.ledger.compiling().saturating_add(pending)
    }

    pub fn held_ready(&self) -> bool {
        self.run.as_ref().and_then(|run| run.tallies.held) == Some(false)
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
        if let Ok(kept) = &self.core.pipeline_cache {
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

    pub fn compiled_per_pass(&self) -> Vec<(&'static str, u32)> {
        self.core.ledger.per_pass().to_vec()
    }
}
