use super::super::making::Finished;
use super::super::run::{CountedAsk, Live, Swap, Work};
use super::*;
use crate::linker::counting::{self, Counter, Counting};
use crate::readback::Reader;

fn counted_key(plain: (u64, u64), generation: u64) -> (u64, u64) {
    let mut hasher = DefaultHasher::new();
    (plain, generation, counting::GROUP).hash(&mut hasher);
    (plain.0, hasher.finish())
}

fn module(
    device: &wgpu::Device,
    shaders: &mut std::collections::BTreeMap<u64, wgpu::ShaderModule>,
    (label, text): (&str, &str),
) -> wgpu::ShaderModule {
    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    shaders
        .entry(hasher.finish())
        .or_insert_with(|| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(text.into()),
            })
        })
        .clone()
}

#[expect(
    clippy::disallowed_methods,
    reason = "a counted twin takes its pipeline's machine layout plus the counter group"
)]
fn counted_layout(
    device: &wgpu::Device,
    built: &BuiltPipeline,
    live: &Live,
) -> Option<wgpu::PipelineLayout> {
    let from = u32::try_from(built.group_layouts.len()).ok()?;
    if from > counting::GROUP {
        return None;
    }
    let layouts: Vec<Option<&wgpu::BindGroupLayout>> = (built.group_layouts.iter())
        .chain((from..counting::GROUP).map(|_| &live.empty))
        .chain(std::iter::once(&live.layout))
        .map(Some)
        .collect();
    Some(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("engine/counted-pipeline"),
        bind_group_layouts: &layouts,
        immediate_size: 0,
    }))
}

type Drawn<'f> = (
    (&'f [Option<wgpu::TextureFormat>], Option<wgpu::TextureFormat>),
    (bool, bool),
);

struct Prepared {
    counted: (u64, u64),
    layout: wgpu::PipelineLayout,
    modules: Vec<wgpu::ShaderModule>,
}

impl Compiler<'_> {
    fn prepare(&mut self, pipeline: u32, plain: (u64, u64)) -> Option<Prepared> {
        let work = self.pipelines.work.as_mut()?;
        let live = work.live.as_ref()?;
        let counted = counted_key(plain, work.generation);
        if work.asked.iter().any(|ask| ask.counted == counted) {
            return None;
        }
        let machine = &mut *self.machine;
        let built = machine.pipelines.get(pipeline as usize)?;
        work.counter.counting().texts(&built.stages)?;
        work.asked.push(CountedAsk {
            counted,
            plain,
            pipeline,
            generation: work.generation,
        });
        work.counter.asked(&built.stages);
        let Some(layout) = counted_layout(self.device, built, live) else {
            work.counter.twin(&built.stages, Err("group_3_taken"));
            return None;
        };
        let texts = work.counter.counting().texts(&built.stages)?;
        let modules = (built.stages.iter().zip(texts))
            .map(|(stage, text)| {
                module(
                    self.device,
                    &mut machine.shaders,
                    (stage.entry.name(), text),
                )
            })
            .collect();
        Some(Prepared {
            counted,
            layout,
            modules,
        })
    }

    pub(super) fn count_ready(
        &mut self,
        pipeline: u32,
        (targets, shown): Drawn<'_>,
        plain: (u64, u64),
        variant: Option<&crate::Variant>,
    ) {
        let Some(keyed) = self.keyed(pipeline, targets, shown, variant) else {
            return;
        };
        if (keyed.program, keyed.key) == plain {
            self.count_render(pipeline, plain, keyed.shape(targets.0, targets.1));
        }
    }

    fn count_render(&mut self, pipeline: u32, plain: (u64, u64), shape: Shape<'_>) {
        let Some(prepared) = self.prepare(pipeline, plain) else {
            return;
        };
        let Some(mut job) = self.render_job(pipeline, shape) else {
            self.twin_failed(pipeline, "twin_unbuilt");
            return;
        };
        let mut modules = prepared.modules.into_iter();
        if let Some(module) = modules.next() {
            job.vertex.0 = module;
        }
        if let (Some(fragment), Some(module)) = (job.fragment.as_mut(), modules.next()) {
            fragment.0 = module;
        }
        job.layout = prepared.layout;
        job.stages.clear();
        let pass = self.machine.pass_name(pipeline);
        self.pipelines.making.submit(
            (prepared.counted, true),
            (pass, Vec::new()),
            move || job.make(),
        );
    }

    pub(super) fn count_compute(&mut self, pipeline: u32, plain: (u64, u64)) {
        let Some(prepared) = self.prepare(pipeline, plain) else {
            return;
        };
        let Some(built) = self.machine.pipelines.get(pipeline as usize) else {
            return;
        };
        let (Some(stage), Some(module)) =
            (built.stages.first(), prepared.modules.into_iter().next())
        else {
            self.twin_failed(pipeline, "twin_unbuilt");
            return;
        };
        let job = ComputeJob {
            device: self.device.clone(),
            label: stage.entry.name().to_owned(),
            layout: prepared.layout,
            module,
            entry: stage.entry.name(),
            constants: stage
                .composed
                .constants(&built.decl.constants)
                .into_iter()
                .map(|(name, value)| (name.to_owned(), value))
                .collect(),
            cache: self.pipelines.cache.clone(),
        };
        let pass = self.machine.pass_name(pipeline);
        self.pipelines.making.submit(
            (prepared.counted, true),
            (pass, Vec::new()),
            move || job.make(),
        );
    }

    fn twin_failed(&mut self, pipeline: u32, reason: &'static str) {
        let (Some(work), Some(built)) = (
            self.pipelines.work.as_mut(),
            self.machine.pipelines.get(pipeline as usize),
        ) else {
            return;
        };
        work.counter.twin(&built.stages, Err(reason));
    }

    pub(super) fn counted_arrived(&mut self, finished: Finished) -> Option<Finished> {
        let done = (finished.program, finished.key);
        let Some(work) = self.pipelines.work.as_mut() else {
            return Some(finished);
        };
        let Some(ask) = work.asked.iter().find(|ask| ask.counted == done) else {
            return Some(finished);
        };
        let current = ask.generation == work.generation && work.live.is_some();
        let ((program, key), pipeline) = (ask.plain, ask.pipeline);
        let pipelines = &self.machine.pipelines;
        let pad = (pipelines.iter())
            .find(|built| built.program_key == program)
            .and_then(|built| u32::try_from(built.group_layouts.len()).ok())
            .unwrap_or(counting::GROUP);
        let outcome = match (finished.made, self.bound.swap.as_mut()) {
            (Ok(Created::Render(counted)), Some(swap)) if current => {
                made_by_program(pipelines, program, key).map(|plain| swap.render.push((plain, counted, pad)))
            }
            (Ok(Created::Compute(counted)), Some(swap)) if current => {
                computed_by_program(pipelines, program, key).map(|plain| swap.compute.push((plain, counted, pad)))
            }
            (Err(_), _) => {
                if current && let Some(built) = pipelines.get(pipeline as usize) {
                    work.counter.twin(&built.stages, Err("twin_refused"));
                }
                return None;
            }
            (Ok(_), _) => return None,
        };
        if let Some(built) = pipelines.get(pipeline as usize) {
            let paired = outcome.map(|()| self.frame + 1).ok_or("twin_unpaired");
            work.counter.twin(&built.stages, paired);
        }
        None
    }
}

impl Run {
    pub(crate) fn count_work(&mut self, on: bool) -> Result<&'static str, &'static str> {
        self.bound.swap = None;
        self.bound.forget();
        if !on {
            if let Some(work) = self.pipelines.work.as_mut() {
                work.live = None;
            }
            return Ok(counting::CHANGES);
        }
        let machine = &self.machine;
        let counting = Counting::new((0..machine.pipelines.len()).map(|index| {
            let asker = machine.asker(index as u32, Asked::Drawn);
            (
                format!("{}/{}", asker.part, asker.entry),
                machine.pipelines[index].stages.as_slice(),
            )
        }));
        if counting.bytes() == 0 {
            if let Some(work) = self.pipelines.work.as_mut() {
                work.live = None;
            }
            return Err("nothing_to_count");
        }
        let buffer = crate::gpu_calls::create_buffer(
            &self.device,
            &wgpu::BufferDescriptor {
                label: Some("engine/counted-work"),
                size: counting.bytes(),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        );
        let layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("engine/counted-work"),
                entries: &[Counting::layout_entry()],
            });
        let empty = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("engine/counted-pad"),
                entries: &[],
            });
        let group = crate::pipelines::bind_group(
            &self.device,
            "engine/counted-work",
            &layout,
            &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        );
        let pad = crate::pipelines::bind_group(&self.device, "engine/counted-pad", &empty, &[]);
        let (generation, asked) = self
            .pipelines
            .work
            .take()
            .map_or((0, Vec::new()), |work| (work.generation + 1, work.asked));
        self.pipelines.work = Some(Work {
            counter: Counter::new(counting, self.frames.asked),
            generation,
            asked,
            live: Some(Live {
                buffer,
                layout,
                empty,
            }),
        });
        self.bound.swap = Some(Swap {
            group,
            empty: pad,
            render: Vec::new(),
            compute: Vec::new(),
        });
        for built in &mut self.machine.pipelines {
            built.ready.clear();
            built.compute_ready = None;
        }
        Ok(counting::CHANGES)
    }

    pub(crate) fn read_work(&mut self) {
        let Some(live) = self.pipelines.work.as_ref().and_then(|work| work.live.as_ref()) else {
            return;
        };
        let buffer = live.buffer.clone();
        let (written, device) = (self.written(), self.device.clone());
        let Some(encoder) = self.recording.encoder() else {
            return;
        };
        self.reading.list.record(
            (&device, encoder),
            (&buffer, 0..buffer.size()),
            (Reader::Work, written),
            1,
        );
    }
}
