use super::making::{Created, Creating};
use super::run::Pipelines;
use super::warming::{Asked, Warm, WarmShape};
use super::*;
use crate::Variant;
use crate::install::RenderDecl;
use crate::machine::{BuiltPipeline, Stage};
use engine_core::DropReason;

fn stage_key(
    stage: &Stage,
    (entry, constants): (&str, &[(String, f64)]),
    (layout, paired): (u64, Option<u64>),
) -> u64 {
    let mut hasher = DefaultHasher::new();
    (stage.text_key(), entry, layout, paired).hash(&mut hasher);
    for (name, value) in constants {
        (name, value.to_bits()).hash(&mut hasher);
    }
    hasher.finish()
}

fn made_by_program(
    pipelines: &[BuiltPipeline],
    program_key: u64,
    key: u64,
) -> Option<wgpu::RenderPipeline> {
    pipelines
        .iter()
        .filter(|other| other.program_key == program_key)
        .find_map(|other| {
            other
                .made
                .iter()
                .find(|(known, _)| *known == key)
                .map(|(_, made)| made.clone())
        })
}

fn stage_constants(stages: &[Stage], offered: &[(&str, f64)]) -> Vec<Vec<(String, f64)>> {
    stages
        .iter()
        .map(|stage| {
            stage
                .composed
                .constants(offered)
                .into_iter()
                .map(|(name, value)| (name.to_owned(), value))
                .collect()
        })
        .collect()
}

struct HeldLayout {
    stride: u64,
    step: wgpu::VertexStepMode,
    attributes: Vec<wgpu::VertexAttribute>,
}

impl HeldLayout {
    fn layout(&self) -> wgpu::VertexBufferLayout<'_> {
        wgpu::VertexBufferLayout {
            array_stride: self.stride,
            step_mode: self.step,
            attributes: &self.attributes,
        }
    }
}

fn vertex_layouts(vertices: &[engine_core::GeometryStream]) -> Vec<HeldLayout> {
    vertices
        .iter()
        .map(|&stream| HeldLayout {
            stride: stream.stride(),
            step: wgpu::VertexStepMode::Vertex,
            attributes: crate::mesh_arena::geometry_attributes(stream),
        })
        .collect()
}

fn depth_stencil_state(
    format: wgpu::TextureFormat,
    depth: DepthUse,
    stencil: &wgpu::StencilState,
) -> wgpu::DepthStencilState {
    let (write, compare, bias) = match depth {
        DepthUse::None => (
            false,
            wgpu::CompareFunction::Always,
            wgpu::DepthBiasState::default(),
        ),
        DepthUse::Compare {
            compare,
            write,
            bias,
        } => (write, compare, bias),
    };
    wgpu::DepthStencilState {
        format,
        depth_write_enabled: Some(write),
        depth_compare: Some(compare),
        stencil: stencil.clone(),
        bias,
    }
}

fn color_targets(
    desc: &RenderDecl,
    formats: &[Option<wgpu::TextureFormat>],
    written: Option<&[u32]>,
) -> Vec<Option<wgpu::ColorTargetState>> {
    formats
        .iter()
        .enumerate()
        .map(|(location, format)| {
            let format = (*format)?;
            let (blend, mask) = desc
                .targets
                .get(location)
                .copied()
                .unwrap_or((None, wgpu::ColorWrites::empty()));
            let outputs = written.is_none_or(|written| written.contains(&(location as u32)));
            Some(wgpu::ColorTargetState {
                format,
                blend: blend.filter(|_| blendable(format)),
                write_mask: if outputs {
                    mask
                } else {
                    wgpu::ColorWrites::empty()
                },
            })
        })
        .collect()
}

fn entry_of(program: &Program) -> &'static str {
    match program {
        Program::Render(desc) => desc.vertex.name(),
        Program::Compute(entry) => entry.name(),
    }
}

impl Machine {
    fn pass_name(&self, pipeline: u32) -> &'static str {
        let Some(built) = self.pipelines.get(pipeline as usize) else {
            return "";
        };
        match built.decl.scope {
            Scoped::Scene(bit) => self.declared.scene_name(bit),
            Scoped::Outside(name) => name,
            Scoped::Pass(pass) => self
                .declared
                .passes
                .get(pass as usize)
                .map_or("", |declared| declared.name),
        }
    }

    fn asker(&self, pipeline: u32, asked: Asked) -> crate::pipelines::Asker {
        let built = self.pipelines.get(pipeline as usize);
        let part = built.and_then(|built| self.declared.parts.get(built.decl.part as usize));
        crate::pipelines::Asker {
            entry: built.map_or("", |built| entry_of(&built.decl.program)),
            part: part.map_or("", |part| part.name),
            brick: part.map_or("", |part| part.package),
            pass: self.pass_name(pipeline),
            by: asked.name(),
        }
    }

    pub(crate) fn draw_key(&self, pass: u32, pipeline: u32) -> crate::profiling::DrawKey {
        let asker = self.asker(pipeline, Asked::Drawn);
        crate::profiling::DrawKey {
            pass: self.declared.passes.get(pass as usize).map_or("", |pass| pass.name),
            part: asker.part,
            pipeline: asker.entry,
        }
    }
}

pub(crate) struct Compiler<'r> {
    device: &'r wgpu::Device,
    machine: &'r mut Machine,
    pipelines: &'r mut Pipelines,
    frame: u64,
    #[cfg(feature = "dev")]
    bound: &'r mut super::run::Bound,
}

struct Shape<'f> {
    formats: &'f [Option<wgpu::TextureFormat>],
    depth: Option<wgpu::TextureFormat>,
    cut: Option<Variant>,
    fragment: Option<engine_core::Entry>,
}

type StageJob = (wgpu::ShaderModule, &'static str, Vec<(String, f64)>);

struct RenderJob {
    device: wgpu::Device,
    label: String,
    layout: wgpu::PipelineLayout,
    vertex: StageJob,
    buffers: Vec<HeldLayout>,
    primitive: wgpu::PrimitiveState,
    depth_stencil: Option<wgpu::DepthStencilState>,
    samples: u32,
    fragment: Option<StageJob>,
    targets: Vec<Option<wgpu::ColorTargetState>>,
    cache: Option<wgpu::PipelineCache>,
    stages: Vec<u64>,
}

struct ComputeJob {
    device: wgpu::Device,
    label: String,
    layout: wgpu::PipelineLayout,
    module: wgpu::ShaderModule,
    entry: &'static str,
    constants: Vec<(String, f64)>,
    cache: Option<wgpu::PipelineCache>,
}

fn named(constants: &[(String, f64)]) -> Vec<(&str, f64)> {
    constants
        .iter()
        .map(|(name, value)| (name.as_str(), *value))
        .collect()
}

fn options<'a>(constants: &'a [(&'a str, f64)]) -> wgpu::PipelineCompilationOptions<'a> {
    wgpu::PipelineCompilationOptions {
        constants,
        zero_initialize_workgroup_memory: true,
    }
}

impl RenderJob {
    #[expect(
        clippy::disallowed_methods,
        reason = "the machine keys every pipeline, which the pipeline list records"
    )]
    fn make(self) -> Creating {
        #[cfg(not(target_arch = "wasm32"))]
        let scope = crate::gpu_calls::validation_scope(&self.device);
        let vertex_constants = named(&self.vertex.2);
        let buffers: Vec<Option<wgpu::VertexBufferLayout<'_>>> = self
            .buffers
            .iter()
            .map(|held| Some(held.layout()))
            .collect();
        let fragment_constants = self
            .fragment
            .as_ref()
            .map(|(_, _, constants)| named(constants));
        let desc = wgpu::RenderPipelineDescriptor {
            label: Some(&self.label),
            layout: Some(&self.layout),
            vertex: wgpu::VertexState {
                module: &self.vertex.0,
                entry_point: Some(self.vertex.1),
                compilation_options: options(&vertex_constants),
                buffers: &buffers,
            },
            primitive: self.primitive,
            depth_stencil: self.depth_stencil.clone(),
            multisample: wgpu::MultisampleState {
                count: self.samples,
                ..Default::default()
            },
            fragment: self.fragment.as_ref().zip(fragment_constants.as_ref()).map(
                |((module, entry, _), constants)| wgpu::FragmentState {
                    module,
                    entry_point: Some(*entry),
                    compilation_options: options(constants),
                    targets: &self.targets,
                },
            ),
            multiview_mask: None,
            cache: self.cache.as_ref(),
        };
        #[cfg(not(target_arch = "wasm32"))]
        {
            let made = self.device.create_render_pipeline(&desc);
            if let Some(Some(error)) = crate::platform::ready(scope.pop()) {
                return Err(error);
            }
            Ok(Created::Render(made))
        }
        #[cfg(target_arch = "wasm32")]
        {
            let made = self.device.create_render_pipeline_async(&desc);
            Box::pin(async move { made.await.map(Created::Render) })
        }
    }
}

impl ComputeJob {
    #[expect(
        clippy::disallowed_methods,
        reason = "the machine keys every pipeline, which the pipeline list records"
    )]
    fn make(self) -> Creating {
        #[cfg(not(target_arch = "wasm32"))]
        let scope = crate::gpu_calls::validation_scope(&self.device);
        let constants = named(&self.constants);
        let desc = wgpu::ComputePipelineDescriptor {
            label: Some(&self.label),
            layout: Some(&self.layout),
            module: &self.module,
            entry_point: Some(self.entry),
            compilation_options: options(&constants),
            cache: self.cache.as_ref(),
        };
        #[cfg(not(target_arch = "wasm32"))]
        {
            let made = self.device.create_compute_pipeline(&desc);
            if let Some(Some(error)) = crate::platform::ready(scope.pop()) {
                return Err(error);
            }
            Ok(Created::Compute(made))
        }
        #[cfg(target_arch = "wasm32")]
        {
            let made = self.device.create_compute_pipeline_async(&desc);
            Box::pin(async move { made.await.map(Created::Compute) })
        }
    }
}

fn computed_by_program(
    pipelines: &[BuiltPipeline],
    program_key: u64,
    key: u64,
) -> Option<wgpu::ComputePipeline> {
    pipelines
        .iter()
        .filter(|other| other.program_key == program_key)
        .find_map(|other| {
            other
                .computed
                .iter()
                .find(|(known, _)| *known == key)
                .map(|(_, made)| made.clone())
        })
}

struct Keyed {
    program: u64,
    key: u64,
    cut: Option<Variant>,
    fragment: Option<engine_core::Entry>,
}

impl Keyed {
    fn shape<'f>(
        &self,
        formats: &'f [Option<wgpu::TextureFormat>],
        depth: Option<wgpu::TextureFormat>,
    ) -> Shape<'f> {
        Shape {
            formats,
            depth,
            cut: self.cut.clone(),
            fragment: self.fragment,
        }
    }
}

fn cut_of(built: &BuiltPipeline, variant: Option<&Variant>) -> Option<Variant> {
    let Program::Render(desc) = &built.decl.program else {
        return None;
    };
    variant?.cut(desc, |name| {
        (built.stages.iter()).any(|stage| !stage.composed.constants(&[(name, 0.0)]).is_empty())
    })
}

impl Run {
    pub(super) fn compiler(&mut self) -> Compiler<'_> {
        Compiler {
            device: &self.device,
            machine: &mut self.machine,
            pipelines: &mut self.pipelines,
            frame: self.frames.asked,
            #[cfg(feature = "dev")]
            bound: &mut self.bound,
        }
    }

    pub(crate) fn take_made(&mut self) -> u32 {
        self.compiler().collect(None);
        self.feed_warming();
        let warming = u32::from(self.pipelines.warming.fed.is_some());
        self.pipelines.making.needed().saturating_sub(warming)
    }

    pub(crate) fn wait_made(&mut self) {
        while let Some(first) = self.pipelines.making.first() {
            self.compiler().collect(Some(first));
            if self.pipelines.making.first() == Some(first) {
                return;
            }
        }
    }

    pub(crate) fn adopt_made(
        &mut self,
        shaders: std::collections::BTreeMap<u64, wgpu::ShaderModule>,
        made: Vec<BuiltPipeline>,
        mut pipelines: Pipelines,
    ) {
        for (key, module) in shaders {
            self.machine.shaders.entry(key).or_insert(module);
        }
        for old in made {
            if let Some(built) = self
                .machine
                .pipelines
                .iter_mut()
                .find(|built| built.program_key == old.program_key)
            {
                built.made.extend(old.made);
                built.computed.extend(old.computed);
            }
        }
        pipelines.making.forget_refused();
        let making = &pipelines.making;
        pipelines.standing.retain(|&made| making.waits(made));
        self.pipelines = pipelines;
        #[cfg(feature = "dev")]
        if self.pipelines.work.as_ref().is_some_and(|work| work.live.is_some()) {
            let _ = self.count_work(true);
        }
    }

    pub(crate) fn compile(
        &mut self,
        pipeline: u32,
        targets: (
            &[Option<wgpu::TextureFormat>],
            Option<wgpu::TextureFormat>,
        ),
        shown: (bool, bool),
        variant: Option<&Variant>,
    ) -> Option<(u64, wgpu::RenderPipeline)> {
        self.compiler()
            .compile_with(pipeline, targets, shown, variant, Asked::Drawn)
    }

    pub(super) fn pipeline_of(&self, handle: u32) -> Option<(u32, Option<usize>)> {
        let count = self.machine.pipelines.len() as u32;
        if handle < count {
            return Some((handle, None));
        }
        let filled = (handle - count) as usize;
        Some((self.filled.get(filled)?.base, Some(filled)))
    }

    pub(super) fn render_pipeline(
        &mut self,
        pipeline: u32,
        drawn: Option<(&Variant, u64)>,
    ) -> Result<usize, engine_core::DropReason> {
        let layout = self
            .recording
            .open()
            .ok_or(DropReason::NoRenderPass)?
            .layout;
        let identity = drawn.map_or(0, |(_, identity)| identity);
        let built = self
            .machine
            .pipelines
            .get(pipeline as usize)
            .ok_or(DropReason::UnknownPipeline)?;
        if let Some(at) = built
            .ready
            .iter()
            .position(|ready| ready.layout == layout && ready.variant == identity)
        {
            return Ok(at);
        }
        let variant = drawn.map(|(variant, _)| variant);
        let program = built.program_key;
        let pass = self
            .machine
            .pass_layout(layout)
            .ok_or(DropReason::NoRenderPass)?;
        let targets = (pass.formats(), pass.depth);
        let shown = (pass.plain, pass.main);
        let found = match variant.and_then(|variant| Some((variant, variant.stand_in.as_deref()?))) {
            Some(variants) => self
                .compiler()
                .standing_in(pipeline, targets, shown, variants),
            None => self
                .compile(pipeline, targets, shown, variant)
                .map(|(key, made)| (key, made, true)),
        };
        let Some((key, made, exact)) = found else {
            return Err(if self.pipelines.making.program_waits(program) {
                DropReason::PipelineCompiling
            } else {
                DropReason::PipelineRefused
            });
        };
        self.pipelines.making.book.used((program, key), self.frames.asked);
        #[cfg(feature = "dev")]
        if self.pipelines.work.is_some() {
            let variant = if exact { variant } else { variant.and_then(|variant| variant.stand_in.as_deref()) };
            self.compiler().count_ready(pipeline, (targets, shown), (program, key), variant);
        }
        let built = self
            .machine
            .pipelines
            .get_mut(pipeline as usize)
            .ok_or(DropReason::UnknownPipeline)?;
        built.ready.push(Ready {
            layout,
            variant: identity,
            made,
            exact,
        });
        Ok(built.ready.len() - 1)
    }

    pub(super) fn compute_pipeline(
        &mut self,
        pipeline: u32,
    ) -> Result<wgpu::ComputePipeline, DropReason> {
        let built = self.machine.pipelines.get(pipeline as usize);
        let program = built.ok_or(DropReason::UnknownPipeline)?.program_key;
        let Some(found) = self.compute_with(pipeline, Asked::Drawn) else {
            return Err(if self.pipelines.making.program_waits(program) {
                DropReason::PipelineCompiling
            } else {
                DropReason::PipelineRefused
            });
        };
        self.pipelines.making.book.used((program, 0), self.frames.asked);
        Ok(found)
    }

    pub(super) fn compute_with(&mut self, pipeline: u32, asked: Asked) -> Option<wgpu::ComputePipeline> {
        self.compiler().collect(None);
        let built = self.machine.pipelines.get(pipeline as usize)?;
        let (Program::Compute(_), Some(stage)) = (&built.decl.program, built.stages.first()) else {
            return None;
        };
        let key = 0;
        let program = built.program_key;
        let made = (program, key);
        if computed_by_program(&self.machine.pipelines, program, key).is_none()
            && !self.pipelines.making.refused(made)
            && !self.pipelines.making.waits(made)
        {
            let constants: Vec<(String, f64)> = stage
                .composed
                .constants(&built.decl.constants)
                .into_iter()
                .map(|(name, value)| (name.to_owned(), value))
                .collect();
            let stages = vec![stage_key(
                stage,
                (stage.entry.name(), &constants),
                (built.layout_key, None),
            )];
            let job = ComputeJob {
                device: self.device.clone(),
                label: stage.entry.name().to_owned(),
                layout: built.layout.clone(),
                module: stage.shader(&self.device, &mut self.machine.shaders),
                entry: stage.entry.name(),
                constants,
                cache: self.pipelines.cache.clone(),
            };
            let pass = self.machine.pass_name(pipeline);
            let machine = &self.machine;
            self.pipelines
                .making
                .book
                .asked(made, self.frames.asked, || machine.asker(pipeline, asked));
            self.pipelines
                .making
                .submit((made, false), (pass, stages), move || job.make());
            if asked == Asked::Warmed {
                self.pipelines.warming.fed = Some(made);
            }
        } else if asked != Asked::Warmed {
            self.pipelines.warming.promote(made);
        }
        self.compiler().collect(None);
        let found = computed_by_program(&self.machine.pipelines, program, key)?;
        if let Some(built) = self.machine.pipelines.get_mut(pipeline as usize)
            && !built.computed.iter().any(|(known, _)| *known == key)
        {
            built.computed.push((key, found.clone()));
        }
        #[cfg(feature = "dev")]
        if asked != Asked::Warmed {
            self.compiler().count_compute(pipeline, made);
        }
        Some(found)
    }
}

impl Compiler<'_> {
    pub(crate) fn collect(&mut self, waited: Option<(u64, u64)>) {
        for finished in self.pipelines.making.take(waited) {
            #[cfg(feature = "dev")]
            let Some(finished) = self.counted_arrived(finished) else {
                continue;
            };
            let done = (finished.program, finished.key);
            let warmed = self.pipelines.warming.fed == Some(done);
            if warmed {
                self.pipelines.warming.fed = None;
            }
            let stood = self.pipelines.standing.contains(&done);
            self.pipelines.standing.retain(|known| *known != done);
            let made = match finished.made {
                Ok(made) => made,
                Err(error) => {
                    self.pipelines.making.refuse(done);
                    let error = error.to_string();
                    self.pipelines.making.book.refused(done, self.frame, &error);
                    self.pipelines.refused.push((finished.pass, finished.program, error));
                    continue;
                }
            };
            let Some(built) = self
                .machine
                .pipelines
                .iter_mut()
                .find(|built| built.program_key == finished.program)
            else {
                continue;
            };
            match made {
                Created::Render(made) => built.made.push((finished.key, made)),
                Created::Compute(made) => built.computed.push((finished.key, made)),
            }
            if warmed {
                self.pipelines.warming.made += 1;
            } else {
                crate::pipelines::count(finished.pass, 1);
            }
            if warmed || stood {
                for built in &mut self.machine.pipelines {
                    if built.program_key == finished.program {
                        built.ready.clear();
                    }
                }
            }
            self.pipelines.making.book.ready(done, self.frame);
        }
    }

    fn made(&mut self, pipeline: u32, (program, key): (u64, u64)) -> Option<wgpu::RenderPipeline> {
        let found = made_by_program(&self.machine.pipelines, program, key)?;
        if let Some(built) = self.machine.pipelines.get_mut(pipeline as usize)
            && !built.made.iter().any(|(known, _)| *known == key)
        {
            built.made.push((key, found.clone()));
        }
        Some(found)
    }

    fn standing_in(
        &mut self,
        pipeline: u32,
        targets: (
            &[Option<wgpu::TextureFormat>],
            Option<wgpu::TextureFormat>,
        ),
        shown: (bool, bool),
        (specialized, dynamic): (&Variant, &Variant),
    ) -> Option<(u64, wgpu::RenderPipeline, bool)> {
        self.collect(None);
        let keyed = self.keyed(pipeline, targets, shown, Some(specialized))?;
        let made = (keyed.program, keyed.key);
        if let Some(found) = self.made(pipeline, made) {
            return Some((keyed.key, found, true));
        }
        let refused = self.pipelines.making.refused(made);
        let standing = self.keyed(pipeline, targets, shown, Some(dynamic))?;
        let exact = standing.key == keyed.key || !self.pipelines.settled;
        if !refused && exact {
            return self
                .compile_with(pipeline, targets, shown, Some(specialized), Asked::Drawn)
                .map(|(key, made)| (key, made, true));
        }
        if !refused {
            if !self.pipelines.standing.contains(&made) {
                self.pipelines.standing.push(made);
                self.pipelines.warming.stood_in += 1;
            }
            let warm = Warm {
                pipeline,
                shape: Some(WarmShape {
                    formats: targets.0.to_vec(),
                    depth: targets.1,
                    shown,
                    variant: keyed.cut,
                }),
            };
            let warming = &mut self.pipelines.warming;
            let queued = warming.queue.iter().any(|known| known.same(&warm));
            if !queued && !self.pipelines.making.waits(made) {
                warming.queue.push_front(warm);
            }
        }
        self.compile_with(pipeline, targets, shown, Some(dynamic), Asked::Drawn)
            .map(|(key, made)| (key, made, false))
    }

    fn keyed(
        &self,
        pipeline: u32,
        (formats, depth): (
            &[Option<wgpu::TextureFormat>],
            Option<wgpu::TextureFormat>,
        ),
        (plain, main): (bool, bool),
        variant: Option<&Variant>,
    ) -> Option<Keyed> {
        let built = self.machine.pipelines.get(pipeline as usize)?;
        let Program::Render(desc) = &built.decl.program else {
            return None;
        };
        let cut = cut_of(built, variant);
        let fragment = desc.fragment.filter(|_| {
            !desc.per_batch.fragment || cut.as_ref().is_some_and(|cut| cut.fragment)
        });
        if fragment.is_none() && formats.iter().any(Option::is_some) {
            return None;
        }
        let mut hasher = DefaultHasher::new();
        (formats, depth, plain, main).hash(&mut hasher);
        (cut.as_ref().map(Variant::key), fragment.is_some()).hash(&mut hasher);
        Some(Keyed {
            program: built.program_key,
            key: hasher.finish(),
            cut,
            fragment,
        })
    }

    pub(super) fn compile_with(
        &mut self,
        pipeline: u32,
        (formats, depth): (
            &[Option<wgpu::TextureFormat>],
            Option<wgpu::TextureFormat>,
        ),
        (plain, main): (bool, bool),
        variant: Option<&Variant>,
        asked: Asked,
    ) -> Option<(u64, wgpu::RenderPipeline)> {
        self.collect(None);
        let keyed = self.keyed(pipeline, (formats, depth), (plain, main), variant)?;
        let (program, key) = (keyed.program, keyed.key);
        let made = (program, key);
        if self.made(pipeline, made).is_none()
            && !self.pipelines.making.refused(made)
            && !self.pipelines.making.waits(made)
        {
            let Some(mut job) = self.render_job(pipeline, keyed.shape(formats, depth)) else {
                self.pipelines.standing.retain(|known| *known != made);
                return None;
            };
            let stages = std::mem::take(&mut job.stages);
            let pass = self.machine.pass_name(pipeline);
            let machine = &*self.machine;
            self.pipelines
                .making
                .book
                .asked(made, self.frame, || machine.asker(pipeline, asked));
            self.pipelines
                .making
                .submit((made, false), (pass, stages), move || job.make());
            if asked == Asked::Warmed {
                self.pipelines.warming.fed = Some(made);
            }
            self.collect(None);
        } else if asked != Asked::Warmed {
            self.pipelines.warming.promote(made);
        }
        let found = self.made(pipeline, made)?;
        Some((key, found))
    }

    fn render_job(&mut self, pipeline: u32, shape: Shape<'_>) -> Option<RenderJob> {
        let machine = &mut *self.machine;
        let built = machine.pipelines.get(pipeline as usize)?;
        let Program::Render(desc) = &built.decl.program else {
            return None;
        };
        let offered: Vec<(&str, f64)> = (shape.cut.iter())
            .flat_map(|cut| cut.constants.iter().map(|(name, value)| (name.as_str(), *value)))
            .chain(built.decl.constants.iter().copied())
            .collect();
        let mut constants = stage_constants(&built.stages, &offered).into_iter();
        let fragment_stage = shape.fragment.and_then(|_| built.stages.get(1));
        let written = fragment_stage
            .zip(shape.fragment)
            .and_then(|(stage, entry)| written_locations(&stage.composed, entry.name()));
        let mut targets = color_targets(desc, shape.formats, written.as_deref());
        if let Some(cut) = &shape.cut {
            cut_targets(&mut targets, cut);
        }
        let vertex = built
            .stages
            .first()?
            .shader(self.device, &mut machine.shaders);
        let fragment = fragment_stage.zip(shape.fragment).map(|(stage, entry)| {
            (
                stage.shader(self.device, &mut machine.shaders),
                entry.name(),
            )
        });
        let vertex_constants = constants.next().unwrap_or_default();
        let fragment_constants = constants.next().unwrap_or_default();
        let vertex_stage = built.stages.first()?;
        let indexed = ["@builtin(vertex_index)", "@builtin(instance_index)"]
            .iter()
            .any(|builtin| vertex_stage.composed.text.contains(builtin));
        let paired = fragment_stage.zip(shape.fragment).map(|(stage, entry)| {
            stage_key(
                stage,
                (entry.name(), &fragment_constants),
                (built.layout_key, Some(u64::from(indexed))),
            )
        });
        let inputs = fragment_stage
            .zip(shape.fragment)
            .map(|(stage, entry)| {
                (stage.composed.fragment(entry.name())).map(|fragment| fragment.inputs.clone())
            });
        let mut partner = DefaultHasher::new();
        inputs.hash(&mut partner);
        let stages = std::iter::once(stage_key(
            vertex_stage,
            (desc.vertex.name(), &vertex_constants),
            (built.layout_key, Some(partner.finish())),
        ))
        .chain(paired)
        .collect();
        Some(RenderJob {
            device: self.device.clone(),
            label: desc.vertex.name().to_owned(),
            layout: built.layout.clone(),
            vertex: (vertex, desc.vertex.name(), vertex_constants),
            buffers: vertex_layouts(&desc.vertices),
            primitive: wgpu::PrimitiveState {
                cull_mode: shape.cut.as_ref().map_or(desc.primitive.cull_mode, |cut| cut.cull),
                ..desc.primitive
            },
            depth_stencil: shape.depth.map(|format| {
                let mut state = depth_stencil_state(format, desc.depth, &desc.stencil);
                if let Some(cut) = &shape.cut {
                    cut_depth(&mut state, cut);
                }
                state
            }),
            samples: desc.samples,
            fragment: fragment.map(|(module, entry)| (module, entry, fragment_constants)),
            targets,
            cache: self.pipelines.cache.clone(),
            stages,
        })
    }
}

type PassFormats = (
    Vec<Option<wgpu::TextureFormat>>,
    Option<wgpu::TextureFormat>,
    (bool, bool),
);

impl Run {
    pub(super) fn pass_formats(&self, index: u32) -> PassFormats {
        let Some(pass) = self.machine.declared.passes.get(index as usize) else {
            return (Vec::new(), None, (false, false));
        };
        let mut formats: Vec<Option<wgpu::TextureFormat>> = Vec::new();
        let mut depth = None;
        let mut shown = (false, false);
        let declared = &self.machine.declared;
        if shared(pass) {
            let set = declared.pass_set(pass);
            let main = lit_set(&set);
            shown = (!set.is_empty() && !main, main);
            for (_, format, _) in set {
                formats.push(Some(format));
            }
            depth = declared
                .depth_target()
                .and_then(|target| declared.target_format(target));
        }
        for &(name, _) in &pass.targets {
            formats.push(match declared.target_of(name) {
                Some(Target::Canvas) => Some(self.canvas.surface),
                Some(target @ Target::Shared(_)) => declared.target_format(target),
                None => None,
            });
        }
        (formats, depth, shown)
    }

    pub(crate) fn make_all_shaders(&mut self) {
        let machine = &mut self.machine;
        for built in &machine.pipelines {
            for stage in &built.stages {
                stage.shader(&self.device, &mut machine.shaders);
            }
        }
    }

    pub(super) fn writes_fit(&self, pipeline: usize, formats: &[Option<wgpu::TextureFormat>]) -> bool {
        let Some(built) = self.machine.pipelines.get(pipeline) else {
            return false;
        };
        let Program::Render(desc) = &built.decl.program else {
            return true;
        };
        let Some(kinds) = built
            .stages
            .get(1)
            .zip(desc.fragment)
            .and_then(|(stage, entry)| stage.composed.fragment(entry.name()))
            .map(|fragment| fragment.outputs.clone())
        else {
            return true;
        };
        kinds.iter().all(|&(location, kind)| {
            formats
                .get(location as usize)
                .copied()
                .flatten()
                .is_none_or(|format| {
                    matches!(
                        (format.sample_type(None, None), kind),
                        (
                            Some(wgpu::TextureSampleType::Uint),
                            wgpu::TextureSampleType::Uint
                        ) | (
                            Some(wgpu::TextureSampleType::Sint),
                            wgpu::TextureSampleType::Sint
                        ) | (
                            Some(wgpu::TextureSampleType::Float { .. }),
                            wgpu::TextureSampleType::Float { .. }
                        )
                    )
                })
        })
    }
}

fn blendable(format: wgpu::TextureFormat) -> bool {
    format
        .sample_type(None, None)
        .is_some_and(|sample| matches!(sample, wgpu::TextureSampleType::Float { filterable: true }))
}

fn written_locations(composed: &crate::linker::Composed, entry: &str) -> Option<Vec<u32>> {
    let fragment = composed.fragment(entry)?;
    Some(fragment.outputs.iter().map(|(at, _)| *at).collect())
}

fn cut_targets(targets: &mut [Option<wgpu::ColorTargetState>], cut: &Variant) {
    let Some(taken) = &cut.targets else {
        return;
    };
    for (target, &(blend, writes)) in targets.iter_mut().zip(taken) {
        let Some(target) = target else { continue };
        target.blend = blend.filter(|_| blendable(target.format));
        target.write_mask &= writes;
    }
}

fn cut_depth(state: &mut wgpu::DepthStencilState, cut: &Variant) {
    if let Some(write) = cut.depth_write {
        state.depth_write_enabled = Some(write);
    }
    if let Some(compare) = cut.depth_compare {
        state.depth_compare = Some(compare);
    }
    if let Some(bias) = cut.depth_bias {
        state.bias = bias;
    }
}

#[cfg(feature = "dev")]
mod counted;
