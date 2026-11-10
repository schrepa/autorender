use super::making::{Created, Finished};
use super::run::Pipelines;
use super::*;
use crate::install::{Faces, FragmentUse, RenderDecl};
use crate::machine::{BuiltPipeline, Stage, output_location};

fn stage_switches(switches: &[(u32, bool)], stages: &[Stage]) -> Vec<(u32, bool)> {
    switches
        .iter()
        .filter(|(id, _)| {
            stages
                .iter()
                .any(|stage| stage.composed.switches.contains(id))
        })
        .copied()
        .collect()
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

fn stage_constants(
    stages: &[Stage],
    switches: &[(u32, bool)],
    material: &[(&str, f64)],
) -> Vec<Vec<(String, f64)>> {
    stages
        .iter()
        .map(|stage| {
            let mut values: Vec<(String, f64)> = switches
                .iter()
                .filter(|(id, _)| stage.composed.switches.contains(id))
                .map(|(id, on)| (id.to_string(), f64::from(u8::from(*on))))
                .collect();
            values.extend(
                stage
                    .composed
                    .constants(material)
                    .into_iter()
                    .map(|(name, value)| (name.to_owned(), value)),
            );
            values
        })
        .collect()
}

fn vertex_layouts(vertices: &[VertexLayout]) -> Vec<Option<wgpu::VertexBufferLayout<'static>>> {
    vertices
        .iter()
        .map(|layout| match layout {
            VertexLayout::Geometry(stream) => Some(crate::mesh_arena::geometry_layout(*stream)),
            VertexLayout::Buffer(layout) => Some(layout.clone()),
        })
        .collect()
}

fn depth_stencil_state(
    format: wgpu::TextureFormat,
    view_depth: bool,
    depth: DepthUse,
) -> wgpu::DepthStencilState {
    let (write, compare, bias) = match depth {
        DepthUse::None => (
            false,
            wgpu::CompareFunction::Always,
            wgpu::DepthBiasState::default(),
        ),
        DepthUse::Test => (
            false,
            if view_depth {
                wgpu::CompareFunction::GreaterEqual
            } else {
                wgpu::CompareFunction::LessEqual
            },
            wgpu::DepthBiasState::default(),
        ),
        DepthUse::Write { bias } => (
            true,
            if view_depth {
                wgpu::CompareFunction::GreaterEqual
            } else {
                wgpu::CompareFunction::LessEqual
            },
            bias,
        ),
        DepthUse::Overwrite => (
            true,
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
        stencil: wgpu::StencilState::default(),
        bias,
    }
}

fn color_targets(
    desc: &RenderDecl,
    formats: &[Option<wgpu::TextureFormat>],
    written: Option<&[u32]>,
    main: bool,
) -> Vec<Option<wgpu::ColorTargetState>> {
    formats
        .iter()
        .enumerate()
        .map(|(location, format)| {
            let format = (*format)?;
            let (blend, mask) = match &desc.targets {
                Some(targets) => targets
                    .get(location)
                    .copied()
                    .unwrap_or((None, wgpu::ColorWrites::empty())),
                None => (
                    (!main || location == 0).then_some(desc.blend).flatten(),
                    desc.writes,
                ),
            };
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

const OVER: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
        operation: wgpu::BlendOperation::Add,
    },
    alpha: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::Zero,
        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
        operation: wgpu::BlendOperation::Add,
    },
};

const ADDED: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    },
    alpha: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::Zero,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    },
};

const COVERED: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::OneMinusSrc,
        operation: wgpu::BlendOperation::Add,
    },
    alpha: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::OneMinusSrc,
        operation: wgpu::BlendOperation::Add,
    },
};

const SUMMED: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    },
    alpha: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    },
};

fn blend_targets(targets: &mut [Option<wgpu::ColorTargetState>], alpha: crate::surface::Alpha) {
    let additive = alpha == crate::surface::Alpha::Additive;
    for (location, target) in targets.iter_mut().enumerate() {
        let Some(target) = target else { continue };
        if location == 0 {
            target.blend = blendable(target.format).then_some(if additive { ADDED } else { OVER });
        } else {
            target.write_mask = wgpu::ColorWrites::empty();
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Weighing {
    Covers,
    Adds,
    Chosen,
}

fn weigh_targets(
    targets: &mut [Option<wgpu::ColorTargetState>],
    weighing: Weighing,
    (cover, accumulated): (usize, usize),
) {
    let adds = matches!(weighing, Weighing::Adds | Weighing::Chosen);
    let covers = matches!(weighing, Weighing::Covers | Weighing::Chosen);
    for (location, target) in targets.iter_mut().enumerate() {
        let Some(target) = target else { continue };
        let blend = if location == 0 {
            adds.then_some(ADDED)
        } else if location == cover {
            covers.then_some(COVERED)
        } else if location == accumulated {
            covers.then_some(SUMMED)
        } else {
            None
        };
        match blend.filter(|_| blendable(target.format)) {
            Some(blend) => target.blend = Some(blend),
            None => target.write_mask = wgpu::ColorWrites::empty(),
        }
    }
}

impl Machine {
    fn part_label(&self, part: u32, entry: &str) -> String {
        match self
            .declared
            .parts
            .get(part as usize)
            .and_then(|part| part.labels.first())
        {
            Some(prefix) => format!("{prefix} {entry}"),
            None => entry.to_owned(),
        }
    }

    fn pass_name(&self, pipeline: u32) -> &'static str {
        let Some(built) = self.pipelines.get(pipeline as usize) else {
            return "";
        };
        match built.decl.scope {
            Scoped::Scene(name) | Scoped::Outside(name) => name,
            Scoped::Pass(pass) => self
                .declared
                .passes
                .get(pass as usize)
                .map_or("", |declared| declared.group.unwrap_or(declared.name)),
        }
    }

    fn makes_records(&self, pipeline: u32) -> bool {
        let Some(built) = self.pipelines.get(pipeline as usize) else {
            return false;
        };
        match built.decl.scope {
            Scoped::Scene(_) | Scoped::Outside(_) => false,
            Scoped::Pass(pass) => self
                .declared
                .passes
                .get(pass as usize)
                .is_some_and(|declared| declared.makes_records),
        }
    }
}

pub(crate) struct Compiler<'r> {
    device: &'r wgpu::Device,
    machine: &'r mut Machine,
    parts: &'r [Option<Box<dyn Installed>>],
    pipelines: &'r mut Pipelines,
}

struct Shape<'f> {
    formats: &'f [Option<wgpu::TextureFormat>],
    depth: Option<(wgpu::TextureFormat, bool)>,
    main: bool,
    switches: Vec<(u32, bool)>,
    material: Vec<(&'static str, f64)>,
    styling: Option<crate::surface::MaterialVariant>,
    raster: engine_core::PassRaster,
    fragment: Option<engine_core::Entry>,
    faces: Option<crate::surface::FaceCull>,
}

type StageJob = (wgpu::ShaderModule, &'static str, Vec<(String, f64)>);

struct RenderJob {
    device: wgpu::Device,
    label: String,
    layout: wgpu::PipelineLayout,
    vertex: StageJob,
    buffers: Vec<Option<wgpu::VertexBufferLayout<'static>>>,
    primitive: wgpu::PrimitiveState,
    depth_stencil: Option<wgpu::DepthStencilState>,
    samples: u32,
    fragment: Option<StageJob>,
    targets: Vec<Option<wgpu::ColorTargetState>>,
    cache: Option<wgpu::PipelineCache>,
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
    fn make(self) -> Result<wgpu::RenderPipeline, wgpu::Error> {
        #[cfg(not(target_arch = "wasm32"))]
        let scope = crate::gpu_calls::validation_scope(&self.device);
        let vertex_constants = named(&self.vertex.2);
        let fragment_constants = self
            .fragment
            .as_ref()
            .map(|(_, _, constants)| named(constants));
        let made = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(&self.label),
                layout: Some(&self.layout),
                vertex: wgpu::VertexState {
                    module: &self.vertex.0,
                    entry_point: Some(self.vertex.1),
                    compilation_options: options(&vertex_constants),
                    buffers: &self.buffers,
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
            });
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(Some(error)) = crate::platform::ready(scope.pop()) {
            return Err(error);
        }
        Ok(made)
    }
}

impl ComputeJob {
    #[expect(
        clippy::disallowed_methods,
        reason = "the machine keys every pipeline, which the pipeline list records"
    )]
    fn make(self) -> Result<wgpu::ComputePipeline, wgpu::Error> {
        #[cfg(not(target_arch = "wasm32"))]
        let scope = crate::gpu_calls::validation_scope(&self.device);
        let constants = named(&self.constants);
        let made = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&self.label),
                layout: Some(&self.layout),
                module: &self.module,
                entry_point: Some(self.entry),
                compilation_options: options(&constants),
                cache: self.cache.as_ref(),
            });
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(Some(error)) = crate::platform::ready(scope.pop()) {
            return Err(error);
        }
        Ok(made)
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

fn note(pipelines: &mut Pipelines, compiled: Compiled, from_list: bool) {
    if from_list {
        pipelines.up_front += 1;
    } else if pipelines
        .listed
        .as_ref()
        .is_some_and(|listed| !listed.contains(&compiled))
    {
        pipelines.unlisted += 1;
    }
    pipelines.compiled.push(compiled);
}

impl Run {
    pub(crate) fn split(&mut self) -> (Compiler<'_>, &mut Assets) {
        (
            Compiler {
                device: &self.device,
                machine: &mut self.machine,
                parts: &self.parts,
                pipelines: &mut self.pipelines,
            },
            &mut self.assets,
        )
    }

    pub(crate) fn take_made(&mut self) -> u32 {
        self.split().0.collect(None);
        self.pipelines.making.pending()
    }

    pub(crate) fn wait_made(&mut self) {
        while let Some(first) = self.pipelines.making.first() {
            self.split().0.collect(Some(first));
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
        let programs: Vec<u64> = self
            .machine
            .pipelines
            .iter()
            .map(|built| built.program_key)
            .collect();
        pipelines
            .compiled
            .retain(|compiled| programs.contains(&compiled.program));
        pipelines.making.forget_refused();
        self.pipelines = pipelines;
    }

    pub(crate) fn compile(
        &mut self,
        pipeline: u32,
        targets: (
            &[Option<wgpu::TextureFormat>],
            Option<(wgpu::TextureFormat, bool)>,
        ),
        shown: (bool, bool),
        variant: Option<crate::surface::MaterialVariant>,
    ) -> Option<(u64, wgpu::RenderPipeline)> {
        self.split().0.compile(pipeline, targets, shown, variant)
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
        variant: Option<crate::surface::MaterialVariant>,
    ) -> Result<usize, engine_core::DropReason> {
        use engine_core::DropReason;
        let layout = self
            .recording
            .open()
            .ok_or(DropReason::NoRenderPass)?
            .layout;
        let bits = variant;
        let built = self
            .machine
            .pipelines
            .get(pipeline as usize)
            .ok_or(DropReason::UnknownPipeline)?;
        if built.skipped {
            return Err(DropReason::PipelineSkipped);
        }
        if let Some(at) = built
            .ready
            .iter()
            .position(|ready| ready.layout == layout && ready.bits == bits)
        {
            return Ok(at);
        }
        let program = built.program_key;
        let Some((key, made)) = self.compile(
            pipeline,
            (layout.formats(), layout.depth),
            (layout.plain, layout.main),
            variant,
        ) else {
            return Err(if self.pipelines.making.program_waits(program) {
                DropReason::PipelineCompiling
            } else {
                DropReason::PipelineRefused
            });
        };
        self.tallies.rebuilt.compiles += 1;
        let built = self
            .machine
            .pipelines
            .get_mut(pipeline as usize)
            .ok_or(DropReason::UnknownPipeline)?;
        built.ready.push(Ready {
            layout,
            bits,
            key,
            made,
        });
        Ok(built.ready.len() - 1)
    }

    pub(crate) fn set_switches(&mut self, switches: Vec<(u32, bool)>) {
        if self.pipelines.switches == switches {
            return;
        }
        self.pipelines.switches = switches;
        for built in &mut self.machine.pipelines {
            built.ready.clear();
            built.compute_ready = None;
        }
    }

    pub(super) fn compute_pipeline(
        &mut self,
        pipeline: u32,
        wait: bool,
    ) -> Option<wgpu::ComputePipeline> {
        self.split().0.collect(None);
        let built = self.machine.pipelines.get(pipeline as usize)?;
        let (Program::Compute(_), Some(stage), false) =
            (&built.decl.program, built.stages.first(), built.skipped)
        else {
            return None;
        };
        let switches: Vec<(u32, bool)> = self
            .pipelines
            .switches
            .iter()
            .filter(|(id, _)| stage.composed.switches.contains(id))
            .copied()
            .collect();
        let mut hasher = DefaultHasher::new();
        switches.hash(&mut hasher);
        let key = hasher.finish();
        let program = built.program_key;
        let made = (program, key);
        if computed_by_program(&self.machine.pipelines, program, key).is_none()
            && !self.pipelines.making.refused(made)
            && !self.pipelines.making.waits(made)
        {
            let job = ComputeJob {
                device: self.device.clone(),
                label: self.machine.part_label(built.decl.part, stage.entry.name()),
                layout: built.layout.clone(),
                module: stage.shader(&self.device, &mut self.machine.shaders),
                entry: stage.entry.name(),
                constants: switches
                    .iter()
                    .map(|(id, on)| (id.to_string(), f64::from(u8::from(*on))))
                    .collect(),
                cache: self.pipelines.cache.clone(),
            };
            let pass = self.machine.pass_name(pipeline);
            self.pipelines.making.submit(made, move || Finished {
                program,
                key,
                pass,
                noted: None,
                made: job.make().map(Created::Compute),
            });
        }
        self.split().0.collect(wait.then_some(made));
        let found = computed_by_program(&self.machine.pipelines, program, key)?;
        if let Some(built) = self.machine.pipelines.get_mut(pipeline as usize)
            && !built.computed.iter().any(|(known, _)| *known == key)
        {
            built.computed.push((key, found.clone()));
        }
        Some(found)
    }
}

impl Compiler<'_> {
    pub(crate) fn part<P: 'static>(&self) -> Option<&P> {
        self.parts
            .iter()
            .flatten()
            .find_map(|held| held.part().downcast_ref::<P>())
    }

    pub(crate) fn group_layout(&self, pipeline: u32, index: u32) -> Option<wgpu::BindGroupLayout> {
        self.machine
            .pipelines
            .get(pipeline as usize)?
            .groups
            .iter()
            .find(|group| group.index == index)
            .map(|group| group.layout.clone())
    }

    pub(crate) fn collect(&mut self, waited: Option<(u64, u64)>) {
        for finished in self.pipelines.making.take(waited) {
            let made = match finished.made {
                Ok(made) => made,
                Err(error) => {
                    self.pipelines
                        .making
                        .refuse((finished.program, finished.key));
                    self.pipelines.refused.push(error.to_string());
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
            crate::pipelines::count(finished.pass, 1);
            if let Some((compiled, from_list)) = finished.noted {
                note(self.pipelines, compiled, from_list);
            }
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

    pub(crate) fn compile(
        &mut self,
        pipeline: u32,
        targets: (
            &[Option<wgpu::TextureFormat>],
            Option<(wgpu::TextureFormat, bool)>,
        ),
        shown: (bool, bool),
        variant: Option<crate::surface::MaterialVariant>,
    ) -> Option<(u64, wgpu::RenderPipeline)> {
        self.compile_with(pipeline, targets, shown, variant, false)
    }

    pub(crate) fn compile_now(
        &mut self,
        pipeline: u32,
        targets: (
            &[Option<wgpu::TextureFormat>],
            Option<(wgpu::TextureFormat, bool)>,
        ),
        shown: (bool, bool),
        variant: Option<crate::surface::MaterialVariant>,
    ) -> Option<(u64, wgpu::RenderPipeline)> {
        self.compile_with(pipeline, targets, shown, variant, true)
    }

    fn compile_with(
        &mut self,
        pipeline: u32,
        (formats, depth): (
            &[Option<wgpu::TextureFormat>],
            Option<(wgpu::TextureFormat, bool)>,
        ),
        (plain, main): (bool, bool),
        variant: Option<crate::surface::MaterialVariant>,
        wait: bool,
    ) -> Option<(u64, wgpu::RenderPipeline)> {
        self.collect(None);
        let built = self.machine.pipelines.get(pipeline as usize)?;
        let Program::Render(desc) = &built.decl.program else {
            return None;
        };
        let switches = stage_switches(&self.pipelines.switches, &built.stages);
        let used = variant.map_or([false; 5], |variant| {
            variant.constants().map(|offered| {
                built
                    .stages
                    .iter()
                    .any(|stage| !stage.composed.constants(&[offered]).is_empty())
            })
        });
        let specialized = used.contains(&true);
        let styled = !plain && !desc.writes.is_empty();
        let faced = desc.faces != Faces::Declared;
        let bits = variant.and_then(|variant| variant.keyed(used, styled, faced));
        let fragment = match desc.faces {
            Faces::Declared | Faces::Material(FragmentUse::Always) => desc.fragment,
            Faces::Material(FragmentUse::Masked) => desc
                .fragment
                .filter(|_| bits.is_some_and(|bits| bits.draw().alpha_test)),
        };
        if built.skipped || (fragment.is_none() && formats.iter().any(Option::is_some)) {
            return None;
        }
        let faces = faced.then(|| bits.map_or(crate::surface::FaceCull::Back, |bits| bits.cull));
        let raster = raster_of(self.machine, pipeline);
        let mut hasher = DefaultHasher::new();
        (formats, depth, plain, main, &switches, bits).hash(&mut hasher);
        (fragment.is_some(), faces).hash(&mut hasher);
        (
            raster.bias,
            raster.slope.to_bits(),
            raster.clamp_near,
            raster.cull,
        )
            .hash(&mut hasher);
        let key = hasher.finish();
        let program = built.program_key;
        let made = (program, key);
        if self.made(pipeline, made).is_none()
            && !self.pipelines.making.refused(made)
            && !self.pipelines.making.waits(made)
        {
            let shape = Shape {
                formats,
                depth,
                main,
                switches,
                material: material_constants(bits.filter(|_| specialized), raster.clamp_near),
                styling: bits.filter(|_| styled),
                raster,
                fragment,
                faces,
            };
            let job = self.render_job(pipeline, shape)?;
            let noted = (!self.machine.makes_records(pipeline)).then(|| {
                (
                    Compiled {
                        program,
                        formats: formats.to_vec(),
                        depth,
                        plain,
                        main,
                        variant: bits,
                    },
                    self.pipelines.listing,
                )
            });
            let pass = self.machine.pass_name(pipeline);
            self.pipelines.making.submit(made, move || Finished {
                program,
                key,
                pass,
                noted,
                made: job.make().map(Created::Render),
            });
            self.collect(wait.then_some(made));
        } else if wait {
            self.collect(Some(made));
        }
        self.made(pipeline, made).map(|found| (key, found))
    }

    fn render_job(&mut self, pipeline: u32, shape: Shape<'_>) -> Option<RenderJob> {
        let machine = &mut *self.machine;
        let built = machine.pipelines.get(pipeline as usize)?;
        let Program::Render(desc) = &built.decl.program else {
            return None;
        };
        let mut constants =
            stage_constants(&built.stages, &shape.switches, &shape.material).into_iter();
        let fragment_stage = shape.fragment.and_then(|_| built.stages.get(1));
        let written = fragment_stage
            .zip(shape.fragment)
            .and_then(|(stage, entry)| fragment_locations(&stage.composed.text, entry.name()));
        let alpha = shape
            .styling
            .map(|styling| styling.draw().alpha)
            .filter(|alpha| {
                matches!(
                    alpha,
                    crate::surface::Alpha::Blend | crate::surface::Alpha::Additive
                )
            });
        let weighed = machine
            .declared
            .passes
            .get(built.home as usize)
            .filter(|_| shape.main)
            .and_then(|pass| {
                let at = |name| output_location(&machine.outputs, pass, name);
                Some((
                    at(engine_core::TargetName::TRANSPARENCY)?,
                    at(engine_core::TargetName::ACCUMULATED)?,
                ))
            });
        let mut targets = color_targets(desc, shape.formats, written.as_deref(), shape.main);
        let declared = desc.blend.is_some() || desc.targets.is_some();
        match (weighed, alpha) {
            (Some(layout), alpha) => {
                use crate::surface::Alpha;
                let weighing = match (alpha, desc.faces) {
                    (Some(Alpha::Additive), Faces::Declared | Faces::Material(_)) => {
                        Some(Weighing::Adds)
                    }
                    (None, Faces::Declared) => (!declared).then_some(Weighing::Chosen),
                    (
                        Some(Alpha::Opaque | Alpha::Mask | Alpha::Blend),
                        Faces::Declared | Faces::Material(_),
                    )
                    | (None, Faces::Material(_)) => Some(Weighing::Covers),
                };
                if let Some(weighing) = weighing {
                    weigh_targets(&mut targets, weighing, layout);
                }
            }
            (None, Some(alpha)) => blend_targets(&mut targets, alpha),
            (None, None) => {}
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
        Some(RenderJob {
            device: self.device.clone(),
            label: machine.part_label(built.decl.part, desc.vertex.name()),
            layout: built.layout.clone(),
            vertex: (vertex, desc.vertex.name(), vertex_constants),
            buffers: vertex_layouts(&desc.vertices),
            primitive: culled(faced(desc.primitive, shape.faces), shape.raster.cull),
            depth_stencil: depth_state(shape.depth, desc.depth, shape.styling, shape.raster),
            samples: desc.samples,
            fragment: fragment.map(|(module, entry)| (module, entry, fragment_constants)),
            targets,
            cache: self.pipelines.cache.clone(),
        })
    }
}

type PassFormats = (
    Vec<Option<wgpu::TextureFormat>>,
    Option<(wgpu::TextureFormat, bool)>,
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
        if pass_layer(pass).is_some() {
            let set = pass_set(&self.machine.outputs, pass);
            let main = lit_set(&set);
            shown = (!set.is_empty() && !main, main);
            for (_, format, _) in set {
                formats.push(Some(format));
            }
            depth = Some((DEPTH_FORMAT, true));
        }
        if crate::shared::draws_canvas(pass) {
            formats.insert(0, Some(self.canvas.surface));
        }
        for &view in &pass.targets {
            formats.push(target_format(final_target(view), &self.machine.outputs));
        }
        for &(texture, _) in &pass.colors {
            formats.push(
                self.machine
                    .declared
                    .textures
                    .get(texture as usize)
                    .map(|texture| texture.format),
            );
        }
        if depth.is_none()
            && let Some(&(texture, access)) = pass.depths.first()
        {
            let view = pass.view_depth(texture, access);
            depth = self
                .machine
                .declared
                .textures
                .get(texture as usize)
                .map(|texture| (texture.format, view));
        }
        (formats, depth, shown)
    }

    fn part_reached(&self, part: u32) -> bool {
        self.reached
            .parts
            .get(part as usize)
            .copied()
            .unwrap_or(true)
    }

    pub(crate) fn warm_reached(&mut self, world: &World) {
        let states = world.get_resource::<engine_registry::BrickStates>();
        let in_play: Vec<engine_registry::GateId> = self
            .installs
            .gates
            .iter()
            .zip(&self.machine.declared.parts)
            .filter(|(_, part)| !crate::DEV && part.turns_on_in_play)
            .filter_map(|(gate, _)| *gate)
            .collect();
        let side = |forced: bool| -> Vec<(u32, bool)> {
            self.machine
                .switches
                .iter()
                .zip(&self.pipelines.switches)
                .map(|(&(id, gate), &(_, on))| {
                    (id, if in_play.contains(&gate) { forced } else { on })
                })
                .collect()
        };
        self.reached = Reached {
            generation: Some(states.map_or(0, engine_registry::BrickStates::generation)),
            parts: self
                .installs
                .gates
                .iter()
                .map(|gate| {
                    gate.is_none_or(|gate| {
                        in_play.contains(&gate) || states.is_none_or(|states| states.on(gate))
                    })
                })
                .collect(),
            sides: [side(true), side(false)],
        };
        self.at_reached_switches(|run| {
            run.compile_listed();
            run.warm();
        });
    }

    pub(super) fn at_reached_switches(&mut self, work: impl Fn(&mut Self)) {
        let [on, off] = self.reached.sides.clone();
        let differ = on != off;
        let real = std::mem::replace(&mut self.pipelines.switches, on);
        work(self);
        if differ {
            self.pipelines.switches = off;
            work(self);
        }
        self.pipelines.switches = real;
    }

    pub(super) fn compile_listed(&mut self) {
        let Some(keys) = self.pipelines.listed.take() else {
            return;
        };
        self.pipelines.listing = true;
        for key in &keys {
            let found = self
                .machine
                .pipelines
                .iter()
                .position(|built| built.program_key == key.program)
                .filter(|&at| self.part_reached(self.machine.pipelines[at].decl.part));
            if let Some(pipeline) = found {
                let _ = self.compile(
                    pipeline as u32,
                    (&key.formats, key.depth),
                    (key.plain, key.main),
                    specialized(key),
                );
            }
        }
        self.pipelines.listing = false;
        self.pipelines.listed = Some(keys);
    }

    pub(crate) fn make_all_shaders(&mut self) {
        let machine = &mut self.machine;
        for built in &machine.pipelines {
            for stage in &built.stages {
                stage.shader(&self.device, &mut machine.shaders);
            }
        }
    }

    pub(crate) fn warm(&mut self) {
        for pipeline in 0..self.machine.pipelines.len() {
            if !self.part_reached(self.machine.pipelines[pipeline].decl.part) {
                continue;
            }
            if matches!(
                self.machine.pipelines[pipeline].decl.program,
                Program::Compute(_)
            ) {
                let _ = self.compute_pipeline(pipeline as u32, false);
                continue;
            }
            for pass in self.scope_passes(pipeline) {
                let Some(declared) = self.machine.declared.passes.get(pass as usize) else {
                    continue;
                };
                if !self.part_reached(declared.part) {
                    continue;
                }
                let (formats, depth, shown) = self.pass_formats(pass);
                if self.writes_fit(pipeline, &formats) {
                    let _ = self.compile(pipeline as u32, (&formats, depth), shown, None);
                }
                for layered in self.layered_formats(pipeline, pass, formats.len()) {
                    if self.writes_fit(pipeline, &layered) {
                        let _ = self.compile(pipeline as u32, (&layered, None), shown, None);
                    }
                }
            }
        }
    }

    fn writes_fit(&self, pipeline: usize, formats: &[Option<wgpu::TextureFormat>]) -> bool {
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
            .and_then(|(stage, entry)| fragment_kinds(&stage.composed.text, entry.name()))
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

    pub(super) fn layered_formats(
        &self,
        pipeline: usize,
        pass: u32,
        declared: usize,
    ) -> Vec<Vec<Option<wgpu::TextureFormat>>> {
        let (Some(built), Some(decl)) = (
            self.machine.pipelines.get(pipeline),
            self.machine.declared.passes.get(pass as usize),
        ) else {
            return Vec::new();
        };
        let Program::Render(desc) = &built.decl.program else {
            return Vec::new();
        };
        if pass_layer(decl).is_some() {
            return Vec::new();
        }
        let written = built
            .stages
            .get(1)
            .zip(desc.fragment)
            .and_then(|(stage, entry)| fragment_locations(&stage.composed.text, entry.name()));
        let count = written
            .and_then(|written| written.iter().max().copied())
            .map_or(0, |last| last as usize + 1);
        if count <= declared {
            return Vec::new();
        }
        let mut formats: Vec<wgpu::TextureFormat> = Vec::new();
        for &(texture, _) in &decl.colors {
            if let Some(found) = self.machine.declared.textures.get(texture as usize)
                && !formats.contains(&found.format)
            {
                formats.push(found.format);
            }
        }
        formats
            .into_iter()
            .map(|format| vec![Some(format); count])
            .collect()
    }
}

fn blendable(format: wgpu::TextureFormat) -> bool {
    format
        .sample_type(None, None)
        .is_some_and(|sample| matches!(sample, wgpu::TextureSampleType::Float { filterable: true }))
}

fn fragment_outputs<'a>(text: &'a str, entry: &str) -> Option<&'a str> {
    let at = text.find(&format!("fn {entry}("))?;
    let rest = &text[at..];
    let brace = rest.find('{')?;
    let Some(arrow) = rest[..brace].find("->") else {
        return Some("");
    };
    let returns = rest[arrow + 2..brace].trim();
    if returns.starts_with('@') {
        return Some(returns);
    }
    let name: String = returns
        .chars()
        .take_while(|character| character.is_ascii_alphanumeric() || *character == '_')
        .collect();
    let start = text.find(&format!("struct {name} {{"))?;
    let body = &text[start..];
    Some(&body[..body.find('}')?])
}

fn fragment_locations(text: &str, entry: &str) -> Option<Vec<u32>> {
    fragment_outputs(text, entry).map(locations)
}

fn fragment_kinds(text: &str, entry: &str) -> Option<Vec<(u32, wgpu::TextureSampleType)>> {
    let outputs = fragment_outputs(text, entry)?;
    let kinds = outputs.match_indices("@location(").map(|(at, _)| {
        let field = &outputs[at..];
        let field = &field[..field.find(',').unwrap_or(field.len())];
        if field.contains("u32") {
            wgpu::TextureSampleType::Uint
        } else if field.contains("i32") {
            wgpu::TextureSampleType::Sint
        } else {
            wgpu::TextureSampleType::Float { filterable: true }
        }
    });
    Some(locations(outputs).into_iter().zip(kinds).collect())
}

fn locations(text: &str) -> Vec<u32> {
    text.match_indices("@location(")
        .filter_map(|(at, found)| {
            let after = &text[at + found.len()..];
            after[..after.find(')')?].trim().parse().ok()
        })
        .collect()
}

const CLAMP_NEAR: &str = "PASS_CLAMP_NEAR";

fn material_constants(
    keyed: Option<crate::surface::MaterialVariant>,
    clamp_near: bool,
) -> Vec<(&'static str, f64)> {
    let mut material = keyed.map_or_else(Vec::new, |keyed| keyed.constants().to_vec());
    if clamp_near {
        material.push((CLAMP_NEAR, 1.0));
    }
    material
}

fn depth_state(
    depth: Option<(wgpu::TextureFormat, bool)>,
    used: DepthUse,
    styling: Option<crate::surface::MaterialVariant>,
    raster: engine_core::PassRaster,
) -> Option<wgpu::DepthStencilState> {
    depth.map(|(format, view)| {
        let mut state = depth_stencil_state(format, view, used);
        if let Some(styling) = styling {
            styling.restyle(&mut state);
        }
        state.bias.constant += raster.bias;
        state.bias.slope_scale += raster.slope;
        state
    })
}

fn raster_of(machine: &Machine, pipeline: u32) -> engine_core::PassRaster {
    let scene = machine
        .pipelines
        .get(pipeline as usize)
        .and_then(|built| match built.decl.scope {
            Scoped::Scene(name) => Some(name),
            Scoped::Pass(_) | Scoped::Outside(_) => None,
        });
    scene
        .and_then(|name| {
            machine
                .declared
                .passes
                .iter()
                .filter_map(|pass| pass.scene)
                .find(|scene| scene.name == name)
        })
        .map_or(engine_core::PassRaster::PLAIN, |scene| scene.raster)
}

fn faced(
    primitive: wgpu::PrimitiveState,
    faces: Option<crate::surface::FaceCull>,
) -> wgpu::PrimitiveState {
    match faces {
        Some(cull) => wgpu::PrimitiveState {
            cull_mode: cull.face(),
            ..primitive
        },
        None => primitive,
    }
}

fn culled(primitive: wgpu::PrimitiveState, cull: engine_core::PassCull) -> wgpu::PrimitiveState {
    match cull {
        engine_core::PassCull::Material => primitive,
        engine_core::PassCull::None => wgpu::PrimitiveState {
            cull_mode: None,
            ..primitive
        },
        engine_core::PassCull::Front => wgpu::PrimitiveState {
            cull_mode: Some(wgpu::Face::Front),
            ..primitive
        },
    }
}
