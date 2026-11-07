use super::*;
use crate::install::RenderDecl;
use crate::machine::{BuiltPipeline, Stage};

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

fn named_constants(constants: &[Vec<(String, f64)>]) -> Vec<Vec<(&str, f64)>> {
    constants
        .iter()
        .map(|values| {
            values
                .iter()
                .map(|(name, value)| (name.as_str(), *value))
                .collect()
        })
        .collect()
}

fn vertex_layouts(vertices: &[VertexLayout]) -> Vec<Option<wgpu::VertexBufferLayout<'static>>> {
    let mut layouts = Vec::new();
    let mut meshes = 0;
    for layout in vertices {
        match layout {
            VertexLayout::Mesh => {
                layouts.push(if meshes == 0 {
                    crate::mesh_arena::vertex_layout()
                } else {
                    crate::mesh_arena::extra_layout()
                });
                meshes += 1;
            }
            VertexLayout::Buffer(layout) => layouts.push(Some(layout.clone())),
        }
    }
    layouts
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
    (debug, main): (bool, bool),
) -> Vec<Option<wgpu::ColorTargetState>> {
    let add = wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    };
    formats
        .iter()
        .enumerate()
        .map(|(location, format)| {
            let format = (*format)?;
            let (blend, mask) = if debug {
                let blend = (location == DEBUG_COUNT).then_some(wgpu::BlendState {
                    color: add,
                    alpha: add,
                });
                (blend, wgpu::ColorWrites::ALL)
            } else {
                match &desc.targets {
                    Some(targets) => targets
                        .get(location)
                        .copied()
                        .unwrap_or((None, wgpu::ColorWrites::empty())),
                    None => (
                        (!main || location == 0).then_some(desc.blend).flatten(),
                        desc.writes,
                    ),
                }
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

fn blend_targets(
    targets: &mut [Option<wgpu::ColorTargetState>],
    alpha: crate::surface::Alpha,
    cover: Option<usize>,
) {
    let additive = alpha == crate::surface::Alpha::Additive;
    for (location, target) in targets.iter_mut().enumerate() {
        let Some(target) = target else { continue };
        if location == 0 {
            target.blend = blendable(target.format).then_some(if additive { ADDED } else { OVER });
        } else if cover == Some(location) && !additive {
            target.blend = blendable(target.format).then_some(COVERED);
        } else {
            target.write_mask = wgpu::ColorWrites::empty();
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
}

pub(crate) struct Compiler<'r> {
    device: &'r wgpu::Device,
    machine: &'r mut Machine,
    parts: &'r [Option<Box<dyn Installed>>],
    switches: &'r [(u32, bool)],
    compiled: &'r mut Vec<Compiled>,
    listed: Option<&'r [Compiled]>,
    unlisted: &'r mut u32,
}

impl Run {
    pub(crate) fn split(&mut self) -> (Compiler<'_>, &mut Assets) {
        (
            Compiler {
                device: &self.device,
                machine: &mut self.machine,
                parts: &self.parts,
                switches: &self.pipelines.switches,
                compiled: &mut self.pipelines.compiled,
                listed: self.pipelines.listed.as_deref(),
                unlisted: &mut self.pipelines.unlisted,
            },
            &mut self.assets,
        )
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
    ) -> Option<usize> {
        let layout = self.recording.open()?.layout;
        let bits = variant;
        let built = self.machine.pipelines.get(pipeline as usize)?;
        if built.skipped {
            return None;
        }
        if let Some(at) = built
            .ready
            .iter()
            .position(|ready| ready.layout == layout && ready.bits == bits)
        {
            return Some(at);
        }
        let (key, made) = self.compile(
            pipeline,
            (layout.formats(), layout.depth),
            (layout.debug, layout.main),
            variant,
        )?;
        self.tallies.rebuilt.compiles += 1;
        let built = self.machine.pipelines.get_mut(pipeline as usize)?;
        built.ready.push(Ready {
            layout,
            bits,
            key,
            made,
        });
        Some(built.ready.len() - 1)
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

    #[expect(
        clippy::disallowed_methods,
        reason = "the machine keys every pipeline, which the pipeline list records"
    )]
    pub(super) fn compute_pipeline(&mut self, pipeline: u32) -> Option<wgpu::ComputePipeline> {
        let built = self.machine.pipelines.get(pipeline as usize)?;
        let (Program::Compute(_), Some(stage), false) =
            (&built.decl.program, built.stages.first(), built.skipped)
        else {
            return None;
        };
        let switches: Vec<(u32, bool)> = self
            .pipelines.switches
            .iter()
            .filter(|(id, _)| stage.composed.switches.contains(id))
            .copied()
            .collect();
        let mut hasher = DefaultHasher::new();
        switches.hash(&mut hasher);
        let key = hasher.finish();
        if let Some((_, made)) = built.computed.iter().find(|(known, _)| *known == key) {
            return Some(made.clone());
        }
        let program_key = built.program_key;
        let shared = self
            .machine
            .pipelines
            .iter()
            .filter(|other| other.program_key == program_key)
            .find_map(|other| {
                other
                    .computed
                    .iter()
                    .find(|(known, _)| *known == key)
                    .map(|(_, made)| made.clone())
            });
        if let Some(made) = shared {
            if let Some(built) = self.machine.pipelines.get_mut(pipeline as usize) {
                built.computed.push((key, made.clone()));
            }
            return Some(made);
        }
        let constants: Vec<(String, f64)> = switches
            .iter()
            .map(|(id, on)| (id.to_string(), f64::from(u8::from(*on))))
            .collect();
        let named: Vec<(&str, f64)> = constants
            .iter()
            .map(|(name, value)| (name.as_str(), *value))
            .collect();
        let module = stage.shader(&self.device, &mut self.machine.shaders);
        let label = self.machine.part_label(built.decl.part, stage.entry.name());
        let made = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&label),
                layout: Some(&built.layout),
                module: &module,
                entry_point: Some(stage.entry.name()),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &named,
                    zero_initialize_workgroup_memory: true,
                },
                cache: None,
            });
        crate::pipelines::count(self.machine.pass_name(pipeline), 1);
        if let Some(built) = self.machine.pipelines.get_mut(pipeline as usize) {
            built.computed.push((key, made.clone()));
        }
        Some(made)
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

    fn note_compiled(&mut self, compiled: Compiled) {
        if self
            .listed
            .is_some_and(|listed| !listed.contains(&compiled))
        {
            *self.unlisted += 1;
        }
        self.compiled.push(compiled);
    }

    #[expect(
        clippy::disallowed_methods,
        reason = "the machine keys every pipeline, which the pipeline list records"
    )]
    pub(crate) fn compile(
        &mut self,
        pipeline: u32,
        (formats, depth): (
            &[Option<wgpu::TextureFormat>],
            Option<(wgpu::TextureFormat, bool)>,
        ),
        (debug, main): (bool, bool),
        variant: Option<crate::surface::MaterialVariant>,
    ) -> Option<(u64, wgpu::RenderPipeline)> {
        let built = self.machine.pipelines.get(pipeline as usize)?;
        let Program::Render(desc) = &built.decl.program else {
            return None;
        };
        if built.skipped || (debug && desc.debug.is_none()) {
            return None;
        }
        let switches = stage_switches(self.switches, &built.stages);
        let used = variant.map_or([false; 5], |variant| {
            variant.constants().map(|offered| {
                built
                    .stages
                    .iter()
                    .any(|stage| !stage.composed.constants(&[offered]).is_empty())
            })
        });
        let specialized = used.contains(&true);
        let styled = !debug && !desc.writes.is_empty();
        let bits = variant.and_then(|variant| variant.keyed(used, styled));
        let raster = raster_of(self.machine, pipeline);
        let material = material_constants(bits.filter(|_| specialized), raster.clamp_near);
        let mut hasher = DefaultHasher::new();
        (formats, depth, debug, main, &switches, bits).hash(&mut hasher);
        (raster.bias, raster.slope.to_bits(), raster.clamp_near, raster.cull).hash(&mut hasher);
        let key = hasher.finish();
        if let Some((_, made)) = built.made.iter().find(|(known, _)| *known == key) {
            return Some((key, made.clone()));
        }
        let shared = made_by_program(&self.machine.pipelines, built.program_key, key);
        if let Some(made) = shared {
            if let Some(built) = self.machine.pipelines.get_mut(pipeline as usize) {
                built.made.push((key, made.clone()));
            }
            return Some((key, made));
        }
        let constants = stage_constants(&built.stages, &switches, &material);
        let named = named_constants(&constants);
        let options = |stage: usize| wgpu::PipelineCompilationOptions {
            constants: named.get(stage).map_or(&[][..], Vec::as_slice),
            zero_initialize_workgroup_memory: true,
        };
        let fragment_at = if debug {
            1 + usize::from(desc.fragment.is_some())
        } else {
            1
        };
        let fragment_entry = if debug { desc.debug } else { desc.fragment };
        let layouts = vertex_layouts(&desc.vertices);
        let styling = bits.filter(|_| styled);
        let depth_stencil = depth_state(depth, desc.depth, styling, raster);
        let fragment_stage = fragment_entry.and_then(|_| built.stages.get(fragment_at));
        let written = fragment_stage
            .zip(fragment_entry)
            .and_then(|(stage, entry)| fragment_locations(&stage.composed.text, entry.name()));
        let blended = styling
            .map(|styling| styling.draw().alpha)
            .filter(|alpha| {
                matches!(
                    alpha,
                    crate::surface::Alpha::Blend | crate::surface::Alpha::Additive
                )
            })
            .map(|alpha| (alpha, crate::machine::cover_location(&self.machine.outputs)));
        let mut targets = color_targets(desc, formats, written.as_deref(), (debug, main));
        if let Some((alpha, cover)) = blended {
            blend_targets(&mut targets, alpha, cover.filter(|_| main));
        }
        let vertex_module = built
            .stages
            .first()?
            .shader(self.device, &mut self.machine.shaders);
        let fragment_module =
            fragment_stage.map(|stage| stage.shader(self.device, &mut self.machine.shaders));
        let label = self.machine.part_label(built.decl.part, desc.vertex.name());
        let made =
            self.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(&label),
                    layout: Some(&built.layout),
                    vertex: wgpu::VertexState {
                        module: &vertex_module,
                        entry_point: Some(desc.vertex.name()),
                        compilation_options: options(0),
                        buffers: &layouts,
                    },
                    primitive: culled(desc.primitive, raster.cull),
                    depth_stencil,
                    multisample: wgpu::MultisampleState {
                        count: desc.samples,
                        ..Default::default()
                    },
                    fragment: fragment_module.as_ref().zip(fragment_entry).map(
                        |(module, entry)| wgpu::FragmentState {
                            module,
                            entry_point: Some(entry.name()),
                            compilation_options: options(fragment_at),
                            targets: &targets,
                        },
                    ),
                    multiview_mask: None,
                    cache: None,
                });
        let program = built.program_key;
        if let Some(built) = self.machine.pipelines.get_mut(pipeline as usize) {
            built.made.push((key, made.clone()));
        }
        crate::pipelines::count(self.machine.pass_name(pipeline), 1);
        self.note_compiled(Compiled {
            program,
            formats: formats.to_vec(),
            depth,
            debug,
            main,
            variant: bits,
        });
        Some((key, made))
    }
}

impl Run {
    pub(super) fn pass_formats(
        &self,
        index: u32,
    ) -> (
        Vec<Option<wgpu::TextureFormat>>,
        Option<(wgpu::TextureFormat, bool)>,
        bool,
    ) {
        let Some(pass) = self.machine.declared.passes.get(index as usize) else {
            return (Vec::new(), None, false);
        };
        let mut formats: Vec<Option<wgpu::TextureFormat>> = Vec::new();
        let mut depth = None;
        let mut main = false;
        if let Some(layer) = pass_layer(pass) {
            if !own_colors_only(pass) {
                main = true;
                for (_, format, _, written) in set_formats(&self.machine.outputs, layer) {
                    formats.push(written.then_some(format));
                }
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
            && let Some(&(texture, _)) = pass.depths.first()
        {
            depth = self
                .machine
                .declared
                .textures
                .get(texture as usize)
                .map(|texture| (texture.format, false));
        }
        (formats, depth, main)
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
            .installs.gates
            .iter()
            .zip(&self.machine.declared.parts)
            .filter(|(_, part)| part.turns_on_in_play)
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
                .installs.gates
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
            run.warm(false);
            run.compile_listed();
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
        let before = self.pipelines.compiled.len();
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
                    (key.debug, key.main),
                    specialized(key),
                );
            }
        }
        self.pipelines.up_front += (self.pipelines.compiled.len() - before) as u32;
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

    pub(crate) fn warm(&mut self, debug: bool) {
        for pipeline in 0..self.machine.pipelines.len() {
            if !self.part_reached(self.machine.pipelines[pipeline].decl.part) {
                continue;
            }
            if matches!(
                self.machine.pipelines[pipeline].decl.program,
                Program::Compute(_)
            ) {
                let _ = self.compute_pipeline(pipeline as u32);
                continue;
            }
            for pass in self.scope_passes(pipeline) {
                let Some(declared) = self.machine.declared.passes.get(pass as usize) else {
                    continue;
                };
                let (lazy, shares) = (declared.name == crate::shared::DEBUG_VIEW, shared(declared));
                if (lazy && !debug) || !self.part_reached(declared.part) {
                    continue;
                }
                let (formats, depth, main) = self.pass_formats(pass);
                if self.writes_fit(pipeline, &formats, false) {
                    let _ = self.compile(pipeline as u32, (&formats, depth), (false, main), None);
                }
                for layered in self.layered_formats(pipeline, pass, formats.len()) {
                    if self.writes_fit(pipeline, &layered, false) {
                        let _ =
                            self.compile(pipeline as u32, (&layered, None), (false, main), None);
                    }
                }
                let shown: Vec<Option<wgpu::TextureFormat>> =
                    DEBUG_FORMATS.iter().copied().map(Some).collect();
                if debug && shares && self.writes_fit(pipeline, &shown, true) {
                    let _ = self.compile(
                        pipeline as u32,
                        (&shown, Some((DEPTH_FORMAT, true))),
                        (true, false),
                        None,
                    );
                }
            }
        }
    }

    fn writes_fit(
        &self,
        pipeline: usize,
        formats: &[Option<wgpu::TextureFormat>],
        debug: bool,
    ) -> bool {
        let Some(built) = self.machine.pipelines.get(pipeline) else {
            return false;
        };
        let Program::Render(desc) = &built.decl.program else {
            return true;
        };
        let (at, entry) = if debug {
            (1 + usize::from(desc.fragment.is_some()), desc.debug)
        } else {
            (1, desc.fragment)
        };
        let Some(kinds) = built
            .stages
            .get(at)
            .zip(entry)
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
    let (arrow, brace) = (rest.find("->")?, rest.find('{')?);
    if arrow > brace {
        return None;
    }
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
