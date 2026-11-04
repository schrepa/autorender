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
                wgpu::CompareFunction::Greater
            } else {
                wgpu::CompareFunction::Less
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

impl Run {
    fn part_label(&self, part: u32, entry: &str) -> String {
        match self
            .machine
            .declared
            .parts
            .get(part as usize)
            .and_then(|part| part.labels.first())
        {
            Some(prefix) => format!("{prefix} {entry}"),
            None => entry.to_owned(),
        }
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
        variant: Option<crate::materials::Variant>,
    ) -> Option<usize> {
        let layout = self.open.as_ref()?.layout;
        let bits = variant.map(crate::materials::Variant::bits);
        let built = self.machine.pipelines.get(pipeline as usize)?;
        if let Some(at) = built
            .ready
            .iter()
            .position(|ready| ready.layout == layout && ready.bits == bits)
        {
            return Some(at);
        }
        let specialized = variant.map(|variant| (variant.bits(), variant.constants()));
        let (key, made) = self.compile(
            pipeline,
            (layout.formats(), layout.depth),
            (layout.debug, layout.main),
            specialized,
        )?;
        self.rebuilt.compiles += 1;
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
        if self.switches == switches {
            return;
        }
        self.switches = switches;
        for built in &mut self.machine.pipelines {
            built.ready.clear();
            built.compute_ready = None;
        }
    }

    fn note_compiled(&mut self, compiled: Compiled) {
        if self
            .listed
            .as_ref()
            .is_some_and(|listed| !listed.contains(&compiled))
        {
            self.unlisted += 1;
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
        variant: Option<Specialized>,
    ) -> Option<(u64, wgpu::RenderPipeline)> {
        let built = self.machine.pipelines.get(pipeline as usize)?;
        let Program::Render(desc) = &built.decl.program else {
            return None;
        };
        if debug && desc.debug.is_none() {
            return None;
        }
        let switches = stage_switches(&self.switches, &built.stages);
        let material: Vec<(&str, f64)> =
            variant.map_or_else(Vec::new, |(_, constants)| constants.to_vec());
        let specialized = built
            .stages
            .iter()
            .any(|stage| !stage.composed.constants(&material).is_empty());
        let bits = variant.filter(|_| specialized).map(|(bits, _)| bits);
        let mut hasher = DefaultHasher::new();
        formats.hash(&mut hasher);
        depth.hash(&mut hasher);
        debug.hash(&mut hasher);
        main.hash(&mut hasher);
        switches.hash(&mut hasher);
        bits.hash(&mut hasher);
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
        let named: Vec<Vec<(&str, f64)>> = constants
            .iter()
            .map(|values| {
                values
                    .iter()
                    .map(|(name, value)| (name.as_str(), *value))
                    .collect()
            })
            .collect();
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
        let depth_stencil =
            depth.map(|(format, view)| depth_stencil_state(format, view, desc.depth));
        let fragment_stage = fragment_entry.and_then(|_| built.stages.get(fragment_at));
        let written = fragment_stage
            .zip(fragment_entry)
            .and_then(|(stage, entry)| fragment_locations(&stage.composed.text, entry.name()));
        let targets = color_targets(desc, formats, written.as_deref(), (debug, main));
        let vertex_module = built
            .stages
            .first()?
            .shader(&self.device, &mut self.machine.shaders);
        let fragment_module =
            fragment_stage.map(|stage| stage.shader(&self.device, &mut self.machine.shaders));
        let label = self.part_label(built.decl.part, desc.vertex.name());
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
                    primitive: desc.primitive,
                    depth_stencil,
                    multisample: wgpu::MultisampleState::default(),
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
        if let Some(built) = self.machine.pipelines.get_mut(pipeline as usize) {
            built.made.push((key, made.clone()));
        }
        crate::pipelines::count(1);
        self.note_compiled(Compiled {
            pipeline,
            formats: formats.to_vec(),
            depth,
            debug,
            main,
            variant: bits,
            constants: if bits.is_some() {
                material
                    .iter()
                    .map(|&(name, value)| (name.to_owned(), value))
                    .collect()
            } else {
                Vec::new()
            },
        });
        Some((key, made))
    }

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
                for (_, format, _, written) in set_formats(&self.machine.outputs, Some(layer)) {
                    formats.push(written.then_some(format));
                }
            }
            depth = Some((DEPTH_FORMAT, true));
        }
        if crate::shared::draws_canvas(pass.name) {
            formats.insert(0, Some(self.surface));
        }
        if pass.name == crate::shared::COMPOSITE {
            for (_, format, _, written) in
                set_formats(&self.machine.outputs, None).into_iter().skip(1)
            {
                if written {
                    formats.push(Some(format));
                }
            }
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

    #[expect(
        clippy::disallowed_methods,
        reason = "the machine keys every pipeline, which the pipeline list records"
    )]
    pub(super) fn compute_pipeline(&mut self, pipeline: u32) -> Option<wgpu::ComputePipeline> {
        let built = self.machine.pipelines.get(pipeline as usize)?;
        let (Program::Compute(_), Some(stage)) = (&built.decl.program, built.stages.first()) else {
            return None;
        };
        let switches: Vec<(u32, bool)> = self
            .switches
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
        let label = self.part_label(built.decl.part, stage.entry.name());
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
        crate::pipelines::count(1);
        if let Some(built) = self.machine.pipelines.get_mut(pipeline as usize) {
            built.computed.push((key, made.clone()));
        }
        Some(made)
    }

    fn part_reached(&self, part: u32) -> bool {
        self.reached
            .parts
            .get(part as usize)
            .copied()
            .unwrap_or(true)
    }

    pub(crate) fn warm_reached(&mut self, world: &World) {
        let gates = world.get_resource::<engine_registry::Gates>();
        let in_play: Vec<engine_registry::GateId> = self
            .gates
            .iter()
            .zip(&self.machine.declared.parts)
            .filter(|(_, part)| part.turns_on_in_play)
            .filter_map(|(gate, _)| *gate)
            .collect();
        let side = |forced: bool| -> Vec<(u32, bool)> {
            self.machine
                .switches
                .iter()
                .zip(&self.switches)
                .map(|(&(id, gate), &(_, on))| {
                    (id, if in_play.contains(&gate) { forced } else { on })
                })
                .collect()
        };
        self.reached = Reached {
            generation: Some(gates.map_or(0, engine_registry::Gates::generation)),
            parts: self
                .gates
                .iter()
                .map(|gate| {
                    gate.is_none_or(|gate| {
                        in_play.contains(&gate) || gates.is_none_or(|gates| gates.on(gate))
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
        let real = std::mem::replace(&mut self.switches, on);
        work(self);
        if differ {
            self.switches = off;
            work(self);
        }
        self.switches = real;
    }

    pub(crate) fn release_off_parts(&mut self, world: &World) {
        let gates = world.get_resource::<engine_registry::Gates>();
        for part in 0..self.gates.len() {
            let on = self.gates[part].is_none_or(|gate| gates.is_some_and(|gates| gates.on(gate)));
            if !on {
                self.live[part] = false;
                self.release_part(part as u32);
            }
        }
    }

    pub(super) fn compile_listed(&mut self) {
        let Some(keys) = self.listed.take() else {
            return;
        };
        let before = self.compiled.len();
        for key in &keys {
            let reached = self
                .machine
                .pipelines
                .get(key.pipeline as usize)
                .is_some_and(|built| self.part_reached(built.decl.part));
            if reached {
                let _ = self.compile(
                    key.pipeline,
                    (&key.formats, key.depth),
                    (key.debug, key.main),
                    specialized(key),
                );
            }
        }
        self.up_front += (self.compiled.len() - before) as u32;
        self.listed = Some(keys);
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
