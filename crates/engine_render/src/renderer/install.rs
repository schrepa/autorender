use bevy_ecs::world::World;
use engine_core::Error;
use engine_registry::{BrickId, BrickStates, Registry};

use super::*;
use crate::install::{InstallCore, PartInfo, Reinstall, Scoped};
use crate::linker::{CORE_PACKAGE, Linker, Part};
use crate::machine::Machine;
use crate::registration::{Generated, Installed, KindDraw, KindMember, PartRecord};

struct Listed {
    record: PartRecord,
    package: &'static str,
    brick: Option<BrickId>,
}

pub(crate) struct Built {
    pub(crate) run_parts: Vec<Option<Box<dyn Installed>>>,
    pub(crate) gates: Vec<Option<BrickId>>,
    pub(crate) pass_gates: Vec<(u32, BrickId)>,
    pub(crate) reinstalls: Vec<Reinstall>,
    pub(crate) draws: Vec<KindMember>,
    pub(crate) machine: Machine,
    pub(crate) passes: Vec<&'static str>,
    pub(crate) material: crate::surface::MaterialLayout,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Installation {
    pub modules: Vec<String>,
    pub texts: Vec<String>,
    pub stages: Vec<(&'static str, Vec<&'static str>, String)>,
}

fn stage_texts(machine: &Machine) -> Vec<(&'static str, Vec<&'static str>, String)> {
    let mut seen = std::collections::BTreeMap::new();
    let mut stages: Vec<(&'static str, Vec<&'static str>, String)> = Vec::new();
    for built in &machine.pipelines {
        let pass = match built.decl.scope {
            Scoped::Scene(name) | Scoped::Outside(name) => name,
            Scoped::Pass(pass) => machine
                .declared
                .passes
                .get(pass as usize)
                .map_or("", |declared| declared.group.unwrap_or(declared.name)),
        };
        for stage in &built.stages {
            let at = *seen.entry(stage.text_key()).or_insert_with(|| {
                stages.push((stage.entry.name(), Vec::new(), stage.composed.text.clone()));
                stages.len() - 1
            });
            if !stages[at].1.contains(&pass) {
                stages[at].1.push(pass);
            }
        }
    }
    stages
}

fn listing(built: &mut Built) -> Installation {
    let modules = built.machine.linker.paths();
    let texts = modules
        .iter()
        .map(|path| match built.machine.linker.compose_module(path) {
            Ok(composed) => composed.text.clone(),
            Err(error) => error.to_string(),
        })
        .collect();
    Installation {
        modules,
        texts,
        stages: Vec::new(),
    }
}

fn core_part<P: crate::GpuPart>() -> PartRecord {
    PartRecord::of::<P>()
}

fn material_layout(
    registry: &Registry,
    linker: &mut Linker,
    roots: &mut Vec<(String, String)>,
    problems: &mut Vec<Error>,
) -> crate::surface::MaterialLayout {
    use crate::surface::{
        LightingModule, MaterialLayout, OpenSlot, SlotModule, SlotPrepare, SurfaceDecls,
        SurfaceKind, SurfaceTerms,
    };
    use engine_place::{PointInput, PointSet};
    let package = |brick: engine_registry::BrickId| {
        registry
            .record(brick)
            .map_or(CORE_PACKAGE, |record| record.package)
    };
    let mut declared = SurfaceDecls {
        slots: crate::surface::core_slots(),
        ..SurfaceDecls::default()
    };
    for (slot, path, _) in &declared.slots {
        roots.push((path.clone(), slot.slot.to_owned()));
    }
    for (brick, slot) in registry.render_entries::<SlotModule>() {
        linker.add_source(slot.module.source());
        let (path, name) = crate::linker::path_of((slot.module, slot.slot));
        roots.push((path.to_owned(), name));
        declared
            .slots
            .push((*slot, path.to_owned(), package(brick)));
    }
    for (brick, open) in registry.render_entries::<OpenSlot>() {
        linker.add_source(open.standard.source());
        let (path, name) = crate::linker::path_of((open.standard, open.input.name));
        roots.push((path.to_owned(), name));
        declared.open.push((*open, path.to_owned(), package(brick)));
    }
    for (brick, input) in registry.render_entries::<PointInput>() {
        declared.inputs.push((*input, package(brick)));
    }
    for (brick, set) in registry.render_entries::<PointSet>() {
        declared.sets.push((*set, package(brick)));
    }
    for (_, terms) in registry.render_entries::<SurfaceTerms>() {
        linker.add_source(terms.derive.source());
        let (path, name) = crate::linker::path_of((terms.derive, crate::surface::DERIVE_ENTRY));
        roots.push((path.to_owned(), name));
        declared.terms.push((*terms, path.to_owned()));
    }
    for (_, model) in registry.render_entries::<LightingModule>() {
        linker.add_source(model.module.source());
        let (path, name) = crate::linker::path_of((model.module, model.form.entry()));
        roots.push((path.to_owned(), name));
        declared.models.push((*model, path.to_owned()));
    }
    for (_, prepare) in registry.render_entries::<SlotPrepare>() {
        linker.add_source(prepare.module.source());
        let (path, name) = crate::linker::path_of((prepare.module, crate::surface::PREPARE_ENTRY));
        roots.push((path.to_owned(), name));
        declared.prepares.push(path.to_owned());
    }
    declared.kinds = registry
        .render_entries::<SurfaceKind>()
        .map(|(_, kind)| *kind)
        .collect();
    let layout = MaterialLayout::of(&declared, problems);
    for (path, text) in [
        (crate::surface::SHADING_MODULE, layout.shading_source()),
        (crate::surface::MATERIALS_MODULE, layout.materials_source()),
        (crate::surface::SLOTS_MODULE, layout.slots_source(true)),
        (crate::surface::SURFACES_MODULE, layout.surfaces_source()),
    ] {
        linker.add_generated(path.to_owned(), CORE_PACKAGE, text);
    }
    layout
}

fn prefix_of(
    registry: &Registry,
    linker: &Linker,
    brick: engine_registry::BrickId,
) -> Option<&'static str> {
    let record = registry.record(brick)?;
    if let Some(prefix) = record.prefix() {
        return Some(prefix);
    }
    linker
        .sources()
        .iter()
        .find(|source| source.package() == record.package)
        .and_then(|source| {
            let first = source.modules().first()?.path();
            Some(first.split("::").next().unwrap_or(first))
        })
}

type Handed = Vec<(&'static str, &'static str, Vec<Part>)>;

fn hand_generated(registry: &Registry, linker: &mut Linker) {
    let mut handed_over: Handed = Vec::new();
    for (brick, record) in registry.iter() {
        let texts: Vec<Part> = record
            .render_entries::<Generated>()
            .flat_map(|entry| entry.0.iter().map(|text| (None, *text)))
            .collect();
        if texts.is_empty() {
            continue;
        }
        let Some(prefix) = prefix_of(registry, linker, brick) else {
            continue;
        };
        handed_over.push((prefix, record.package, texts));
    }
    for (prefix, package, texts) in handed_over {
        linker.hand(format!("generated::{prefix}"), package, texts);
    }
}

fn pass_gates(registry: &Registry, passes: &[crate::install::PassDecl]) -> Vec<(u32, BrickId)> {
    let scoped: Vec<(&'static str, BrickId)> = registry
        .render_entries::<engine_core::PassEntry>()
        .map(|(brick, entry)| (entry.name, brick))
        .collect();
    passes
        .iter()
        .enumerate()
        .filter_map(|(index, pass)| {
            let scene = pass.scene.filter(|scene| scene.core)?;
            let &(_, gate) = scoped.iter().find(|(name, _)| *name == scene.name)?;
            Some((index as u32, gate))
        })
        .collect()
}

pub fn device_limits(
    limits: &wgpu::Limits,
    features: wgpu::Features,
    cores: u32,
) -> engine_core::DeviceLimits {
    engine_core::DeviceLimits {
        cores,
        storage_room: limits
            .max_storage_buffer_binding_size
            .min(limits.max_buffer_size),
        texture_layers: limits.max_texture_array_layers,
        max_texture_dimension_2d: limits.max_texture_dimension_2d,
        max_compute_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
        indirect_first_instance: features.contains(wgpu::Features::INDIRECT_FIRST_INSTANCE),
    }
}

fn scene_passes(world: &World) -> engine_core::ScenePasses {
    world
        .get_resource::<engine_core::ScenePasses>()
        .cloned()
        .unwrap_or_default()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Lint {
    Launch,
    Check,
}

pub struct Ahead {
    edited: std::collections::BTreeMap<String, String>,
    lint: Lint,
    problems: Vec<Error>,
    roots: Vec<(String, String)>,
    composing: crate::machine::Composing,
    parts: Vec<Option<Box<dyn Installed>>>,
    gates: Vec<Option<BrickId>>,
    reinstalls: Vec<Reinstall>,
    draws: Vec<KindMember>,
    passes: Vec<&'static str>,
    material: crate::surface::MaterialLayout,
}

pub(crate) fn install(
    device: &wgpu::Device,
    world: &mut World,
    (edited, lint): (std::collections::BTreeMap<String, String>, Lint),
) -> Result<Built, Vec<Error>> {
    let ahead = ahead(world, (edited, lint));
    finish(device, world, ahead)
}

pub(crate) fn ahead(
    world: &mut World,
    (edited, lint): (std::collections::BTreeMap<String, String>, Lint),
) -> Ahead {
    world.resource_scope::<Registry, _>(|world, registry| {
        install_with(world, &registry, (edited, lint))
    })
}

pub(crate) fn finish(
    device: &wgpu::Device,
    world: &mut World,
    ahead: Ahead,
) -> Result<Built, Vec<Error>> {
    let built =
        world.resource_scope::<Registry, _>(|_, registry| built(device, &registry, ahead))?;
    if let Some(mut registry) = world.get_resource_mut::<Registry>() {
        for notice in built.material.notices() {
            registry.report_notice(notice.clone());
        }
    }
    Ok(built)
}

#[expect(
    clippy::too_many_lines,
    reason = "longer than the line ceiling; it may only shrink"
)]
fn install_with(
    world: &mut World,
    registry: &Registry,
    (edited, lint): (std::collections::BTreeMap<String, String>, Lint),
) -> Ahead {
    let mut problems = Vec::new();
    let mut roots = Vec::new();
    let mut linker = Linker::edited(crate::shared::core_source(), edited.clone());
    for (path, text) in crate::shaders::generated_core() {
        linker.add_generated(path.to_owned(), CORE_PACKAGE, text);
    }
    let material = material_layout(registry, &mut linker, &mut roots, &mut problems);
    linker.add_generated(
        crate::shared::PASSES_MODULE.to_owned(),
        CORE_PACKAGE,
        scene_passes(world).wgsl(),
    );
    linker.add_generated(
        crate::shared::KINDS_MODULE.to_owned(),
        CORE_PACKAGE,
        world
            .get_resource::<engine_core::DrawnTable>()
            .map(engine_core::DrawnTable::wgsl)
            .unwrap_or_default(),
    );
    hand_generated(registry, &mut linker);
    let mut listed: Vec<Listed> = vec![
        Listed {
            record: core_part::<crate::shared::ScenePart>(),
            package: CORE_PACKAGE,
            brick: None,
        },
        Listed {
            record: core_part::<crate::render_bricks::Composite>(),
            package: CORE_PACKAGE,
            brick: None,
        },
        Listed {
            record: core_part::<crate::drawn::DrawnGpu>(),
            package: CORE_PACKAGE,
            brick: None,
        },
        Listed {
            record: core_part::<crate::mipmaps::Mipmaps>(),
            package: CORE_PACKAGE,
            brick: None,
        },
    ];
    let mut draws: Vec<&KindDraw> = Vec::new();
    for (brick, record) in registry.iter() {
        for entry in &record.render {
            if let Some(part) = entry.downcast_ref::<PartRecord>() {
                if listed.iter().any(|known| known.record.part == part.part) {
                    continue;
                }
                listed.push(Listed {
                    record: PartRecord {
                        part: part.part,
                        name: part.name,
                        install: part.install,
                    },
                    package: record.package,
                    brick: Some(brick),
                });
            }
            if let Some(draw) = entry.downcast_ref::<KindDraw>() {
                draws.push(draw);
            }
        }
    }
    let mut core = InstallCore::new(linker);
    core.passes = scene_passes(world);
    for entry in &listed {
        core.declared.parts.push(PartInfo {
            ty: entry.record.part,
            package: entry.package,
            name: entry.record.name,
        });
    }
    for draw in &draws {
        let index = listed
            .iter()
            .position(|entry| entry.record.part == draw.part);
        if let (Some(index), Some(kind)) = (index, core.passes.kind(draw.kind).copied()) {
            core.draws.push(KindMember::new(
                index as u32,
                kind,
                std::sync::Arc::clone(&draw.call),
            ));
        }
    }
    let mut parts: Vec<Option<Box<dyn Installed>>> = Vec::with_capacity(listed.len());
    let mut gates = Vec::with_capacity(listed.len());
    let mut reinstalls = Vec::with_capacity(listed.len());
    for (index, entry) in listed.iter().enumerate() {
        core.part = index as u32;
        core.package = entry.package;
        let from = core.declared.counts();
        match (entry.record.install)(world, &mut core) {
            Ok(held) => parts.push(Some(held)),
            Err(error) => {
                problems.push(error);
                parts.push(None);
            }
        }
        reinstalls.push(Reinstall {
            install: entry.record.install,
            package: entry.package,
            from,
            to: core.declared.counts(),
        });
        gates.push(entry.brick);
    }
    let InstallCore {
        declared,
        linker,
        draws: entries,
        problems: mut declared_problems,
        ..
    } = core;
    problems.append(&mut declared_problems);
    problems.extend(linker.check());
    let mut passes: Vec<&'static str> = Vec::new();
    for pass in &declared.passes {
        let name = pass.group.unwrap_or(pass.name);
        if !passes.contains(&name) {
            passes.push(name);
        }
    }
    let composing = crate::machine::compose(linker, declared, &mut problems);
    Ahead {
        edited,
        lint,
        problems,
        roots,
        composing,
        parts,
        gates,
        reinstalls,
        draws: entries,
        passes,
        material,
    }
}

fn built(device: &wgpu::Device, registry: &Registry, ahead: Ahead) -> Result<Built, Vec<Error>> {
    let mut problems = ahead.problems;
    let scope = crate::gpu_calls::validation_scope(device);
    let machine = Machine::build(device, ahead.composing, &mut problems);
    if let Some(Some(error)) = crate::platform::ready(scope.pop()) {
        problems.push(failed_with("$.pipelines", error));
    }
    let Some(mut machine) = machine.filter(|_| problems.is_empty()) else {
        return Err(problems);
    };
    if ahead.lint == Lint::Check {
        let composed: Vec<std::sync::Arc<crate::linker::Composed>> = machine
            .pipelines
            .iter()
            .flat_map(|pipeline| {
                pipeline
                    .stages
                    .iter()
                    .map(|stage| std::sync::Arc::clone(&stage.composed))
            })
            .collect();
        problems.extend(machine.linker.unused(&composed, &ahead.roots));
    }
    machine.linker.forget_wholes();
    let pass_gates = pass_gates(registry, &machine.declared.passes);
    if problems.is_empty() {
        Ok(Built {
            run_parts: ahead.parts,
            gates: ahead.gates,
            pass_gates,
            reinstalls: ahead.reinstalls,
            draws: ahead.draws,
            machine,
            passes: ahead.passes,
            material: ahead.material,
        })
    } else {
        Err(problems)
    }
}

impl Renderer {
    pub fn install(&mut self, world: &mut World) -> Result<(), Vec<Error>> {
        let installed = install(&self.core.device, world, (self.edited(), Lint::Launch))?;
        self.adopt_installed(installed, world);
        Ok(())
    }

    pub fn ahead(world: &mut World) -> Option<Ahead> {
        let launch = (std::collections::BTreeMap::new(), Lint::Launch);
        world.contains_resource::<Registry>().then(|| ahead(world, launch))
    }

    pub fn install_ahead(&mut self, world: &mut World, ahead: Ahead) -> Result<(), Vec<Error>> {
        if ahead.edited != self.edited() {
            return self.install(world);
        }
        let installed = finish(&self.core.device, world, ahead)?;
        self.adopt_installed(installed, world);
        Ok(())
    }

    #[cfg(all(feature = "dev", not(target_arch = "wasm32")))]
    fn edited(&self) -> std::collections::BTreeMap<String, String> {
        self.watched.edited()
    }

    #[cfg(not(all(feature = "dev", not(target_arch = "wasm32"))))]
    fn edited(&self) -> std::collections::BTreeMap<String, String> {
        std::collections::BTreeMap::new()
    }

    pub(super) fn adopt_installed(&mut self, installed: Built, world: &mut World) {
        let (mut assets, kept) = match (self.run.take(), self.assets.take()) {
            (Some(run), _) => {
                let (assets, kept) = kept_of(run);
                (assets, Some(kept))
            }
            (None, Some(assets)) => (assets, None),
            (None, None) => (new_assets(&self.core.device, &self.core.queue), None),
        };
        assets
            .materials
            .configure(&self.core.device, installed.material.clone());
        crate::mesh_arena::configure(
            &self.core.device,
            world,
            (&mut assets.meshes, &mut assets.materials),
        );
        let mut run = Run::new(
            (self.core.device.clone(), self.core.queue.clone()),
            installed.machine,
            installed.run_parts,
            assets,
        );
        if let Some(kept) = kept {
            adopt(&mut run, kept, world);
        }
        run.installs.gates = installed.gates;
        run.installs.pass_gates = installed.pass_gates;
        run.installs.reinstalls = installed.reinstalls;
        run.installs.draw_entries = installed.draws;
        run.take_states(world.get_resource::<BrickStates>());
        run.canvas.surface = self.core.display.format();
        run.canvas.format = run.canvas.surface;
        let mut stages: Vec<(&'static str, &'static str)> = Vec::new();
        self.spans = run
            .machine
            .declared
            .passes
            .iter()
            .map(|pass| {
                let name = pass.group.unwrap_or(pass.name);
                match stages.iter().find(|(known, _)| *known == name) {
                    Some(&(_, stage)) => stage,
                    None => {
                        let stage: &'static str =
                            Box::leak(format!("stage_{name}").into_boxed_str());
                        stages.push((name, stage));
                        stage
                    }
                }
            })
            .collect();
        self.passes = Box::leak(installed.passes.into_boxed_slice());
        self.plans.clear();
        #[cfg(not(target_arch = "wasm32"))]
        {
            run.pipelines.cache = self
                .core
                .pipeline_cache
                .as_ref()
                .map(|kept| kept.cache.clone());
        }
        self.run = Some(run);
    }

    pub fn check(world: &mut World) -> Result<(), Vec<Error>> {
        install_noop(world).map(drop)
    }
}

#[doc(hidden)]
pub fn install_noop(world: &mut World) -> Result<Installation, Vec<Error>> {
    let (device, queue) = noop_device().map_err(|error| vec![error])?;
    let mut installed = install(
        &device,
        world,
        (std::collections::BTreeMap::new(), Lint::Check),
    )?;
    let mut listed = listing(&mut installed);
    listed.stages = stage_texts(&installed.machine);
    let queue = crate::gpu_calls::Queue::new(queue);
    let mut assets = new_assets(&device, &queue);
    assets
        .materials
        .configure(&device, installed.material.clone());
    let scope = crate::gpu_calls::validation_scope(&device);
    let mut run = Run::new(
        (device.clone(), queue.clone()),
        installed.machine,
        installed.run_parts,
        assets,
    );
    run.installs.gates = installed.gates;
    run.warm();
    run.wait_made();
    run.make_all_shaders();
    let refused: Vec<Error> = std::mem::take(&mut run.pipelines.refused)
        .into_iter()
        .map(|(_, _, message)| failed_with("$.pipelines", message))
        .collect();
    match crate::platform::ready(scope.pop()) {
        Some(Some(error)) => Err(vec![failed_with("$.pipelines", error)]),
        Some(None) | None if refused.is_empty() => Ok(listed),
        Some(None) | None => Err(refused),
    }
}

struct Kept {
    declared: crate::install::Declared,
    parts: Vec<Option<Box<dyn Installed>>>,
    tables: crate::encode::Tables,
    view: crate::encode::Tables,
    filled: Vec<crate::encode::Filled>,
    frames: (u64, u64),
    installed: Vec<engine_core::FrameId>,
    readbacks: crate::readback::Readbacks,
    shaders: std::collections::BTreeMap<u64, wgpu::ShaderModule>,
    made: Vec<crate::machine::BuiltPipeline>,
    pipelines: crate::encode::Pipelines,
}

fn kept_of(run: Run) -> (Assets, Kept) {
    let Run {
        assets,
        machine,
        parts,
        tables,
        sets,
        filled,
        frames,
        installs,
        reading,
        pipelines,
        ..
    } = run;
    let at = sets.at;
    let view = sets
        .list
        .into_iter()
        .nth(at)
        .map(|set| set.tables)
        .unwrap_or_default();
    let kept = Kept {
        declared: machine.declared,
        parts,
        tables,
        view,
        filled,
        frames: (frames.current, frames.asked),
        installed: installs.installed,
        readbacks: reading.list,
        shaders: machine.shaders,
        made: machine.pipelines,
        pipelines,
    };
    (assets, kept)
}

fn same_shape(old: &crate::install::Declared, new: &crate::install::Declared) -> bool {
    let parts = |declared: &crate::install::Declared| {
        declared
            .parts
            .iter()
            .map(|part| part.ty)
            .collect::<Vec<_>>()
    };
    let textures = |declared: &crate::install::Declared| {
        declared
            .textures
            .iter()
            .map(|texture| {
                (
                    texture.label,
                    texture.format,
                    texture.lifetime,
                    texture.part,
                )
            })
            .collect::<Vec<_>>()
    };
    let buffers = |declared: &crate::install::Declared| {
        declared
            .buffers
            .iter()
            .map(|buffer| (buffer.label, buffer.lifetime, buffer.part, buffer.usage))
            .collect::<Vec<_>>()
    };
    let passes = |declared: &crate::install::Declared| {
        declared
            .passes
            .iter()
            .map(|pass| (pass.name, pass.part))
            .collect::<Vec<_>>()
    };
    let pipelines = |declared: &crate::install::Declared| {
        declared
            .pipelines
            .iter()
            .map(|pipeline| (pipeline.part, pipeline.scope))
            .collect::<Vec<_>>()
    };
    let inputs = |declared: &crate::install::Declared| {
        declared
            .inputs
            .iter()
            .map(|input| (input.part, input.scope, input.buffer))
            .collect::<Vec<_>>()
    };
    parts(old) == parts(new)
        && textures(old) == textures(new)
        && buffers(old) == buffers(new)
        && old.samplers == new.samplers
        && passes(old) == passes(new)
        && pipelines(old) == pipelines(new)
        && inputs(old) == inputs(new)
}

fn adopt(run: &mut Run, kept: Kept, world: &mut World) {
    (run.frames.current, run.frames.asked) = kept.frames;
    run.reading.list = kept.readbacks;
    run.adopt_made(kept.shaders, kept.made, kept.pipelines);
    let next = run.written().next();
    run.installs.installed.fill(next);
    if !same_shape(&kept.declared, &run.machine.declared) {
        return;
    }
    run.installs.installed = kept.installed;
    for (slot, held) in run.parts.iter_mut().zip(kept.parts) {
        if let Some(mut held) = held {
            held.rebind(world);
            *slot = Some(held);
        }
    }
    run.tables = kept.tables;
    run.set_mut().tables = kept.view;
    run.filled = kept
        .filled
        .into_iter()
        .map(|mut filled| {
            filled.groups.clear();
            filled
        })
        .collect();
}

pub(crate) fn reinstall(
    run: &mut Run,
    world: &mut World,
    part: u32,
) -> Result<Box<dyn Installed>, Error> {
    let name = run
        .machine
        .declared
        .parts
        .get(part as usize)
        .map_or("", |info| info.name);
    let Some(again) = run.installs.reinstalls.get(part as usize).cloned() else {
        return Err(failed(&format!("$.parts.{name}")));
    };
    let linker = std::mem::replace(
        &mut run.machine.linker,
        Linker::new(crate::shared::core_source()),
    );
    let mut core = InstallCore::new(linker);
    core.passes = scene_passes(world);
    core.part = part;
    core.package = again.package;
    core.declared = run.machine.declared.before(again.from);
    core.draws.clone_from(&run.installs.draw_entries);
    let made = (again.install)(world, &mut core);
    run.machine.linker = core.linker;
    let held = made?;
    if let Some(problem) = core.problems.into_iter().next() {
        return Err(problem);
    }
    if core.declared.counts() != again.to {
        return Err(failed(&format!("$.parts.{name}.install")).got("declared_differently"));
    }
    Ok(held)
}

#[cfg(not(target_arch = "wasm32"))]
fn noop_device() -> Result<(wgpu::Device, wgpu::Queue), Error> {
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = wgpu::Backends::NOOP;
    descriptor.backend_options.noop.enable = true;
    let instance = wgpu::Instance::new(descriptor);
    let adapter =
        crate::platform::ready(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .ok_or_else(|| failed("$.adapter"))?
            .map_err(|error| failed_with("$.adapter", error))?;
    let (device, queue) = crate::platform::ready(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("engine/noop-device"),
        required_limits: super::startup::device_limits(&adapter.limits()),
        ..Default::default()
    }))
    .ok_or_else(|| failed("$.device"))?
    .map_err(|error| failed_with("$.device", error))?;
    Ok((device, queue))
}

#[cfg(target_arch = "wasm32")]
fn noop_device() -> Result<(wgpu::Device, wgpu::Queue), Error> {
    Err(failed("$.device").expected(Expected::Supported))
}
