use std::sync::Arc;

use bevy_ecs::world::World;
use engine_core::Error;
use engine_registry::{BrickId, BrickStates, Registry};

use super::*;
use crate::install::{InstallCore, PartInfo, Reinstall};
use crate::linker::{CORE_PACKAGE, Shaders};
use crate::machine::Machine;
use crate::registration::{Installed, KindDraw, KindMember, PartRecord};

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
}

fn core_part<P: crate::GpuPart>() -> PartRecord {
    PartRecord::of::<P>()
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

pub fn device_limits(limits: &wgpu::Limits) -> engine_core::DeviceLimits {
    engine_core::DeviceLimits {
        storage_room: limits
            .max_storage_buffer_binding_size
            .min(limits.max_buffer_size),
    }
}

fn scene_passes(world: &World) -> engine_core::ScenePasses {
    world
        .get_resource::<engine_core::ScenePasses>()
        .cloned()
        .unwrap_or_default()
}

#[derive(Clone, Default, PartialEq, Eq)]
pub(crate) struct Linking {
    #[cfg(feature = "dev")]
    pub(crate) edited: std::collections::BTreeMap<String, String>,
    #[cfg(feature = "dev")]
    pub(crate) check: bool,
}

impl Linking {
    #[cfg(feature = "dev")]
    fn shaders(&self, world: &World) -> Result<Shaders, Vec<Error>> {
        let inputs = crate::linker::ShaderInputs::of(world.resource::<Registry>(), &self.edited);
        Shaders::link(inputs, self.check)
    }

    #[cfg(not(feature = "dev"))]
    fn shaders(&self, world: &World) -> Result<Shaders, Vec<Error>> {
        let code = engine_core::ErrorCode::MissingResource;
        let missing = || vec![Error::new(code, "$.shaders").got("LinkedShaders")];
        let text = world.get_resource::<crate::LinkedShaders>().ok_or_else(missing)?;
        Shaders::embedded(text.0, world.resource::<Registry>()).map_err(|error| vec![error])
    }
}

#[derive(Default)]
pub struct Ahead {
    linking: Linking,
    problems: Vec<Error>,
    composing: Option<crate::machine::Composing>,
    parts: Vec<Option<Box<dyn Installed>>>,
    gates: Vec<Option<BrickId>>,
    reinstalls: Vec<Reinstall>,
    draws: Vec<KindMember>,
    passes: Vec<&'static str>,
}

pub(crate) fn install(
    device: &wgpu::Device,
    world: &mut World,
    linking: Linking,
) -> Result<Built, Vec<Error>> {
    let ahead = ahead(world, linking);
    finish(device, world, ahead)
}

pub(crate) fn ahead(world: &mut World, linking: Linking) -> Ahead {
    let linked = match linking.shaders(world) {
        Ok(linked) => Arc::new(linked),
        Err(problems) => {
            return Ahead {
                linking,
                problems,
                ..Ahead::default()
            };
        }
    };
    world.resource_scope::<Registry, _>(|world, registry| {
        install_with(world, &registry, linking, linked)
    })
}

pub(crate) fn finish(
    device: &wgpu::Device,
    world: &mut World,
    ahead: Ahead,
) -> Result<Built, Vec<Error>> {
    world.resource_scope::<Registry, _>(|_, registry| built(device, &registry, ahead))
}

fn install_with(
    world: &mut World,
    registry: &Registry,
    linking: Linking,
    linked: Arc<Shaders>,
) -> Ahead {
    let mut problems = Vec::new();
    let mut listed: Vec<Listed> = vec![
        Listed {
            record: core_part::<crate::shared::ScenePart>(),
            package: CORE_PACKAGE,
            brick: None,
        },
        Listed {
            record: core_part::<crate::drawn::DrawnGpu>(),
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
    let mut core = InstallCore::new(linked);
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
        let kind = (core.passes.kinds().iter())
            .position(|decl| decl.kind == draw.kind)
            .and_then(|at| Some((at as u32, *core.passes.kinds().get(at)?)));
        if let (Some(index), Some(kind)) = (index, kind) {
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
        linked,
        draws: entries,
        problems: mut declared_problems,
        ..
    } = core;
    problems.append(&mut declared_problems);
    let mut passes: Vec<&'static str> = Vec::new();
    for pass in &declared.passes {
        if !passes.contains(&pass.name) {
            passes.push(pass.name);
        }
    }
    let composing = crate::machine::compose(linked, declared, &mut problems);
    Ahead {
        linking,
        problems,
        composing: Some(composing),
        parts,
        gates,
        reinstalls,
        draws: entries,
        passes,
    }
}

fn built(device: &wgpu::Device, registry: &Registry, ahead: Ahead) -> Result<Built, Vec<Error>> {
    let mut problems = ahead.problems;
    let Some(composing) = ahead.composing else {
        return Err(problems);
    };
    let scope = crate::gpu_calls::validation_scope(device);
    let machine = Machine::build(device, composing, &mut problems);
    if let Some(Some(error)) = crate::platform::ready(scope.pop()) {
        problems.push(failed_with("$.pipelines", error));
    }
    let Some(machine) = machine.filter(|_| problems.is_empty()) else {
        return Err(problems);
    };
    let pass_gates = pass_gates(registry, &machine.declared.passes);
    Ok(Built {
        run_parts: ahead.parts,
        gates: ahead.gates,
        pass_gates,
        reinstalls: ahead.reinstalls,
        draws: ahead.draws,
        machine,
        passes: ahead.passes,
    })
}

impl Renderer {
    pub fn install(&mut self, world: &mut World) -> Result<(), Vec<Error>> {
        let installed = install(&self.core.device, world, self.linking())?;
        self.adopt_installed(installed, world);
        Ok(())
    }

    pub fn ahead(world: &mut World) -> Option<Ahead> {
        (world.contains_resource::<Registry>()).then(|| ahead(world, Linking::default()))
    }

    pub fn install_ahead(&mut self, world: &mut World, ahead: Ahead) -> Result<(), Vec<Error>> {
        if ahead.linking != self.linking() {
            return self.install(world);
        }
        let installed = finish(&self.core.device, world, ahead)?;
        self.adopt_installed(installed, world);
        Ok(())
    }

    #[cfg(all(feature = "dev", not(target_arch = "wasm32")))]
    fn linking(&self) -> Linking {
        Linking {
            edited: self.watched.edited(),
            check: false,
        }
    }

    #[cfg(not(all(feature = "dev", not(target_arch = "wasm32"))))]
    fn linking(&self) -> Linking {
        Linking::default()
    }

    pub(super) fn adopt_installed(&mut self, installed: Built, world: &mut World) {
        let (assets, kept) = match (self.run.take(), self.assets.take()) {
            (Some(run), _) => {
                let (assets, kept) = kept_of(run);
                (assets, Some(kept))
            }
            (None, Some(assets)) => (assets, None),
            (None, None) => (new_assets(&self.core.device), None),
        };
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
        run.installs.settle_members(&run.machine.declared);
        if let Some(passes) = world.get_resource::<engine_core::ScenePasses>() {
            run.tallies.kinds = crate::encode::KindTallies::of(passes, &run.machine.declared.passes);
        }
        run.take_states(world.get_resource::<BrickStates>());
        run.canvas.surface = self.core.display.format();
        run.canvas.format = run.canvas.surface;
        let canvas_size = self.core.display.size().pixels();
        run.canvas.size = (canvas_size.0.max(1), canvas_size.1.max(1));
        let mut stages: Vec<(&'static str, &'static str)> = Vec::new();
        self.spans = run
            .machine
            .declared
            .passes
            .iter()
            .map(|pass| {
                let name = pass.name;
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
        #[cfg(not(target_arch = "wasm32"))]
        {
            run.pipelines.cache = self
                .core
                .pipeline_cache
                .as_ref()
                .ok()
                .map(|kept| kept.cache.clone());
        }
        #[cfg(feature = "dev")]
        crate::whatif::install(world, &run.machine);
        self.run = Some(run);
    }
}

#[cfg(feature = "dev")]
pub fn install_noop(
    world: &mut World,
    gpu: crate::GpuRequest,
) -> Result<String, Vec<Error>> {
    let (device, queue) = noop_device(gpu).map_err(|error| vec![error])?;
    let linking = Linking {
        check: true,
        ..Linking::default()
    };
    let installed = install(&device, world, linking)?;
    let linked = serde_json::to_string_pretty(&*installed.machine.linked)
        .map_err(|error| vec![failed_with("$.shaders", error)])?;
    let queue = crate::gpu_calls::Queue::new(queue);
    let assets = new_assets(&device);
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
        Some(None) | None if refused.is_empty() => Ok(linked + "\n"),
        Some(None) | None => Err(refused),
    }
}

struct Kept {
    declared: crate::install::Declared,
    parts: Vec<Option<Box<dyn Installed>>>,
    tables: crate::encode::Tables,
    filled: Vec<crate::encode::Filled>,
    frames: (u64, u64),
    readbacks: crate::readback::Readbacks,
    arrived: Vec<Option<(engine_core::FrameId, Vec<u8>)>>,
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
        filled,
        frames,
        reading,
        pipelines,
        ..
    } = run;
    let kept = Kept {
        declared: machine.declared,
        parts,
        tables,
        filled,
        frames: (frames.current, frames.asked),
        readbacks: reading.list,
        arrived: reading.arrived,
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
                    texture.dimension,
                    texture.lifetime,
                    texture.part,
                    texture.usage,
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
    let readbacks = |declared: &crate::install::Declared| {
        declared
            .readbacks
            .iter()
            .map(|readback| (readback.label, readback.part))
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
    let batch_groups = |declared: &crate::install::Declared| {
        declared
            .batch_groups
            .iter()
            .map(|batch| (batch.module.index(), batch.index))
            .collect::<Vec<_>>()
    };
    parts(old) == parts(new)
        && textures(old) == textures(new)
        && buffers(old) == buffers(new)
        && readbacks(old) == readbacks(new)
        && old.samplers == new.samplers
        && passes(old) == passes(new)
        && pipelines(old) == pipelines(new)
        && batch_groups(old) == batch_groups(new)
}

fn adopt(run: &mut Run, kept: Kept, world: &mut World) {
    (run.frames.current, run.frames.asked) = kept.frames;
    run.reading.list = kept.readbacks;
    run.adopt_made(kept.shaders, kept.made, kept.pipelines);
    if !same_shape(&kept.declared, &run.machine.declared) {
        return;
    }
    run.reading.arrived = kept.arrived;
    for (slot, held) in run.parts.iter_mut().zip(kept.parts) {
        if let Some(mut held) = held {
            held.rebind(world);
            *slot = Some(held);
        }
    }
    run.tables = kept.tables;
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
    let mut core = InstallCore::new(Arc::clone(&run.machine.linked));
    core.passes = scene_passes(world);
    core.part = part;
    core.package = again.package;
    core.declared = run.machine.declared.before(again.from);
    core.draws.clone_from(&run.installs.draw_entries);
    let held = (again.install)(world, &mut core)?;
    if let Some(problem) = core.problems.into_iter().next() {
        return Err(problem);
    }
    if core.declared.counts() != again.to {
        return Err(failed(&format!("$.parts.{name}.install")).got("declared_differently"));
    }
    Ok(held)
}

#[cfg(all(feature = "dev", not(target_arch = "wasm32")))]
fn noop_device(gpu: crate::GpuRequest) -> Result<(wgpu::Device, wgpu::Queue), Error> {
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = wgpu::Backends::NOOP;
    descriptor.backend_options.noop.enable = true;
    let instance = wgpu::Instance::new(descriptor);
    let opened = crate::platform::ready(super::startup::open(instance, gpu))
        .ok_or_else(|| failed("$.device"))??;
    Ok((opened.device, opened.queue))
}

#[cfg(all(feature = "dev", target_arch = "wasm32"))]
fn noop_device(_: crate::GpuRequest) -> Result<(wgpu::Device, wgpu::Queue), Error> {
    Err(failed("$.device").expected(Expected::Supported))
}

#[cfg(feature = "dev")]
pub(super) fn declared_names(world: &mut World) {
    use engine_registry::{Kind, NameEntry, register, unavailable};

    type Read = fn(&Run) -> serde_json::Value;
    let names: [(&'static str, Read); 2] = [("gpu.targets", targets), ("links", links)];
    for (name, read) in names {
        register(
            world,
            NameEntry::read(name, Kind::State, move |reach, _| {
                Ok(reach.part::<Renderer>().map_or_else(
                    || unavailable("no_renderer"),
                    |renderer| {
                        renderer
                            .run
                            .as_ref()
                            .map_or_else(|| unavailable("renderer_not_installed"), read)
                    },
                ))
            }),
        );
    }
}

#[cfg(feature = "dev")]
fn targets(run: &Run) -> serde_json::Value {
    let mut facts = run.machine.declared.target_facts(&run.installs.reinstalls);
    let set = (run.sets.list.get(run.sets.at)).or_else(|| run.sets.list.first());
    if let serde_json::Value::Object(listed) = &mut facts {
        for (at, target) in run.machine.declared.targets.iter().enumerate() {
            if let Some(serde_json::Value::Object(fields)) = listed.get_mut(target.name.name()) {
                let size = (set.and_then(|set| set.shared.get(at)))
                    .map(|made| [made.size.width, made.size.height]);
                let fraction = run.canvas.fraction(target.name);
                fields.insert("fraction".to_owned(), serde_json::json!(fraction));
                fields.insert("size".to_owned(), serde_json::json!(size));
            }
        }
    }
    facts
}

#[cfg(feature = "dev")]
fn links(run: &Run) -> serde_json::Value {
    use serde_json::{Map, Value, json};

    use crate::graph::keyed;
    use crate::install::Program;

    let machine = &run.machine;
    let declared = &machine.declared;
    let part_name = |part: u32| {
        declared
            .parts
            .get(part as usize)
            .map_or("", |info| info.name)
    };
    let mut passes = Map::new();
    for (index, pass) in declared.passes.iter().enumerate() {
        let mut parts = Map::new();
        parts.insert(part_name(pass.part).to_owned(), json!({}));
        let pipelines = machine
            .pipelines
            .iter()
            .filter(|built| crate::machine::in_pass(built, index as u32, pass));
        for built in pipelines {
            let mut entries = Map::new();
            for stage in &built.stages {
                keyed(
                    &mut entries,
                    stage.entry.name(),
                    json!(stage.composed.module),
                );
            }
            let entry = match &built.decl.program {
                Program::Render(render) => render.vertex.name(),
                Program::Compute(entry) => entry.name(),
            };
            let held = parts
                .entry(part_name(built.decl.part))
                .or_insert_with(|| json!({}));
            if let Value::Object(held) = held {
                let key = format!("{entry}/{:x}", built.program_key);
                keyed(held, &key, Value::Object(entries));
            }
        }
        keyed(&mut passes, pass.name, Value::Object(parts));
    }
    let mut links = machine.graph.links();
    links.insert("passes".to_owned(), Value::Object(passes));
    links.insert(
        "kinds".to_owned(),
        crate::registration::kind_passes(declared, &run.installs.draw_entries),
    );
    Value::Object(links)
}
