use std::any::TypeId;

use bevy_ecs::world::World;
use engine_core::{Error, Module, PlaceKind};
use engine_registry::{GateId, Registry};

use super::*;
use crate::install::{DrawEntry, InstallCore, PartInfo};
use crate::linker::{CORE_PACKAGE, Linker};
use crate::machine::{Machine, OutputDecl};
use crate::places::{Answered, Fed, checked, face_module, face_source, place_module};
use crate::registration::{DrawAny, Installed, PartRecord};

struct Listed {
    record: PartRecord,
    package: &'static str,
    brick: Option<engine_registry::BrickId>,
    fills: Vec<TypeId>,
}

pub(crate) struct Installation {
    pub(crate) run_parts: Vec<Option<Box<dyn Installed>>>,
    pub(crate) gates: Vec<Option<GateId>>,
    pub(crate) machine: Machine,
    pub(crate) pieces: Vec<(usize, crate::pieces::PieceAt)>,
    pub(crate) passes: Vec<&'static str>,
}

fn module_path(module: &Module) -> Option<String> {
    let text = module.source().modules().get(module.index() as usize)?;
    Some(text.path().to_owned())
}

fn core_part<P: crate::GpuPart>() -> PartRecord {
    PartRecord::of::<P>(engine_registry::Scope::BRICK)
}

fn faces(registry: &Registry, linker: &mut Linker, problems: &mut Vec<Error>) -> Vec<(u32, GateId)> {
    let mut switches = Vec::new();
    let mut catalog: Vec<_> = registry.catalog().iter().filter(|decl| decl.wgsl().is_some()).collect();
    catalog.sort_by_key(|decl| decl.name());
    for (_, record) in registry.iter() {
        for filled in record.fills() {
            if let Some(module) = filled.wgsl.as_deref().and_then(|wgsl| wgsl.downcast_ref::<Module>()) {
                linker.add_source(module.source());
            }
        }
        for joined in record.joins() {
            if let Some(module) = joined.wgsl.as_deref().and_then(|wgsl| wgsl.downcast_ref::<Module>()) {
                linker.add_source(module.source());
            }
        }
    }
    for decl in catalog {
        let Some(source) = decl.wgsl() else { continue };
        let name = decl.name();
        let contract = match checked(name, source, decl.kind(), &|module| module.starts_with("engine::")) {
            Ok(contract) => contract,
            Err(error) => {
                problems.push(error);
                continue;
            }
        };
        linker.add_generated(place_module(name), CORE_PACKAGE, source.to_owned());
        let mut fed = |brick, scope, wgsl: Option<&Module>| -> Option<Fed> {
            let module = module_path(wgsl?)?;
            let switch = registry.gate(brick, scope).map(|gate| {
                let id = switches.len() as u32;
                switches.push((id, gate));
                id
            });
            let implements = contract
                .answers
                .iter()
                .filter(|answer| linker.defines(&module, &answer.name))
                .map(|answer| answer.name.clone())
                .collect();
            Some(Fed {
                module,
                switch,
                implements,
            })
        };
        let answered = match decl.kind() {
            PlaceKind::Slot | PlaceKind::Component => {
                let filler = registry.iter().find_map(|(brick, record)| {
                    let filled = record.fills().iter().find(|filled| filled.slot == decl.type_id())?;
                    Some((brick, filled.scope, filled.wgsl.as_deref().and_then(|wgsl| wgsl.downcast_ref::<Module>())))
                });
                match filler.and_then(|(brick, scope, wgsl)| fed(brick, scope, wgsl)) {
                    Some(filler) => Answered::Filler(filler),
                    None => Answered::Nobody,
                }
            }
            PlaceKind::List => {
                let mut members = Vec::new();
                for (brick, record) in registry.iter() {
                    for joined in record.joins().iter().filter(|joined| joined.list == decl.type_id()) {
                        let wgsl = joined.wgsl.as_deref().and_then(|wgsl| wgsl.downcast_ref::<Module>());
                        if let Some(member) = fed(brick, joined.scope, wgsl) {
                            members.push(member);
                        }
                    }
                }
                Answered::Members(members)
            }
        };
        linker.add_generated(face_module(name), CORE_PACKAGE, face_source(name, &contract, &answered));
    }
    switches
}

fn outputs(registry: &Registry) -> Vec<OutputDecl> {
    let mut found = Vec::new();
    for (_, record) in registry.iter() {
        for joined in record.joins().iter().filter(|joined| joined.list == TypeId::of::<crate::ViewOutputs>()) {
            let Some(output) = joined.member::<crate::Output>() else { continue };
            found.push(OutputDecl {
                name: output.name,
                format: output.format,
                layers: output.layers.to_vec(),
                module: joined
                    .wgsl
                    .as_deref()
                    .and_then(|wgsl| wgsl.downcast_ref::<Module>())
                    .and_then(module_path),
            });
        }
    }
    found
}

fn pieces(registry: &Registry, linker: &mut Linker, problems: &mut Vec<Error>) -> Vec<(usize, crate::pieces::PieceAt)> {
    let mut listed = Vec::new();
    let mut indices = Vec::new();
    for (index, (brick, entry)) in registry.pieces().enumerate() {
        let Some(package) = registry.record(brick).map(|record| record.package) else { continue };
        let prefix = linker.sources().iter().find(|source| source.package() == package).and_then(|source| {
            let first = source.modules().first()?.path();
            Some(first.split("::").next().unwrap_or(first))
        });
        let Some(prefix) = prefix else { continue };
        listed.push(crate::pieces::Piece {
            brick: package,
            prefix,
            ty: entry.wgsl_name,
            size: entry.size as u64,
        });
        indices.push(index);
        linker.add_generated(format!("generated::{prefix}"), CORE_PACKAGE, entry.wgsl.to_owned());
    }
    let (placed, _) = match crate::pieces::place(&listed, crate::pieces::PIECES_BYTES) {
        Ok(placed) => placed,
        Err(error) => {
            problems.push(error);
            return Vec::new();
        }
    };
    linker.add_generated(crate::pieces::PIECES_MODULE.to_owned(), CORE_PACKAGE, crate::pieces::pieces_source(&listed));
    for piece in &listed {
        linker.add_generated(crate::pieces::piece_path(piece), CORE_PACKAGE, crate::pieces::piece_source(piece));
    }
    indices.into_iter().zip(placed).collect()
}

pub(crate) fn install(device: &wgpu::Device, world: &mut World) -> Result<Installation, Vec<Error>> {
    world.resource_scope::<Registry, _>(|world, registry| install_with(device, world, &registry))
}

fn install_with(device: &wgpu::Device, world: &mut World, registry: &Registry) -> Result<Installation, Vec<Error>> {
    let mut problems = Vec::new();
    let mut linker = Linker::new(crate::shared::core_source());
    let switches = faces(registry, &mut linker, &mut problems);
    let outputs = outputs(registry);
    linker.add_generated(crate::shared::TARGETS_MODULE.to_owned(), CORE_PACKAGE, crate::shared::targets_source(&outputs));
    linker.add_generated(crate::shared::COMPOSITE_MODULE.to_owned(), CORE_PACKAGE, crate::shared::composite_source());
    let core_fills = vec![TypeId::of::<crate::MainView>()];
    let mut listed: Vec<Listed> = vec![
        Listed { record: core_part::<crate::shared::Layers>(), package: CORE_PACKAGE, brick: None, fills: core_fills.clone() },
        Listed { record: core_part::<crate::shared::Composite>(), package: CORE_PACKAGE, brick: None, fills: core_fills },
    ];
    let mut draws: Vec<(TypeId, &'static str, &DrawAny)> = Vec::new();
    for (brick, record) in registry.iter() {
        let fills: Vec<TypeId> = record.fills().iter().map(|filled| filled.slot).collect();
        for entry in &record.render {
            if let Some(part) = entry.downcast_ref::<PartRecord>() {
                if listed.iter().any(|known| known.record.part == part.part) {
                    continue;
                }
                listed.push(Listed {
                    record: PartRecord { part: part.part, name: part.name, scope: part.scope, install: part.install },
                    package: record.package,
                    brick: Some(brick),
                    fills: fills.clone(),
                });
            }
            if let Some(draw) = entry.downcast_ref::<DrawAny>() {
                draws.push((draw.part, record.package, draw));
            }
        }
    }
    let mut core = InstallCore::new(linker);
    for (index, entry) in listed.iter().enumerate() {
        core.declared.parts.push(PartInfo {
            package: entry.package,
            name: entry.record.name,
            brick: entry.brick.map_or(u32::MAX, |brick| brick.index() as u32),
            registered: index as u32,
        });
    }
    for (part, package, draw) in &draws {
        if let Some(index) = listed.iter().position(|entry| entry.record.part == *part) {
            core.draws.push(DrawEntry { package: *package, part: index as u32, record: std::sync::Arc::clone(&draw.record) });
        }
    }
    let mut parts: Vec<Option<Box<dyn Installed>>> = Vec::with_capacity(listed.len());
    let mut gates = Vec::with_capacity(listed.len());
    for (index, entry) in listed.iter().enumerate() {
        core.part = index as u32;
        core.package = entry.package;
        core.fills.clone_from(&entry.fills);
        let _owner = crate::memory::owned(entry.package);
        match (entry.record.install)(world, &mut core) {
            Ok(held) => parts.push(Some(held)),
            Err(error) => {
                problems.push(error);
                parts.push(None);
            }
        }
        gates.push(entry.brick.and_then(|brick| registry.gate(brick, entry.record.scope)));
    }
    let InstallCore { declared, mut linker, problems: mut declared_problems, .. } = core;
    problems.append(&mut declared_problems);
    let pieces = pieces(registry, &mut linker, &mut problems);
    let fills = |package: &'static str, place: &str| {
        registry.brick(package).is_some_and(|record| record.places().any(|decl| decl.name() == place))
    };
    problems.extend(linker.check(&fills));
    let passes: Vec<&'static str> = declared.passes.iter().map(|pass| pass.name).collect();
    let limits = crate::pipeline_layouts::REQUIRED_LIMITS;
    let scope = crate::gpu_calls::validation_scope(device);
    let machine = Machine::build(device, linker, declared, outputs, switches, &limits, &mut problems);
    if let Some(Some(error)) = crate::platform::ready(scope.pop()) {
        problems.push(failed_with("$.pipelines", error));
    }
    match machine {
        Some(machine) if problems.is_empty() => Ok(Installation { run_parts: parts, gates, machine, pieces, passes }),
        Some(_) | None => Err(problems),
    }
}

impl Renderer {
    #[doc(hidden)]
    pub fn register_places(registry: &mut Registry) {
        registry.places(crate::shared::PLACES);
        registry.fill_core::<crate::MainView>();
    }

    pub fn install(&mut self, world: &mut World) -> Result<(), Vec<Error>> {
        let installed = install(&self.core.device, world)?;
        let assets = match (self.run.take(), self.assets.take()) {
            (Some(run), _) => run.assets,
            (None, Some(assets)) => assets,
            (None, None) => new_assets(&self.core.device, &self.core.queue),
        };
        let mut run = Run::new(
            self.core.device.clone(),
            wgpu::Queue::clone(&self.core.queue),
            installed.machine,
            installed.run_parts,
            assets,
        );
        run.gates = installed.gates;
        run.piece_layout = installed.pieces;
        run.surface = self.core.display.format();
        let scope = crate::gpu_calls::validation_scope(&self.core.device);
        run.warm();
        self.installing = Some(Box::pin(scope.pop()));
        self.passes = installed.passes;
        self.plans.clear();
        self.run = Some(run);
        Ok(())
    }

    pub fn check(world: &mut World) -> Result<(), Vec<Error>> {
        let (device, queue) = noop_device().map_err(|error| vec![error])?;
        let installed = install(&device, world)?;
        let queue = crate::gpu_calls::Queue::new(queue);
        let assets = new_assets(&device, &queue);
        let scope = crate::gpu_calls::validation_scope(&device);
        let mut run = Run::new(device.clone(), wgpu::Queue::clone(&queue), installed.machine, installed.run_parts, assets);
        run.gates = installed.gates;
        run.warm();
        match crate::platform::ready(scope.pop()) {
            Some(Some(error)) => Err(vec![failed_with("$.pipelines", error)]),
            Some(None) | None => Ok(()),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn noop_device() -> Result<(wgpu::Device, wgpu::Queue), Error> {
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = wgpu::Backends::NOOP;
    descriptor.backend_options.noop.enable = true;
    let instance = wgpu::Instance::new(descriptor);
    let adapter = crate::platform::ready(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
        .ok_or_else(|| failed("$.adapter"))?
        .map_err(|error| failed_with("$.adapter", error))?;
    let (device, queue) = crate::platform::ready(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("engine/noop-device"),
        required_limits: adapter.limits(),
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
