use std::any::TypeId;

use bevy_ecs::world::World;
use engine_core::{Error, Module, Section};
use engine_place::PlaceKind;
use engine_registry::{GateId, MembersOf, Registry, SettingsEntry};

use super::*;
use crate::install::{DrawEntry, InstallCore, PartInfo};
use crate::linker::{CORE_PACKAGE, Linker, Part};
use crate::machine::{Machine, OutputDecl};
use crate::places::{Answered, Fed, checked, face_module, face_source, place_module};
use crate::registration::{DrawAny, Generated, Installed, PartRecord, ResolvePass};

struct Listed {
    record: PartRecord,
    package: &'static str,
    brick: Option<engine_registry::BrickId>,
    fills: Vec<TypeId>,
}

pub(crate) struct Built {
    pub(crate) run_parts: Vec<Option<Box<dyn Installed>>>,
    pub(crate) gates: Vec<Option<GateId>>,
    pub(crate) machine: Machine,
    pub(crate) pieces: Vec<(usize, crate::pieces::PieceAt)>,
    pub(crate) passes: Vec<&'static str>,
    pub(crate) linked: engine_core::LinkedStats,
    pub(crate) placed: Placed,
    pub(crate) resolve: Option<(crate::PassNames, &'static str)>,
}

#[derive(Default)]
pub(crate) struct Placed {
    pieces: Vec<InstalledPiece>,
    arena: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InstalledPart {
    pub brick: &'static str,
    pub part: &'static str,
    pub passes: Vec<&'static str>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InstalledPiece {
    pub brick: &'static str,
    pub name: &'static str,
    pub offset: u64,
    pub size: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Installation {
    pub parts: Vec<InstalledPart>,
    pub modules: Vec<String>,
    pub texts: Vec<String>,
    pub layouts: usize,
    pub frame_variants: usize,
    pub pieces: Vec<InstalledPiece>,
    pub arena: u64,
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
    let machine = &built.machine;
    let parts = machine
        .declared
        .parts
        .iter()
        .enumerate()
        .map(|(index, part)| InstalledPart {
            brick: part.owner,
            part: part.name,
            passes: machine
                .declared
                .passes
                .iter()
                .filter(|pass| pass.part as usize == index)
                .map(|pass| pass.name)
                .collect(),
        })
        .collect();
    let mut layouts: Vec<String> = machine.pipelines.iter().map(crate::machine::layout_key).collect();
    layouts.sort_unstable();
    layouts.dedup();
    Installation {
        parts,
        modules,
        texts,
        layouts: layouts.len(),
        frame_variants: machine.variants.len(),
        pieces: built.placed.pieces.clone(),
        arena: built.placed.arena,
    }
}

#[derive(Default)]
struct Links {
    faces: Vec<(&'static str, &'static str)>,
    nobody: Vec<&'static str>,
    lists: Vec<(&'static str, &'static [&'static str])>,
}

impl Links {
    fn stats(mut self) -> engine_core::LinkedStats {
        self.faces.sort_unstable();
        self.nobody.sort_unstable();
        self.lists.sort_unstable_by_key(|&(place, _)| place);
        engine_core::LinkedStats {
            faces: Box::leak(self.faces.into_boxed_slice()),
            nobody: Box::leak(self.nobody.into_boxed_slice()),
            lists: Box::leak(self.lists.into_boxed_slice()),
        }
    }
}

fn module_path(module: &Module) -> Option<String> {
    let text = module.source().modules().get(module.index() as usize)?;
    Some(text.path().to_owned())
}

fn core_part<P: crate::GpuPart>() -> PartRecord {
    PartRecord::of::<P>(engine_registry::Scope::BRICK)
}

fn wgsl_of(wgsl: Option<&(dyn std::any::Any + Send + Sync)>) -> Option<&Module> {
    wgsl.and_then(|wgsl| wgsl.downcast_ref::<Module>())
}

fn faces(
    registry: &Registry,
    linker: &mut Linker,
    roots: &mut Vec<(String, String)>,
    problems: &mut Vec<Error>,
) -> (Vec<(u32, GateId)>, Links) {
    let mut switches = Vec::new();
    let mut links = Links::default();
    let mut catalog: Vec<_> = registry.catalog().iter().filter(|decl| decl.wgsl().is_some()).collect();
    catalog.sort_by_key(|decl| decl.name());
    for (_, record) in registry.iter() {
        for filled in record.fills() {
            if let Some(module) = wgsl_of(filled.wgsl.as_deref()) {
                linker.add_source(module.source());
            }
        }
        for joined in record.joins() {
            if let Some(module) = wgsl_of(joined.wgsl.as_deref()) {
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
            let implements: Vec<String> = contract
                .answers
                .iter()
                .filter(|answer| {
                    if answer.constant {
                        linker.provides(&module, &answer.name)
                    } else {
                        linker.defines(&module, &answer.name)
                    }
                })
                .map(|answer| answer.name.clone())
                .collect();
            for answer in &implements {
                roots.push((module.clone(), answer.clone()));
            }
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
                    Some((brick, filled.scope, wgsl_of(filled.wgsl.as_deref())))
                });
                match filler.and_then(|(brick, scope, wgsl)| Some((brick, fed(brick, scope, wgsl)?))) {
                    Some((brick, filler)) => {
                        if let Some(record) = registry.record(brick) {
                            links.faces.push((name, record.package));
                        }
                        Answered::Filler(filler)
                    }
                    None => {
                        links.nobody.push(name);
                        Answered::Nobody
                    }
                }
            }
            PlaceKind::List => {
                let mut members = Vec::new();
                let mut bricks: Vec<&'static str> = Vec::new();
                for (brick, record) in registry.iter() {
                    for joined in record.joins().iter().filter(|joined| joined.list == decl.type_id()) {
                        if let Some(member) = fed(brick, joined.scope, wgsl_of(joined.wgsl.as_deref())) {
                            members.push(member);
                            bricks.push(record.package);
                        }
                    }
                }
                links.lists.push((name, Box::leak(bricks.into_boxed_slice())));
                Answered::Members(members)
            }
        };
        linker.add_generated(face_module(name), CORE_PACKAGE, face_source(name, &contract, &answered));
    }
    (switches, links)
}

fn outputs(registry: &Registry, world: &World, roots: &mut Vec<(String, String)>) -> Vec<OutputDecl> {
    let members = world
        .get_resource::<MembersOf<crate::ViewOutputs>>()
        .map_or(&[][..], |members| members.all());
    let joins = registry
        .iter()
        .flat_map(|(_, record)| record.joins().iter())
        .filter(|joined| joined.list == TypeId::of::<crate::ViewOutputs>());
    let mut found = Vec::new();
    for (output, joined) in members.iter().zip(joins) {
        let module = wgsl_of(joined.wgsl.as_deref()).and_then(module_path);
        if let Some(module) = &module {
            roots.push((module.clone(), output.name.to_owned()));
        }
        found.push(OutputDecl {
            name: output.name,
            format: output.format,
            layers: output.layers.to_vec(),
            module,
        });
    }
    found
}

fn prefix_of(registry: &Registry, linker: &Linker, brick: engine_registry::BrickId) -> Option<&'static str> {
    let record = registry.record(brick)?;
    if let Some(prefix) = record.prefix() {
        return Some(prefix);
    }
    linker.sources().iter().find(|source| source.package() == record.package).and_then(|source| {
        let first = source.modules().first()?.path();
        Some(first.split("::").next().unwrap_or(first))
    })
}

type Handed = Vec<(&'static str, &'static str, Vec<Part>)>;

fn hand_generated(registry: &Registry, linker: &mut Linker, pieces: Handed) {
    let mut handed_over: Handed = Vec::new();
    for (brick, record) in registry.iter() {
        let texts: Vec<Part> =
            record.render_entries::<Generated>().flat_map(|entry| entry.0.iter().map(|text| (None, *text))).collect();
        if texts.is_empty() {
            continue;
        }
        let Some(prefix) = prefix_of(registry, linker, brick) else { continue };
        handed_over.push((prefix, record.package, texts));
    }
    for (prefix, package, piece) in pieces {
        match handed_over.iter_mut().find(|(known, _, _)| *known == prefix) {
            Some((_, _, texts)) => {
                let mut joined = piece;
                joined.append(texts);
                *texts = joined;
            }
            None => handed_over.push((prefix, package, piece)),
        }
    }
    for (prefix, package, texts) in handed_over {
        linker.hand(format!("generated::{prefix}"), package, texts);
    }
}

fn pieces(
    registry: &Registry,
    linker: &mut Linker,
    problems: &mut Vec<Error>,
) -> (Vec<(usize, crate::pieces::PieceAt)>, Placed, Handed) {
    let mut listed = Vec::new();
    let mut indices = Vec::new();
    let mut handed_over: Handed = Vec::new();
    for (index, (brick, entry)) in registry.pieces().enumerate() {
        let Some(package) = registry.record(brick).map(|record| record.package) else { continue };
        let Some(prefix) = prefix_of(registry, linker, brick) else { continue };
        listed.push(crate::pieces::Piece {
            brick: package,
            prefix,
            ty: entry.wgsl_name,
            size: entry.size as u64,
        });
        indices.push(index);
        match handed_over.iter_mut().find(|(known, _, _)| *known == prefix) {
            Some((_, _, texts)) => texts.insert(0, (Some(entry.wgsl_name), entry.wgsl)),
            None => handed_over.push((prefix, package, vec![(Some(entry.wgsl_name), entry.wgsl)])),
        }
    }
    let (placed, arena) = match crate::pieces::place(&listed, crate::pieces::PIECES_BYTES) {
        Ok(placed) => placed,
        Err(error) => {
            problems.push(error);
            return (Vec::new(), Placed::default(), handed_over);
        }
    };
    let named = Placed {
        pieces: listed
            .iter()
            .zip(&placed)
            .map(|(piece, at)| InstalledPiece {
                brick: piece.brick,
                name: piece.ty,
                offset: at.offset,
                size: at.size,
            })
            .collect(),
        arena,
    };
    linker.add_generated(crate::pieces::PIECES_MODULE.to_owned(), CORE_PACKAGE, crate::pieces::pieces_source(&listed));
    for piece in &listed {
        linker.add_generated(crate::pieces::piece_path(piece), CORE_PACKAGE, crate::pieces::piece_source(piece));
    }
    (indices.into_iter().zip(placed).collect(), named, handed_over)
}

fn names_twice(world: &World, problems: &mut Vec<Error>) {
    if let Some(members) = world.get_resource::<MembersOf<crate::ViewOutputs>>() {
        problems.extend(members.names_twice(|member| member.name));
    }
    if let Some(members) = world.get_resource::<MembersOf<crate::DebugModes>>() {
        problems.extend(members.names_twice(|member| member.name));
    }
}

fn device_limits(device: &wgpu::Device, world: &World) -> engine_core::DeviceLimits {
    let limits = device.limits();
    let cores = world.get_resource::<engine_core::DeviceLimits>().map_or(0, |held| held.cores);
    engine_core::DeviceLimits {
        storage_room: limits.max_storage_buffer_binding_size.min(limits.max_buffer_size),
        texture_layers: limits.max_texture_array_layers,
        max_texture_dimension_2d: limits.max_texture_dimension_2d,
        max_compute_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
        indirect_first_instance: device.features().contains(wgpu::Features::INDIRECT_FIRST_INSTANCE),
        ..engine_core::DeviceLimits::without_device(cores)
    }
}

pub(crate) fn install(device: &wgpu::Device, world: &mut World) -> Result<Built, Vec<Error>> {
    world.resource_scope::<Registry, _>(|world, registry| install_with(device, world, &registry))
}

#[expect(clippy::too_many_lines, reason = "longer than the line ceiling; it may only shrink")]
fn install_with(device: &wgpu::Device, world: &mut World, registry: &Registry) -> Result<Built, Vec<Error>> {
    let mut problems = Vec::new();
    names_twice(world, &mut problems);
    let mut roots = Vec::new();
    let mut linker = Linker::new(crate::shared::core_source());
    for (path, text) in crate::shaders::generated_core() {
        linker.add_generated(path.to_owned(), CORE_PACKAGE, text);
    }
    let (switches, links) = faces(registry, &mut linker, &mut roots, &mut problems);
    let outputs = outputs(registry, world, &mut roots);
    linker.add_generated(crate::shared::TARGETS_MODULE.to_owned(), CORE_PACKAGE, crate::shared::targets_source(&outputs));
    linker.add_generated(crate::shared::COMPOSITE_MODULE.to_owned(), CORE_PACKAGE, crate::shared::composite_source());
    let (pieces, placed, piece_texts) = pieces(registry, &mut linker, &mut problems);
    hand_generated(registry, &mut linker, piece_texts);
    let resolve = registry.iter().find_map(|(_, record)| {
        let fills = record.fills().iter().any(|filled| filled.slot == TypeId::of::<crate::Resolve>());
        record.render_entries::<ResolvePass>().next().filter(|_| fills).map(|entry| (entry.0, record.package))
    });
    let core_fills = vec![TypeId::of::<crate::MainView>()];
    let mut listed: Vec<Listed> = vec![
        Listed { record: core_part::<crate::shared::Layers>(), package: CORE_PACKAGE, brick: None, fills: core_fills.clone() },
        Listed { record: core_part::<crate::shared::Composite>(), package: CORE_PACKAGE, brick: None, fills: core_fills },
    ];
    let mut draws: Vec<(TypeId, &DrawAny)> = Vec::new();
    for (brick, record) in registry.iter() {
        let fills: Vec<TypeId> = record.fills().iter().map(|filled| filled.slot).collect();
        for entry in &record.render {
            if let Some(part) = entry.downcast_ref::<PartRecord>() {
                if listed.iter().any(|known| known.record.part == part.part) {
                    continue;
                }
                listed.push(Listed {
                    record: PartRecord {
                        part: part.part,
                        name: part.name,
                        lazy: part.lazy,
                        labels: part.labels,
                        scope: part.scope,
                        install: part.install,
                    },
                    package: record.package,
                    brick: Some(brick),
                    fills: fills.clone(),
                });
            }
            if let Some(draw) = entry.downcast_ref::<DrawAny>() {
                draws.push((draw.part, draw));
            }
        }
    }
    let mut core = InstallCore::new(linker, device.features());
    core.limits = device_limits(device, world);
    for entry in &listed {
        core.declared.parts.push(PartInfo {
            ty: entry.record.part,
            package: entry.package,
            name: entry.record.name,
            lazy: entry.record.lazy,
            labels: entry.record.labels,
            owner: entry.brick.map_or(engine_registry::CORE_ROW, |_| entry.package),
            profile: entry.brick.map_or(0, engine_registry::BrickId::profile_index),
        });
    }
    for (part, draw) in &draws {
        if let Some(index) = listed.iter().position(|entry| entry.record.part == *part) {
            core.draws.push(DrawEntry { part: index as u32, record: std::sync::Arc::clone(&draw.record) });
        }
    }
    let mut parts: Vec<Option<Box<dyn Installed>>> = Vec::with_capacity(listed.len());
    let mut gates = Vec::with_capacity(listed.len());
    for (index, entry) in listed.iter().enumerate() {
        core.part = index as u32;
        core.package = entry.package;
        core.fills.clone_from(&entry.fills);
        let _owner = core.declared.parts[index].owned();
        match (entry.record.install)(world, &mut core) {
            Ok(held) => parts.push(Some(held)),
            Err(error) => {
                problems.push(error);
                parts.push(None);
            }
        }
        gates.push(entry.brick.and_then(|brick| registry.gate(brick, entry.record.scope)));
    }
    let InstallCore { declared, linker, problems: mut declared_problems, .. } = core;
    problems.append(&mut declared_problems);
    let fills = |package: &'static str, place: &str| {
        registry.brick(package).is_some_and(|record| record.places().any(|decl| decl.name() == place))
    };
    problems.extend(linker.check(&fills));
    let mut passes: Vec<&'static str> = Vec::new();
    for pass in &declared.passes {
        let name = pass.group.unwrap_or(pass.name);
        if !passes.contains(&name) {
            passes.push(name);
        }
    }
    let limits = crate::pipeline_layouts::REQUIRED_LIMITS;
    let scope = crate::gpu_calls::validation_scope(device);
    let machine = Machine::build(device, linker, declared, outputs, switches, &limits, &mut problems);
    if let Some(Some(error)) = crate::platform::ready(scope.pop()) {
        problems.push(failed_with("$.pipelines", error));
    }
    let Some(mut machine) = machine else {
        return Err(problems);
    };
    let composed: Vec<std::sync::Arc<crate::linker::Composed>> = machine
        .pipelines
        .iter()
        .flat_map(|pipeline| pipeline.stages.iter().map(|stage| std::sync::Arc::clone(&stage.composed)))
        .collect();
    problems.extend(machine.linker.unused(&composed, &roots));
    if problems.is_empty() {
        Ok(Built { run_parts: parts, gates, machine, pieces, passes, linked: links.stats(), placed, resolve })
    } else {
        Err(problems)
    }
}

fn check_render(value: &serde_json::Value, path: &str) -> Result<(), Error> {
    engine_registry::parse_section::<crate::RenderSettings>(value, path, crate::RenderSettings::check).map(drop)
}

fn set_render(world: &mut World, value: &serde_json::Value, path: &str) -> Result<(), Error> {
    engine_registry::set_section::<crate::RenderSettings>(world, value, path, crate::RenderSettings::check)
}

impl Renderer {
    #[doc(hidden)]
    pub fn register_places(registry: &mut Registry) {
        registry.places(crate::shared::PLACES);
        registry.fill_core::<crate::MainView>();
        registry.core_settings(SettingsEntry::of::<crate::RenderSettings>(
            crate::RenderSettings::KEY,
            check_render,
            set_render,
            crate::RenderSettings::schema,
        ));
    }

    pub fn install(&mut self, world: &mut World) -> Result<(), Vec<Error>> {
        let installed = install(&self.core.device, world)?;
        let (mut assets, kept) = match (self.run.take(), self.assets.take()) {
            (Some(run), _) => {
                let (assets, kept) = kept_of(run);
                (assets, Some(kept))
            }
            (None, Some(assets)) => (assets, None),
            (None, None) => (new_assets(&self.core.device, &self.core.queue), None),
        };
        crate::mesh_arena::configure(
            &self.core.device,
            world,
            (&mut assets.meshes, &mut assets.materials, &mut assets.impostors),
        );
        let mut run = Run::new(
            self.core.device.clone(),
            wgpu::Queue::clone(&self.core.queue),
            installed.machine,
            installed.run_parts,
            assets,
        );
        if let Some(kept) = kept {
            adopt(&mut run, kept, world);
        }
        run.gates = installed.gates;
        run.piece_layout = installed.pieces;
        run.surface = self.core.display.format();
        run.resolve = installed.resolve.map(|(names, _)| {
            let stage: &'static str = Box::leak(format!("stage_{}", names.pass).into_boxed_str());
            (names, stage)
        });
        self.resolver = installed.resolve.map(|(names, package)| (names.pass, package));
        run.switches = switches_of(&run.machine, world, &[]);
        let scope = crate::gpu_calls::validation_scope(&self.core.device);
        run.warm(false);
        self.installing = Some(Box::pin(scope.pop()));
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
                        let stage: &'static str = Box::leak(format!("stage_{name}").into_boxed_str());
                        stages.push((name, stage));
                        stage
                    }
                }
            })
            .collect();
        self.passes = Box::leak(installed.passes.into_boxed_slice());
        self.linked = installed.linked;
        let names: Vec<&'static str> = run.machine.declared.parts.iter().map(|part| part.name).collect();
        self.part_names = Box::leak(names.into_boxed_slice());
        self.ran.clear();
        self.plans.clear();
        self.run = Some(run);
        Ok(())
    }

    pub fn check(world: &mut World) -> Result<(), Vec<Error>> {
        install_noop(world).map(drop)
    }
}

#[doc(hidden)]
pub fn install_noop(world: &mut World) -> Result<Installation, Vec<Error>> {
    let (device, queue) = noop_device().map_err(|error| vec![error])?;
    let mut installed = install(&device, world)?;
    let listed = listing(&mut installed);
    let queue = crate::gpu_calls::Queue::new(queue);
    let assets = new_assets(&device, &queue);
    let scope = crate::gpu_calls::validation_scope(&device);
    let mut run = Run::new(device.clone(), wgpu::Queue::clone(&queue), installed.machine, installed.run_parts, assets);
    run.gates = installed.gates;
    run.switches = switches_of(&run.machine, world, &[]);
    run.warm(true);
    match crate::platform::ready(scope.pop()) {
        Some(Some(error)) => Err(vec![failed_with("$.pipelines", error)]),
        Some(None) | None => Ok(listed),
    }
}

struct Kept {
    declared: crate::install::Declared,
    parts: Vec<Option<Box<dyn Installed>>>,
    textures: Vec<[Option<crate::encode::Allocated>; 2]>,
    sizes: Vec<Option<(wgpu::Extent3d, u32)>>,
    buffers: Vec<Option<(wgpu::Buffer, u64)>>,
    filled: Vec<crate::encode::Filled>,
    rings: Vec<crate::encode::Ring>,
}

fn kept_of(run: Run) -> (Assets, Kept) {
    let Run {
        assets,
        machine,
        parts,
        textures,
        sizes,
        buffers,
        filled,
        rings,
        ..
    } = run;
    let kept = Kept {
        declared: machine.declared,
        parts,
        textures,
        sizes,
        buffers,
        filled,
        rings,
    };
    (assets, kept)
}

fn same_shape(old: &crate::install::Declared, new: &crate::install::Declared) -> bool {
    let parts = |declared: &crate::install::Declared| declared.parts.iter().map(|part| part.ty).collect::<Vec<_>>();
    let textures = |declared: &crate::install::Declared| {
        declared
            .textures
            .iter()
            .map(|texture| (texture.label, texture.format, texture.lifetime, texture.part))
            .collect::<Vec<_>>()
    };
    let buffers = |declared: &crate::install::Declared| {
        declared
            .buffers
            .iter()
            .map(|buffer| (buffer.label, buffer.lifetime, buffer.part, buffer.usage))
            .collect::<Vec<_>>()
    };
    let passes = |declared: &crate::install::Declared| declared.passes.iter().map(|pass| (pass.name, pass.part)).collect::<Vec<_>>();
    let pipelines = |declared: &crate::install::Declared| {
        declared.pipelines.iter().map(|pipeline| (pipeline.part, pipeline.scope)).collect::<Vec<_>>()
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
    if !same_shape(&kept.declared, &run.machine.declared) {
        return;
    }
    for (slot, held) in run.parts.iter_mut().zip(kept.parts) {
        if let Some(mut held) = held {
            held.rebind(world);
            *slot = Some(held);
        }
    }
    run.textures = kept.textures;
    run.sizes = kept.sizes;
    run.buffers = kept.buffers;
    run.filled = kept
        .filled
        .into_iter()
        .map(|mut filled| {
            filled.groups.clear();
            filled
        })
        .collect();
    run.rings = kept.rings;
}

pub(crate) fn switches_of(machine: &Machine, world: &World, faulted: &[GateId]) -> Vec<(u32, bool)> {
    let gates = world.get_resource::<engine_registry::Gates>();
    machine
        .switches
        .iter()
        .map(|&(id, gate)| (id, !faulted.contains(&gate) && gates.is_none_or(|gates| gates.on(gate))))
        .collect()
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
