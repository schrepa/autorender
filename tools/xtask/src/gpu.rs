//! The renderer's shaders checked against its Rust side: every module
//! parses and validates with naga, and every WGSL struct mirroring an upload
//! struct (`engine_render::gpu_layouts`) matches its Rust layout.

use std::path::Path;

use engine_render::bindings::{
    Binding, Format, Kind, SHARED_GROUPS, Sample, Sampling, Stages, TABLES, Table, View,
};
use engine_render::gpu_layouts::{Layout, constants, layouts};
use engine_render::pipeline_layouts::{
    LAYOUT_BUDGET, PIPELINE_LAYOUTS, PipelineLayout, REQUIRED_LIMITS, StageLimits,
};
use engine_render::shaders::{ENTRIES, Stage};
use naga::valid::{Capabilities, ValidationFlags, Validator};

mod inter_stage;

/// A module naga parsed and validated, and what its functions use.
pub struct Checked {
    pub name: &'static str,
    pub module: naga::Module,
    pub info: naga::valid::ModuleInfo,
}

/// Every module of `engine_render::shaders`, parsed and validated.
pub fn modules() -> Result<Vec<Checked>, String> {
    let mut checked = Vec::new();
    let mut failures = Vec::new();
    for &(name, source) in engine_render::shaders::MODULES {
        let module = match naga::front::wgsl::parse_str(source) {
            Ok(module) => module,
            Err(error) => {
                failures.push(format!("{name}: {}", error.emit_to_string(source)));
                continue;
            }
        };
        let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
        match validator.validate(&module) {
            Ok(info) => checked.push(Checked { name, module, info }),
            Err(error) => failures.push(format!("{name}: {}", error.emit_to_string(source))),
        }
    }
    if failures.is_empty() {
        Ok(checked)
    } else {
        Err(failures.join("\n"))
    }
}

/// The members of struct `name` in `module`: name, offset and size each.
fn wgsl_struct(module: &naga::Module, name: &str) -> Option<Vec<(String, u32, u32)>> {
    module.types.iter().find_map(|(_, ty)| {
        let naga::TypeInner::Struct { members, .. } = &ty.inner else {
            return None;
        };
        (ty.name.as_deref() == Some(name)).then(|| {
            members
                .iter()
                .map(|member| {
                    (
                        member.name.clone().unwrap_or_default(),
                        member.offset,
                        module.types[member.ty].inner.size(module.to_ctx()),
                    )
                })
                .collect()
        })
    })
}

/// Mismatches between `layout` and its mirror `members` in module `module`.
fn compare(
    layout: &Layout,
    mirror: &str,
    module: &str,
    members: &[(String, u32, u32)],
) -> Vec<String> {
    let at = format!("{module} `{mirror}` mirroring {}", layout.rust);
    let mut errors = Vec::new();
    if members.len() > layout.fields.len() {
        errors.push(format!(
            "{at}: {} members beyond the Rust struct's {} fields",
            members.len(),
            layout.fields.len()
        ));
    }
    for ((name, offset, size), field) in members.iter().zip(layout.fields) {
        if name != field.name || *offset as usize != field.offset || *size as usize != field.size {
            errors.push(format!(
                "{at}: member `{name}` at {offset} ({size} bytes) where Rust has `{}` at {} ({} bytes)",
                field.name, field.offset, field.size
            ));
        }
    }
    errors
}

/// Every upload struct's WGSL mirrors against its Rust layout.
fn check_layouts(modules: &[Checked]) -> Result<usize, String> {
    let mut errors = Vec::new();
    let mut mirrors = 0;
    for layout in layouts() {
        for mirror in layout.mirrors {
            let mut found = false;
            for checked in modules {
                if mirror.module.is_some_and(|module| module != checked.name) {
                    continue;
                }
                if let Some(members) = wgsl_struct(&checked.module, mirror.name) {
                    found = true;
                    mirrors += 1;
                    errors.extend(compare(layout, mirror.name, checked.name, &members));
                }
            }
            if !found {
                errors.push(format!(
                    "{}: no shader module {} declares its mirror `{}`",
                    layout.rust,
                    mirror.module.unwrap_or("of engine_render::shaders"),
                    mirror.name
                ));
            }
        }
    }
    if errors.is_empty() {
        Ok(mirrors)
    } else {
        Err(errors.join("\n"))
    }
}

/// A resource a shader module declares: its group, binding, name and the
/// global naga holds it in.
struct Declared<'a> {
    group: u32,
    binding: u32,
    name: &'a str,
    handle: naga::Handle<naga::GlobalVariable>,
}

fn declared(module: &naga::Module) -> Vec<Declared<'_>> {
    module
        .global_variables
        .iter()
        .filter_map(|(handle, global)| {
            let binding = global.binding.as_ref()?;
            Some(Declared {
                group: binding.group,
                binding: binding.binding,
                name: global.name.as_deref().unwrap_or_default(),
                handle,
            })
        })
        .collect()
}

/// How a table's minimum buffer size fits a shader's type.
#[derive(Clone, Copy, PartialEq)]
enum Fit {
    /// The type's size, or a whole number of a flat array's elements.
    Exact,
    /// More than the type needs: a shared table sized for another module.
    Larger,
}

/// How `min_size` fits the type of `global` (0 leaves it to draws).
fn fit(module: &naga::Module, global: &naga::GlobalVariable, min_size: u64) -> Result<Fit, String> {
    let inner = &module.types[global.ty].inner;
    let size = inner.size(module.to_ctx()) as u64;
    // Words read as a flat runtime-sized array may be bound a record at a time.
    let flat_stride = match inner {
        naga::TypeInner::Array {
            base,
            size: naga::ArraySize::Dynamic,
            stride,
        } if !matches!(module.types[*base].inner, naga::TypeInner::Struct { .. }) => {
            Some(u64::from(*stride))
        }
        _ => None,
    };
    if min_size == 0
        || min_size == size
        || flat_stride.is_some_and(|stride| min_size > size && min_size.is_multiple_of(stride))
    {
        Ok(Fit::Exact)
    } else if min_size > size {
        Ok(Fit::Larger)
    } else {
        Err(format!(
            "minimum size {min_size} bytes where the shader's type spans {size}"
        ))
    }
}

/// Why `binding` of a table cannot hold `global` of `module`, if it cannot:
/// its kind, and for buffers how its minimum size fits.
fn mismatch(
    module: &naga::Module,
    global: &naga::GlobalVariable,
    binding: &Binding,
) -> Result<Option<Fit>, String> {
    let inner = &module.types[global.ty].inner;
    let sized = |min_size: u64| fit(module, global, min_size).map(Some);
    let view = |dim: naga::ImageDimension, arrayed: bool| match (dim, arrayed) {
        (naga::ImageDimension::D2, false) => Some(View::D2),
        (naga::ImageDimension::D2, true) => Some(View::D2Array),
        (naga::ImageDimension::D3, false) => Some(View::D3),
        (naga::ImageDimension::Cube, false) => Some(View::Cube),
        (naga::ImageDimension::Cube, true) => Some(View::CubeArray),
        _ => None,
    };
    let format = |format: naga::StorageFormat| match format {
        naga::StorageFormat::R32Float => Some(Format::R32Float),
        naga::StorageFormat::R32Uint => Some(Format::R32Uint),
        naga::StorageFormat::Rgba8Snorm => Some(Format::Rgba8Snorm),
        naga::StorageFormat::Rgba8Unorm => Some(Format::Rgba8Unorm),
        naga::StorageFormat::Rgba16Float => Some(Format::Rgba16Float),
        _ => None,
    };
    let expected = match (global.space, inner) {
        (naga::AddressSpace::Uniform, _) => match binding.kind {
            Kind::Uniform { min_size, .. } => return sized(min_size),
            _ => "a uniform buffer",
        },
        (naga::AddressSpace::Storage { access }, _)
            if !access.contains(naga::StorageAccess::STORE) =>
        {
            match binding.kind {
                Kind::Read { min_size } => return sized(min_size),
                _ => "a read-only storage buffer",
            }
        }
        (naga::AddressSpace::Storage { .. }, _) => match binding.kind {
            Kind::Write => return Ok(None),
            _ => "a writable storage buffer",
        },
        (naga::AddressSpace::Handle, naga::TypeInner::Sampler { comparison }) => match binding.kind
        {
            Kind::Sampler(Sampling::Comparison) if *comparison => return Ok(None),
            Kind::Sampler(Sampling::Filtering | Sampling::NonFiltering) if !*comparison => {
                return Ok(None);
            }
            _ if *comparison => "a comparison sampler",
            _ => "a filtering or non-filtering sampler",
        },
        (
            naga::AddressSpace::Handle,
            naga::TypeInner::Image {
                dim,
                arrayed,
                class,
            },
        ) => {
            let view = view(*dim, *arrayed);
            match (class, binding.kind) {
                (naga::ImageClass::Storage { format: stored, .. }, Kind::Storage(table))
                    if view == Some(View::D2) && format(*stored) == Some(table) =>
                {
                    return Ok(None);
                }
                (naga::ImageClass::Storage { format: stored, .. }, Kind::Storage3d(table))
                    if view == Some(View::D3) && format(*stored) == Some(table) =>
                {
                    return Ok(None);
                }
                (naga::ImageClass::Storage { .. }, _) => {
                    "a storage texture of the shader's format and dimension"
                }
                (
                    class,
                    Kind::Texture {
                        sample,
                        view: table,
                    },
                ) if view == Some(table) => match (class, sample) {
                    (
                        naga::ImageClass::Sampled {
                            kind: naga::ScalarKind::Float,
                            multi: false,
                        },
                        Sample::Float | Sample::Unfilterable,
                    )
                    | (
                        naga::ImageClass::Sampled {
                            kind: naga::ScalarKind::Uint,
                            multi: false,
                        },
                        Sample::Uint,
                    )
                    | (naga::ImageClass::Depth { multi: false }, Sample::Depth) => return Ok(None),
                    _ => "a texture of the shader's sample type",
                },
                _ => "a texture of the shader's dimension",
            }
        }
        _ => "a kind of binding tables cannot declare",
    };
    Err(format!(
        "the shader declares {expected}, the table {:?}",
        binding.kind
    ))
}

fn stage_of(stage: naga::ShaderStage) -> Stages {
    match stage {
        naga::ShaderStage::Vertex => Stages::VERTEX,
        naga::ShaderStage::Fragment => Stages::FRAGMENT,
        _ => Stages::COMPUTE,
    }
}

fn covers(stages: Stages, stage: Stages) -> bool {
    (stage.vertex() && stages.vertex())
        || (stage.fragment() && stages.fragment())
        || (stage.compute() && stages.compute())
}

/// Every bind group table (`engine_render::bindings`) against the shader
/// groups it serves: each binding declared by one of them, under its
/// number and name, of a matching kind and size; and every entry point's
/// resources in one table serving their group, visible to its stage.
#[expect(
    clippy::too_many_lines,
    reason = "longer than the line ceiling; it may only shrink"
)]
fn check_bindings(modules: &[Checked]) -> Result<usize, String> {
    let mut errors = Vec::new();
    let module = |name: &str| modules.iter().find(|checked| checked.name == name);
    for &(table_name, table) in TABLES {
        for serves in table.serves {
            if module(serves.module).is_none() {
                errors.push(format!(
                    "{table_name}: serves {}, which is not a shader module",
                    serves.module
                ));
            }
        }
        for binding in table.bindings {
            let mut found = false;
            let mut fits = Vec::new();
            for serves in table.serves {
                let Some(checked) = module(serves.module) else {
                    continue;
                };
                for resource in declared(&checked.module) {
                    if resource.group != serves.group || resource.binding != binding.binding {
                        continue;
                    }
                    if resource.name != binding.name {
                        continue;
                    }
                    found = true;
                    let global = &checked.module.global_variables[resource.handle];
                    match mismatch(&checked.module, global, binding) {
                        Ok(fit) => fits.extend(fit),
                        Err(why) => errors.push(format!(
                            "{table_name} binding {} `{}` in {} group {}: {why}",
                            binding.binding, binding.name, checked.name, serves.group
                        )),
                    }
                }
            }
            if !fits.is_empty() && !fits.contains(&Fit::Exact) {
                errors.push(format!(
                    "{table_name} binding {} `{}`: its minimum size is no shader's type's size",
                    binding.binding, binding.name
                ));
            }
            if !found {
                let names: Vec<_> = table
                    .serves
                    .iter()
                    .filter_map(|serves| {
                        let checked = module(serves.module)?;
                        let names: Vec<_> = declared(&checked.module)
                            .into_iter()
                            .filter(|resource| {
                                resource.group == serves.group
                                    && resource.binding == binding.binding
                            })
                            .map(|resource| format!("`{}` in {}", resource.name, checked.name))
                            .collect();
                        (!names.is_empty()).then(|| names.join(", "))
                    })
                    .collect();
                errors.push(format!(
                    "{table_name} binding {} `{}`: no shader group it serves declares it{}",
                    binding.binding,
                    binding.name,
                    if names.is_empty() {
                        String::new()
                    } else {
                        format!(" (binding {} is {})", binding.binding, names.join(", "))
                    }
                ));
            }
        }
    }
    // Each entry point's resources of a group, in one table serving it.
    let mut entry_points = 0;
    for checked in modules {
        let resources = declared(&checked.module);
        for (index, entry) in checked.module.entry_points.iter().enumerate() {
            entry_points += 1;
            let info = checked.info.get_entry_point(index);
            let stage = stage_of(entry.stage);
            let used: Vec<&Declared> = resources
                .iter()
                .filter(|resource| !info[resource.handle].is_empty())
                .collect();
            let mut groups: Vec<u32> = used.iter().map(|resource| resource.group).collect();
            groups.sort_unstable();
            groups.dedup();
            for group in groups {
                let wanted: Vec<&&Declared> = used
                    .iter()
                    .filter(|resource| resource.group == group)
                    .collect();
                let lacking = |table: &Table| -> Vec<String> {
                    wanted
                        .iter()
                        .filter(|resource| {
                            !table.bindings.iter().any(|binding| {
                                binding.binding == resource.binding
                                    && binding.name == resource.name
                                    && covers(binding.stages, stage)
                            })
                        })
                        .map(|resource| format!("{} `{}`", resource.binding, resource.name))
                        .collect()
                };
                let candidates: Vec<(&str, Vec<String>)> = TABLES
                    .iter()
                    .filter(|(_, table)| {
                        table
                            .serves
                            .iter()
                            .any(|serves| serves.module == checked.name && serves.group == group)
                    })
                    .map(|&(name, table)| (name, lacking(table)))
                    .collect();
                if candidates.iter().any(|(_, lacking)| lacking.is_empty()) {
                    continue;
                }
                let closest = candidates.iter().min_by_key(|(_, lacking)| lacking.len());
                errors.push(match closest {
                    None => format!(
                        "{} `{}` uses group {group}, which no table serves",
                        checked.name, entry.name
                    ),
                    Some((name, lacking)) => format!(
                        "{} `{}` group {group}: no table serving it holds all it uses for its stage; {name} lacks {}",
                        checked.name,
                        entry.name,
                        lacking.join(", ")
                    ),
                });
            }
        }
    }
    if errors.is_empty() {
        Ok(entry_points)
    } else {
        Err(errors.join("\n"))
    }
}

/// Every entry point `engine_render::shaders::ENTRIES` declares against its
/// module: an entry point of the module, of the stage declared, declared
/// once; and every entry point of every module declared there, so a stage
/// no pipeline names does not linger. Returns how many there are.
fn check_entries(modules: &[Checked]) -> Result<usize, String> {
    let mut errors = Vec::new();
    for (index, entry) in ENTRIES.iter().enumerate() {
        if ENTRIES[..index]
            .iter()
            .any(|earlier| earlier.module == entry.module && earlier.name == entry.name)
        {
            errors.push(format!(
                "{} `{}` is declared twice in entry_points!",
                entry.module, entry.name
            ));
        }
        let Some(checked) = modules.iter().find(|checked| checked.name == entry.module) else {
            errors.push(format!(
                "entry point `{}`: {} is not a shader module",
                entry.name, entry.module
            ));
            continue;
        };
        match checked
            .module
            .entry_points
            .iter()
            .find(|point| point.name == entry.name)
        {
            None => errors.push(format!(
                "{} has no entry point `{}` (entry_points! declares one)",
                entry.module, entry.name
            )),
            Some(point) => {
                let declared = match entry.stage {
                    Stage::Vertex => Stages::VERTEX,
                    Stage::Fragment => Stages::FRAGMENT,
                    Stage::Compute => Stages::COMPUTE,
                };
                if stage_of(point.stage) != declared {
                    errors.push(format!(
                        "{} `{}` is a {:?} stage, declared {:?} in entry_points!",
                        entry.module, entry.name, point.stage, entry.stage
                    ));
                }
            }
        }
    }
    for checked in modules {
        for point in &checked.module.entry_points {
            if !ENTRIES
                .iter()
                .any(|entry| entry.module == checked.name && entry.name == point.name)
            {
                errors.push(format!(
                    "{} `{}` is in no entry_points! list (crates/engine_render/src/shaders.rs)",
                    checked.name, point.name
                ));
            }
        }
    }
    if errors.is_empty() {
        Ok(ENTRIES.len())
    } else {
        Err(errors.join(
            "
",
        ))
    }
}

/// Every table's binding numbers, and the shared groups' ranges
/// (`engine_render::bindings::SHARED_GROUPS`): no table holds one number
/// twice; in a shared group, features' ranges do not overlap, each binding
/// of each table serving the group belongs to one feature and lies in its
/// range, and every name a range lists is some table's. Returns how many
/// bindings the ranges place.
fn check_ranges() -> Result<usize, String> {
    let mut errors = Vec::new();
    for &(table_name, table) in TABLES {
        for (index, binding) in table.bindings.iter().enumerate() {
            if table.bindings[..index]
                .iter()
                .any(|earlier| earlier.binding == binding.binding)
            {
                errors.push(format!(
                    "{table_name} holds binding {} twice (`{}`)",
                    binding.binding, binding.name
                ));
            }
        }
    }
    let mut placed = 0;
    for shared in SHARED_GROUPS {
        let at = format!("{} @ {}", shared.module, shared.group);
        for (index, range) in shared.ranges.iter().enumerate() {
            if range.numbers.is_empty() {
                errors.push(format!("{at}: {}'s range is empty", range.feature));
            }
            for earlier in &shared.ranges[..index] {
                if range.numbers.start < earlier.numbers.end
                    && earlier.numbers.start < range.numbers.end
                {
                    errors.push(format!(
                        "{at}: {} ({:?}) overlaps {} ({:?})",
                        range.feature, range.numbers, earlier.feature, earlier.numbers
                    ));
                }
            }
        }
        let serving = TABLES.iter().filter(|(_, table)| {
            table
                .serves
                .iter()
                .any(|serves| serves.module == shared.module && serves.group == shared.group)
        });
        for &(table_name, table) in serving {
            for binding in table.bindings {
                let owners: Vec<_> = shared
                    .ranges
                    .iter()
                    .filter(|range| range.names.contains(&binding.name))
                    .collect();
                match owners.as_slice() {
                    [] => errors.push(format!(
                        "{table_name} binding {} `{}`: no feature of {at} lists it (bindings::SHARED_GROUPS)",
                        binding.binding, binding.name
                    )),
                    [range] if !range.numbers.contains(&binding.binding) => errors.push(format!(
                        "{table_name} binding {} `{}` lies outside {}'s range {:?} in {at}",
                        binding.binding, binding.name, range.feature, range.numbers
                    )),
                    [_] => placed += 1,
                    [..] => errors.push(format!(
                        "{at}: `{}` is listed by several features",
                        binding.name
                    )),
                }
            }
        }
        for range in shared.ranges {
            for name in range.names {
                let bound = TABLES.iter().any(|(_, table)| {
                    table.serves.iter().any(|serves| {
                        serves.module == shared.module && serves.group == shared.group
                    }) && table.bindings.iter().any(|binding| binding.name == *name)
                });
                if !bound {
                    errors.push(format!(
                        "{at}: {} lists `{name}`, which no table serving it binds",
                        range.feature
                    ));
                }
            }
        }
    }
    if errors.is_empty() {
        Ok(placed)
    } else {
        Err(errors.join(
            "
",
        ))
    }
}

/// What a binding takes of the limits per stage.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Resource {
    UniformBuffer,
    StorageBuffer,
    SampledTexture,
    Sampler,
    StorageTexture,
}

impl Resource {
    const ALL: [Resource; 5] = [
        Resource::UniformBuffer,
        Resource::StorageBuffer,
        Resource::SampledTexture,
        Resource::Sampler,
        Resource::StorageTexture,
    ];

    fn of(kind: Kind) -> Resource {
        match kind {
            Kind::Uniform { .. } => Resource::UniformBuffer,
            Kind::Read { .. } | Kind::Write => Resource::StorageBuffer,
            Kind::Texture { .. } => Resource::SampledTexture,
            Kind::Sampler(_) => Resource::Sampler,
            Kind::Storage(_) | Kind::Storage3d(_) => Resource::StorageTexture,
        }
    }

    fn limit(self, limits: &StageLimits) -> u32 {
        match self {
            Resource::UniformBuffer => limits.uniform_buffers,
            Resource::StorageBuffer => limits.storage_buffers,
            Resource::SampledTexture => limits.sampled_textures,
            Resource::Sampler => limits.samplers,
            Resource::StorageTexture => limits.storage_textures,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Resource::UniformBuffer => "uniform buffers",
            Resource::StorageBuffer => "storage buffers",
            Resource::SampledTexture => "sampled textures",
            Resource::Sampler => "samplers",
            Resource::StorageTexture => "storage textures",
        }
    }
}

/// How many of `resource` the bindings of `layout`'s tables visible to
/// `stage` take: every one counts, used or not.
fn stage_count(layout: &PipelineLayout, stage: Stages, resource: Resource) -> u32 {
    let count = layout
        .groups
        .iter()
        .flat_map(|table| table.bindings)
        .filter(|binding| covers(binding.stages, stage) && Resource::of(binding.kind) == resource)
        .count();
    u32::try_from(count).unwrap_or(u32::MAX)
}

/// Every pipeline layout (`engine_render::pipeline_layouts`): its tables
/// serve one shader module's groups at their slots (a table binding nothing
/// stands in anywhere), within the bind groups every device grants; each
/// stage's buffers, textures and samplers, counted over its tables'
/// bindings visible to the stage, within the limits every device grants
/// ([`REQUIRED_LIMITS`]), or within larger ones it names, which must be what
/// it takes; its dynamic uniform buffers within the limit; no two layouts of
/// the same tables, every table in some layout, no two tables of one label
/// (pipeline layouts check their groups by it); and as many layouts as
/// [`LAYOUT_BUDGET`]. Returns how many layouts there are.
fn check_pipeline_layouts() -> Result<usize, String> {
    let mut errors = Vec::new();
    for (index, &(name, table)) in TABLES.iter().enumerate() {
        if let Some((earlier, _)) = TABLES[..index]
            .iter()
            .find(|(_, other)| other.label == table.label)
        {
            errors.push(format!(
                "{name} and {earlier} share the label {}",
                table.label
            ));
        }
        if !PIPELINE_LAYOUTS
            .iter()
            .any(|(_, layout)| layout.groups.iter().any(|group| group.label == table.label))
        {
            errors.push(format!(
                "{name} is in no pipeline layout (engine_render::pipeline_layouts)"
            ));
        }
    }
    let stages = [
        (Stages::VERTEX, "vertex"),
        (Stages::FRAGMENT, "fragment"),
        (Stages::COMPUTE, "compute"),
    ];
    for (index, &(name, layout)) in PIPELINE_LAYOUTS.iter().enumerate() {
        let limits = &layout.limits;
        if layout.groups.len() > limits.bind_groups as usize {
            errors.push(format!(
                "{name}: {} groups, beyond the {} every device grants",
                layout.groups.len(),
                limits.bind_groups
            ));
        }
        let serves_slots = |module: &str| {
            layout.groups.iter().zip(0u32..).all(|(table, slot)| {
                table.bindings.is_empty()
                    || table
                        .serves
                        .iter()
                        .any(|serves| serves.module == module && serves.group == slot)
            })
        };
        if !layout
            .groups
            .iter()
            .flat_map(|table| table.serves)
            .any(|serves| serves_slots(serves.module))
        {
            errors.push(format!(
                "{name}: no shader module's groups are its tables at their slots (the tables' `serves`)"
            ));
        }
        if let Some((earlier, _)) = PIPELINE_LAYOUTS[..index].iter().find(|(_, other)| {
            other.groups.len() == layout.groups.len()
                && other
                    .groups
                    .iter()
                    .zip(layout.groups)
                    .all(|(a, b)| a.label == b.label)
        }) {
            errors.push(format!("{name} binds the same tables as {earlier}"));
        }
        let dynamic = layout
            .groups
            .iter()
            .flat_map(|table| table.bindings)
            .filter(|binding| matches!(binding.kind, Kind::Uniform { dynamic: true, .. }))
            .count();
        if dynamic > limits.dynamic_uniform_buffers as usize {
            errors.push(format!(
                "{name}: {dynamic} dynamic uniform buffers, beyond the {} every device grants",
                limits.dynamic_uniform_buffers
            ));
        }
        for resource in Resource::ALL {
            let (limit, required) = (resource.limit(limits), resource.limit(&REQUIRED_LIMITS));
            let mut most = 0;
            for (stage, stage_name) in stages {
                let used = stage_count(layout, stage, resource);
                most = most.max(used);
                if used > limit {
                    errors.push(format!(
                        "{name}: its {stage_name} stage binds {used} {}, beyond the {limit} {}",
                        resource.name(),
                        if limit == required {
                            "every device grants (name a larger limit with `needing`, and check the device where the layout is built)"
                        } else {
                            "it names"
                        }
                    ));
                }
            }
            if limit > required && most != limit {
                errors.push(format!(
                    "{name} names {limit} {} but binds at most {most} to a stage",
                    resource.name()
                ));
            }
        }
    }
    if PIPELINE_LAYOUTS.len() != LAYOUT_BUDGET {
        errors.push(format!(
            "{} pipeline layouts, against a budget of {LAYOUT_BUDGET}: each distinct layout costs Firefox's D3D12 backend about 150 ms of startup, so the budget follows the layouts only deliberately (engine_render::pipeline_layouts::LAYOUT_BUDGET)",
            PIPELINE_LAYOUTS.len()
        ));
    }
    if errors.is_empty() {
        Ok(PIPELINE_LAYOUTS.len())
    } else {
        Err(errors.join("\n"))
    }
}

/// `word` spelled out where it is a vector or matrix alias: `vec3f` as
/// `vec3<f32>`.
fn spelled_out(word: &str) -> Option<[String; 4]> {
    let scalar = |suffix: u8| match suffix {
        b'f' => Some("f32"),
        b'i' => Some("i32"),
        b'u' => Some("u32"),
        b'h' => Some("f16"),
        _ => None,
    };
    let (generic, suffix) = match word.as_bytes() {
        [b'v', b'e', b'c', b'2'..=b'4', suffix] => (&word[..4], *suffix),
        [b'm', b'a', b't', b'2'..=b'4', b'x', b'2'..=b'4', suffix] => (&word[..6], *suffix),
        _ => return None,
    };
    Some([
        generic.to_string(),
        "<".to_string(),
        scalar(suffix)?.to_string(),
        ">".to_string(),
    ])
}

/// The tokens of WGSL `source`, without comments, aliases spelled out.
fn wgsl_tokens(source: &str) -> Vec<String> {
    const OPERATORS: [&str; 18] = [
        "->", "<<=", ">>=", "<=", ">=", "==", "!=", "&&", "||", "<<", ">>", "+=", "-=", "*=", "/=",
        "%=", "&=", "|=",
    ];
    let bytes = source.as_bytes();
    let mut tokens = Vec::new();
    let mut at = 0;
    while at < bytes.len() {
        let rest = &source[at..];
        let byte = bytes[at];
        if byte.is_ascii_whitespace() {
            at += 1;
        } else if rest.starts_with("//") {
            at += rest.find('\n').unwrap_or(rest.len());
        } else if rest.starts_with("/*") {
            // Block comments nest.
            let mut depth = 0;
            while at < bytes.len() {
                if source[at..].starts_with("/*") {
                    depth += 1;
                    at += 2;
                } else if source[at..].starts_with("*/") {
                    depth -= 1;
                    at += 2;
                    if depth == 0 {
                        break;
                    }
                } else {
                    at += 1;
                }
            }
        } else if byte.is_ascii_alphanumeric()
            || byte == b'_'
            || (byte == b'.' && bytes.get(at + 1).is_some_and(u8::is_ascii_digit))
        {
            // A word or a number, exponent signs included.
            let number = byte.is_ascii_digit() || byte == b'.';
            let mut end = at + 1;
            while let Some(&next) = bytes.get(end) {
                let sign = number
                    && matches!(next, b'+' | b'-')
                    && matches!(bytes[end - 1], b'e' | b'E' | b'p' | b'P');
                if next.is_ascii_alphanumeric() || next == b'_' || (number && next == b'.') || sign
                {
                    end += 1;
                } else {
                    break;
                }
            }
            let word = &source[at..end];
            match spelled_out(word) {
                Some(spelled) => tokens.extend(spelled),
                None => tokens.push(word.to_string()),
            }
            at = end;
        } else {
            let length = OPERATORS
                .iter()
                .find(|operator| rest.starts_with(**operator))
                .map_or_else(
                    || rest.chars().next().map_or(1, char::len_utf8),
                    |operator| operator.len(),
                );
            tokens.push(rest[..length].to_string());
            at += length;
        }
    }
    tokens
}

/// The functions of WGSL `tokens`: each one's name, and its signature and
/// body with its parameters and locals renamed in the order they are
/// declared, so the same code under other names compares equal.
fn wgsl_functions(tokens: &[String]) -> Vec<(String, Vec<String>)> {
    let identifier = |token: &str| {
        token
            .chars()
            .next()
            .is_some_and(|first| first.is_ascii_alphabetic() || first == '_')
    };
    let mut functions = Vec::new();
    let mut at = 0;
    while at + 1 < tokens.len() {
        if tokens[at] != "fn" {
            at += 1;
            continue;
        }
        let name = tokens[at + 1].clone();
        // The signature runs to the body's brace; the body to its match.
        let Some(open) = tokens[at..].iter().position(|token| token == "{") else {
            break;
        };
        let code_start = at + 2;
        let open = at + open;
        let mut depth = 0usize;
        let mut close = tokens.len() - 1;
        for (index, token) in tokens.iter().enumerate().skip(open) {
            match token.as_str() {
                "{" => depth += 1,
                "}" => {
                    depth -= 1;
                    if depth == 0 {
                        close = index;
                        break;
                    }
                }
                _ => {}
            }
        }
        let code = &tokens[code_start..=close];
        let signature = open - code_start;
        // Parameters: names before a colon directly inside the signature's
        // parentheses. Locals: names after let, const and var (or var<..>).
        let mut renamed: Vec<(&str, String)> = Vec::new();
        let mut parentheses = 0usize;
        for (index, token) in code.iter().enumerate() {
            if index < signature {
                match token.as_str() {
                    "(" => parentheses += 1,
                    ")" => parentheses = parentheses.saturating_sub(1),
                    _ => {}
                }
                if parentheses == 1
                    && code.get(index + 1).is_some_and(|next| next == ":")
                    && identifier(token)
                {
                    renamed.push((token.as_str(), format!("$p{}", renamed.len())));
                }
            } else if matches!(token.as_str(), "let" | "var" | "const") {
                let mut local = index + 1;
                if code.get(local).is_some_and(|next| next == "<") {
                    while code.get(local).is_some_and(|next| next != ">") {
                        local += 1;
                    }
                    local += 1;
                }
                if let Some(name) = code.get(local).filter(|name| identifier(name)) {
                    renamed.push((name.as_str(), format!("$l{}", renamed.len())));
                }
            }
        }
        let normalized = code
            .iter()
            .enumerate()
            .map(|(index, token)| {
                // Fields keep their names.
                let field = index > 0 && code[index - 1] == ".";
                renamed
                    .iter()
                    .find(|(name, _)| !field && *name == token.as_str())
                    .map_or_else(|| token.clone(), |(_, renaming)| renaming.clone())
            })
            .collect();
        functions.push((name, normalized));
        at = close + 1;
    }
    functions
}

/// No WGSL function written twice: functions of the same signature and
/// body, whatever the names of the functions, their parameters and their
/// locals, in one file or two, are one function, which a file the modules
/// using it list (`engine_render::shaders`) holds once. Returns how many
/// functions there are.
fn check_duplicate_bodies() -> Result<usize, String> {
    let mut files: Vec<(&str, &str)> = Vec::new();
    for &(file, source) in engine_render::shaders::FILES {
        if !files.iter().any(|(seen, _)| *seen == file) {
            files.push((file, source));
        }
    }
    let mut functions: Vec<(&str, String, Vec<String>)> = Vec::new();
    let mut errors = Vec::new();
    for (file, source) in files {
        for (name, code) in wgsl_functions(&wgsl_tokens(source)) {
            if let Some((other_file, other, _)) = functions
                .iter()
                .find(|(_, _, other_code)| *other_code == code)
            {
                errors.push(format!(
                    "{file}'s `{name}` is {other_file}'s `{other}` written again: keep one, in a file the modules using it list (engine_render::shaders)"
                ));
            }
            functions.push((file, name, code));
        }
    }
    if errors.is_empty() {
        Ok(functions.len())
    } else {
        Err(errors.join("\n"))
    }
}

/// The module-scope constants of `module` with a literal value.
fn wgsl_constants(module: &naga::Module) -> Vec<(&str, f64)> {
    module
        .constants
        .iter()
        .filter_map(|(_, constant)| {
            let value = match module.global_expressions[constant.init] {
                naga::Expression::Literal(literal) => match literal {
                    naga::Literal::F64(value) | naga::Literal::AbstractFloat(value) => value,
                    naga::Literal::F32(value) => f64::from(value),
                    naga::Literal::U32(value) => f64::from(value),
                    naga::Literal::I32(value) => f64::from(value),
                    naga::Literal::U16(value) => f64::from(value),
                    naga::Literal::I16(value) => f64::from(value),
                    naga::Literal::AbstractInt(value) | naga::Literal::I64(value) => value as f64,
                    naga::Literal::U64(value) => value as f64,
                    _ => return None,
                },
                _ => return None,
            };
            Some((constant.name.as_deref()?, value))
        })
        .collect()
}

/// The value a mirror names in `checked`: a constant `NAME`, the workgroup
/// side of entry point `@entry` (each dimension the side or 1), or the size
/// of struct `size_of Name`.
fn mirrored(checked: &Checked, mirror: &str) -> Option<f64> {
    if let Some(entry) = mirror.strip_prefix('@') {
        let entry = checked
            .module
            .entry_points
            .iter()
            .find(|point| point.name == entry)?;
        let side = entry.workgroup_size[0];
        let square = entry
            .workgroup_size
            .iter()
            .all(|&size| size == side || size == 1);
        return Some(if square { f64::from(side) } else { f64::NAN });
    }
    if let Some(name) = mirror.strip_prefix("size_of ") {
        return checked.module.types.iter().find_map(|(_, ty)| {
            (ty.name.as_deref() == Some(name))
                .then(|| f64::from(ty.inner.size(checked.module.to_ctx())))
        });
    }
    wgsl_constants(&checked.module)
        .into_iter()
        .find_map(|(name, value)| (name == mirror).then_some(value))
}

/// Every shared constant (`engine_render::gpu_layouts::constants`) against
/// its WGSL mirrors; WGSL constants of one name in several modules against
/// each other; and Rust constants of the renderer named as a WGSL constant
/// is, which belong with the shared ones.
fn check_constants(root: &Path, modules: &[Checked]) -> Result<usize, String> {
    let mut errors = Vec::new();
    let declared: Vec<(&str, Vec<(&str, f64)>)> = modules
        .iter()
        .map(|checked| (checked.name, wgsl_constants(&checked.module)))
        .collect();
    let mut mirrors = 0;
    for constant in constants() {
        for mirror in constant.mirrors {
            let mut found = false;
            for checked in modules {
                if mirror.module.is_some_and(|name| name != checked.name) {
                    continue;
                }
                let Some(value) = mirrored(checked, mirror.name) else {
                    continue;
                };
                found = true;
                mirrors += 1;
                if value != constant.value {
                    errors.push(format!(
                        "{} `{}` is {value} where its Rust mirror {} is {}",
                        checked.name, mirror.name, constant.rust, constant.value
                    ));
                }
            }
            if !found {
                errors.push(format!(
                    "{}: no shader module {} declares its mirror `{}`",
                    constant.rust,
                    mirror.module.unwrap_or("of engine_render::shaders"),
                    mirror.name
                ));
            }
        }
    }
    // One name, one value, in every module.
    let mut by_name: std::collections::BTreeMap<&str, Vec<(&str, f64)>> = Default::default();
    for (module, values) in &declared {
        for &(name, value) in values {
            by_name.entry(name).or_default().push((module, value));
        }
    }
    for (name, uses) in &by_name {
        if uses.iter().any(|&(_, value)| value != uses[0].1) {
            let spelled: Vec<String> = uses
                .iter()
                .map(|(module, value)| format!("{value} in {module}"))
                .collect();
            errors.push(format!(
                "WGSL `{name}` differs between modules ({}); name differing values apart",
                spelled.join(", ")
            ));
        }
    }
    // Rust constants named as WGSL ones are the registered ones, declared
    // in gpu_layouts, or are named apart.
    fn walk(directory: &Path, files: &mut Vec<std::path::PathBuf>) {
        for entry in std::fs::read_dir(directory).into_iter().flatten().flatten() {
            let path = entry.path();
            if path.is_dir() {
                if path.file_name().is_some_and(|name| name != "gpu_layouts") {
                    walk(&path, files);
                }
            } else if path.extension().is_some_and(|extension| extension == "rs") {
                files.push(path);
            }
        }
    }
    let mut files = Vec::new();
    walk(&root.join("crates/engine_render/src"), &mut files);
    for file in files {
        let Ok(text) = std::fs::read_to_string(&file) else {
            continue;
        };
        for (index, line) in text.lines().enumerate() {
            let item = match line.strip_prefix("pub") {
                Some(rest) if rest.starts_with('(') => {
                    rest.split_once(") ").map_or(line, |(_, item)| item)
                }
                Some(rest) => rest.strip_prefix(' ').unwrap_or(line),
                None => line,
            };
            let Some(rest) = item.strip_prefix("const ") else {
                continue;
            };
            let name: String = rest
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            if rest[name.len()..].trim_start().starts_with(':')
                && by_name.contains_key(name.as_str())
            {
                errors.push(format!(
                    "{}:{}: `{name}` shares its name with a WGSL constant; declare it once in \
                     engine_render::gpu_layouts::constants with its mirrors, or name it apart",
                    file.strip_prefix(root).unwrap_or(&file).display(),
                    index + 1
                ));
            }
        }
    }
    if errors.is_empty() {
        Ok(mirrors)
    } else {
        Err(errors.join("\n"))
    }
}

/// The words of `source` outside its comments, counted.
fn word_counts(source: &str) -> engine_core::collections::HashMap<&str, usize> {
    let mut counts = engine_core::collections::HashMap::default();
    for line in source.lines() {
        let code = line.split("//").next().unwrap_or_default();
        for word in code.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_')) {
            if !word.is_empty() {
                *counts.entry(word).or_default() += 1;
            }
        }
    }
    counts
}

/// Shader items nothing uses: functions, variables, structs and overrides
/// no entry point of any module holding them reaches (naga's compaction
/// drops them), and constants no module holding them names again.
fn check_unused(modules: &[Checked]) -> Result<(), String> {
    use std::collections::{BTreeMap, BTreeSet};
    fn reachable(module: &naga::Module) -> Vec<String> {
        let mut items = Vec::new();
        items.extend(module.functions.iter().filter_map(|(_, function)| {
            function
                .name
                .as_ref()
                .map(|name| format!("function `{name}`"))
        }));
        items.extend(module.global_variables.iter().filter_map(|(_, global)| {
            global
                .name
                .as_ref()
                .map(|name| format!("variable `{name}`"))
        }));
        items.extend(
            module.overrides.iter().filter_map(|(_, value)| {
                value.name.as_ref().map(|name| format!("override `{name}`"))
            }),
        );
        items.extend(module.types.iter().filter_map(|(_, ty)| match ty.inner {
            naga::TypeInner::Struct { .. } => {
                ty.name.as_ref().map(|name| format!("struct `{name}`"))
            }
            _ => None,
        }));
        items
    }
    let mut held: BTreeMap<String, BTreeSet<&str>> = BTreeMap::new();
    let mut used = BTreeSet::new();
    for checked in modules {
        for item in reachable(&checked.module) {
            held.entry(item).or_default().insert(checked.name);
        }
        let mut compacted = checked.module.clone();
        naga::compact::compact(&mut compacted, naga::compact::KeepUnused::No);
        used.extend(reachable(&compacted));
        let source = engine_render::shaders::MODULES
            .iter()
            .find_map(|&(name, source)| (name == checked.name).then_some(source))
            .unwrap_or_default();
        let words = word_counts(source);
        for name in checked
            .module
            .constants
            .iter()
            .filter_map(|(_, constant)| constant.name.as_ref())
        {
            let item = format!("constant `{name}`");
            held.entry(item.clone()).or_default().insert(checked.name);
            if words.get(name.as_str()).copied().unwrap_or_default() > 1 {
                used.insert(item);
            }
        }
    }
    let unused: Vec<String> = held
        .iter()
        .filter(|(item, _)| !used.contains(*item))
        .map(|(item, modules)| {
            format!(
                "{item} of {} is used nowhere; remove it",
                modules.iter().copied().collect::<Vec<_>>().join(", ")
            )
        })
        .collect();
    if unused.is_empty() {
        Ok(())
    } else {
        Err(unused.join(
            "
",
        ))
    }
}

/// Upload structs (`Pod` derives) outside `gpu_layouts`, which no check
/// would see.
fn check_upload_structs(root: &Path) -> Result<(), String> {
    fn walk(directory: &Path, found: &mut Vec<String>) {
        for entry in std::fs::read_dir(directory).into_iter().flatten().flatten() {
            let path = entry.path();
            if path.is_dir() {
                if path.file_name().is_some_and(|name| name != "gpu_layouts") {
                    walk(&path, found);
                }
            } else if path.extension().is_some_and(|extension| extension == "rs")
                && std::fs::read_to_string(&path).is_ok_and(|text| {
                    text.contains("Pod") && text.contains("derive(") && derives_pod(&text)
                })
            {
                found.push(path.display().to_string());
            }
        }
    }
    fn derives_pod(text: &str) -> bool {
        text.split("#[derive(").skip(1).any(|rest| {
            rest.split(')')
                .next()
                .is_some_and(|list| list.split(',').any(|item| item.trim() == "Pod"))
        })
    }
    let mut found = Vec::new();
    walk(&root.join("crates/engine_render/src"), &mut found);
    let mut errors = Vec::new();
    if !found.is_empty() {
        errors.push(format!(
            "upload structs belong in engine_render::gpu_layouts, where the GPU check sees them: {}",
            found.join(", ")
        ));
    }
    // Sizes are this check's to know: docs restating them go stale.
    let layouts = root.join("crates/engine_render/src/gpu_layouts");
    for entry in std::fs::read_dir(&layouts).into_iter().flatten().flatten() {
        let Ok(text) = std::fs::read_to_string(entry.path()) else {
            continue;
        };
        for (index, line) in text.lines().enumerate() {
            let Some(doc) = line.trim_start().strip_prefix("///") else {
                continue;
            };
            let words: Vec<&str> = doc
                .split(|c: char| c.is_whitespace() || c == '-' || c == '(' || c == ')')
                .collect();
            if words.windows(2).any(|pair| {
                pair[0].parse::<u64>().is_ok()
                    && pair[1]
                        .trim_end_matches(['.', ',', ';', ':'])
                        .starts_with("byte")
            }) {
                errors.push(format!(
                    "{}:{}: a doc restating a byte size; the GPU check holds sizes",
                    entry
                        .path()
                        .strip_prefix(root)
                        .unwrap_or(&entry.path())
                        .display(),
                    index + 1
                ));
            }
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join(
            "
",
        ))
    }
}

/// Runs every check of the renderer's shaders.
pub fn check(root: &Path) -> Result<(), String> {
    let modules = modules()?;
    let mirrors = check_layouts(&modules);
    let entry_points = check_bindings(&modules);
    let entries = check_entries(&modules);
    let inter_stage = inter_stage::check(&modules);
    let ranges = check_ranges();
    let pipeline_layouts = check_pipeline_layouts();
    let functions = check_duplicate_bodies();
    let shared = check_constants(root, &modules);
    let unused = check_unused(&modules);
    let structs = check_upload_structs(root);
    let errors: Vec<String> = [
        mirrors.as_ref().err(),
        entry_points.as_ref().err(),
        entries.as_ref().err(),
        inter_stage.as_ref().err(),
        ranges.as_ref().err(),
        pipeline_layouts.as_ref().err(),
        functions.as_ref().err(),
        shared.as_ref().err(),
        unused.as_ref().err(),
        structs.as_ref().err(),
    ]
    .into_iter()
    .flatten()
    .cloned()
    .collect();
    if !errors.is_empty() {
        return Err(errors.join(
            "
",
        ));
    }
    println!(
        "Validated {} shader modules with naga; {} WGSL mirrors match their {} upload structs; {} tables hold the bindings of {} entry points, each declared with its stage ({}); {} fragment stages read only locations a vertex stage of their module writes alike; {} bindings of shared groups within their features' ranges; {} pipeline layouts within every device's limits and their budget; {} WGSL functions, none written twice; {} WGSL constants mirror {} Rust ones.",
        modules.len(),
        mirrors?,
        layouts().len(),
        TABLES.len(),
        entry_points?,
        entries?,
        inter_stage?,
        ranges?,
        pipeline_layouts?,
        functions?,
        shared?,
        constants().count()
    );
    Ok(())
}
