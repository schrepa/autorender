//! The renderer's shaders checked against its Rust side: every module
//! parses and validates with naga, and every WGSL struct mirroring an upload
//! struct (`engine_render::gpu_layouts`) matches its Rust layout.

use std::path::Path;

use engine_render::bindings::{
    Binding, Format, Kind, Sample, Sampling, Stages, TABLES, Table, View,
};
use engine_render::gpu_layouts::{Layout, constants, layouts};
use naga::valid::{Capabilities, ValidationFlags, Validator};

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
fn word_counts(source: &str) -> std::collections::HashMap<&str, usize> {
    let mut counts = std::collections::HashMap::new();
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
    let shared = check_constants(root, &modules);
    let unused = check_unused(&modules);
    let structs = check_upload_structs(root);
    let errors: Vec<String> = [
        mirrors.as_ref().err(),
        entry_points.as_ref().err(),
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
        "Validated {} shader modules with naga; {} WGSL mirrors match their {} upload structs; {} tables hold the bindings of {} entry points; {} WGSL constants mirror {} Rust ones.",
        modules.len(),
        mirrors?,
        layouts().len(),
        TABLES.len(),
        entry_points?,
        shared?,
        constants().count()
    );
    Ok(())
}
