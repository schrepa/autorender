use std::collections::{BTreeMap, BTreeSet};

use engine_render::bindings::{
    Format, GroupRef, Kind, ResolvedLayout, Sample, Sampling, SharedGroups, StageCounts, Stages,
    TABLES, View,
};
use engine_render::linker::declared_bindings;
use engine_render::pipeline_layouts::{
    LAYOUT_BUDGET, PIPELINE_LAYOUTS, REQUIRED_LIMITS, StageLimits, UNIFORM_BINDING_BYTES,
};
use engine_render::shaders::MODULES;

use super::{Checked, plain, stage_of};

pub(super) struct Resolved {
    name: &'static str,
    limits: StageLimits,
    layout: ResolvedLayout,
}

pub(super) fn resolve(shared: &SharedGroups) -> (Vec<Resolved>, Vec<String>) {
    let mut resolved = Vec::new();
    let mut errors = Vec::new();
    for &(name, layout) in PIPELINE_LAYOUTS {
        match ResolvedLayout::resolve(layout.label, layout.groups, shared) {
            Ok(found) => resolved.push(Resolved {
                name,
                limits: layout.limits,
                layout: found,
            }),
            Err(error) => errors.push(format!("{name}: {error}")),
        }
    }
    (resolved, errors)
}

#[derive(Clone, Copy, PartialEq)]
enum Fit {
    Exact,
    Larger,
}

fn fit(module: &naga::Module, global: &naga::GlobalVariable, min_size: u64) -> Result<Fit, String> {
    let inner = &module.types[global.ty].inner;
    let size = inner.size(module.to_ctx()) as u64;
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

fn view_of(dim: naga::ImageDimension, arrayed: bool) -> Option<View> {
    match (dim, arrayed) {
        (naga::ImageDimension::D2, false) => Some(View::D2),
        (naga::ImageDimension::D2, true) => Some(View::D2Array),
        (naga::ImageDimension::D3, false) => Some(View::D3),
        (naga::ImageDimension::Cube, false) => Some(View::Cube),
        (naga::ImageDimension::Cube, true) => Some(View::CubeArray),
        _ => None,
    }
}

fn format_of(format: naga::StorageFormat) -> Option<Format> {
    match format {
        naga::StorageFormat::R32Float => Some(Format::R32Float),
        naga::StorageFormat::R32Uint => Some(Format::R32Uint),
        naga::StorageFormat::Rgba8Snorm => Some(Format::Rgba8Snorm),
        naga::StorageFormat::Rgba8Unorm => Some(Format::Rgba8Unorm),
        naga::StorageFormat::Rgba16Float => Some(Format::Rgba16Float),
        _ => None,
    }
}

fn image(class: naga::ImageClass, view: Option<View>, kind: Kind) -> Result<(), &'static str> {
    match (class, kind) {
        (naga::ImageClass::Storage { format: stored, .. }, Kind::Storage(table))
            if view == Some(View::D2) && format_of(stored) == Some(table) =>
        {
            Ok(())
        }
        (naga::ImageClass::Storage { format: stored, .. }, Kind::StorageArray(table))
            if view == Some(View::D2Array) && format_of(stored) == Some(table) =>
        {
            Ok(())
        }
        (naga::ImageClass::Storage { .. }, _) => {
            Err("a storage texture of the shader's format and dimension")
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
            | (naga::ImageClass::Depth { multi: false }, Sample::Depth) => Ok(()),
            _ => Err("a texture of the shader's sample type"),
        },
        _ => Err("a texture of the shader's dimension"),
    }
}

fn mismatch(
    module: &naga::Module,
    global: &naga::GlobalVariable,
    kind: Kind,
) -> Result<Option<Fit>, String> {
    let inner = &module.types[global.ty].inner;
    let sized = |min_size: u64| fit(module, global, min_size).map(Some);
    let expected = match (global.space, inner) {
        (naga::AddressSpace::Uniform, _) => match kind {
            Kind::Uniform { .. } if inner.size(module.to_ctx()) as u64 > UNIFORM_BINDING_BYTES => {
                return Err(format!(
                    "{} bytes, more than the {UNIFORM_BINDING_BYTES} a uniform binding holds on \
                     every device: bind it as read-only storage",
                    inner.size(module.to_ctx())
                ));
            }
            Kind::Uniform { min_size, .. } => return sized(min_size),
            _ => "a uniform buffer",
        },
        (naga::AddressSpace::Storage { access }, _)
            if !access.contains(naga::StorageAccess::STORE) =>
        {
            match kind {
                Kind::Read { min_size } => return sized(min_size),
                _ => "a read-only storage buffer",
            }
        }
        (naga::AddressSpace::Storage { .. }, _) => match kind {
            Kind::Write => return Ok(None),
            _ => "a writable storage buffer",
        },
        (naga::AddressSpace::Handle, naga::TypeInner::Sampler { comparison }) => match kind {
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
        ) => match image(*class, view_of(*dim, *arrayed), kind) {
            Ok(()) => return Ok(None),
            Err(expected) => expected,
        },
        _ => "a kind of binding tables cannot declare",
    };
    Err(format!("the shader declares {expected}, the layout {kind:?}"))
}

type Judged<'r> = (&'r Resolved, Vec<String>, Vec<((&'static str, &'static str), Fit)>);

fn judge<'r>(
    checked: &Checked,
    used: &[(String, &naga::GlobalVariable)],
    resolved: &'r Resolved,
) -> Judged<'r> {
    let mut wrong = Vec::new();
    let mut sized = Vec::new();
    for (name, global) in used {
        let Some(binding) = resolved.layout.binding(name) else {
            continue;
        };
        match mismatch(&checked.module, global, binding.kind) {
            Ok(fit) => sized.extend(fit.map(|fit| ((binding.owner, binding.name), fit))),
            Err(why) => wrong.push(format!("`{name}`: {why}")),
        }
    }
    (resolved, wrong, sized)
}

fn declared_everywhere(shared: &SharedGroups) -> Vec<String> {
    let declared: BTreeSet<&str> = MODULES
        .iter()
        .flat_map(|module| declared_bindings(module.source))
        .collect();
    let mut errors = Vec::new();
    for &(table_name, table) in TABLES {
        for binding in table.bindings {
            if !declared.contains(binding.name) {
                errors.push(format!(
                    "{table_name} binding `{}`: no shader module declares it",
                    binding.name
                ));
            }
        }
    }
    for (group, bindings) in shared.groups() {
        for binding in bindings {
            if !declared.contains(binding.decl.name) {
                errors.push(format!(
                    "shared group {group} ({} {}) binding `{}`: no shader module declares it",
                    binding.brick, binding.set, binding.decl.name
                ));
            }
        }
    }
    errors
}

pub(super) fn check(
    modules: &[Checked],
    resolved: &[Resolved],
    shared: &SharedGroups,
) -> Result<usize, String> {
    let mut errors = declared_everywhere(shared);
    let mut fits: BTreeMap<(&str, &str), Vec<Fit>> = BTreeMap::new();
    let mut entry_points = 0;
    for checked in modules {
        for (index, entry) in checked.module.entry_points.iter().enumerate() {
            entry_points += 1;
            let info = checked.info.get_entry_point(index);
            let stage = stage_of(entry.stage);
            let used: Vec<(String, &naga::GlobalVariable)> = checked
                .module
                .global_variables
                .iter()
                .filter(|(handle, global)| global.binding.is_some() && !info[*handle].is_empty())
                .map(|(_, global)| (plain(global.name.as_deref().unwrap_or_default()), global))
                .collect();
            if used.is_empty() {
                continue;
            }
            let lacking = |layout: &ResolvedLayout| -> Vec<String> {
                used.iter()
                    .filter(|(name, _)| {
                        !layout
                            .binding(name)
                            .is_some_and(|binding| binding.stages.covers(stage))
                    })
                    .map(|(name, _)| format!("`{name}`"))
                    .collect()
            };
            let candidates: Vec<&Resolved> = resolved
                .iter()
                .filter(|candidate| lacking(&candidate.layout).is_empty())
                .collect();
            if candidates.is_empty() {
                let closest = resolved
                    .iter()
                    .map(|candidate| (candidate.name, lacking(&candidate.layout)))
                    .min_by_key(|(_, lacking)| lacking.len());
                errors.push(match closest {
                    None => format!("{} `{}`: no pipeline layout exists", checked.name, entry.name),
                    Some((name, lacking)) => format!(
                        "{} `{}`: no pipeline layout holds every binding it uses, visible to its stage; the closest, {name}, lacks {}",
                        checked.name,
                        entry.name,
                        lacking.join(", ")
                    ),
                });
                continue;
            }
            let judged: Vec<Judged<'_>> = candidates
                .into_iter()
                .map(|candidate| judge(checked, &used, candidate))
                .collect();
            let matching: Vec<&Judged<'_>> =
                judged.iter().filter(|(_, wrong, _)| wrong.is_empty()).collect();
            if matching.is_empty() {
                let (candidate, wrong, _) = &judged[0];
                errors.push(format!(
                    "{} `{}` under {}: {}",
                    checked.name,
                    entry.name,
                    candidate.name,
                    wrong.join("; ")
                ));
            }
            for (_, _, sized) in matching {
                for &(key, found) in sized {
                    fits.entry(key).or_default().push(found);
                }
            }
        }
    }
    for ((owner, name), found) in &fits {
        if !found.contains(&Fit::Exact) {
            errors.push(format!(
                "{owner} binding `{name}`: its minimum size is no shader's type's size"
            ));
        }
    }
    if errors.is_empty() {
        Ok(entry_points)
    } else {
        Err(errors.join("\n"))
    }
}

fn same(a: &GroupRef, b: &GroupRef) -> bool {
    match (a, b) {
        (GroupRef::Own(a), GroupRef::Own(b)) => a.label == b.label,
        (
            GroupRef::Shared {
                group: a,
                without: x,
            },
            GroupRef::Shared {
                group: b,
                without: y,
            },
        ) => a == b && x == y,
        _ => false,
    }
}

fn views(shared: &SharedGroups) -> Vec<String> {
    let mut errors = Vec::new();
    for &(name, layout) in PIPELINE_LAYOUTS {
        for group in layout.groups {
            let GroupRef::Shared { group, without } = *group else {
                continue;
            };
            let Some(bindings) = shared.group(group) else {
                errors.push(format!("{name}: no part contributes the shared group {group}"));
                continue;
            };
            for set in without {
                if !bindings.iter().any(|binding| binding.set == *set) {
                    errors.push(format!(
                        "{name}: the shared group {group} has no set {set} to leave out"
                    ));
                }
            }
        }
    }
    errors
}

fn tables() -> Vec<String> {
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
        let placed = PIPELINE_LAYOUTS.iter().any(|(_, layout)| {
            layout
                .groups
                .iter()
                .any(|group| matches!(group, GroupRef::Own(own) if own.label == table.label))
        });
        if !placed {
            errors.push(format!(
                "{name} is in no pipeline layout (engine_render::pipeline_layouts)"
            ));
        }
    }
    for (index, &(name, layout)) in PIPELINE_LAYOUTS.iter().enumerate() {
        if let Some((earlier, _)) = PIPELINE_LAYOUTS[..index].iter().find(|(_, other)| {
            other.groups.len() == layout.groups.len()
                && other
                    .groups
                    .iter()
                    .zip(layout.groups)
                    .all(|(a, b)| same(a, b))
        }) {
            errors.push(format!("{name} binds the same groups as {earlier}"));
        }
    }
    errors
}

const RESOURCES: [(&str, fn(&StageCounts) -> u32, fn(&StageLimits) -> u32); 5] = [
    (
        "uniform buffers",
        |counts| counts.uniform_buffers,
        |limits| limits.uniform_buffers,
    ),
    (
        "storage buffers",
        |counts| counts.storage_buffers,
        |limits| limits.storage_buffers,
    ),
    (
        "sampled textures",
        |counts| counts.sampled_textures,
        |limits| limits.sampled_textures,
    ),
    ("samplers", |counts| counts.samplers, |limits| limits.samplers),
    (
        "storage textures",
        |counts| counts.storage_textures,
        |limits| limits.storage_textures,
    ),
];

pub(super) fn check_layouts(
    resolved: &[Resolved],
    unresolved: Vec<String>,
    shared: &SharedGroups,
) -> Result<usize, String> {
    let mut errors = unresolved;
    errors.extend(views(shared));
    errors.extend(tables());
    for layout in resolved {
        if let Err(error) = layout.layout.check(&layout.limits) {
            errors.push(format!(
                "{}: {error} (name a larger limit with `needing` only where the layout is built on a device that grants it)",
                layout.name
            ));
        }
        for (what, counted, limit) in RESOURCES {
            let (named, required) = (limit(&layout.limits), limit(&REQUIRED_LIMITS));
            let most = [Stages::VERTEX, Stages::FRAGMENT, Stages::COMPUTE]
                .into_iter()
                .map(|stage| counted(&layout.layout.counts(stage)))
                .max()
                .unwrap_or_default();
            if named > required && most != named {
                errors.push(format!(
                    "{} names {named} {what} but binds at most {most} to a stage",
                    layout.name
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
