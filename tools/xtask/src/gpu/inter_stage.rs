//! Inter-stage locations: what each fragment stage reads, against what
//! the vertex stages of its module write.

use super::Checked;

/// A value a vertex stage passes to a fragment stage at a location: its
/// type, interpolation and sampling, as naga parsed them (its defaults
/// applied).
#[derive(Clone, Debug, PartialEq)]
struct Varying {
    ty: naga::TypeInner,
    interpolation: Option<naga::Interpolation>,
    sampling: Option<naga::Sampling>,
}

/// The locations a value of `ty` bound by `binding` carries: the value
/// when bound to a location, else its struct's members bound to locations.
fn varyings(
    module: &naga::Module,
    ty: naga::Handle<naga::Type>,
    binding: Option<&naga::Binding>,
) -> Vec<(u32, Varying)> {
    match binding {
        Some(naga::Binding::Location {
            location,
            interpolation,
            sampling,
            ..
        }) => vec![(
            *location,
            Varying {
                ty: module.types[ty].inner.clone(),
                interpolation: *interpolation,
                sampling: *sampling,
            },
        )],
        Some(naga::Binding::BuiltIn(_)) => Vec::new(),
        None => match &module.types[ty].inner {
            naga::TypeInner::Struct { members, .. } => members
                .iter()
                .flat_map(|member| varyings(module, member.ty, member.binding.as_ref()))
                .collect(),
            _ => Vec::new(),
        },
    }
}

/// Inter-stage locations: every fragment stage reads only locations some
/// vertex stage of its module writes all of, each of the same type,
/// interpolation and sampling, so some pipeline can pair them (WebGPU
/// refuses a pipeline whose fragment stage reads a location its vertex
/// stage does not write alike). Returns how many fragment stages read
/// locations.
pub(super) fn check(modules: &[Checked]) -> Result<usize, String> {
    let mut errors = Vec::new();
    let mut fragments = 0;
    for checked in modules {
        let module = &checked.module;
        let stages = |stage| {
            module
                .entry_points
                .iter()
                .filter(move |point| point.stage == stage)
        };
        let written: Vec<(&str, Vec<(u32, Varying)>)> = stages(naga::ShaderStage::Vertex)
            .map(|point| {
                let result = point.function.result.as_ref();
                let values = result
                    .map(|result| varyings(module, result.ty, result.binding.as_ref()))
                    .unwrap_or_default();
                (point.name.as_str(), values)
            })
            .collect();
        for point in stages(naga::ShaderStage::Fragment) {
            let read: Vec<(u32, Varying)> = point
                .function
                .arguments
                .iter()
                .flat_map(|argument| varyings(module, argument.ty, argument.binding.as_ref()))
                .collect();
            if read.is_empty() {
                continue;
            }
            fragments += 1;
            let missing = |values: &[(u32, Varying)]| -> Vec<u32> {
                read.iter()
                    .filter(|value| !values.contains(value))
                    .map(|(location, _)| *location)
                    .collect()
            };
            if written.iter().any(|(_, values)| missing(values).is_empty()) {
                continue;
            }
            let closest = written
                .iter()
                .map(|(name, values)| (name, missing(values)))
                .min_by_key(|(_, missing)| missing.len());
            errors.push(match closest {
                Some((name, missing)) => format!(
                    "{} `{}` reads locations no vertex stage of the module writes alike: the closest, `{name}`, does not write {missing:?} with their type, interpolation and sampling",
                    checked.name, point.name
                ),
                None => format!(
                    "{} `{}` reads locations, but the module has no vertex stage",
                    checked.name, point.name
                ),
            });
        }
    }
    if errors.is_empty() {
        Ok(fragments)
    } else {
        Err(errors.join(
            "
",
        ))
    }
}
