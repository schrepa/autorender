use super::Checked;

#[derive(Clone, Debug, PartialEq)]
struct Varying {
    ty: naga::TypeInner,
    interpolation: Option<naga::Interpolation>,
    sampling: Option<naga::Sampling>,
}

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

type Written<'m> = (&'m str, &'m str, Vec<(u32, Varying)>);

fn written(modules: &[Checked]) -> Vec<Written<'_>> {
    let mut written = Vec::new();
    for checked in modules {
        let module = &checked.module;
        for point in module
            .entry_points
            .iter()
            .filter(|point| point.stage == naga::ShaderStage::Vertex)
        {
            let result = point.function.result.as_ref();
            let values = result
                .map(|result| varyings(module, result.ty, result.binding.as_ref()))
                .unwrap_or_default();
            written.push((checked.name.as_str(), point.name.as_str(), values));
        }
    }
    written
}

pub(super) fn check(modules: &[Checked]) -> Result<usize, String> {
    let mut errors = Vec::new();
    let mut fragments = 0;
    let written = written(modules);
    for checked in modules {
        let module = &checked.module;
        for point in module
            .entry_points
            .iter()
            .filter(|point| point.stage == naga::ShaderStage::Fragment)
        {
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
            if written
                .iter()
                .any(|(_, _, values)| missing(values).is_empty())
            {
                continue;
            }
            let closest = written
                .iter()
                .map(|(owner, name, values)| (owner, name, missing(values)))
                .min_by_key(|(_, _, missing)| missing.len());
            errors.push(match closest {
                Some((owner, name, missing)) => format!(
                    "{} `{}` reads locations no vertex stage writes alike: the closest, {owner} `{name}`, does not write {missing:?} with their type, interpolation and sampling",
                    checked.name, point.name
                ),
                None => format!(
                    "{} `{}` reads locations, but no module has a vertex stage",
                    checked.name, point.name
                ),
            });
        }
    }
    if errors.is_empty() {
        Ok(fragments)
    } else {
        Err(errors.join("\n"))
    }
}
