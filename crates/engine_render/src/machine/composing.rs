use super::*;

pub(crate) struct Composing {
    pub(super) linker: Linker,
    pub(super) declared: Declared,
    pub(super) outputs: Vec<OutputDecl>,
    pub(super) switches: Vec<(u32, engine_registry::GateId)>,
    pub(super) core_frame: Vec<(String, String, Ty)>,
    pub(super) numbers: Vec<u32>,
    pub(super) variables: Vec<FrameVariable>,
    pub(super) frame_bound: BTreeMap<u32, Bound>,
    pub(super) material_entries: Vec<wgpu::BindGroupLayoutEntry>,
    pub(super) depth_writes: Vec<bool>,
    pub(super) reaches: Vec<Reach>,
    pub(super) staged: Vec<(PipelineDecl, Vec<Stage>, Reached)>,
}

pub(crate) fn compose(
    mut linker: Linker,
    (declared, outputs): (Declared, Vec<OutputDecl>),
    switches: Vec<(u32, engine_registry::GateId)>,
    problems: &mut Vec<Error>,
) -> Composing {
    let numbered = linker.number();
    let core_frame: Vec<(String, String, Ty)> = numbered
        .iter()
        .map(|(path, name, _, ty)| (path.clone(), name.clone(), *ty))
        .collect();
    let numbers: Vec<u32> = numbered
        .iter()
        .filter_map(|(path, name, _, _)| linker.binding_of(path, name))
        .collect();
    let variables: Vec<FrameVariable> = numbered
        .iter()
        .map(|(_, name, package, ty)| FrameVariable {
            brick: package,
            name: Box::leak(name.clone().into_boxed_str()),
            kind: class(*ty, name == crate::linker::CAMERA),
        })
        .collect();
    let mut frame_bound = BTreeMap::new();
    for bind in &declared.frame {
        let (path, name) = crate::linker::path_of((bind.var.module(), bind.var.name()));
        if let Some(number) = linker.binding_of(path, &name) {
            frame_bound.insert(number, bind.bound.clone());
        }
    }
    let material_group = crate::linker::declarations(
        linker
            .text(crate::surface::MATERIALS_MODULE)
            .unwrap_or_default(),
    );
    let material_entries: Vec<wgpu::BindGroupLayoutEntry> = material_group
        .iter()
        .map(crate::surface::material_entry)
        .collect();
    let depth_writes = depth_writes_of(&declared);
    let mut reaches = Vec::new();
    let mut staged: Vec<(PipelineDecl, Vec<Stage>, Reached)> = Vec::new();
    for (index, decl) in declared.pipelines.iter().enumerate() {
        let path = format!("$.pipelines.{index}");
        let named = format!(
            "{path}.{}",
            decl.entries()
                .iter()
                .map(|entry| entry.name())
                .collect::<Vec<_>>()
                .join("+")
        );
        let Some((stages, reached)) = linked(
            (&declared, decl),
            (&outputs, &mut linker),
            (&named, &mut *problems),
        ) else {
            continue;
        };
        let label: &'static str = Box::leak(named.into_boxed_str());
        let is_material = |module: &str, group: u32| group == 1 && linker.is_core(module);
        let mut own: Vec<(Class, ShaderStages)> = reached
            .iter()
            .filter(|(module, found, _)| found.group > 0 && !is_material(module, found.group))
            .map(|(_, found, stages)| (class(found.ty, false), *stages))
            .collect();
        if reached
            .iter()
            .any(|(module, found, _)| is_material(module, found.group))
        {
            own.extend(
                material_group
                    .iter()
                    .zip(&material_entries)
                    .map(|(declared, entry)| (class(declared.ty, false), entry.visibility)),
            );
        }
        let groups = reached
            .iter()
            .filter(|(_, found, _)| found.group > 0)
            .map(|(_, found, _)| found.group)
            .max()
            .unwrap_or(0);
        let frame_reached: Vec<(u32, ShaderStages)> = reached
            .iter()
            .filter(|(_, found, _)| found.group == 0)
            .filter_map(|(module, found, stages)| {
                let number = linker.binding_of(module, &found.name)?;
                let at = numbers.iter().position(|known| *known == number)?;
                Some((at as u32, *stages))
            })
            .collect();
        let pass = match decl.scope {
            Scoped::Pass(pass) => pass,
            Scoped::Scene(_) | Scoped::Outside(_) => u32::MAX - decl.part,
        };
        reaches.push(Reach {
            label,
            pass,
            alone: matches!(decl.scope, Scoped::Outside(_)),
            own_groups: groups,
            own,
            reached: frame_reached,
            fragment: fragment_key(decl, &stages),
        });
        staged.push((decl.clone(), stages, reached));
    }
    Composing {
        linker,
        declared,
        outputs,
        switches,
        core_frame,
        numbers,
        variables,
        frame_bound,
        material_entries,
        depth_writes,
        reaches,
        staged,
    }
}

fn fragment_key(decl: &PipelineDecl, stages: &[Stage]) -> Option<u64> {
    let rendered = matches!(decl.program, Program::Render(_));
    stages.get(1).filter(|_| rendered).map(|stage| {
        let mut hasher = DefaultHasher::new();
        (stage.entry.name(), &stage.composed.text).hash(&mut hasher);
        hasher.finish()
    })
}

fn linked(
    (declared, decl): (&Declared, &PipelineDecl),
    (outputs, linker): (&[OutputDecl], &mut Linker),
    (named, problems): (&str, &mut Vec<Error>),
) -> Option<(Vec<Stage>, Reached)> {
    let writes = written_module(declared, decl, (outputs, &*linker), (named, &mut *problems))?;
    let (mut stages, mut reached, mut failed) = (Vec::new(), Reached::new(), false);
    for entry in decl.entries() {
        match linker.compose_in(entry, writes.as_ref()) {
            Ok(composed) => {
                let at = stage_of(entry, linker);
                for (module, declaration) in linker.reached(&composed) {
                    match reached.iter_mut().find(|(known, found, _)| {
                        *known == module && found.name == declaration.name
                    }) {
                        Some((_, _, stages)) => *stages |= at,
                        None => reached.push((module, declaration, at)),
                    }
                }
                stages.push(Stage { entry, composed });
            }
            Err(error) => {
                problems.push(error);
                failed = true;
            }
        }
    }
    if failed {
        return None;
    }
    for (var, _) in &decl.bound {
        let (module, name) = crate::linker::path_of((var.module(), var.name()));
        if !reached
            .iter()
            .any(|(known, found, _)| known == module && found.name == name)
        {
            problems.push(Error::new(ErrorCode::UnknownVariable, named).got(var.name()));
            failed = true;
        }
    }
    for (module, found, _) in &reached {
        let core = linker.is_core(module);
        let bound = decl.bound.iter().any(|(var, _)| {
            let (known, name) = crate::linker::path_of((var.module(), var.name()));
            known == module && name == found.name
        });
        if found.group > 0 && !core && !bound {
            problems.push(Error::new(ErrorCode::UnknownVariable, named).got(&found.name));
            failed = true;
        }
    }
    (!failed).then_some((stages, reached))
}
