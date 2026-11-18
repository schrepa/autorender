use super::*;

type Reach = (String, bool, Vec<(u32, ShaderStages)>);

#[derive(Clone, Debug)]
pub(crate) struct BatchEntry {
    pub(crate) binding: u32,
    pub(crate) name: String,
    pub(crate) ty: Ty,
    pub(crate) stages: ShaderStages,
}

impl BatchEntry {
    pub(crate) fn layout(&self) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding: self.binding,
            visibility: self.stages,
            ty: self.ty.binding(true, false),
            count: None,
        }
    }
}

pub(crate) struct BatchLayout {
    pub(crate) layout: wgpu::BindGroupLayout,
    pub(crate) entries: Vec<BatchEntry>,
}

pub(crate) struct Composing {
    pub(super) linker: Linker,
    pub(super) declared: Declared,
    pub(super) core_frame: Vec<(String, String, Ty)>,
    pub(super) numbers: Vec<u32>,
    pub(super) frame_bound: BTreeMap<u32, Bound>,
    pub(super) batch_entries: Vec<Vec<BatchEntry>>,
    pub(super) depth_writes: Vec<bool>,
    pub(super) reaches: Vec<Reach>,
    pub(super) staged: Vec<(PipelineDecl, Vec<Stage>, Reached)>,
}

pub(crate) fn compose(
    mut linker: Linker,
    declared: Declared,
    problems: &mut Vec<Error>,
) -> Composing {
    let core_frame = linker.number();
    let numbers: Vec<u32> = core_frame
        .iter()
        .filter_map(|(path, name, _)| linker.binding_of(path, name))
        .collect();
    let mut frame_bound = BTreeMap::new();
    for bind in &declared.frame {
        let (path, name) = crate::linker::path_of((bind.var.module(), bind.var.name()));
        if let Some(number) = linker.binding_of(path, &name) {
            frame_bound.insert(number, bind.bound.clone());
        }
    }
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
            (&declared.batch_groups, decl),
            &mut linker,
            (&named, &mut *problems),
        ) else {
            continue;
        };
        let mut frame_reached: Vec<(u32, ShaderStages)> = reached
            .iter()
            .filter(|(_, found, _)| found.group == 0)
            .filter_map(|(module, found, stages)| {
                let number = linker.binding_of(module, &found.name)?;
                let at = numbers.iter().position(|known| *known == number)?;
                Some((at as u32, *stages))
            })
            .collect();
        frame_reached.sort_by_key(|&(variable, _)| variable);
        let alone = matches!(decl.scope, Scoped::Outside(_));
        reaches.push((named, alone, frame_reached));
        staged.push((decl.clone(), stages, reached));
    }
    let batch_entries = batch_entries(&declared, &linker, &staged);
    Composing {
        linker,
        declared,
        core_frame,
        numbers,
        frame_bound,
        batch_entries,
        depth_writes,
        reaches,
        staged,
    }
}

pub(crate) fn batch_path(batch: &crate::install::BatchDecl) -> &'static str {
    crate::linker::path_of((batch.module, "")).0
}

fn batch_entries(
    declared: &Declared,
    linker: &Linker,
    staged: &[(PipelineDecl, Vec<Stage>, Reached)],
) -> Vec<Vec<BatchEntry>> {
    declared
        .batch_groups
        .iter()
        .map(|batch| {
            let path = batch_path(batch);
            crate::linker::declarations(linker.text(path).unwrap_or_default())
                .into_iter()
                .filter(|found| found.group == batch.index)
                .map(|found| {
                    let stages = staged
                        .iter()
                        .flat_map(|(_, _, reached)| reached)
                        .filter(|(module, known, _)| {
                            module.as_str() == path
                                && known.group == found.group
                                && known.binding == found.binding
                        })
                        .fold(ShaderStages::NONE, |all, (_, _, stages)| all | *stages);
                    BatchEntry {
                        binding: found.binding,
                        name: found.name,
                        ty: found.ty,
                        stages: if stages.is_empty() {
                            ShaderStages::VERTEX_FRAGMENT
                        } else {
                            stages
                        },
                    }
                })
                .collect()
        })
        .collect()
}

fn linked(
    (batches, decl): (&[crate::install::BatchDecl], &PipelineDecl),
    linker: &mut Linker,
    (named, problems): (&str, &mut Vec<Error>),
) -> Option<(Vec<Stage>, Reached)> {
    let (mut stages, mut reached, mut failed) = (Vec::new(), Reached::new(), false);
    for entry in decl.entries() {
        match linker.compose_in(entry) {
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
        let batched = batches.iter().any(|batch| batch_path(batch) == module.as_str() && batch.index == found.group);
        if found.group > 0 && !core && !bound && !batched {
            problems.push(Error::new(ErrorCode::UnknownVariable, named).got(&found.name));
            failed = true;
        }
    }
    (!failed).then_some((stages, reached))
}
