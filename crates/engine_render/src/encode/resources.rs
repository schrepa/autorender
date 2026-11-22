use engine_core::Var;

use super::compile::Resolve;
use super::*;

pub(crate) struct HeldBatch {
    group: wgpu::BindGroup,
    resolves: Option<Vec<(u32, Resolve)>>,
}

#[derive(Default)]
pub(crate) struct HeldBatches(Vec<Vec<Option<HeldBatch>>>);

fn key_slot(key: Option<Asset<Material>>) -> usize {
    key.map_or(0, |handle| handle.slot() + 1)
}

fn keyed_get(keyed: &[Option<HeldBatch>], key: Option<Asset<Material>>) -> Option<&HeldBatch> {
    keyed.get(key_slot(key)).and_then(Option::as_ref)
}

impl HeldBatches {
    pub(crate) fn group(&self, batch: u32, key: Option<Asset<Material>>) -> Option<&wgpu::BindGroup> {
        self.0
            .get(batch as usize)
            .and_then(|keyed| keyed_get(keyed, key))
            .map(|held| &held.group)
    }

    pub(crate) fn holds(&self, key: Option<Asset<Material>>) -> bool {
        self.0.iter().any(|keyed| keyed_get(keyed, key).is_some())
    }

    pub(crate) fn resolve_mut(
        &mut self,
        batches: &[(u32, u32)],
        pipeline: u32,
        key: Option<Asset<Material>>,
    ) -> Option<&mut Resolve> {
        let at = batches.iter().map(|&(_, batch)| batch as usize).find(|&batch| {
            self.0
                .get(batch)
                .and_then(|keyed| keyed_get(keyed, key))
                .is_some_and(|held| held.resolves.is_some())
        })?;
        let held = self.0.get_mut(at)?.get_mut(key_slot(key))?.as_mut()?;
        (held.resolves.as_mut()?.iter_mut())
            .find(|(known, _)| *known == pipeline)
            .map(|(_, resolve)| resolve)
    }
}

pub(crate) enum BatchEntryRaw {
    Range { buffer: u32, offset: u64, size: u64 },
    Texture(u32),
    TextureMip(u32, u32),
    Asset(Asset<engine_core::Texture>),
    AssetSampler(engine_core::Sampler),
}

enum Held {
    Range(wgpu::Buffer, u64, u64),
    View(wgpu::TextureView),
    Sampler(wgpu::Sampler),
}

impl Run {
    fn resolve_batch_entry(
        &mut self,
        entry: &crate::machine::BatchEntry,
        given: &BatchEntryRaw,
        group: u32,
        path: &str,
    ) -> Result<Held, Error> {
        match given {
            BatchEntryRaw::Range {
                buffer,
                offset,
                size,
            } => match self.buffer_held(*buffer) {
                Some((made, bytes)) if offset + size <= *bytes => {
                    Ok(Held::Range(made.clone(), *offset, *size))
                }
                Some(_) | None => {
                    Err(fault(path.to_owned(), ErrorCode::InvalidValue).got(entry.name.as_str()))
                }
            },
            BatchEntryRaw::Texture(texture) => {
                let dimension = view_dimension(entry.ty);
                match self
                    .tables
                    .textures
                    .get_mut(*texture as usize)
                    .and_then(Option::as_mut)
                {
                    Some(made) => Ok(Held::View(made.viewed(dimension).clone())),
                    None => self.batch_stand_in(entry, group, path),
                }
            }
            BatchEntryRaw::TextureMip(texture, mip) => {
                let dimension = view_dimension(entry.ty);
                match self
                    .tables
                    .textures
                    .get_mut(*texture as usize)
                    .and_then(Option::as_mut)
                {
                    Some(made) => match made.viewed_mip(dimension, *mip) {
                        Some(view) => Ok(Held::View(view.clone())),
                        None => Err(fault(path.to_owned(), ErrorCode::OutOfRange)
                            .allowed(engine_core::Allowed::Range {
                                min: 0.0,
                                max: f64::from(made.mip_levels.saturating_sub(1)),
                            })
                            .got(*mip)),
                    },
                    None => self.batch_stand_in(entry, group, path),
                }
            }
            BatchEntryRaw::Asset(texture) => match self.assets.textures.texture_view(*texture) {
                Some(view) => Ok(Held::View(view.clone())),
                None => self.batch_stand_in(entry, group, path),
            },
            BatchEntryRaw::AssetSampler(sampler) => {
                let device = self.device.clone();
                Ok(Held::Sampler(
                    self.assets.textures.sampler(&device, *sampler).clone(),
                ))
            }
        }
    }

    pub(crate) fn set_batch(
        &mut self,
        batch: u32,
        key: Option<Asset<Material>>,
        entries: &[(Var, BatchEntryRaw)],
        variant: Option<crate::Variant>,
    ) -> Result<(), Error> {
        let path = self.path("set_batch");
        let (Some(decl), Some(layout)) = (
            self.machine.declared.batch_groups.get(batch as usize).copied(),
            self.machine.batches.get(batch as usize),
        ) else {
            return Err(fault(path, ErrorCode::UnknownVariable).got(batch));
        };
        let module = crate::machine::batch_path(&decl);
        let wanted = layout.entries.clone();
        let layout = layout.layout.clone();
        let pipelines: Vec<u32> = (0..self.machine.pipelines.len() as u32)
            .filter(|&pipeline| {
                (self.machine.pipelines[pipeline as usize].batches.iter())
                    .any(|&(_, known)| known == batch)
            })
            .collect();
        let groups = &self.assets.batches.0;
        let doubled = variant.is_some()
            && pipelines.iter().any(|&pipeline| {
                (self.machine.pipelines[pipeline as usize].batches.iter()).any(|&(_, other)| {
                    other != batch
                        && (groups.get(other as usize))
                            .and_then(|keyed| keyed_get(keyed, key))
                            .is_some_and(|held| held.resolves.is_some())
                })
            });
        if doubled {
            return Err(fault(path, ErrorCode::NameTwice).got("variant"));
        }
        let mut held = Vec::with_capacity(wanted.len());
        for entry in &wanted {
            let given = entries.iter().find(|(var, _)| {
                let (known, name) = crate::linker::path_of((var.module(), var.name()));
                known == module && name == entry.name
            });
            let Some((_, given)) = given else {
                return Err(fault(path, ErrorCode::UnknownVariable).got(entry.name.as_str()));
            };
            let resolved = self.resolve_batch_entry(entry, given, decl.index, &path)?;
            held.push((entry.binding, resolved));
        }
        let bound: Vec<wgpu::BindGroupEntry<'_>> = held
            .iter()
            .map(|(binding, resolved)| wgpu::BindGroupEntry {
                binding: *binding,
                resource: match resolved {
                    Held::Range(buffer, offset, size) => {
                        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer,
                            offset: *offset,
                            size: wgpu::BufferSize::new(*size),
                        })
                    }
                    Held::View(view) => wgpu::BindingResource::TextureView(view),
                    Held::Sampler(sampler) => wgpu::BindingResource::Sampler(sampler),
                },
            })
            .collect();
        let group = crate::pipelines::bind_group(&self.device, "engine/batch-group", &layout, &bound);
        let resolves = variant.map(|variant| {
            pipelines
                .into_iter()
                .map(|pipeline| (pipeline, self.resolve(pipeline, Some(&variant))))
                .collect()
        });
        let groups = &mut self.assets.batches.0;
        if groups.len() <= batch as usize {
            groups.resize_with(batch as usize + 1, Default::default);
        }
        let keyed = &mut groups[batch as usize];
        let slot = key_slot(key);
        if keyed.len() <= slot {
            keyed.resize_with(slot + 1, || None);
        }
        keyed[slot] = Some(HeldBatch { group, resolves });
        self.forget_records(key);
        Ok(())
    }

    pub(crate) fn clear_batch(&mut self, batch: u32, key: Option<Asset<Material>>) {
        if let Some(held) = self
            .assets
            .batches
            .0
            .get_mut(batch as usize)
            .and_then(|keyed| keyed.get_mut(key_slot(key)))
        {
            *held = None;
        }
        self.forget_records(key);
    }

    pub(super) fn refresh_held(&mut self) {
        let mut groups = std::mem::take(&mut self.assets.batches.0);
        for held in groups.iter_mut().flatten().flatten() {
            for (_, resolve) in held.resolves.iter_mut().flatten() {
                self.refresh(resolve);
            }
        }
        self.assets.batches.0 = groups;
        self.refresh_plain();
    }

    fn batch_stand_in(
        &mut self,
        entry: &crate::machine::BatchEntry,
        group: u32,
        path: &str,
    ) -> Result<Held, Error> {
        match self.reported_stand_in(Some(self.part), entry.ty, (group, entry.binding)) {
            Resolved::View(view) => Ok(Held::View(view)),
            Resolved::Sampler(sampler) => Ok(Held::Sampler(sampler)),
            Resolved::Buffer(_) | Resolved::Dynamic(..) | Resolved::Missing => {
                Err(fault(path.to_owned(), ErrorCode::InvalidValue).got(entry.name.as_str()))
            }
        }
    }
}
