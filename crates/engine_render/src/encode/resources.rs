use engine_core::Var;

use super::*;

pub(crate) struct HeldBatch {
    group: wgpu::BindGroup,
    variant: Option<Drawn>,
}

pub(crate) type Drawn = (std::sync::Arc<crate::Variant>, u64);

#[derive(Default)]
pub(crate) struct HeldBatches(Vec<Vec<Option<HeldBatch>>>);

fn key_slot(key: Option<MaterialHandle>) -> usize {
    key.map_or(0, |handle| handle.slot() + 1)
}

fn keyed_get(keyed: &[Option<HeldBatch>], key: Option<MaterialHandle>) -> Option<&HeldBatch> {
    keyed.get(key_slot(key)).and_then(Option::as_ref)
}

impl HeldBatches {
    pub(crate) fn group(&self, batch: u32, key: Option<MaterialHandle>) -> Option<&wgpu::BindGroup> {
        self.0
            .get(batch as usize)
            .and_then(|keyed| keyed_get(keyed, key))
            .map(|held| &held.group)
    }

    pub(crate) fn held(
        &self,
        key: Option<MaterialHandle>,
    ) -> Option<Option<Drawn>> {
        let mut found = None;
        for keyed in &self.0 {
            if let Some(held) = keyed_get(keyed, key) {
                if held.variant.is_some() {
                    return Some(held.variant.clone());
                }
                found = Some(None);
            }
        }
        found
    }
}

pub(crate) enum BatchEntryRaw {
    Range { buffer: u32, offset: u64, size: u64 },
    Texture(u32),
    Asset(TextureHandle),
    AssetSampler(engine_core::Sampler),
}

enum Held {
    Range(wgpu::Buffer, u64, u64),
    View(wgpu::TextureView),
    Sampler(wgpu::Sampler),
}

impl Run {
    pub(crate) fn set_batch(
        &mut self,
        batch: u32,
        key: Option<MaterialHandle>,
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
        let others = self.assets.batches.0.iter().enumerate();
        let doubled = variant.is_some()
            && others
                .filter(|(at, _)| *at != batch as usize)
                .any(|(_, keyed)| keyed_get(keyed, key).is_some_and(|held| held.variant.is_some()));
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
            let resolved = match given {
                BatchEntryRaw::Range {
                    buffer,
                    offset,
                    size,
                } => match self.buffer_held(*buffer) {
                    Some((made, bytes)) if offset + size <= *bytes => {
                        Held::Range(made.clone(), *offset, *size)
                    }
                    Some(_) | None => {
                        return Err(fault(path, ErrorCode::InvalidValue).got(entry.name.as_str()));
                    }
                },
                BatchEntryRaw::Texture(texture) => {
                    let dimension = view_dimension(entry.ty);
                    match self.tables.textures.get_mut(*texture as usize).and_then(Option::as_mut) {
                        Some(made) => Held::View(made.viewed(dimension).clone()),
                        None => self.batch_placeholder(entry.ty),
                    }
                }
                BatchEntryRaw::Asset(texture) => match self.assets.textures.texture_view(*texture) {
                    Some(view) => Held::View(view.clone()),
                    None => self.batch_placeholder(entry.ty),
                },
                BatchEntryRaw::AssetSampler(sampler) => {
                    let device = self.device.clone();
                    Held::Sampler(self.assets.textures.sampler(&device, *sampler).clone())
                }
            };
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
        let groups = &mut self.assets.batches.0;
        if groups.len() <= batch as usize {
            groups.resize_with(batch as usize + 1, Default::default);
        }
        let keyed = &mut groups[batch as usize];
        let slot = key_slot(key);
        if keyed.len() <= slot {
            keyed.resize_with(slot + 1, || None);
        }
        keyed[slot] = Some(HeldBatch {
            group,
            variant: variant.map(|variant| {
                let identity = variant.identity();
                (std::sync::Arc::new(variant), identity)
            }),
        });
        Ok(())
    }

    pub(crate) fn clear_batch(&mut self, batch: u32, key: Option<MaterialHandle>) {
        if let Some(held) = self
            .assets
            .batches
            .0
            .get_mut(batch as usize)
            .and_then(|keyed| keyed.get_mut(key_slot(key)))
        {
            *held = None;
        }
    }

    fn batch_placeholder(&mut self, ty: Ty) -> Held {
        self.placeholder((ty, 0));
        let made = self
            .bindings
            .placeholders
            .iter()
            .find(|(known, _)| *known == (ty, 0))
            .map(|(_, made)| made);
        match made {
            Some(Placeholder::Texture(view)) => Held::View(view.clone()),
            Some(Placeholder::Sampler(sampler)) => Held::Sampler(sampler.clone()),
            None => Held::Range(self.uniforms.camera.clone(), 0, self.uniforms.camera.size()),
        }
    }
}
