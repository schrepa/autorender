use engine_core::{Expected, Var};

use super::*;

impl Run {
    pub(crate) fn mipmaps(&mut self, texture: u32, layers: Range<u32>) -> bool {
        let Some(made) = self
            .written_texture(texture)
            .map(|made| made.texture.clone())
        else {
            let error = fault(self.path("mipmaps"), ErrorCode::InvalidValue);
            self.fault(error);
            return false;
        };
        if !crate::mipmaps::drawn(made.format(), made.dimension()) {
            let error = fault(self.path("mipmaps"), ErrorCode::UnsupportedFeature)
                .expected(Expected::Supported)
                .got(made.format());
            self.fault(error);
            return false;
        }
        self.owed_mips.retain(|(owed, _)| *owed != texture);
        self.owed_mips.push((texture, layers));
        self.record_mips();
        !self.owed_mips.iter().any(|(owed, _)| *owed == texture)
    }

    pub(crate) fn record_mips(&mut self) {
        let device = self.device.clone();
        let mut encoder = self.queued.before.take().unwrap_or_else(|| {
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("engine/part-mips"),
            })
        });
        for (texture, layers) in std::mem::take(&mut self.owed_mips) {
            let Some(made) = self
                .written_texture(texture)
                .map(|made| made.texture.clone())
            else {
                continue;
            };
            let (mut compiler, assets) = self.split();
            let levels = made.mip_level_count();
            let recorded = layers.clone().all(|layer| {
                assets
                    .textures
                    .mips()
                    .record((&device, &mut encoder), &mut compiler, &made, Some(layer))
                    .is_some_and(|written| written + 1 >= levels)
            });
            if !recorded {
                self.owed_mips.push((texture, layers));
            }
        }
        self.queued.before = Some(encoder);
    }

    pub(crate) fn grow_buffer(&mut self, buffer: u32, bytes: u64) -> Allocation {
        let Some(held) = self
            .buffer_tables_mut(buffer)
            .buffers
            .get_mut(buffer as usize)
        else {
            return Allocation::Fresh;
        };
        if held.as_ref().is_some_and(|(_, known)| *known >= bytes) {
            return Allocation::Kept;
        }
        let previous = held.take();
        let allocation = self.allocate_buffer(buffer, bytes.max(16).next_power_of_two());
        let Some((old, kept)) = previous else {
            return allocation;
        };
        match self.buffer_held(buffer) {
            Some((new, _)) => {
                let mut encoder =
                    self.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("engine/part-grow"),
                        });
                encoder.copy_buffer_to_buffer(&old, 0, new, 0, kept);
                self.queue.submit_now([encoder.finish()]);
                Allocation::Kept
            }
            None => Allocation::Fresh,
        }
    }
}

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
                } => {
                    let per_view = self
                        .machine
                        .declared
                        .buffers
                        .get(*buffer as usize)
                        .is_none_or(|declared| declared.per_view);
                    match self.buffer_held(*buffer) {
                        Some((made, bytes)) if !per_view && offset + size <= *bytes => {
                            Held::Range(made.clone(), *offset, *size)
                        }
                        Some(_) | None => {
                            return Err(
                                fault(path, ErrorCode::InvalidValue).got(entry.name.as_str())
                            );
                        }
                    }
                }
                BatchEntryRaw::Texture(texture) => {
                    match self.texture_view(&Bound::Texture(*texture), entry.ty) {
                        Some(view) => Held::View(view),
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
