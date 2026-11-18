use std::collections::BTreeMap;

use bevy_ecs::prelude::*;
use engine_core::{
    AlphaMode, AssetLibrary, AssetResidency, BitValue, Error, ErrorCode, GpuBits, Material,
    MaterialHandle, Sampler, Setting, TextureHandle, UvTransform, Var,
};
use engine_render::{
    Allocation, BatchGroup, Buffer, GpuPart, GroupEntry, Install, Lifetime, Prepare, Texture,
    Variant, wgpu,
};

use super::layout::{Alpha, DrawState, Lighting, MATERIALS, RECORD_VAR, SurfaceLayout};

const RANGE_ALIGN: u64 = 256;

const COLOR_LOCATIONS: usize = 8;

const TEXEL: wgpu::Extent3d = wgpu::Extent3d {
    width: 1,
    height: 1,
    depth_or_array_layers: 1,
};

#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum FaceCull {
    #[default]
    Back,
    Front,
    None,
}

impl FaceCull {
    pub(crate) fn of(cull: engine_core::Cull) -> Self {
        match cull {
            engine_core::Cull::Back => Self::Back,
            engine_core::Cull::Front => Self::Front,
            engine_core::Cull::None => Self::None,
        }
    }

    pub(crate) fn face(self) -> Option<wgpu::Face> {
        match self {
            Self::Back => Some(wgpu::Face::Back),
            Self::Front => Some(wgpu::Face::Front),
            Self::None => None,
        }
    }
}

const OVER: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
        operation: wgpu::BlendOperation::Add,
    },
    alpha: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::Zero,
        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
        operation: wgpu::BlendOperation::Add,
    },
};

const ADDED: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    },
    alpha: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::Zero,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    },
};

type Targets = Vec<(Option<wgpu::BlendState>, wgpu::ColorWrites)>;

fn blend_targets(alpha: Alpha) -> Option<Targets> {
    let additive = alpha == Alpha::Additive;
    matches!(alpha, Alpha::Blend | Alpha::Additive).then(|| {
        (0..COLOR_LOCATIONS)
            .map(|location| {
                if location == 0 {
                    (
                        Some(if additive { ADDED } else { OVER }),
                        wgpu::ColorWrites::ALL,
                    )
                } else {
                    (None, wgpu::ColorWrites::empty())
                }
            })
            .collect()
    })
}

fn staged(
    draw: DrawState,
    (maps, model): (u32, u32),
    state: &engine_core::RenderState,
) -> Variant {
    let bias = [state.bias.constant, state.bias.slope.to_bits() as i32];
    Variant {
        constants: vec![
            ("DRAW_STATE".to_owned(), f64::from(draw.pack())),
            ("MATERIAL_MAPS".to_owned(), f64::from(maps)),
            ("MATERIAL_MODEL".to_owned(), f64::from(model)),
        ],
        targets: blend_targets(draw.alpha),
        depth_write: draw.no_depth_write.then_some(false),
        depth_compare: draw
            .no_depth_test
            .then_some(wgpu::CompareFunction::Always),
        depth_bias: (bias != [0, 0]).then(|| wgpu::DepthBiasState {
            constant: bias[0],
            slope_scale: f32::from_bits(bias[1] as u32),
            clamp: 0.0,
        }),
        cull: FaceCull::of(state.cull).face(),
        fragment: draw.alpha_test,
        stand_in: None,
    }
}

impl SurfaceLayout {
    pub(super) fn record(
        &self,
        handle: MaterialHandle,
        material: &Material,
    ) -> (Vec<u32>, Vec<Error>) {
        let path = format!("$.materials.{}", handle.slot());
        let mut problems = Vec::new();
        let mut words = vec![0_u32; self.record_bytes() / 4];
        let [r, g, b] = material.emissive();
        words[..4].copy_from_slice(&[r, g, b, 0.0].map(f32::to_bits));
        let (alpha, cutoff) = alpha_of(material.alpha_mode());
        words[4..8].copy_from_slice(&[material.normal_scale(), 0.0, cutoff, 0.0].map(f32::to_bits));
        let model = self.model(material.lighting()).unwrap_or_else(|| {
            problems.push(
                Error::new(ErrorCode::UnknownVariable, format!("{path}.lighting"))
                    .got(format!("{:?}", material.lighting())),
            );
            Lighting::Unlit.bits()
        });
        words[8] = model;
        words[9] = alpha.bits();
        for (slot, module) in material.slots() {
            let Some(index) = self.slot_index(slot) else {
                if self.skipped_slot(slot) {
                    continue;
                }
                problems.push(
                    Error::new(ErrorCode::UnknownVariable, format!("{path}.slots.{slot}"))
                        .got(slot.as_str())
                        .variants(self.slot_names()),
                );
                continue;
            };
            match self.slot_module(slot, module) {
                Some(id) => words[12 + index] = id,
                None => problems.push(
                    Error::new(ErrorCode::UnknownVariable, format!("{path}.slots.{slot}"))
                        .got(module.as_str()),
                ),
            }
        }
        for (name, map) in material.maps() {
            let index = self
                .maps
                .iter()
                .position(|known| known == name)
                .filter(|_| self.reads(material, name));
            let Some(index) = index else {
                if self.skipped_reads(material, name) {
                    continue;
                }
                problems.push(
                    Error::new(ErrorCode::UnknownVariable, format!("{path}.maps.{name}"))
                        .got(name.as_str())
                        .variants(self.read_maps(material)),
                );
                continue;
            };
            let [first, second] = transform_rows(&map.transform);
            let at = self.maps_at() + index * 8;
            words[at..at + 4]
                .copy_from_slice(&[first[0], first[1], first[2], 1.0].map(f32::to_bits));
            words[at + 4..at + 8].copy_from_slice(
                &[second[0], second[1], second[2], map.tex_coord as f32].map(f32::to_bits),
            );
        }
        let values_at = self.maps_at() + self.maps.len() * 8;
        for (index, declared) in self.values.iter().enumerate() {
            let value = material.value(declared.name).unwrap_or(declared.standard);
            if let Some(rule) = declared.rule
                && let Err(invalid) = value.check(declared.name, rule)
            {
                problems.push(invalid.error(ErrorCode::InvalidMaterial, &format!("{path}.values")));
            }
            words[values_at + index] = value.to_bits();
        }
        (words, problems)
    }

    pub(super) fn plain_record(&self) -> Vec<u32> {
        let mut words = vec![0_u32; self.record_bytes() / 4];
        words[4..8].copy_from_slice(&[1.0_f32, 0.0, 0.5, 0.0].map(f32::to_bits));
        let values_at = self.maps_at() + self.maps.len() * 8;
        for (index, declared) in self.values.iter().enumerate() {
            words[values_at + index] = declared.standard.to_bits();
        }
        words
    }

    pub(super) fn variant(&self, material: Option<&Material>, words: &[u32]) -> Variant {
        let fixed = self.maps.len() <= u32::BITS as usize;
        let maps_at = self.maps_at();
        let mut maps = 0_u32;
        for index in 0..self.maps.len().min(u32::BITS as usize) {
            if f32::from_bits(words[maps_at + index * 8 + 3]) != 0.0 {
                maps |= 1 << index;
            }
        }
        let state = material.map_or_else(engine_core::RenderState::default, Material::render_state);
        let draw = DrawState {
            fixed,
            ..draw_of(state)
        };
        let chosen = if fixed { (maps, words[8]) } else { (0, 0) };
        Variant {
            stand_in: draw.fixed.then(|| {
                let dynamic = DrawState {
                    fixed: false,
                    ..draw
                };
                Box::new(staged(dynamic, (0, 0), &state))
            }),
            ..staged(draw, chosen, &state)
        }
    }
}

fn alpha_of(mode: AlphaMode) -> (Alpha, f32) {
    match mode {
        AlphaMode::Opaque => (Alpha::Opaque, 0.5),
        AlphaMode::Mask { cutoff } => (Alpha::Mask, cutoff),
        AlphaMode::Blend => (Alpha::Blend, 0.5),
        AlphaMode::Additive => (Alpha::Additive, 0.5),
    }
}

fn draw_of(state: engine_core::RenderState) -> DrawState {
    DrawState {
        alpha_test: state.alpha.masked(),
        fixed: false,
        alpha: alpha_of(state.alpha).0,
        no_depth_test: !state.depth_test,
        no_depth_write: !state.depth_write,
    }
}

fn transform_rows(transform: &UvTransform) -> [[f32; 3]; 2] {
    let (sin, cos) = engine_core::math::sin_cos(transform.rotation);
    let [sx, sy] = transform.scale;
    let [ox, oy] = transform.offset;
    [[cos * sx, sin * sy, ox], [-sin * sx, cos * sy, oy]]
}

type MapBind = Option<(TextureHandle, Sampler)>;

struct Held {
    slot: u32,
    revision: u64,
    words: Vec<u32>,
    maps: Vec<MapBind>,
    textures: Vec<(TextureHandle, u64)>,
    variant: Variant,
}

pub(super) struct SurfaceRecords {
    layout: SurfaceLayout,
    group: BatchGroup<Self>,
    records: Buffer<Self>,
    white: Texture<Self>,
    map_vars: Vec<(Var, Var)>,
    stride: u64,
    plain: Vec<u32>,
    held: BTreeMap<MaterialHandle, Held>,
    free: Vec<u32>,
    slots: u32,
    synced: Option<u64>,
}

impl GpuPart for SurfaceRecords {
    type Reads = (
        Option<Res<'static, AssetResidency>>,
        Option<Res<'static, AssetLibrary>>,
    );
    const NAME: &'static str = "surface_records";

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        let layout = SurfaceLayout::of(cx.entries());
        let records = cx.buffer("surface_records", Lifetime::Kept);
        let white = cx.texture(
            "surface_white",
            wgpu::TextureFormat::Rgba8Unorm,
            Lifetime::Kept,
        );
        let group = cx.batch_group(MATERIALS, 1, &[(RECORD_VAR, records)])?;
        let map_vars = layout
            .map_vars()
            .iter()
            .map(|(_, texture, sampler)| (*texture, *sampler))
            .collect();
        Ok(Self {
            stride: (layout.record_bytes() as u64).next_multiple_of(RANGE_ALIGN),
            plain: layout.plain_record(),
            layout,
            group,
            records,
            white,
            map_vars,
            held: BTreeMap::new(),
            free: Vec::new(),
            slots: 1,
            synced: None,
        })
    }

    fn prepare(
        &mut self,
        (residency, library): SystemParamItem<'_, '_, Self::Reads>,
        cx: &mut Prepare<'_, Self>,
    ) {
        let mut changed = Vec::new();
        if let (Some(residency), Some(library)) = (residency, library) {
            self.sync(&residency, &library, cx, &mut changed);
        }
        let bytes = u64::from(self.slots.next_power_of_two()) * self.stride;
        let fresh = cx.allocate_buffer(self.records, bytes) == Allocation::Fresh;
        let white = cx.allocate(self.white, TEXEL, 1) == Allocation::Fresh;
        if white {
            let layout = wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            };
            cx.write_texture(self.white, wgpu::Origin3d::ZERO, &[255_u8; 4], layout, TEXEL);
        }
        if fresh || white {
            changed = self.held.keys().copied().collect();
            cx.write(self.records, 0, &self.plain);
            let entries = self.entries(0, &vec![None; self.map_vars.len()]);
            if let Err(error) = cx.set_batch(self.group, None, &entries, None) {
                cx.report(error);
            }
        }
        for handle in changed {
            let Some(held) = self.held.get(&handle) else {
                continue;
            };
            cx.write(self.records, u64::from(held.slot) * self.stride, &held.words);
            let entries = self.entries(held.slot, &held.maps);
            let variant = Some(held.variant.clone());
            if let Err(error) = cx.set_batch(self.group, Some(handle), &entries, variant) {
                cx.report(error);
            }
        }
    }
}

impl SurfaceRecords {
    fn sync(
        &mut self,
        residency: &AssetResidency,
        library: &AssetLibrary,
        cx: &mut Prepare<'_, Self>,
        changed: &mut Vec<MaterialHandle>,
    ) {
        let generation = residency.revision();
        if self.synced == Some(generation) {
            return;
        }
        let gone: Vec<MaterialHandle> = self
            .held
            .iter()
            .filter(|(handle, held)| {
                !residency.material_resident(**handle)
                    || held
                        .textures
                        .iter()
                        .any(|(texture, revision)| library.revision_of(*texture) != Some(*revision))
            })
            .map(|(handle, _)| *handle)
            .collect();
        for handle in gone {
            if let Some(held) = self.held.remove(&handle) {
                self.free.push(held.slot);
            }
            cx.clear_batch(self.group, Some(handle));
        }
        let mut waiting = false;
        for (handle, revision) in residency.materials() {
            if self
                .held
                .get(&handle)
                .is_some_and(|held| held.revision == revision)
            {
                continue;
            }
            let Some(material) = library.material(handle) else {
                continue;
            };
            let maps: Vec<MapBind> = self
                .layout
                .maps()
                .iter()
                .map(|name| {
                    material
                        .map(name)
                        .filter(|_| self.layout.reads(material, name))
                        .map(|map| (map.texture, map.sampler))
                })
                .collect();
            if maps
                .iter()
                .flatten()
                .any(|(texture, _)| !cx.texture_resident(*texture))
            {
                waiting = true;
                continue;
            }
            let (words, problems) = self.layout.record(handle, material);
            for problem in problems {
                cx.report(problem);
            }
            let variant = self.layout.variant(Some(material), &words);
            let textures = maps
                .iter()
                .flatten()
                .map(|(texture, _)| (*texture, library.revision_of(*texture).unwrap_or_default()))
                .collect();
            let slot = match self.held.get(&handle) {
                Some(held) => held.slot,
                None => self.free.pop().unwrap_or_else(|| {
                    self.slots += 1;
                    self.slots - 1
                }),
            };
            self.held.insert(
                handle,
                Held {
                    slot,
                    revision,
                    words,
                    maps,
                    textures,
                    variant,
                },
            );
            changed.push(handle);
        }
        if !waiting {
            self.synced = Some(generation);
        }
    }

    fn entries(&self, slot: u32, maps: &[MapBind]) -> Vec<(Var, GroupEntry<Self>)> {
        let mut entries = vec![(
            RECORD_VAR,
            GroupEntry::Range {
                buffer: self.records,
                offset: u64::from(slot) * self.stride,
                size: self.layout.record_bytes() as u64,
            },
        )];
        for ((texture, sampler), map) in self.map_vars.iter().zip(maps) {
            match map {
                Some((handle, kind)) => {
                    entries.push((*texture, GroupEntry::Asset(*handle)));
                    entries.push((*sampler, GroupEntry::AssetSampler(*kind)));
                }
                None => {
                    entries.push((*texture, GroupEntry::Texture(self.white)));
                    entries.push((*sampler, GroupEntry::AssetSampler(Sampler::CLAMP)));
                }
            }
        }
        entries
    }
}
