use engine_core::collections::HashMap;
use engine_core::{
    AlphaMode, Filter, Material, Mesh, MeshHandle, Texture, TextureFormat, TextureHandle,
    TextureRef, Wrap, content_hash,
};

#[derive(Default)]
pub(crate) struct Contents {
    meshes: HashMap<MeshHandle, u128>,
    textures: HashMap<TextureHandle, u128>,
}

impl Contents {
    pub(crate) fn mesh(&mut self, handle: MeshHandle, mesh: &Mesh) -> u128 {
        *self.meshes.entry(handle).or_insert_with(|| {
            let mut bytes = Vec::with_capacity(mesh.byte_len() + 16);
            words(
                &mut bytes,
                &[mesh.vertices().len() as u32, mesh.indices().len() as u32],
            );
            for vertex in mesh.vertices() {
                floats(&mut bytes, &vertex.position);
                floats(&mut bytes, &vertex.normal);
                floats(&mut bytes, &vertex.uv);
            }
            words(&mut bytes, mesh.indices());
            bytes.push(u8::from(mesh.colors().is_some()));
            for color in mesh.colors().unwrap_or_default() {
                floats(&mut bytes, color);
            }
            bytes.push(u8::from(mesh.uv1().is_some()));
            for uv in mesh.uv1().unwrap_or_default() {
                floats(&mut bytes, uv);
            }
            content_hash(&bytes)
        })
    }

    pub(crate) fn material<'a>(
        &mut self,
        material: Option<&Material>,
        texture: impl Fn(TextureHandle) -> Option<&'a Texture>,
    ) -> u128 {
        let Some(material) = material else {
            return 0;
        };
        let mut bytes = Vec::with_capacity(256);
        floats(&mut bytes, &material.color());
        for map in material.maps() {
            let Some(map) = map else {
                bytes.push(0);
                continue;
            };
            bytes.push(1);
            bytes.extend_from_slice(&self.texture(map.texture, &texture).to_le_bytes());
            texture_ref(&mut bytes, &map);
        }
        let surface = material.surface();
        floats(
            &mut bytes,
            &[
                surface.roughness,
                surface.metalness,
                surface.emission,
                surface.ripples,
                material.normal_scale(),
                material.occlusion_strength(),
            ],
        );
        floats(&mut bytes, &material.emissive());
        match material.alpha_mode() {
            AlphaMode::Opaque => bytes.push(0),
            AlphaMode::Mask { cutoff } => {
                bytes.push(1);
                floats(&mut bytes, &[cutoff]);
            }
        }
        bytes.push(u8::from(material.is_double_sided()));
        floats(&mut bytes, &[material.world_tiling().unwrap_or(-1.0)]);
        match material.top_layer() {
            Some(top) => {
                bytes.push(1);
                floats(&mut bytes, &top.color);
                floats(&mut bytes, &[top.coverage, top.roughness]);
            }
            None => bytes.push(0),
        }
        match material.grounding() {
            Some(grounding) => {
                bytes.push(1);
                floats(&mut bytes, &[grounding.height]);
            }
            None => bytes.push(0),
        }
        content_hash(&bytes)
    }

    fn texture<'a>(
        &mut self,
        handle: TextureHandle,
        texture: &impl Fn(TextureHandle) -> Option<&'a Texture>,
    ) -> u128 {
        if let Some(&content) = self.textures.get(&handle) {
            return content;
        }
        let Some(found) = texture(handle) else {
            return 0;
        };
        let mut bytes = Vec::with_capacity(found.byte_len() + 12);
        let format = match found.format() {
            TextureFormat::Rgba8 => 0,
            TextureFormat::Bc7 => 1,
            TextureFormat::Etc2Rgba8 => 2,
            TextureFormat::Astc4x4 => 3,
        };
        words(&mut bytes, &[found.width(), found.height(), format]);
        for level in found.levels() {
            bytes.extend_from_slice(level);
        }
        let content = content_hash(&bytes);
        self.textures.insert(handle, content);
        content
    }

    pub(crate) fn retain(
        &mut self,
        mut mesh: impl FnMut(MeshHandle) -> bool,
        mut texture: impl FnMut(TextureHandle) -> bool,
    ) {
        self.meshes.retain(|&handle, _| mesh(handle));
        self.textures.retain(|&handle, _| texture(handle));
    }
}

fn floats(bytes: &mut Vec<u8>, values: &[f32]) {
    for value in values {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
}

fn words(bytes: &mut Vec<u8>, values: &[u32]) {
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
}

fn texture_ref(bytes: &mut Vec<u8>, map: &TextureRef) {
    let wrap = |wrap: Wrap| match wrap {
        Wrap::Repeat => 0,
        Wrap::MirroredRepeat => 1,
        Wrap::ClampToEdge => 2,
    };
    let filter = |filter: Filter| match filter {
        Filter::Nearest => 0,
        Filter::Linear => 1,
    };
    let sampler = map.sampler;
    bytes.extend_from_slice(&[
        wrap(sampler.wrap_u),
        wrap(sampler.wrap_v),
        filter(sampler.mag_filter),
        filter(sampler.min_filter),
        sampler.mipmap_filter.map_or(2, filter),
    ]);
    let transform = map.transform;
    floats(bytes, &transform.offset);
    floats(bytes, &[transform.rotation]);
    floats(bytes, &transform.scale);
    words(bytes, &[map.tex_coord]);
}
