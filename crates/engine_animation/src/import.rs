use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use engine_core::math::max;
use engine_core::{
    AlphaMode, AssetLibrary, Error, ErrorCode, Expected, Filter, Material, MaterialHandle, Mesh,
    Mesh3d, MeshVertex, Sampler, Surface, Texture, TextureHandle, TextureRef, UvTransform, Wrap,
};
use engine_places::animation::{Bound, Clip, Joint, Keys, Rig, RigPart, Track, VertexSkin};
use glam::{Mat3, Mat4, Quat, Vec3};
use gltf::animation::util::ReadOutputs;
use gltf::animation::{Interpolation, Property};
use gltf::texture::{MagFilter, MinFilter, WrappingMode};

const WHOLE: [f32; 4] = [1.0, 0.0, 0.0, 0.0];

pub struct Model {
    pub statics: Vec<Mesh3d>,
    pub rig: Option<Arc<Rig>>,
}

#[derive(Default)]
pub struct ImportShared {
    textures: BTreeMap<Vec<u8>, TextureHandle>,
    materials: Vec<(Material, MaterialHandle)>,
}

struct Placed<'a> {
    node: gltf::Node<'a>,
    parent: Option<usize>,
    world: Mat4,
}

#[derive(Clone, Copy)]
enum Binding {
    Skinned(u32),
    Rigid(u32),
    Still,
}

#[derive(Default)]
struct Soup {
    vertices: Vec<MeshVertex>,
    indices: Vec<u32>,
    skin: Vec<VertexSkin>,
}

struct Library<'a> {
    assets: &'a mut AssetLibrary,
    shared: &'a mut ImportShared,
    binary: &'a [u8],
    textures: BTreeMap<usize, TextureHandle>,
    materials: BTreeMap<Option<usize>, MaterialHandle>,
}

fn unsupported(path: impl Into<String>) -> Error {
    Error::new(ErrorCode::UnsupportedFeature, path)
}

fn malformed(path: impl Into<String>) -> Error {
    Error::new(ErrorCode::InvalidGltf, path)
}

pub fn import_glb(
    bytes: &[u8],
    assets: &mut AssetLibrary,
    shared: &mut ImportShared,
) -> Result<Model, Error> {
    let gltf =
        gltf::Gltf::from_slice(bytes).map_err(|error| malformed("$").got(error.to_string()))?;
    let binary = gltf.blob.as_deref().unwrap_or_default();
    let document = &gltf.document;
    let placed = placed_nodes(document);
    let skins: BTreeSet<usize> = placed
        .iter()
        .filter(|held| held.node.mesh().is_some())
        .filter_map(|held| held.node.skin())
        .map(|skin| skin.index())
        .collect();
    let joints = joint_nodes(&placed, document, &skins);
    let (mut palette, starts) = palette_of(document, binary, &skins, &joints)?;
    let mut library = Library {
        assets,
        shared,
        binary,
        textures: BTreeMap::new(),
        materials: BTreeMap::new(),
    };
    let mut rigid = BTreeMap::new();
    let mut parts = Vec::new();
    let mut statics = Vec::new();
    for (at, held) in placed.iter().enumerate() {
        let Some(mesh) = held.node.mesh() else {
            continue;
        };
        let skin_start = held
            .node
            .skin()
            .and_then(|skin| starts.get(&skin.index()).copied());
        let (binding, world) = match (skin_start, holder(&placed, at, &joints)) {
            (Some(start), _) => (Binding::Skinned(start), Mat4::IDENTITY),
            (None, Some((holder_at, joint))) => {
                let entry = *rigid.entry(joint).or_insert_with(|| {
                    palette.push(Bound {
                        joint,
                        inverse_bind: placed[holder_at].world.inverse().to_cols_array_2d(),
                    });
                    palette.len() as u32 - 1
                });
                (Binding::Rigid(entry), held.world)
            }
            (None, None) => (Binding::Still, held.world),
        };
        for primitive in mesh.primitives() {
            let path = format!(
                "$.meshes[{}].primitives[{}]",
                mesh.index(),
                primitive.index()
            );
            let soup = read_primitive(&primitive, binary, world, binding, &path)?;
            let material = library
                .material(&primitive.material())
                .map_err(|error| error.within(&path))?;
            let handle = Mesh::new(soup.vertices, soup.indices)
                .and_then(|built| library.assets.insert_mesh(built))
                .map_err(|error| error.within(&path))?;
            match binding {
                Binding::Skinned(_) | Binding::Rigid(_) => parts.push(RigPart {
                    mesh: handle,
                    material,
                    skin: soup.skin,
                }),
                Binding::Still => statics.push(Mesh3d {
                    mesh: handle,
                    material,
                }),
            }
        }
    }
    let rig = if joints.is_empty() {
        None
    } else {
        let rig = Rig {
            joints: rig_joints(&placed, &joints),
            palette,
            clips: clips_of(document, binary, &joints),
            parts,
        };
        rig.check()?;
        Some(Arc::new(rig))
    };
    Ok(Model { statics, rig })
}

fn placed_nodes(document: &gltf::Document) -> Vec<Placed<'_>> {
    let mut placed = Vec::new();
    let scene = document
        .default_scene()
        .or_else(|| document.scenes().next());
    let Some(scene) = scene else {
        return placed;
    };
    let mut stack: Vec<_> = scene
        .nodes()
        .map(|node| (node, None, Mat4::IDENTITY))
        .collect();
    while let Some((node, parent, above)) = stack.pop() {
        let world = above * Mat4::from_cols_array_2d(&node.transform().matrix());
        let at = placed.len();
        stack.extend(node.children().map(|child| (child, Some(at), world)));
        placed.push(Placed {
            node,
            parent,
            world,
        });
    }
    placed
}

fn joint_nodes(
    placed: &[Placed<'_>],
    document: &gltf::Document,
    skins: &BTreeSet<usize>,
) -> BTreeMap<usize, u32> {
    let here: BTreeSet<usize> = placed.iter().map(|held| held.node.index()).collect();
    let mut wanted = BTreeSet::new();
    for skin in document
        .skins()
        .filter(|skin| skins.contains(&skin.index()))
    {
        wanted.extend(skin.joints().map(|joint| joint.index()));
    }
    for animation in document.animations() {
        for channel in animation.channels() {
            let target = channel.target();
            if !matches!(target.property(), Property::MorphTargetWeights) {
                wanted.insert(target.node().index());
            }
        }
    }
    wanted.retain(|node| here.contains(node));
    for held in placed.iter().rev() {
        if let Some(parent) = held.parent
            && wanted.contains(&held.node.index())
        {
            wanted.insert(placed[parent].node.index());
        }
    }
    let mut joints = BTreeMap::new();
    for held in placed {
        let node = held.node.index();
        if wanted.contains(&node) {
            joints.insert(node, joints.len() as u32);
        }
    }
    joints
}

fn holder(placed: &[Placed<'_>], at: usize, joints: &BTreeMap<usize, u32>) -> Option<(usize, u32)> {
    let mut cursor = Some(at);
    while let Some(index) = cursor {
        let held = placed.get(index)?;
        if let Some(&joint) = joints.get(&held.node.index()) {
            return Some((index, joint));
        }
        cursor = held.parent;
    }
    None
}

fn rig_joints(placed: &[Placed<'_>], joints: &BTreeMap<usize, u32>) -> Vec<Joint> {
    placed
        .iter()
        .filter(|held| joints.contains_key(&held.node.index()))
        .map(|held| {
            let (translation, rotation, scale) = held.node.transform().decomposed();
            Joint {
                name: held
                    .node
                    .name()
                    .map_or_else(|| format!("node{}", held.node.index()), str::to_owned),
                parent: held
                    .parent
                    .and_then(|parent| joints.get(&placed[parent].node.index()).copied()),
                translation,
                rotation: Quat::from_array(rotation).normalize().to_array(),
                scale,
            }
        })
        .collect()
}

fn palette_of(
    document: &gltf::Document,
    binary: &[u8],
    skins: &BTreeSet<usize>,
    joints: &BTreeMap<usize, u32>,
) -> Result<(Vec<Bound>, BTreeMap<usize, u32>), Error> {
    let mut palette = Vec::new();
    let mut starts = BTreeMap::new();
    for skin in document
        .skins()
        .filter(|skin| skins.contains(&skin.index()))
    {
        starts.insert(skin.index(), palette.len() as u32);
        let reader = skin.reader(|buffer| (buffer.index() == 0).then_some(binary));
        let inverses: Vec<[[f32; 4]; 4]> = reader
            .read_inverse_bind_matrices()
            .map(|matrices| matrices.collect())
            .unwrap_or_default();
        for (at, joint) in skin.joints().enumerate() {
            let Some(&index) = joints.get(&joint.index()) else {
                return Err(malformed(format!("$.skins[{}].joints[{at}]", skin.index())));
            };
            palette.push(Bound {
                joint: index,
                inverse_bind: inverses
                    .get(at)
                    .copied()
                    .unwrap_or_else(|| Mat4::IDENTITY.to_cols_array_2d()),
            });
        }
    }
    Ok((palette, starts))
}

fn vertex_skin(
    binding: Binding,
    joints: &[[u16; 4]],
    weights: &[[f32; 4]],
    at: usize,
) -> VertexSkin {
    match binding {
        Binding::Still => VertexSkin {
            joints: [0; 4],
            weights: WHOLE,
        },
        Binding::Rigid(entry) => VertexSkin {
            joints: [entry; 4],
            weights: WHOLE,
        },
        Binding::Skinned(start) => {
            let held = joints.get(at).copied().unwrap_or_default();
            let weights = weights.get(at).copied().unwrap_or_default();
            if weights.iter().sum::<f32>() <= 0.0 {
                return VertexSkin {
                    joints: [start; 4],
                    weights: WHOLE,
                };
            }
            let mut joined = [start; 4];
            for (slot, (&joint, &weight)) in joined.iter_mut().zip(held.iter().zip(&weights)) {
                if weight > 0.0 {
                    *slot = start + u32::from(joint);
                }
            }
            VertexSkin {
                joints: joined,
                weights,
            }
        }
    }
}

fn read_primitive(
    primitive: &gltf::Primitive<'_>,
    binary: &[u8],
    world: Mat4,
    binding: Binding,
    path: &str,
) -> Result<Soup, Error> {
    if primitive.mode() != gltf::mesh::Mode::Triangles {
        return Err(unsupported(format!("{path}.mode")).expected(Expected::Triangles));
    }
    let reader = primitive.reader(|buffer| (buffer.index() == 0).then_some(binary));
    let positions: Vec<Vec3> = reader
        .read_positions()
        .ok_or_else(|| {
            Error::new(
                ErrorCode::MissingField,
                format!("{path}.attributes.POSITION"),
            )
        })?
        .map(|position| world.transform_point3(Vec3::from_array(position)))
        .collect();
    let normals: Option<Vec<[f32; 3]>> = reader.read_normals().map(|normals| normals.collect());
    let uvs: Vec<[f32; 2]> = reader
        .read_tex_coords(0)
        .map(|uvs| uvs.into_f32().collect())
        .unwrap_or_default();
    let joints: Vec<[u16; 4]> = reader
        .read_joints(0)
        .map(|joints| joints.into_u16().collect())
        .unwrap_or_default();
    let weights: Vec<[f32; 4]> = reader
        .read_weights(0)
        .map(|weights| weights.into_f32().collect())
        .unwrap_or_default();
    let mut triangles: Vec<u32> = match reader.read_indices() {
        Some(indices) => indices.into_u32().collect(),
        None => (0..positions.len() as u32).collect(),
    };
    if !triangles.len().is_multiple_of(3)
        || triangles
            .iter()
            .any(|&index| index as usize >= positions.len())
    {
        return Err(Error::new(
            ErrorCode::InvalidMesh,
            format!("{path}.indices"),
        ));
    }
    let linear = Mat3::from_mat4(world);
    if linear.determinant() < 0.0 {
        for triangle in triangles.as_chunks_mut::<3>().0 {
            triangle.swap(1, 2);
        }
    }
    let mut soup = Soup::default();
    let mut push = |at: usize, normal: Vec3| {
        let normal = if normal.abs().max_element() > 0.0 {
            normal
        } else {
            Vec3::Y
        };
        soup.vertices.push(MeshVertex {
            position: positions[at].to_array(),
            normal: normal.to_array(),
            uv: uvs.get(at).copied().unwrap_or_default(),
        });
        soup.skin.push(vertex_skin(binding, &joints, &weights, at));
    };
    match normals {
        Some(normals) => {
            let turn = linear.inverse().transpose();
            for at in 0..positions.len() {
                let normal = normals.get(at).copied().unwrap_or_default();
                push(at, turn * Vec3::from_array(normal));
            }
            soup.indices = triangles;
        }
        None => {
            for triangle in triangles.as_chunks::<3>().0 {
                let [a, b, c] = triangle.map(|index| positions[index as usize]);
                let normal = (b - a).cross(c - a);
                for &index in triangle {
                    push(index as usize, normal);
                }
            }
            soup.indices = (0..soup.vertices.len() as u32).collect();
        }
    }
    Ok(soup)
}

fn keys_of<T>(times: Vec<f32>, values: Vec<T>, interpolation: Interpolation) -> Keys<T> {
    let values = if interpolation == Interpolation::CubicSpline {
        values.into_iter().skip(1).step_by(3).collect()
    } else {
        values
    };
    Keys {
        times,
        values,
        step: interpolation == Interpolation::Step,
    }
}

fn clips_of(document: &gltf::Document, binary: &[u8], joints: &BTreeMap<usize, u32>) -> Vec<Clip> {
    let mut clips = Vec::new();
    for (at, animation) in document.animations().enumerate() {
        let mut tracks: BTreeMap<u32, Track> = BTreeMap::new();
        let mut duration = 0.0;
        for channel in animation.channels() {
            let Some(&joint) = joints.get(&channel.target().node().index()) else {
                continue;
            };
            let reader = channel.reader(|buffer| (buffer.index() == 0).then_some(binary));
            let (Some(times), Some(outputs)) = (reader.read_inputs(), reader.read_outputs()) else {
                continue;
            };
            let times: Vec<f32> = times.collect();
            duration = times
                .iter()
                .fold(duration, |longest, &time| max(longest, time));
            let interpolation = channel.sampler().interpolation();
            let track = tracks.entry(joint).or_insert(Track {
                joint,
                translation: None,
                rotation: None,
                scale: None,
            });
            match outputs {
                ReadOutputs::Translations(values) => {
                    track.translation = Some(keys_of(times, values.collect(), interpolation));
                }
                ReadOutputs::Rotations(values) => {
                    let values = values
                        .into_f32()
                        .map(|value| Quat::from_array(value).normalize().to_array())
                        .collect();
                    track.rotation = Some(keys_of(times, values, interpolation));
                }
                ReadOutputs::Scales(values) => {
                    track.scale = Some(keys_of(times, values.collect(), interpolation));
                }
                ReadOutputs::MorphTargetWeights(_) => {}
            }
        }
        clips.push(Clip {
            name: animation
                .name()
                .map_or_else(|| format!("animation{at}"), str::to_owned),
            duration,
            tracks: tracks.into_values().collect(),
        });
    }
    clips
}

fn sampler_of(texture: &gltf::Texture<'_>) -> Sampler {
    let held = texture.sampler();
    let wrap = |mode: WrappingMode| match mode {
        WrappingMode::Repeat => Wrap::Repeat,
        WrappingMode::MirroredRepeat => Wrap::MirroredRepeat,
        WrappingMode::ClampToEdge => Wrap::ClampToEdge,
    };
    let (min_filter, mipmap_filter) = match held.min_filter() {
        None | Some(MinFilter::LinearMipmapLinear) => (Filter::Linear, Some(Filter::Linear)),
        Some(MinFilter::Nearest) => (Filter::Nearest, None),
        Some(MinFilter::Linear) => (Filter::Linear, None),
        Some(MinFilter::NearestMipmapNearest) => (Filter::Nearest, Some(Filter::Nearest)),
        Some(MinFilter::LinearMipmapNearest) => (Filter::Linear, Some(Filter::Nearest)),
        Some(MinFilter::NearestMipmapLinear) => (Filter::Nearest, Some(Filter::Linear)),
    };
    let mag_filter = if matches!(held.mag_filter(), Some(MagFilter::Nearest)) {
        Filter::Nearest
    } else {
        Filter::Linear
    };
    Sampler {
        wrap_u: wrap(held.wrap_s()),
        wrap_v: wrap(held.wrap_t()),
        mag_filter,
        min_filter,
        mipmap_filter,
    }
}

impl Library<'_> {
    fn material(&mut self, material: &gltf::Material<'_>) -> Result<MaterialHandle, Error> {
        let index = material.index();
        if let Some(&held) = self.materials.get(&index) {
            return Ok(held);
        }
        let built = match index {
            Some(index) => self
                .build_material(material)
                .map_err(|error| error.within(&format!("$.materials[{index}]")))?,
            None => Material::new([1.0; 4])?,
        };
        let shared = self
            .shared
            .materials
            .iter()
            .find(|(held, _)| *held == built);
        let handle = match shared {
            Some(&(_, handle)) => handle,
            None => {
                let handle = self.assets.insert_material(built)?;
                self.shared.materials.push((built, handle));
                handle
            }
        };
        self.materials.insert(index, handle);
        Ok(handle)
    }

    fn build_material(&mut self, material: &gltf::Material<'_>) -> Result<Material, Error> {
        let pbr = material.pbr_metallic_roughness();
        let [red, green, blue, alpha] = pbr.base_color_factor();
        let mut built = Material::new([red, green, blue, 1.0])?.with_surface(Surface {
            roughness: pbr.roughness_factor(),
            metalness: pbr.metallic_factor(),
            emission: 0.0,
            ripples: 0.0,
        })?;
        match material.alpha_mode() {
            gltf::material::AlphaMode::Opaque => {}
            gltf::material::AlphaMode::Mask => {
                let cutoff = material.alpha_cutoff().unwrap_or(0.5);
                built = built.with_alpha_mode(AlphaMode::Mask { cutoff })?;
                if alpha < 1.0 {
                    built = built.with_opacity(max(alpha, 0.001))?;
                }
            }
            gltf::material::AlphaMode::Blend => {
                return Err(unsupported("$.alphaMode")
                    .got("BLEND")
                    .variants(["OPAQUE", "MASK"]));
            }
        }
        if let Some(info) = pbr.base_color_texture() {
            built = built.with_base_color_texture(self.texture_ref(&info)?)?;
        }
        let strength = material.emissive_strength().unwrap_or(1.0);
        let emissive = material.emissive_factor().map(|value| value * strength);
        if emissive != [0.0; 3] {
            built = built.with_emissive(emissive)?;
        }
        if let Some(info) = material.emissive_texture() {
            built = built.with_emissive_texture(self.texture_ref(&info)?)?;
        }
        Ok(built.with_double_sided(material.double_sided()))
    }

    fn texture_ref(&mut self, info: &gltf::texture::Info<'_>) -> Result<TextureRef, Error> {
        let transform = info.texture_transform();
        let tex_coord = transform
            .as_ref()
            .and_then(|transform| transform.tex_coord())
            .unwrap_or_else(|| info.tex_coord());
        if tex_coord != 0 {
            return Err(unsupported("$.texCoord").at_most(0).got(tex_coord));
        }
        let texture = info.texture();
        let Some(image) = texture.source() else {
            return Err(
                unsupported(format!("$.textures[{}].source", texture.index()))
                    .expected(Expected::Present),
            );
        };
        let handle = self.image(&image)?;
        let mut held = TextureRef::new(handle).with_sampler(sampler_of(&texture));
        if let Some(transform) = transform {
            held = held.with_transform(UvTransform {
                offset: transform.offset(),
                rotation: transform.rotation(),
                scale: transform.scale(),
            });
        }
        Ok(held)
    }

    fn image(&mut self, image: &gltf::Image<'_>) -> Result<TextureHandle, Error> {
        let index = image.index();
        if let Some(&held) = self.textures.get(&index) {
            return Ok(held);
        }
        let path = format!("$.images[{index}]");
        let gltf::image::Source::View { view, mime_type } = image.source() else {
            return Err(unsupported(format!("{path}.uri")).expected(Expected::Absent));
        };
        let bytes = self
            .binary
            .get(view.offset()..view.offset() + view.length())
            .ok_or_else(|| malformed(format!("{path}.bufferView")))?;
        if let Some(&held) = self.shared.textures.get(bytes) {
            self.textures.insert(index, held);
            return Ok(held);
        }
        let decoded = match mime_type {
            "image/png" => decode_png(bytes),
            "image/jpeg" => decode_jpeg(bytes),
            _ => {
                return Err(unsupported(format!("{path}.mimeType"))
                    .got(mime_type)
                    .variants(["image/png", "image/jpeg"]));
            }
        };
        let (width, height, pixels) = decoded.map_err(|failed| {
            Error::new(ErrorCode::InvalidTexture, path.clone())
                .expected(Expected::Image)
                .got(failed)
        })?;
        let texture =
            Texture::from_rgba(width, height, pixels).map_err(|error| error.within(&path))?;
        let handle = self.assets.insert_texture(texture)?;
        self.shared.textures.insert(bytes.to_vec(), handle);
        self.textures.insert(index, handle);
        Ok(handle)
    }
}

fn decode_png(bytes: &[u8]) -> Result<(u32, u32, Vec<u8>), String> {
    let mut decoder = png::Decoder::new(std::io::Cursor::new(bytes));
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder.read_info().map_err(|error| error.to_string())?;
    let mut buffer = vec![0; reader.output_buffer_size()];
    let info = reader
        .next_frame(&mut buffer)
        .map_err(|error| error.to_string())?;
    buffer.truncate(info.buffer_size());
    let pixels = match info.color_type {
        png::ColorType::Rgba => buffer,
        png::ColorType::Rgb => buffer
            .as_chunks::<3>()
            .0
            .iter()
            .flat_map(|&[red, green, blue]| [red, green, blue, 255])
            .collect(),
        png::ColorType::GrayscaleAlpha => buffer
            .as_chunks::<2>()
            .0
            .iter()
            .flat_map(|&[gray, alpha]| [gray, gray, gray, alpha])
            .collect(),
        png::ColorType::Grayscale => buffer
            .iter()
            .flat_map(|&gray| [gray, gray, gray, 255])
            .collect(),
        png::ColorType::Indexed => return Err("indexed".into()),
    };
    Ok((info.width, info.height, pixels))
}

fn decode_jpeg(bytes: &[u8]) -> Result<(u32, u32, Vec<u8>), String> {
    let options = zune_jpeg::zune_core::options::DecoderOptions::default()
        .jpeg_set_out_colorspace(zune_jpeg::zune_core::colorspace::ColorSpace::RGBA);
    let mut decoder = zune_jpeg::JpegDecoder::new_with_options(
        zune_jpeg::zune_core::bytestream::ZCursor::new(bytes),
        options,
    );
    let pixels = decoder.decode().map_err(|error| error.to_string())?;
    let (width, height) = decoder.dimensions().ok_or("dimensions")?;
    Ok((width as u32, height as u32, pixels))
}
