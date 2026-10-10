//! Materials of the metro over the generated texture sets, and the library that
//! admits meshes, textures, materials and clips into a workload's asset
//! library. Generated texture data (RGBA8 or BC7 levels) is kept for the page's
//! lifetime: workloads rebuild their assets from it without generating again.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use engine::prelude::*;

use super::bc7;
use super::models::Model;
use super::people;
use super::textures::{self, Image, Set};

/// Every material. Colors come from per-object tints.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Mat {
    // Façades: lit windows, grime over the whole building.
    Glass,
    Office,
    Brick,
    Panel,
    Deco,
    Siding,
    Roof,
    Tiles,
    Road,
    Asphalt,
    Crossing,
    Pavers,
    Plaza,
    Grass,
    Hedge,
    Dirt,
    Water,
    Brushed,
    Chrome,
    Gold,
    Concrete,
    Container,
    Solar,
    Leaves,
    Fronds,
    Wood,
    Canvas,
    Lattice,
    Fence,
    /// Neon in one of six colors.
    Neon(u8),
    /// Advertisement board `slot`, cycling through the ads.
    Ad(u8),
    Fixture,
    Vehicle,
    Seats,
    Pitch,
    Rails,
    Person,
    Flora,
    /// Clear glass that bends what is behind it.
    ClearGlass,
    /// Frosted, tinted glass.
    FrostedGlass,
    /// A plainly blended translucent pane.
    Pane,
}

/// Board slots the city's billboards share; each shows one ad at a time.
pub(crate) const AD_SLOTS: u8 = 3;
pub(crate) const NEON_COLORS: [[f32; 3]; 6] = [
    [1.0, 0.15, 0.55],
    [0.1, 0.9, 1.0],
    [0.65, 0.3, 1.0],
    [1.0, 0.55, 0.1],
    [0.3, 1.0, 0.4],
    [1.0, 0.95, 0.4],
];

/// Which map of a set a texture holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Slot {
    Base,
    Mr,
    Normal,
    Emissive,
}

/// Texture data ready for admission.
enum Payload {
    Rgba8 {
        width: u32,
        height: u32,
        pixels: Vec<u8>,
        alpha: bool,
    },
    Bc7 {
        width: u32,
        height: u32,
        levels: Vec<Vec<u8>>,
    },
}

/// Generated maps per set, and their BC7 levels once compressed: a set's
/// maps go once all of them are compressed, and `None` then answers for
/// maps the set lacks.
#[derive(Default)]
struct Cache {
    maps: HashMap<Set, Arc<textures::Maps>>,
    compressed: HashMap<(Set, Slot), Option<Arc<Compressed>>>,
}

/// A map's size and BC7 levels.
struct Compressed {
    width: u32,
    height: u32,
    levels: Vec<Vec<u8>>,
}

/// Every slot, in the order maps list them.
const SLOTS: [Slot; 4] = [Slot::Base, Slot::Mr, Slot::Normal, Slot::Emissive];

/// `slot`'s map of `maps`, when the set has one.
fn slot_image(maps: &textures::Maps, slot: Slot) -> Option<&Image> {
    match slot {
        Slot::Base => Some(&maps.base),
        Slot::Mr => maps.mr.as_ref(),
        Slot::Normal => maps.normal.as_ref(),
        Slot::Emissive => maps.emissive.as_ref(),
    }
}

fn cache() -> &'static Mutex<Cache> {
    static CACHE: OnceLock<Mutex<Cache>> = OnceLock::new();
    CACHE.get_or_init(Mutex::default)
}

fn maps(set: Set) -> Arc<textures::Maps> {
    let mut cache = cache().lock().expect("texture cache");
    cache
        .maps
        .entry(set)
        .or_insert_with(|| Arc::new(textures::generate(set)))
        .clone()
}

fn payload(set: Set, slot: Slot, compressed: bool) -> Option<Payload> {
    if compressed {
        let cache = cache().lock().expect("texture cache");
        if let Some(compressed) = cache.compressed.get(&(set, slot)) {
            let compressed = compressed.as_ref()?;
            return Some(Payload::Bc7 {
                width: compressed.width,
                height: compressed.height,
                levels: compressed.levels.clone(),
            });
        }
    }
    let maps = maps(set);
    let image = slot_image(&maps, slot)?;
    if !compressed {
        return Some(Payload::Rgba8 {
            width: image.width,
            height: image.height,
            pixels: image.pixels.clone(),
            alpha: slot == Slot::Base && set.masked(),
        });
    }
    let levels: Vec<Vec<u8>> = textures::mip_chain(image)
        .iter()
        .map(|(width, height, pixels)| bc7::encode(*width, *height, pixels))
        .collect();
    {
        let mut cache = cache().lock().expect("texture cache");
        let compressed = Compressed {
            width: image.width,
            height: image.height,
            levels: levels.clone(),
        };
        cache
            .compressed
            .insert((set, slot), Some(Arc::new(compressed)));
        let compressed = |slot| cache.compressed.contains_key(&(set, slot));
        if SLOTS
            .iter()
            .all(|&slot| slot_image(&maps, slot).is_none() || compressed(slot))
        {
            for slot in SLOTS {
                if slot_image(&maps, slot).is_none() {
                    cache.compressed.insert((set, slot), None);
                }
            }
            cache.maps.remove(&set);
        }
    }
    Some(Payload::Bc7 {
        width: image.width,
        height: image.height,
        levels,
    })
}

impl Mat {
    /// The texture set this material samples, if any.
    fn set(self) -> Option<Set> {
        Some(match self {
            Mat::Glass => Set::Glass,
            Mat::Office => Set::Office,
            Mat::Brick => Set::Brick,
            Mat::Panel => Set::Panel,
            Mat::Deco => Set::Deco,
            Mat::Siding => Set::Siding,
            Mat::Roof => Set::Roof,
            Mat::Tiles => Set::Tiles,
            Mat::Road => Set::Road,
            Mat::Asphalt => Set::Asphalt,
            Mat::Crossing => Set::Crossing,
            Mat::Pavers => Set::Pavers,
            Mat::Plaza => Set::Plaza,
            Mat::Grass | Mat::Hedge => Set::Grass,
            Mat::Dirt => Set::Dirt,
            Mat::Water => Set::Water,
            Mat::Brushed => Set::Brushed,
            Mat::Chrome | Mat::Gold => Set::Chrome,
            Mat::Concrete => Set::Concrete,
            Mat::Container => Set::Container,
            Mat::Solar => Set::Solar,
            Mat::Leaves => Set::Leaves,
            Mat::Fronds => Set::Fronds,
            Mat::Wood => Set::Wood,
            Mat::Canvas => Set::Canvas,
            Mat::Lattice => Set::Lattice,
            Mat::Fence => Set::Fence,
            Mat::Neon(_) => Set::Neon,
            Mat::Ad(slot) => Set::Ad(slot),
            Mat::Fixture => Set::Fixture,
            Mat::Vehicle => Set::Vehicle,
            Mat::Seats => Set::Seats,
            Mat::Pitch => Set::Pitch,
            Mat::Rails => Set::Rails,
            Mat::Person | Mat::Flora | Mat::ClearGlass | Mat::FrostedGlass | Mat::Pane => {
                return None;
            }
        })
    }

    fn facade(self) -> bool {
        matches!(
            self,
            Mat::Glass | Mat::Office | Mat::Brick | Mat::Panel | Mat::Deco | Mat::Siding
        )
    }

    /// Factors when a workload draws without textures.
    fn untextured(self) -> Surface {
        let surface = |roughness, metalness| Surface {
            roughness,
            metalness,
            emission: 0.0,
            ripples: 0.0,
        };
        match self {
            Mat::Chrome => surface(0.08, 1.0),
            Mat::Gold => surface(0.2, 1.0),
            Mat::Brushed => surface(0.3, 1.0),
            Mat::Water => Surface {
                ripples: 0.5,
                ..surface(0.05, 0.0)
            },
            Mat::Glass | Mat::Panel => surface(0.15, 0.6),
            Mat::Solar => surface(0.15, 0.3),
            Mat::Vehicle => surface(0.3, 0.4),
            Mat::Person | Mat::Flora | Mat::ClearGlass | Mat::FrostedGlass | Mat::Pane => {
                self.plain().1
            }
            _ => surface(0.75, 0.0),
        }
    }

    /// Color factor and factors (multiplying maps where textured).
    fn plain(self) -> ([f32; 4], Surface) {
        let surface = |roughness, metalness| Surface {
            roughness,
            metalness,
            emission: 0.0,
            ripples: 0.0,
        };
        match self {
            Mat::Gold => ([1.0, 0.78, 0.4, 1.0], surface(1.0, 1.0)),
            Mat::Hedge => ([0.55, 0.75, 0.5, 1.0], surface(1.0, 1.0)),
            Mat::Water => (
                [1.0; 4],
                Surface {
                    ripples: 0.5,
                    ..surface(1.0, 1.0)
                },
            ),
            Mat::Person => ([1.0; 4], surface(0.8, 0.0)),
            Mat::Flora => ([1.0; 4], surface(0.85, 0.0)),
            Mat::ClearGlass => ([0.9, 0.95, 1.0, 1.0], surface(0.05, 0.0)),
            Mat::FrostedGlass => ([0.75, 0.9, 1.0, 1.0], surface(0.35, 0.0)),
            Mat::Pane => ([0.6, 0.85, 1.0, 1.0], surface(0.1, 0.0)),
            _ => ([1.0; 4], surface(1.0, 1.0)),
        }
    }
}

/// Clips of the city's animated models.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Clip {
    Walk,
    Idle,
    Flag,
    Dish,
    Trivision,
    Drawbridge,
    Door,
    Turbine,
    Excavator,
    Carousel,
    Ring,
    /// A window washer's platform riding a tower of height class `n`.
    Gondola(u8),
}

/// A workload's meshes, materials, textures and clips, admitted on first
/// use.
pub(crate) struct Library {
    variety: u32,
    textured: bool,
    compressed: bool,
    meshes: HashMap<Model, (MeshHandle, Vec3)>,
    materials: HashMap<Mat, MaterialHandle>,
    textures: HashMap<(Set, Slot), TextureHandle>,
    clips: HashMap<Clip, ClipHandle>,
}

impl Library {
    pub(crate) fn new(variety: u32, textured: bool, compressed: bool) -> Self {
        Self {
            variety,
            textured,
            compressed,
            meshes: HashMap::new(),
            materials: HashMap::new(),
            textures: HashMap::new(),
            clips: HashMap::new(),
        }
    }

    /// The mesh drawing `model` and its size.
    pub(crate) fn mesh(
        &mut self,
        assets: &mut AssetLibrary,
        model: Model,
    ) -> Result<(MeshHandle, Vec3), String> {
        let model = model.resolve(self.variety);
        if let Some(found) = self.meshes.get(&model) {
            return Ok(*found);
        }
        let mesh = model.build()?;
        let bounds = mesh.bounds();
        let handle = assets
            .insert_mesh(mesh)
            .map_err(|error| error.to_string())?;
        let entry = (handle, bounds.max() - bounds.min());
        self.meshes.insert(model, entry);
        Ok(entry)
    }

    fn texture(
        &mut self,
        assets: &mut AssetLibrary,
        set: Set,
        slot: Slot,
    ) -> Result<Option<TextureRef>, String> {
        if let Some(texture) = self.textures.get(&(set, slot)) {
            return Ok(Some(self.reference(set, *texture)));
        }
        let Some(payload) = payload(set, slot, self.compressed) else {
            return Ok(None);
        };
        let texture = match payload {
            Payload::Rgba8 {
                width,
                height,
                pixels,
                alpha,
            } => {
                if alpha {
                    Texture::from_rgba(width, height, pixels)
                } else {
                    Texture::new(width, height, pixels)
                }
            }
            Payload::Bc7 {
                width,
                height,
                levels,
            } => Texture::compressed(engine::TextureFormat::Bc7, width, height, levels),
        }
        .map_err(|error| error.to_string())?;
        let handle = assets
            .insert_texture(texture)
            .map_err(|error| error.to_string())?;
        self.textures.insert((set, slot), handle);
        Ok(Some(self.reference(set, handle)))
    }

    fn reference(&self, set: Set, texture: TextureHandle) -> TextureRef {
        let clamp = matches!(
            set,
            Set::Leaves | Set::Fronds | Set::Ad(_) | Set::Fixture | Set::Vehicle | Set::Neon
        );
        TextureRef::new(texture).with_sampler(if clamp {
            Sampler::CLAMP
        } else {
            Sampler::REPEAT
        })
    }

    /// The asset material of `mat`; ads start on their own slot's picture.
    pub(crate) fn material(
        &mut self,
        assets: &mut AssetLibrary,
        mat: Mat,
    ) -> Result<MaterialHandle, String> {
        if let Some(handle) = self.materials.get(&mat) {
            return Ok(*handle);
        }
        let material = self.describe(assets, mat, None)?;
        let handle = assets
            .insert_material(material)
            .map_err(|error| error.to_string())?;
        self.materials.insert(mat, handle);
        Ok(handle)
    }

    /// Board slot `slot` showing advertisement `ad` (a set of its own).
    pub(crate) fn advert(
        &mut self,
        assets: &mut AssetLibrary,
        slot: u8,
        ad: u8,
    ) -> Result<Material, String> {
        self.describe(assets, Mat::Ad(slot), Some(ad))
    }

    fn describe(
        &mut self,
        assets: &mut AssetLibrary,
        mat: Mat,
        ad: Option<u8>,
    ) -> Result<Material, String> {
        let error = |error: AssetError| error.to_string();
        let set = match (mat, ad) {
            (Mat::Ad(_), Some(ad)) => Some(Set::Ad(ad)),
            (Mat::Ad(slot), None) => Some(Set::Ad(slot)),
            _ => mat.set(),
        }
        .filter(|_| self.textured);
        let (color, surface) = mat.plain();
        let surface = if set.is_some() {
            surface
        } else {
            mat.untextured()
        };
        let mut material = Material::new(color)
            .map_err(error)?
            .with_surface(surface)
            .map_err(error)?;
        material = match mat {
            Mat::Leaves | Mat::Fronds | Mat::Lattice | Mat::Fence => material
                .with_alpha_mode(AlphaMode::Mask { cutoff: 0.5 })
                .map_err(error)?
                .with_double_sided(true),
            Mat::Canvas => material.with_double_sided(true),
            Mat::ClearGlass => material
                .with_transmission(Transmission {
                    factor: 0.95,
                    ior: 1.5,
                    thickness: 0.4,
                })
                .map_err(error)?,
            Mat::FrostedGlass => material
                .with_transmission(Transmission {
                    factor: 0.85,
                    ior: 1.45,
                    thickness: 0.2,
                })
                .map_err(error)?,
            Mat::Pane => material
                .with_opacity(0.35)
                .map_err(error)?
                .with_double_sided(true),
            _ => material,
        };
        let Some(set) = set else {
            // Untextured workloads keep plain factors; neon and ads glow.
            return match mat {
                Mat::Neon(color) => material
                    .with_emissive(NEON_COLORS[color as usize % NEON_COLORS.len()].map(|c| c * 2.0))
                    .map_err(error),
                Mat::Ad(_) => material.with_emissive([1.2; 3]).map_err(error),
                _ => Ok(material),
            };
        };
        if let Some(base) = self.texture(assets, set, Slot::Base)? {
            material = material.with_base_color_texture(base).map_err(error)?;
        }
        if let Some(mr) = self.texture(assets, set, Slot::Mr)? {
            material = material
                .with_metallic_roughness_texture(mr)
                .map_err(error)?;
        }
        if let Some(normal) = self.texture(assets, set, Slot::Normal)? {
            let scale = if mat == Mat::Water { 0.6 } else { 1.0 };
            material = material.with_normal_texture(normal, scale).map_err(error)?;
        }
        if mat.facade() {
            // Grime spans each building through its second coordinates.
            if let Some(grime) = self.texture(assets, Set::Grime, Slot::Mr)? {
                material = material
                    .with_occlusion_texture(grime.with_tex_coord(1), 1.0)
                    .map_err(error)?;
            }
        } else if matches!(
            mat,
            Mat::Concrete | Mat::Plaza | Mat::Pavers | Mat::Tiles | Mat::Container
        ) {
            // Occlusion packed in the metallic-roughness texture's red.
            if let Some(mr) = self.texture(assets, set, Slot::Mr)? {
                material = material.with_occlusion_texture(mr, 1.0).map_err(error)?;
            }
        }
        if let Some(glow) = self.texture(assets, set, Slot::Emissive)? {
            let factor = match mat {
                Mat::Neon(color) => {
                    NEON_COLORS[color as usize % NEON_COLORS.len()].map(|c| c * 4.0)
                }
                Mat::Ad(_) => [1.6; 3],
                Mat::Fixture | Mat::Vehicle => [3.0; 3],
                _ => [1.6; 3],
            };
            material = material
                .with_emissive(factor)
                .map_err(error)?
                .with_emissive_texture(glow)
                .map_err(error)?;
        }
        Ok(material)
    }

    pub(crate) fn clip(
        &mut self,
        assets: &mut AssetLibrary,
        clip: Clip,
    ) -> Result<ClipHandle, String> {
        if let Some(handle) = self.clips.get(&clip) {
            return Ok(*handle);
        }
        let built = match clip {
            Clip::Walk => people::walk()?,
            Clip::Idle => people::idle()?,
            _ => super::props::clip(clip)?,
        };
        let handle = assets
            .insert_clip(built)
            .map_err(|error| error.to_string())?;
        self.clips.insert(clip, handle);
        Ok(handle)
    }
}
