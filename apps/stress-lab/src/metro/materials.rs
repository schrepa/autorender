//! Materials of the metro over the generated texture sets, and the library that
//! admits meshes, textures, materials and clips into a workload's asset
//! library. Each texture set loads as texture records (`kit::records`),
//! generated and cooked only where no pack or store holds them.

use engine::collections::HashMap;

use engine::prelude::*;

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

impl Slot {
    /// The maps `set` has, in the order its records list them: each one's
    /// slot, name and how its texels read.
    fn of(set: Set) -> Vec<(Slot, &'static str, TextureUsage)> {
        let mut slots = vec![(Slot::Base, "base", TextureUsage::Color)];
        if set.has_mr() {
            slots.push((Slot::Mr, "mr", TextureUsage::Data));
        }
        if set.has_normal() {
            slots.push((Slot::Normal, "normal", TextureUsage::Normal));
        }
        if set.emits() {
            slots.push((Slot::Emissive, "emissive", TextureUsage::Color));
        }
        slots
    }
}

/// `slot`'s map of `maps`, when the set has one.
fn slot_image(maps: &textures::Maps, slot: Slot) -> Option<&Image> {
    match slot {
        Slot::Base => Some(&maps.base),
        Slot::Mr => maps.mr.as_ref(),
        Slot::Normal => maps.normal.as_ref(),
        Slot::Emissive => maps.emissive.as_ref(),
    }
}

/// The sources of `set`'s maps in `slots`, generated.
fn sources(set: Set, slots: &[Slot]) -> Result<Vec<TextureSource>, String> {
    let maps = textures::generate(set);
    slots
        .iter()
        .map(|&slot| {
            let image = slot_image(&maps, slot)
                .ok_or_else(|| format!("the metro's {set:?} set has no {slot:?} map"))?;
            let usage = if image.normal {
                TextureUsage::Normal
            } else if image.srgb {
                TextureUsage::Color
            } else {
                TextureUsage::Data
            };
            Ok(TextureSource {
                width: image.width,
                height: image.height,
                pixels: image.pixels.clone(),
                usage,
            })
        })
        .collect()
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
    meshes: HashMap<Model, (MeshHandle, Vec3)>,
    materials: HashMap<Mat, MaterialHandle>,
    // Each set asked for: its maps' textures by slot.
    textures: HashMap<Set, Vec<(Slot, TextureHandle)>>,
    clips: HashMap<Clip, ClipHandle>,
}

impl Library {
    pub(crate) fn new(variety: u32, textured: bool) -> Self {
        Self {
            variety,
            textured,
            meshes: HashMap::default(),
            materials: HashMap::default(),
            textures: HashMap::default(),
            clips: HashMap::default(),
        }
    }

    /// The mesh drawing `model` and its size.
    pub(crate) fn mesh(
        &mut self,
        assets: &mut AssetLibrary,
        model: Model,
    ) -> engine::Result<(MeshHandle, Vec3)> {
        let model = model.resolve(self.variety);
        if let Some(found) = self.meshes.get(&model) {
            return Ok(*found);
        }
        let mesh = model.build()?;
        let bounds = mesh.bounds();
        let handle = assets.insert_mesh(mesh)?;
        let entry = (handle, bounds.max() - bounds.min());
        self.meshes.insert(model, entry);
        Ok(entry)
    }

    /// `set`'s `slot` map, when the set has one: the set's textures asked
    /// of `records` the first time any of them is.
    fn texture(
        &mut self,
        (records, assets): (&mut Records, &mut AssetLibrary),
        set: Set,
        slot: Slot,
    ) -> engine::Result<Option<TextureRef>> {
        if !self.textures.contains_key(&set) {
            let maps = Slot::of(set);
            let named: Vec<(&str, TextureUsage)> =
                maps.iter().map(|&(_, name, usage)| (name, usage)).collect();
            let slots: Vec<Slot> = maps.iter().map(|&(slot, _, _)| slot).collect();
            let made = slots.clone();
            let name = format!("metro/{set:?}");
            let handles =
                crate::kit::records::textures((records, assets), name, &named, move || {
                    sources(set, &made)
                })?;
            self.textures
                .insert(set, slots.into_iter().zip(handles).collect());
        }
        let found = self.textures[&set]
            .iter()
            .find(|(held, _)| *held == slot)
            .map(|&(_, texture)| texture);
        Ok(found.map(|texture| self.reference(set, texture)))
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
        (records, assets): (&mut Records, &mut AssetLibrary),
        mat: Mat,
    ) -> engine::Result<MaterialHandle> {
        if let Some(handle) = self.materials.get(&mat) {
            return Ok(*handle);
        }
        let material = self.describe((records, &mut *assets), mat, None)?;
        let handle = assets.insert_material(material)?;
        self.materials.insert(mat, handle);
        Ok(handle)
    }

    /// Board slot `slot` showing advertisement `ad` (a set of its own).
    pub(crate) fn advert(
        &mut self,
        stores: (&mut Records, &mut AssetLibrary),
        slot: u8,
        ad: u8,
    ) -> engine::Result<Material> {
        self.describe(stores, Mat::Ad(slot), Some(ad))
    }

    fn describe(
        &mut self,
        (records, assets): (&mut Records, &mut AssetLibrary),
        mat: Mat,
        ad: Option<u8>,
    ) -> engine::Result<Material> {
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
        let mut material = Material::new(color)?.with_surface(surface)?;
        material = match mat {
            Mat::Leaves | Mat::Fronds | Mat::Lattice | Mat::Fence => material
                .with_alpha_mode(AlphaMode::Mask { cutoff: 0.5 })?
                .with_double_sided(true),
            Mat::Canvas => material.with_double_sided(true),
            Mat::ClearGlass => material.with_transmission(Transmission {
                factor: 0.95,
                ior: 1.5,
                thickness: 0.4,
            })?,
            Mat::FrostedGlass => material.with_transmission(Transmission {
                factor: 0.85,
                ior: 1.45,
                thickness: 0.2,
            })?,
            Mat::Pane => material.with_opacity(0.35)?.with_double_sided(true),
            _ => material,
        };
        let Some(set) = set else {
            // Untextured workloads keep plain factors; neon and ads glow.
            return match mat {
                Mat::Neon(color) => material
                    .with_emissive(NEON_COLORS[color as usize % NEON_COLORS.len()].map(|c| c * 2.0))
                    .map_err(engine::Error::from),
                Mat::Ad(_) => material
                    .with_emissive([1.2; 3])
                    .map_err(engine::Error::from),
                _ => Ok(material),
            };
        };
        if let Some(base) = self.texture((&mut *records, &mut *assets), set, Slot::Base)? {
            material = material.with_base_color_texture(base)?;
        }
        if let Some(mr) = self.texture((&mut *records, &mut *assets), set, Slot::Mr)? {
            material = material.with_metallic_roughness_texture(mr)?;
        }
        if let Some(normal) = self.texture((&mut *records, &mut *assets), set, Slot::Normal)? {
            let scale = if mat == Mat::Water { 0.6 } else { 1.0 };
            material = material.with_normal_texture(normal, scale)?;
        }
        if mat.facade() {
            // Grime spans each building through its second coordinates.
            if let Some(grime) =
                self.texture((&mut *records, &mut *assets), Set::Grime, Slot::Mr)?
            {
                material = material.with_occlusion_texture(grime.with_tex_coord(1), 1.0)?;
            }
        } else if matches!(
            mat,
            Mat::Concrete | Mat::Plaza | Mat::Pavers | Mat::Tiles | Mat::Container
        ) {
            // Occlusion packed in the metallic-roughness texture's red.
            if let Some(mr) = self.texture((&mut *records, &mut *assets), set, Slot::Mr)? {
                material = material.with_occlusion_texture(mr, 1.0)?;
            }
        }
        if let Some(glow) = self.texture((&mut *records, &mut *assets), set, Slot::Emissive)? {
            let factor = match mat {
                Mat::Neon(color) => {
                    NEON_COLORS[color as usize % NEON_COLORS.len()].map(|c| c * 4.0)
                }
                Mat::Ad(_) => [1.6; 3],
                Mat::Fixture | Mat::Vehicle => [3.0; 3],
                _ => [1.6; 3],
            };
            material = material
                .with_emissive(factor)?
                .with_emissive_texture(glow)?;
        }
        // Façades' cells are rooms, eight across.
        if mat.facade() {
            material = material.with_windows(8)?;
        }
        Ok(material)
    }

    pub(crate) fn clip(
        &mut self,
        assets: &mut AssetLibrary,
        clip: Clip,
    ) -> engine::Result<ClipHandle> {
        if let Some(handle) = self.clips.get(&clip) {
            return Ok(*handle);
        }
        let built = match clip {
            Clip::Walk => people::walk()?,
            Clip::Idle => people::idle()?,
            _ => super::props::clip(clip)?,
        };
        let handle = assets.insert_clip(built)?;
        self.clips.insert(clip, handle);
        Ok(handle)
    }
}
