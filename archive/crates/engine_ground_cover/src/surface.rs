use bevy_ecs::prelude::*;
use engine_core::{
    AlphaMode, Asset, AssetLibrary, AssetResidency, AssetUses, BASE_COLOR_MAP, ColorSpace, Cull,
    Error, LightingModel, Material, MaterialHandle, NORMAL_MAP, Sampler, SurfaceSlot, Texture,
    TextureHandle, TextureRef,
};
use engine_places::surface::{OCCLUSION, ROUGHNESS, TRANSMISSION};
use engine_places_rendering::lighting::inputs;
use engine_render::SlotModule;

use crate::settings::GroundCover;
use crate::shaders::cover::{card, surface};

pub(crate) const FIELD: &str = "cover_field";

const COVER_SURFACE: &str = "cover";
const CARD_SURFACE: &str = "cover_card";
const COVER_MATERIAL: &str = "$.cover.material";
const CARD_MATERIAL: &str = "$.cover.card_material";
const CARD_MAPS: &[&str] = &[BASE_COLOR_MAP, NORMAL_MAP];

const COVER_SLOTS: [&str; 9] = [
    SurfaceSlot::BaseColor.name(),
    SurfaceSlot::Opacity.name(),
    SurfaceSlot::Normal.name(),
    ROUGHNESS,
    OCCLUSION,
    TRANSMISSION,
    inputs::MIRRORED.name,
    SurfaceSlot::Kind.name(),
    inputs::TANGLED.name,
];

const CARD_SLOTS: [&str; 10] = [
    SurfaceSlot::BaseColor.name(),
    SurfaceSlot::Opacity.name(),
    SurfaceSlot::Cutoff.name(),
    SurfaceSlot::Normal.name(),
    ROUGHNESS,
    OCCLUSION,
    TRANSMISSION,
    inputs::MIRRORED.name,
    SurfaceSlot::Kind.name(),
    inputs::TANGLED.name,
];

type Held = Result<Option<MaterialHandle>, Error>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CardSource {
    albedo: Option<TextureHandle>,
    normal: Option<(TextureHandle, u64)>,
    cutoff: u32,
    relief: u32,
}

struct CardSurface {
    source: CardSource,
    linear: Option<TextureHandle>,
    made: Result<MaterialHandle, Error>,
}

#[derive(Resource)]
pub(crate) struct CoverSurface {
    blades: Option<Result<MaterialHandle, Error>>,
    cards: Vec<CardSurface>,
    pub(crate) blade: Held,
    pub(crate) card: Vec<Held>,
    uses: AssetUses,
}

impl Default for CoverSurface {
    fn default() -> Self {
        Self {
            blades: None,
            cards: Vec::new(),
            blade: Ok(None),
            card: Vec::new(),
            uses: AssetUses::default(),
        }
    }
}

pub(crate) fn slot_modules() -> impl Iterator<Item = SlotModule> {
    let blades = COVER_SLOTS.into_iter().map(|slot| SlotModule {
        slot,
        name: COVER_SURFACE,
        module: surface::MODULE,
        maps: &[],
        values: &[],
        reach: 0.0,
    });
    let cards = CARD_SLOTS.into_iter().map(|slot| SlotModule {
        slot,
        name: CARD_SURFACE,
        module: card::MODULE,
        maps: CARD_MAPS,
        values: &[],
        reach: 0.0,
    });
    blades.chain(cards)
}

fn blade_material() -> Result<Material, Error> {
    COVER_SLOTS.into_iter().try_fold(
        Material::new([1.0, 1.0, 1.0, 1.0])?
            .with_cull(Cull::None)
            .with_lighting(LightingModel::Named(FIELD.to_owned())),
        |material, slot| material.with_slot(slot, COVER_SURFACE),
    )
}

fn card_material(source: CardSource, normal: Option<TextureHandle>) -> Result<Material, Error> {
    let mut material = Material::new([1.0, 1.0, 1.0, 1.0])?
        .with_alpha_mode(AlphaMode::Mask {
            cutoff: f32::from_bits(source.cutoff),
        })?
        .with_normal_scale(f32::from_bits(source.relief))?
        .with_cull(Cull::None)
        .with_lighting(LightingModel::Named(FIELD.to_owned()));
    let maps = [(BASE_COLOR_MAP, source.albedo), (NORMAL_MAP, normal)];
    for (name, texture) in maps {
        if let Some(texture) = texture {
            let map = TextureRef::new(texture).with_sampler(Sampler::CLAMP);
            material = material.with_map(name, map)?;
        }
    }
    CARD_SLOTS
        .into_iter()
        .try_fold(material, |material, slot| {
            material.with_slot(slot, CARD_SURFACE)
        })
}

fn linear(
    library: &mut AssetLibrary,
    handle: TextureHandle,
) -> Result<Option<TextureHandle>, Error> {
    let Some(texture) = library.texture(handle) else {
        return Ok(None);
    };
    if texture.color_space() == ColorSpace::Linear {
        return Ok(None);
    }
    let copy = Texture::from_rgba(texture.width(), texture.height(), texture.pixels().to_vec())?
        .with_color_space(ColorSpace::Linear);
    library.insert_texture(copy).map(Some)
}

impl CardSurface {
    fn of(library: &mut AssetLibrary, source: CardSource) -> Self {
        let copied = match source.normal {
            Some((normal, _)) => linear(library, normal),
            None => Ok(None),
        };
        let linear = copied.as_ref().ok().copied().flatten();
        let normal = linear.or(source.normal.map(|(normal, _)| normal));
        let made = copied
            .and_then(|_| card_material(source, normal))
            .and_then(|material| library.insert_material(material));
        Self {
            source,
            linear,
            made,
        }
    }

    fn kept(&self, library: &AssetLibrary, source: CardSource) -> bool {
        self.source == source
            && self
                .made
                .as_ref()
                .map_or(true, |&made| library.material(made).is_some())
    }

    fn drop_from(self, library: &mut AssetLibrary) {
        if let Ok(made) = self.made {
            library.remove_material(made);
        }
        if let Some(linear) = self.linear {
            library.remove_texture(linear);
        }
    }
}

fn sources(cover: &GroundCover, library: &AssetLibrary) -> Vec<CardSource> {
    cover
        .atlases
        .values()
        .map(|atlas| CardSource {
            albedo: atlas.albedo,
            normal: atlas
                .normal
                .and_then(|normal| Some((normal, library.revision_of(normal)?))),
            cutoff: cover.cards.cutoff.to_bits(),
            relief: cover.cards.relief.to_bits(),
        })
        .collect()
}

pub(crate) fn hold_surface(
    mut library: ResMut<AssetLibrary>,
    residency: Res<AssetResidency>,
    cover: Res<GroundCover>,
    mut surface: ResMut<CoverSurface>,
) {
    let surface = &mut *surface;
    let kept = surface
        .blades
        .as_ref()
        .and_then(|made| made.as_ref().ok().copied())
        .filter(|&made| library.material(made).is_some());
    let blades = match kept {
        Some(made) => Ok(made),
        None => blade_material().and_then(|material| library.insert_material(material)),
    };
    let wanted = sources(&cover, &library);
    let same = surface.cards.len() == wanted.len()
        && surface
            .cards
            .iter()
            .zip(&wanted)
            .all(|(card, &source)| card.kept(&library, source));
    if !same {
        for card in surface.cards.drain(..) {
            card.drop_from(&mut library);
        }
        surface.cards = wanted
            .into_iter()
            .map(|source| CardSurface::of(&mut library, source))
            .collect();
    }
    let made: Vec<Asset> = std::iter::once(&blades)
        .chain(surface.cards.iter().map(|card| &card.made))
        .filter_map(|material| material.as_ref().ok().copied())
        .map(Asset::Material)
        .collect();
    if surface.uses.assets() != made.as_slice() {
        surface.uses = AssetUses::new(&residency, made);
    }
    surface.blade = residency.made_material(Some(&blades), COVER_MATERIAL);
    surface.blades = Some(blades);
    surface.card.clear();
    for card in &surface.cards {
        surface
            .card
            .push(residency.made_material(Some(&card.made), CARD_MATERIAL));
    }
}
