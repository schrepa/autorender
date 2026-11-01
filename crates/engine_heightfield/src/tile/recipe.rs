use std::sync::Arc;

use engine_codec::{Error, ErrorCode, Reader, record};
use glam::Vec2;

use super::{BiomeRules, Biomes};
use crate::layer::LayerKind;
use crate::{Layers, Pyramid, Stamp};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StackView {
    pub offset: Vec2,
    pub low: Vec2,
    pub high: Vec2,
    pub blend: f32,
}

record!(StackView {
    offset,
    low,
    high,
    blend,
});

#[derive(Clone, Debug, PartialEq)]
pub struct TerrainRecipe {
    pub stack: [u8; 32],
    pub view: Option<StackView>,
    pub pyramid: Pyramid,
    pub surface: Biomes,
    pub materials: bool,
}

record!(TerrainRecipe {
    stack,
    view,
    pyramid,
    surface,
    materials,
});

pub struct Recipe {
    pub layers: Arc<Layers>,
    pub pyramid: Pyramid,
    pub rules: Arc<BiomeRules>,
    pub materials: bool,
    pub around: bool,
}

impl Recipe {
    pub fn read<'a>(
        recipe: &[u8],
        stack: &dyn Fn([u8; 32]) -> Option<&'a [u8]>,
        kinds: &[&'static LayerKind],
    ) -> Result<Self, Error> {
        let recipe: TerrainRecipe = engine_codec::decode(recipe)?;
        let shared = stack(recipe.stack).ok_or_else(|| {
            Error::new(ErrorCode::UnknownReference, "$.stack").got(hex(recipe.stack))
        })?;
        let mut reader = Reader::new(shared);
        let layers = Layers::read(&mut reader, kinds)?;
        reader.finish()?;
        let stack = Arc::new(layers);
        let layers = match recipe.view {
            Some(view) => {
                Arc::new(stack.surroundings(view.offset, (view.low, view.high), view.blend))
            }
            None => stack,
        };
        let rules = recipe.surface.compile()?;
        Ok(Self {
            layers,
            pyramid: recipe.pyramid,
            rules: Arc::new(rules),
            materials: recipe.materials,
            around: recipe.view.is_some(),
        })
    }

    pub fn stamped(
        &self,
        stamps: &[Stamp],
        kinds: &[&'static LayerKind],
    ) -> Result<Arc<Layers>, Error> {
        stamped(&self.layers, stamps, kinds)
    }
}

pub fn stamped(
    layers: &Arc<Layers>,
    stamps: &[Stamp],
    kinds: &[&'static LayerKind],
) -> Result<Arc<Layers>, Error> {
    if stamps.is_empty() {
        return Ok(layers.clone());
    }
    layers.with_stamps(stamps, kinds).map(Arc::new)
}

fn hex(key: [u8; 32]) -> String {
    key.iter().map(|byte| format!("{byte:02x}")).collect()
}
