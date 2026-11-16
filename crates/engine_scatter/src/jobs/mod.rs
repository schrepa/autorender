use std::sync::Arc;

use engine_core::{Error, ErrorCode, Job, JobCx, JobKind, RecordKey};
use engine_heightfield::layer::LayerKind;
use engine_heightfield::tile::{Generator, Recipe, shaded_elsewhere};
use engine_open_world_kit::terrain::TerrainLayers;
use engine_tiles::TileKey;

use canopy::{CANOPY_BLOCK, Canopy, CanopyAsset, MAX_CANOPY_TEXELS};

pub(crate) mod canopy;

include!(concat!(env!("OUT_DIR"), "/jobs.rs"));

pub(crate) static CANOPY: JobKind = JobKind::new("scatter::canopy", &SOURCE, start_canopy);

struct CanopyMaking {
    recipe: Arc<Recipe>,
    assets: Vec<CanopyAsset>,
    from: u8,
    tiles: Vec<TileKey>,
    next: usize,
    generator: Option<Box<Generator>>,
    canopy: Option<Canopy>,
}

impl CanopyMaking {
    fn tile(&self, key: TileKey) -> Generator {
        let recipe = &self.recipe;
        let materials = recipe.materials && !shaded_elsewhere(key.level, recipe.around);
        Generator::new(Arc::clone(&recipe.layers), &recipe.pyramid, key)
            .with_surface(Arc::clone(&recipe.rules))
            .with_materials(materials)
            .with_placed_from(self.from)
    }
}

impl Job for CanopyMaking {
    fn step(&mut self, budget: u64, _: &mut JobCx<'_>) -> Result<Option<Vec<u8>>, Error> {
        let mut budget = budget.max(1);
        while let Some(&key) = self.tiles.get(self.next) {
            if self.generator.is_none() {
                self.generator = Some(Box::new(self.tile(key)));
            }
            let Some(generator) = self.generator.as_mut() else {
                continue;
            };
            let Some(tile) = generator.advance(&mut budget) else {
                if generator.failed() {
                    return Err(Error::new(ErrorCode::InvalidValue, "$.recipe"));
                }
                return Ok(None);
            };
            self.generator = None;
            self.next += 1;
            let Some(canopy) = self.canopy.as_mut() else {
                continue;
            };
            for placed in tile.placed() {
                canopy.add(&self.assets, placed);
            }
        }
        Ok(self.canopy.take().map(Canopy::finish))
    }
}

fn start_canopy(input: &[u8], cx: &mut JobCx<'_>) -> Result<Box<dyn Job>, Error> {
    let (ask, assets) =
        canopy::read_ask(input).ok_or_else(|| Error::new(ErrorCode::InvalidValue, "$"))?;
    let sized = |texels: u32| {
        (CANOPY_BLOCK..=MAX_CANOPY_TEXELS).contains(&texels) && texels.is_multiple_of(CANOPY_BLOCK)
    };
    if !sized(ask.size[0]) || !sized(ask.size[1]) {
        return Err(Error::new(ErrorCode::OutOfRange, "$.size")
            .range(f64::from(CANOPY_BLOCK), f64::from(MAX_CANOPY_TEXELS))
            .got(ask.size));
    }
    let kinds: Vec<&'static LayerKind> = cx.members::<TerrainLayers>().copied().collect();
    let recipe = cx.held(ask.recipe, || {
        let key = RecordKey::from_hash(ask.recipe);
        let bytes = cx
            .share(key)
            .ok_or_else(|| Error::new(ErrorCode::UnknownReference, "$.recipe").got(key.hex()))?;
        Recipe::read(bytes, &|stack| cx.share(RecordKey::from_hash(stack)), &kinds)
    })?;
    let pyramid = recipe.pyramid;
    let extent = pyramid.extent();
    let cell = engine_math::max(extent.x / ask.size[0] as f32, extent.y / ask.size[1] as f32);
    let from = (0..pyramid.levels())
        .rev()
        .find(|&level| pyramid.spacing(level) <= cell * std::f32::consts::SQRT_2)
        .unwrap_or(0) as u8;
    let tiles = (u32::from(from)..pyramid.levels())
        .filter(|&level| recipe.rules.places_from(level as u8, Some(from)))
        .flat_map(|level| pyramid.level_tiles(level))
        .collect();
    let canopy = Canopy::new(&ask, extent.to_array());
    Ok(Box::new(CanopyMaking {
        recipe,
        assets,
        from,
        tiles,
        next: 0,
        generator: None,
        canopy: Some(canopy),
    }))
}
