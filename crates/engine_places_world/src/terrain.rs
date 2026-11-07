use engine_place::List;

pub use engine_heightfield::layer::LayerKind;

pub struct TerrainLayers;

impl List for TerrainLayers {
    const NAME: &'static str = "terrain_layers";
    type Member = &'static LayerKind;
    const SMOKES: &'static [engine_place::Smoke] = &[crate::smokes::TERRAIN_LAYERS];
}
