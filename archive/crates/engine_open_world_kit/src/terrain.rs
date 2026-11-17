use engine_heightfield::layer::LayerKind;
use engine_place::List;

pub struct TerrainLayers;

impl List for TerrainLayers {
    const NAME: &'static str = "terrain_layers";
    type Member = &'static LayerKind;
}
