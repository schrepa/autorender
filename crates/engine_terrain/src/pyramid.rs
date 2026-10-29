use engine_core::{Pyramid, TileKey};

crate::codec::record!(TileKey { level, x, z });

impl crate::codec::Codec for Pyramid {
    fn write(&self, out: &mut crate::codec::Writer) {
        out.put(&self.base_tiles()).put(&self.spacing(0));
    }

    fn read(input: &mut crate::codec::Reader<'_>) -> Result<Self, crate::codec::Error> {
        let tiles = input.get()?;
        let spacing = input.get()?;
        Self::new(tiles, spacing)
    }
}
