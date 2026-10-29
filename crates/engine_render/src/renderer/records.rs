use super::*;
use crate::water::WaterPass;

impl Renderer {
    pub fn take_record_wants(&mut self) -> Vec<RecordKey> {
        let mut wants = Vec::new();
        self.core.impostors.atlas.take_wants(&mut wants);
        self.core.impostors.placed.take_wants(&mut wants);
        wants
    }

    pub fn give_record(&mut self, record: RecordKey, bytes: Option<Vec<u8>>) {
        let impostors = &mut self.core.impostors;
        if impostors.atlas.asks(record) {
            impostors.atlas.answer(record, bytes);
        } else if impostors.placed.asks(record) {
            impostors.placed.answer(record, bytes);
        }
    }

    pub fn take_made_records(&mut self) -> Vec<(RecordKey, Vec<u8>, engine_core::RecordParity)> {
        let mut impostors = Vec::new();
        self.core.impostors.atlas.take_made(&mut impostors);
        self.core.impostors.placed.take_made(&mut impostors);
        impostors
            .into_iter()
            .map(|(record, bytes)| (record, bytes, engine_core::RecordParity::Exact))
            .collect()
    }

    pub fn records_left(&self) -> [(&'static str, u64); 2] {
        [
            (
                "impostors",
                self.core.impostors.atlas.left() + self.core.impostors.placed.left(),
            ),
            (
                "water looks",
                self.parts
                    .get::<WaterPass>()
                    .map_or(0, |water| water.settle().looks_left()),
            ),
        ]
    }

    pub fn settle_water(
        &mut self,
        starts: Vec<(
            u64,
            Vec<RecordKey>,
            std::sync::Arc<engine_terrain::SettleStart>,
        )>,
    ) {
        let _owner = crate::memory::owned(crate::water::PART.decl.brick);
        if let Some(water) = self.parts.get_mut::<WaterPass>() {
            water.settle_mut().settle(starts);
        }
    }

    pub fn take_settled_water(&mut self) -> Vec<(RecordKey, Result<Vec<u8>, String>)> {
        self.parts
            .get_mut::<WaterPass>()
            .map(|water| water.settle_mut().take_made())
            .unwrap_or_default()
    }

    pub fn set_water_cooking(&mut self, cooking: bool) {
        if let Some(water) = self.parts.get_mut::<WaterPass>() {
            water.settle_mut().set_cooking(cooking);
        }
    }
}
