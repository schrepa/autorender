use super::*;

impl Renderer {
    pub fn take_record_wants(&mut self) -> Vec<RecordKey> {
        let mut wants = Vec::new();
        self.impostors.atlas.take_wants(&mut wants);
        self.impostors.placed.take_wants(&mut wants);
        wants
    }

    pub fn give_record(&mut self, record: RecordKey, bytes: Option<Vec<u8>>) {
        let impostors = &mut self.impostors;
        if impostors.atlas.asks(record) {
            impostors.atlas.answer(record, bytes);
        } else if impostors.placed.asks(record) {
            impostors.placed.answer(record, bytes);
        }
    }

    pub fn take_made_records(&mut self) -> Vec<(RecordKey, Vec<u8>, engine_core::RecordParity)> {
        let mut impostors = Vec::new();
        self.impostors.atlas.take_made(&mut impostors);
        self.impostors.placed.take_made(&mut impostors);
        impostors
            .into_iter()
            .map(|(record, bytes)| (record, bytes, engine_core::RecordParity::Exact))
            .collect()
    }

    pub fn records_left(&self) -> [(&'static str, u64); 2] {
        [
            (
                "impostors",
                self.impostors.atlas.left() + self.impostors.placed.left(),
            ),
            ("water looks", self.water_settle.looks_left()),
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
        let _owner = crate::memory::feature(crate::render_graph::FeaturePass::Water);
        self.water_settle.settle(starts);
    }

    pub fn take_settled_water(&mut self) -> Vec<(RecordKey, Result<Vec<u8>, String>)> {
        self.water_settle.take_made()
    }

    pub fn set_water_cooking(&mut self, cooking: bool) {
        self.water_settle.set_cooking(cooking);
    }
}
