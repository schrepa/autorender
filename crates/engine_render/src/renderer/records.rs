use super::*;

impl Renderer {
    pub fn take_record_wants(&mut self) -> Vec<RecordKey> {
        let mut wants = Vec::new();
        self.core.impostors.atlas.take_wants(&mut wants);
        wants
    }

    pub fn give_record(&mut self, record: RecordKey, bytes: Option<Vec<u8>>) {
        let impostors = &mut self.core.impostors;
        if impostors.atlas.asks(record) {
            impostors.atlas.answer(record, bytes);
        }
    }

    pub fn take_made_records(&mut self) -> Vec<(RecordKey, Vec<u8>, engine_core::RecordParity)> {
        let mut made = Vec::new();
        self.core.impostors.atlas.take_made(&mut made);
        made.into_iter()
            .map(|(record, bytes)| (record, bytes, engine_core::RecordParity::Exact))
            .collect()
    }

    pub fn records_left(&self) -> Vec<(&'static str, u64)> {
        vec![("impostors", self.core.impostors.atlas.left())]
    }
}
