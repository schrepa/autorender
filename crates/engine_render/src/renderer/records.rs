use super::*;
use crate::parts::{MadeRecord, RecordStart};

impl Renderer {
    fn each_part(&mut self, mut act: impl FnMut(&mut dyn GpuPart)) {
        for index in 0..self.parts.len() {
            let _owner = crate::memory::owned(self.parts.decl(index).brick);
            self.parts.with_part(index, |part, _| act(part));
        }
    }

    pub fn take_record_wants(&mut self) -> Vec<RecordKey> {
        let mut wants = Vec::new();
        self.core.impostors.atlas.take_wants(&mut wants);
        self.each_part(|part| part.record_wants(&mut wants));
        wants
    }

    pub fn give_record(&mut self, record: RecordKey, bytes: Option<Vec<u8>>) {
        let impostors = &mut self.core.impostors;
        if impostors.atlas.asks(record) {
            impostors.atlas.answer(record, bytes);
            return;
        }
        let Some(index) = (0..self.parts.len()).find(|&index| {
            self.parts
                .part(index)
                .is_some_and(|part| part.asks_record(record))
        }) else {
            return;
        };
        let _owner = crate::memory::owned(self.parts.decl(index).brick);
        self.parts
            .with_part(index, |part, _| part.answer_record(record, bytes));
    }

    pub fn take_made_records(&mut self) -> Vec<(RecordKey, Vec<u8>, engine_core::RecordParity)> {
        let mut made = Vec::new();
        self.core.impostors.atlas.take_made(&mut made);
        self.each_part(|part| part.made_records(&mut made));
        made.into_iter()
            .map(|(record, bytes)| (record, bytes, engine_core::RecordParity::Exact))
            .collect()
    }

    pub fn records_left(&self) -> Vec<(&'static str, u64)> {
        let mut left = vec![("impostors", self.core.impostors.atlas.left())];
        for (_, part) in self.parts.iter() {
            let Some((name, count)) = part.records_left() else {
                continue;
            };
            match left.iter_mut().find(|(known, _)| *known == name) {
                Some((_, known)) => *known += count,
                None => left.push((name, count)),
            }
        }
        left
    }

    pub fn start_records(&mut self, mut starts: Vec<RecordStart>) {
        for index in 0..self.parts.len() {
            if starts.is_empty() {
                return;
            }
            let _owner = crate::memory::owned(self.parts.decl(index).brick);
            self.parts
                .with_part(index, |part, _| part.start_records(&mut starts));
        }
    }

    pub fn take_records(&mut self) -> Vec<MadeRecord> {
        let mut made = Vec::new();
        self.each_part(|part| made.extend(part.take_records()));
        made
    }
}
