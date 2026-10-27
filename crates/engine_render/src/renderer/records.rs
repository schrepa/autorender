//! The records the renderer makes, which its host keeps and the cook ships:
//! impostors' layers (`impostors`). The renderer names the records it wants
//! by key and hands over those it made; its host fetches them from its
//! packs and store and keeps what it is handed (`engine_core::Jobs`), as it
//! does every record the engine makes, and answers each fetch with a later
//! frame. Made on the GPU, they are cooked once, and a verifying host never
//! has them made again.

use super::*;

impl Renderer {
    /// Records the renderer asks hosts for since the last call: the layers
    /// of impostors it would otherwise bake. Hosts answer each
    /// ([`Self::give_record`]).
    pub fn take_record_wants(&mut self) -> Vec<RecordKey> {
        let mut wants = Vec::new();
        self.impostors.atlas.take_wants(&mut wants);
        self.impostors.placed.take_wants(&mut wants);
        wants
    }

    /// Hosts' answer for record `record`, which the renderer asked for
    /// ([`Self::take_record_wants`]): its bytes, or `None` where no pack or
    /// store holds it (or the host that would have answered is gone), which
    /// bakes what it holds.
    pub fn give_record(&mut self, record: RecordKey, bytes: Option<Vec<u8>>) {
        let impostors = &mut self.impostors;
        if impostors.atlas.asks(record) {
            impostors.atlas.answer(record, bytes);
        } else if impostors.placed.asks(record) {
            impostors.placed.answer(record, bytes);
        }
    }

    /// Records the renderer made since the last call, for hosts to keep,
    /// each with how a verifying host compares it: the layers of impostors
    /// it baked (never made again for one).
    pub fn take_made_records(&mut self) -> Vec<(RecordKey, Vec<u8>, engine_core::RecordParity)> {
        let mut impostors = Vec::new();
        self.impostors.atlas.take_made(&mut impostors);
        self.impostors.placed.take_made(&mut impostors);
        impostors
            .into_iter()
            .map(|(record, bytes)| (record, bytes, engine_core::RecordParity::Exact))
            .collect()
    }

    /// Records the renderer still has to come (asked of hosts, baking or
    /// reading back), by what makes them: impostors' layers; and the looks
    /// the water settling for records has left at most, which a region
    /// settling long counts down.
    pub fn records_left(&self) -> [(&'static str, u64); 2] {
        [
            (
                "impostors",
                self.impostors.atlas.left() + self.impostors.placed.left(),
            ),
            ("water looks", self.water_settle.looks_left()),
        ]
    }

    /// Settles the water of each of `starts` on the GPU for its records
    /// (`settled_water`, one for each tile it keeps, in its order), those
    /// already settling left as they are: terrains hand them over
    /// ([`engine_core::Terrains::take_water_settles`]).
    pub fn settle_water(
        &mut self,
        starts: Vec<(
            Vec<RecordKey>,
            std::sync::Arc<engine_core::terrain_data::SettleStart>,
        )>,
    ) {
        self.water_settle.settle(starts);
    }

    /// The records of settled water made since the last call, or why one
    /// failed, for hosts to keep within their tolerance
    /// ([`engine_core::SETTLED_PARITY`]) and hand the terrains that asked.
    pub fn take_settled_water(&mut self) -> Vec<(RecordKey, Result<Vec<u8>, String>)> {
        self.water_settle.take_made()
    }

    /// Whether the session cooks, which has water settle with more of the
    /// GPU's memory and more steps a frame.
    pub fn set_water_cooking(&mut self, cooking: bool) {
        self.water_settle.set_cooking(cooking);
    }
}
