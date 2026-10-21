//! The records the renderer makes, which its host keeps and the cook ships:
//! impostors' layers (`impostors`) and the runtime virtual texture's far
//! pages (`terrain_vt`). The renderer names the records it wants by key and
//! hands over those it made; its host fetches them from its packs and store
//! and keeps what it is handed (`engine_core::Jobs`), as it does every
//! record the engine makes, and answers each fetch with a later frame. Made
//! on the GPU, they are cooked once: a verifying host compares those made
//! again with those it handed back by each kind's parity
//! (`engine_core::RecordParity`), far pages' within their peak
//! signal-to-noise ratio.

use super::*;

impl Renderer {
    /// How the host's records stand from this frame on: whether it is
    /// `verifying` the records it hands back (far pages are made again and
    /// handed over to compare), and whether textures are still `loading`
    /// (far pages wait until they show their own records).
    pub fn set_records(&mut self, verifying: bool, loading: bool) {
        self.terrain
            .vt
            .set_records(crate::terrain_vt::RecordMode { verifying, loading });
    }

    /// Records the renderer asks hosts for since the last call: the layers
    /// of impostors it would otherwise bake, and terrains' far pages. Hosts
    /// answer each ([`Self::give_record`]).
    pub fn take_record_wants(&mut self) -> Vec<RecordKey> {
        let mut wants = Vec::new();
        self.impostors.atlas.take_wants(&mut wants);
        self.impostors.placed.take_wants(&mut wants);
        self.terrain.vt.take_wants(&mut wants);
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
        } else if self.terrain.vt.asks(record) {
            self.terrain.vt.answer(record, bytes);
        }
    }

    /// Records the renderer made since the last call, for hosts to keep,
    /// each with how a verifying host compares it: the layers of impostors
    /// it baked (never made again for one) and terrains' far pages.
    pub fn take_made_records(&mut self) -> Vec<(RecordKey, Vec<u8>, engine_core::RecordParity)> {
        let mut impostors = Vec::new();
        self.impostors.atlas.take_made(&mut impostors);
        self.impostors.placed.take_made(&mut impostors);
        let mut made: Vec<_> = impostors
            .into_iter()
            .map(|(record, bytes)| (record, bytes, engine_core::RecordParity::Exact))
            .collect();
        self.terrain.vt.take_made(&mut made);
        made
    }

    /// Records the renderer still has to come (asked of hosts, baking or
    /// reading back), by what makes them: impostors' layers and terrains'
    /// far pages.
    pub fn records_left(&self) -> [(&'static str, u64); 2] {
        [
            (
                "impostors",
                self.impostors.atlas.left() + self.impostors.placed.left(),
            ),
            ("far pages", self.terrain.vt.records_left()),
        ]
    }

    /// Whether far ground still waits for its records, which preparation
    /// waits for.
    pub fn records_loading(&self) -> bool {
        self.terrain.vt.records_loading()
    }

    /// Settles the water of each of `starts` on the GPU for its record
    /// (`settled_water`), those already settling left as they are: terrains
    /// hand them over ([`engine_core::Terrains::take_water_settles`]).
    pub fn settle_water(
        &mut self,
        starts: Vec<(RecordKey, std::sync::Arc<engine_core::terrain_data::SettleStart>)>,
    ) {
        self.water_settle.settle(starts);
    }

    /// The records of settled water made since the last call, or why one
    /// failed, for hosts to keep within their tolerance
    /// ([`engine_core::SETTLED_PARITY`]) and hand the terrains that asked.
    pub fn take_settled_water(&mut self) -> Vec<(RecordKey, Result<Vec<u8>, String>)> {
        self.water_settle.take_made()
    }

    /// Tiles whose water waits to settle or settles.
    pub fn water_settling_left(&self) -> u64 {
        self.water_settle.left()
    }

    /// Whether the session cooks, which has water settle with more of the
    /// GPU's memory and more steps a frame.
    pub fn set_water_cooking(&mut self, cooking: bool) {
        self.water_settle.set_cooking(cooking);
    }
}
