//! The engine's job worker (decision 110): a small Wasm module hosts run in
//! Web Workers beside the engine. It holds what the engine shares with its
//! workers (terrain stacks and recipes) and runs the jobs the engine hands
//! out, terrain tiles, walks and bakes, with the code engines without
//! workers run in place, so outputs are the same bits. Jobs run a slice at a
//! time (decision 112): hosts start one, step it until done, and between
//! steps take up a more urgent one or drop one the engine cancelled; parts
//! of an output ready early go back before the rest (decision 114).

use engine_terrain::jobs::TerrainWorker;

/// One worker's shares and the jobs it runs.
#[derive(Default)]
pub struct Worker {
    terrain: TerrainWorker,
}

impl Worker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Holds `bytes` as `key`, replacing what it held.
    pub fn share(&mut self, key: &str, bytes: &[u8]) -> Result<(), String> {
        self.terrain.share(key, bytes)
    }

    /// Drops what `key` held.
    pub fn unshare(&mut self, key: &str) {
        self.terrain.unshare(key);
    }

    /// Drops everything shared, and every job started.
    pub fn clear(&mut self) {
        self.terrain = TerrainWorker::new();
    }

    /// Runs job `kind` on `input` whole; its output.
    pub fn run(&mut self, kind: &str, input: &[u8]) -> Result<Vec<u8>, String> {
        self.terrain.run(kind, input)
    }

    /// Starts job `id` of `kind` on `input`, to [`Self::step`] until done.
    pub fn start(&mut self, id: u64, kind: &str, input: &[u8]) -> Result<(), String> {
        self.terrain.start(id, kind, input)
    }

    /// Works on job `id` for about `budget` work units: its output once
    /// done, `None` while work is left.
    pub fn step(&mut self, id: u64, budget: u64) -> Result<Option<Vec<u8>>, String> {
        self.terrain.step(id, budget)
    }

    /// A part of job `id`'s output ready before the rest (decision 114),
    /// taken once.
    pub fn take_part(&mut self, id: u64) -> Option<Vec<u8>> {
        self.terrain.take_part(id)
    }

    /// Stops job `id`.
    pub fn cancel(&mut self, id: u64) {
        self.terrain.cancel(id);
    }
}

#[cfg(target_arch = "wasm32")]
mod browser {
    use wasm_bindgen::prelude::*;

    /// [`super::Worker`] for a Web Worker's script. Job ids and budgets are
    /// JavaScript numbers.
    #[wasm_bindgen]
    pub struct JobWorker {
        worker: super::Worker,
    }

    #[wasm_bindgen]
    impl JobWorker {
        #[wasm_bindgen(constructor)]
        pub fn new() -> Self {
            console_error_panic_hook::set_once();
            Self {
                worker: super::Worker::new(),
            }
        }

        pub fn share(&mut self, key: &str, bytes: &[u8]) -> Result<(), JsValue> {
            self.worker
                .share(key, bytes)
                .map_err(|error| JsValue::from_str(&error))
        }

        pub fn unshare(&mut self, key: &str) {
            self.worker.unshare(key);
        }

        pub fn clear(&mut self) {
            self.worker.clear();
        }

        pub fn start(&mut self, id: f64, kind: &str, input: &[u8]) -> Result<(), JsValue> {
            self.worker
                .start(id as u64, kind, input)
                .map_err(|error| JsValue::from_str(&error))
        }

        pub fn step(&mut self, id: f64, budget: f64) -> Result<Option<Vec<u8>>, JsValue> {
            self.worker
                .step(id as u64, budget as u64)
                .map_err(|error| JsValue::from_str(&error))
        }

        pub fn take_part(&mut self, id: f64) -> Option<Vec<u8>> {
            self.worker.take_part(id as u64)
        }

        pub fn cancel(&mut self, id: f64) {
            self.worker.cancel(id as u64);
        }
    }

    impl Default for JobWorker {
        fn default() -> Self {
            Self::new()
        }
    }
}
