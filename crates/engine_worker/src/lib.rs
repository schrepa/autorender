//! The engine's job worker (decision 110): a small Wasm module hosts run in
//! Web Workers beside the engine. It holds what the engine shares with its
//! workers (terrain stacks and recipes) and runs the jobs the engine hands
//! out, terrain tiles, walks and bakes, with the code engines without
//! workers run in place, so outputs are the same bits.

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

    /// Drops everything shared.
    pub fn clear(&mut self) {
        self.terrain = TerrainWorker::new();
    }

    /// Runs job `kind` on `input`; its output.
    pub fn run(&mut self, kind: &str, input: &[u8]) -> Result<Vec<u8>, String> {
        self.terrain.run(kind, input)
    }
}

#[cfg(target_arch = "wasm32")]
mod browser {
    use wasm_bindgen::prelude::*;

    /// [`super::Worker`] for a Web Worker's script.
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

        pub fn run(&mut self, kind: &str, input: &[u8]) -> Result<Vec<u8>, JsValue> {
            self.worker
                .run(kind, input)
                .map_err(|error| JsValue::from_str(&error))
        }
    }

    impl Default for JobWorker {
        fn default() -> Self {
            Self::new()
        }
    }
}
