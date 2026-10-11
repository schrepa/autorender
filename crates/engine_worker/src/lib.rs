//! The engine's job worker: a small Wasm module hosts run in
//! Web Workers beside the engine. It holds what the engine shares with its
//! workers (terrain stacks and recipes) and runs the jobs the engine hands
//! out, terrain tiles, walks and bakes, with the code engines without
//! workers run in place, so outputs are the same bits. Jobs run a slice at a
//! time: hosts start one, step it until done, and between
//! steps take up a more urgent one or drop one the engine cancelled; parts
//! of an output ready early go back before the rest.

use engine_terrain::jobs::TerrainWorker;
pub use engine_terrain::{BAKE_CHILD_JOB, BakeAction, BakeCoordinator, BakeDiagnostics};

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

    /// A part of job `id`'s output ready before the rest,
    /// taken once.
    pub fn take_part(&mut self, id: u64) -> Option<Vec<u8>> {
        self.terrain.take_part(id)
    }

    /// Runs a small fixed job, so the module's hot code
    /// reaches its optimized tier before real jobs.
    pub fn warm(&mut self) {
        self.terrain.warm();
    }

    /// Stops job `id`.
    pub fn cancel(&mut self, id: u64) {
        self.terrain.cancel(id);
    }
}

#[cfg(target_arch = "wasm32")]
mod browser {
    use wasm_bindgen::prelude::*;

    fn integer(value: f64) -> Result<u64, JsValue> {
        if value.is_finite()
            && value >= 0.0
            && value <= 9_007_199_254_740_991.0
            && value.fract() == 0.0
        {
            Ok(value as u64)
        } else {
            Err(JsValue::from_str(
                "expected an exactly represented nonnegative integer",
            ))
        }
    }

    /// The same coordinator native hosts use, instantiated lazily on the
    /// page only for an uncached bake with multiple workers.
    #[wasm_bindgen]
    pub struct BakeCoordinator {
        graph: super::BakeCoordinator,
    }

    #[wasm_bindgen]
    impl BakeCoordinator {
        #[wasm_bindgen(constructor)]
        pub fn new(input: &[u8], epoch: f64) -> Result<Self, JsValue> {
            Ok(Self {
                graph: super::BakeCoordinator::start(input, integer(epoch)?)
                    .map_err(|error| JsValue::from_str(&error))?,
            })
        }

        /// Ordered {share, bytes}, {unshare}, or {id, input} actions.
        pub fn ready(&mut self, limit: f64) -> Result<js_sys::Array, JsValue> {
            let limit = usize::try_from(integer(limit)?)
                .map_err(|_| JsValue::from_str("run count is too large"))?;
            let actions = self
                .graph
                .ready(limit)
                .map_err(|error| JsValue::from_str(&error))?;
            let values = js_sys::Array::new();
            for action in actions {
                let value = js_sys::Object::new();
                let set = |name: &str, content: &JsValue| {
                    js_sys::Reflect::set(&value, &JsValue::from_str(name), content)
                };
                match action {
                    super::BakeAction::Share { key, bytes } => {
                        set("share", &JsValue::from_str(&key))?;
                        set("bytes", &js_sys::Uint8Array::from(bytes.as_slice()))?;
                    }
                    super::BakeAction::Unshare { key } => {
                        set("unshare", &JsValue::from_str(&key))?;
                    }
                    super::BakeAction::Run { id, input } => {
                        if id > 9_007_199_254_740_991 {
                            return Err(JsValue::from_str("child ID is not exactly represented"));
                        }
                        set("id", &JsValue::from_f64(id as f64))?;
                        set("input", &js_sys::Uint8Array::from(input.as_slice()))?;
                    }
                }
                values.push(&value);
            }
            Ok(values)
        }

        pub fn complete(&mut self, id: f64, output: &[u8]) -> Result<(), JsValue> {
            self.graph
                .complete(integer(id)?, output)
                .map_err(|error| JsValue::from_str(&error))
        }

        pub fn take_result(&mut self) -> Option<Vec<u8>> {
            self.graph.take_result()
        }

        pub fn cancel(&mut self) {
            self.graph.cancel();
        }

        /// In-flight, completed, sent/received bytes, shared bytes, joined
        /// buffer capacity, peak retained bytes. Host counters add copies.
        pub fn diagnostics(&self) -> Vec<f64> {
            let value = self.graph.diagnostics();
            vec![
                value.in_flight as f64,
                value.completed as f64,
                value.sent_bytes as f64,
                value.received_bytes as f64,
                value.share_bytes as f64,
                value.buffer_bytes as f64,
                value.peak_bytes as f64,
            ]
        }
    }

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

        pub fn warm(&mut self) {
            self.worker.warm();
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
