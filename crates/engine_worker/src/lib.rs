//! The engine's job worker: a small Wasm module hosts run in
//! Web Workers beside the engine, and natively a thread's. It holds what
//! the engine shares with its workers (terrain stacks and recipes) and runs
//! the jobs it hands out, terrain tiles, walks and bakes, with the code
//! engines without workers run in place, so outputs are the same bits. Jobs
//! run a slice at a time ([`Worker::slice`]): the most urgent held job
//! steps until done or for about [`SLICE_MS`], so between slices hosts read
//! their messages, a more urgent job goes first and one the engine
//! cancelled stops; parts of an output ready early go back before the rest.
#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

use engine_terrain::jobs::TerrainWorker;

mod cache;

pub use cache::JobCache;

/// Milliseconds a worker runs a job before it looks at its messages and
/// more urgent jobs again.
pub const SLICE_MS: f64 = 8.0;
/// Work units a worker asks a job to do at a time, about one a sample.
pub const STEP_UNITS: u64 = 4096;

/// What a slice of a job came to.
#[derive(Debug, PartialEq, Eq)]
pub enum Report {
    /// Job `id` is done: its output.
    Done(u64, Vec<u8>),
    /// A part of job `id`'s output ready before the rest.
    Part(u64, Vec<u8>),
    /// Job `id` failed, for the reason given.
    Failed(u64, String),
}

/// A job a worker holds.
struct Held {
    id: u64,
    kind: String,
    input: Vec<u8>,
    priority: u32,
    started: bool,
}

/// One worker's shares and the jobs it holds and runs.
#[derive(Default)]
pub struct Worker {
    terrain: TerrainWorker,
    held: Vec<Held>,
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

    /// Drops everything shared, and every job held.
    pub fn clear(&mut self) {
        self.terrain = TerrainWorker::new();
        self.held.clear();
    }

    /// Holds job `id` of `kind` on `input` at `priority` (lower first),
    /// which [`Self::slice`] runs.
    pub fn hold(&mut self, id: u64, kind: &str, input: Vec<u8>, priority: u32) {
        self.held.push(Held {
            id,
            kind: kind.to_owned(),
            input,
            priority,
            started: false,
        });
    }

    /// Whether it holds a job to run.
    pub fn busy(&self) -> bool {
        !self.held.is_empty()
    }

    /// Stops job `id`, held or started.
    pub fn cancel(&mut self, id: u64) {
        self.held.retain(|job| job.id != id);
        self.terrain.cancel(id);
    }

    /// One slice of the most urgent job held (the lowest priority, then
    /// the first handed): stepped until done or for about [`SLICE_MS`].
    /// What it came to; nothing where no job is held, or where the job
    /// goes on with no part ready.
    pub fn slice(&mut self) -> Option<Report> {
        let at =
            (0..self.held.len()).min_by_key(|&at| (self.held[at].priority, self.held[at].id))?;
        let job = &mut self.held[at];
        let id = job.id;
        let stepped = (|| {
            if !job.started {
                self.terrain.start(id, &job.kind, &job.input)?;
                job.started = true;
            }
            let start = clock::now_ms();
            loop {
                if let Some(output) = self.terrain.step(id, STEP_UNITS)? {
                    return Ok(Some(output));
                }
                if clock::now_ms() - start >= SLICE_MS {
                    return Ok(None);
                }
            }
        })();
        match stepped {
            Ok(Some(output)) => {
                self.held.remove(at);
                Some(Report::Done(id, output))
            }
            Ok(None) => self
                .terrain
                .take_part(id)
                .map(|part| Report::Part(id, part)),
            Err(error) => {
                self.held.remove(at);
                self.terrain.cancel(id);
                Some(Report::Failed(id, error))
            }
        }
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
}

/// Milliseconds on a clock of the worker's own: the Web Worker's, or
/// natively since its first reading.
mod clock {
    #[cfg(target_arch = "wasm32")]
    #[wasm_bindgen::prelude::wasm_bindgen]
    extern "C" {
        #[wasm_bindgen(js_namespace = performance, js_name = now)]
        fn performance_now() -> f64;
    }

    #[cfg(target_arch = "wasm32")]
    pub fn now_ms() -> f64 {
        performance_now()
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn now_ms() -> f64 {
        static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
        START
            .get_or_init(std::time::Instant::now)
            .elapsed()
            .as_secs_f64()
            * 1000.0
    }
}

#[cfg(target_arch = "wasm32")]
mod browser {
    use wasm_bindgen::prelude::*;

    use crate::{JobCache, Report};

    /// [`super::Worker`] for a Web Worker's script. Job ids and budgets are
    /// JavaScript numbers.
    #[wasm_bindgen]
    pub struct JobWorker {
        worker: super::Worker,
    }

    #[wasm_bindgen]
    impl JobWorker {
        /// A worker whose panics stop it for good, as the engine's do
        /// ([`engine_panic`]'s hook).
        #[wasm_bindgen(constructor)]
        pub fn new() -> Self {
            engine_panic::install();
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

        pub fn hold(&mut self, id: f64, kind: &str, input: Vec<u8>, priority: u32) {
            self.worker.hold(id as u64, kind, input, priority);
        }

        pub fn busy(&self) -> bool {
            self.worker.busy()
        }

        /// One slice ([`super::Worker::slice`]) as the page's pool reads
        /// it: `{type: "done" | "part", id, output}` or `{type: "failed",
        /// id, error}`, or nothing.
        pub fn slice(&mut self) -> JsValue {
            let Some(report) = self.worker.slice() else {
                return JsValue::UNDEFINED;
            };
            let object = js_sys::Object::new();
            let set = |field: &str, value: &JsValue| {
                // Setting a field of a plain new object cannot fail.
                let _ = js_sys::Reflect::set(&object, &JsValue::from_str(field), value);
            };
            let (kind, id) = match &report {
                Report::Done(id, output) => {
                    set("output", &js_sys::Uint8Array::from(output.as_slice()));
                    ("done", id)
                }
                Report::Part(id, part) => {
                    set("output", &js_sys::Uint8Array::from(part.as_slice()));
                    ("part", id)
                }
                Report::Failed(id, error) => {
                    set("error", &JsValue::from_str(error));
                    ("failed", id)
                }
            };
            set("type", &JsValue::from_str(kind));
            set("id", &JsValue::from_f64(*id as f64));
            object.into()
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

    /// [`JobCache`] for the page, which keeps the bytes. Sizes are
    /// JavaScript numbers.
    #[wasm_bindgen]
    pub struct JobCacheIndex {
        cache: JobCache,
    }

    #[wasm_bindgen]
    impl JobCacheIndex {
        /// [`JobCache::open`].
        #[wasm_bindgen(constructor)]
        pub fn new(
            build: &[u8],
            budget: f64,
            available: Option<f64>,
            verify: bool,
            index: Option<String>,
        ) -> Self {
            Self {
                cache: JobCache::open(
                    build,
                    budget as u64,
                    available.map(|bytes| bytes as u64),
                    verify,
                    index.as_deref(),
                ),
            }
        }

        pub fn verify(&self) -> bool {
            self.cache.verify()
        }

        /// Hits, misses, bytes and mismatches.
        pub fn counters(&self) -> Vec<f64> {
            let (hits, misses, bytes, mismatches) = self.cache.counters();
            vec![hits as f64, misses as f64, bytes as f64, mismatches as f64]
        }

        pub fn key(&self, kind: &str, identity: &[u8]) -> String {
            self.cache.key(kind, identity)
        }

        pub fn lists(&self, name: &str) -> bool {
            self.cache.lists(name)
        }

        pub fn holds(&self, key: &str) -> bool {
            self.cache.holds(key)
        }

        /// [`JobCache::hit`]; `bytes` none where the read failed.
        pub fn hit(&mut self, key: &str, bytes: Option<f64>) -> bool {
            self.cache.hit(key, bytes.map(|bytes| bytes as u64))
        }

        pub fn miss(&mut self) {
            self.cache.miss();
        }

        pub fn store(&mut self, key: &str, bytes: f64) -> bool {
            self.cache.store(key, bytes as u64)
        }

        pub fn mismatch(&mut self, key: &str) {
            self.cache.mismatch(key);
        }

        pub fn failed(&mut self) {
            self.cache.failed();
        }

        pub fn take_removed(&mut self) -> Vec<String> {
            self.cache.take_removed()
        }

        pub fn index(&mut self) -> Option<String> {
            self.cache.index()
        }
    }
}
