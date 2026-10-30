#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

use std::sync::Arc;

use engine_navigation::NavigationWorker;
use engine_tables::TableWorker;
use engine_terrain::jobs::TerrainWorker;
use engine_textures::TextureWorker;

mod cache;
mod image;
mod pack;

pub use cache::JobCache;
pub use image::{IMAGE_FORMATS, Image, ImageFormat, decode_image, image_format};
pub use pack::{
    LISTING_KIND, PACK_HEADER, PackEntry, PackIndex, key_from_hex, listing_bytes, listing_key,
    pack_bytes, pack_directory, read_listing, unpack, unpack_bytes,
};

include!(concat!(env!("OUT_DIR"), "/source.rs"));

pub fn record_key(kind: &str, identity: &[u8]) -> String {
    engine_sources::record_key(&SOURCE, kind, identity)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

pub const SLICE_MS: f64 = 8.0;
pub const STEP_UNITS: u64 = 4096;

#[derive(Debug, PartialEq, Eq)]
pub enum Report {
    Done(u64, Vec<u8>),
    Failed(u64, String),
}

struct Held {
    id: u64,
    kind: String,
    input: Vec<u8>,
    priority: u32,
    started: bool,
}

#[derive(Clone, Copy)]
enum Runner {
    Tables,
    Textures,
    Navigation,
    Terrain,
}

impl Runner {
    fn of(kind: &str) -> Self {
        if TableWorker::runs(kind) {
            Self::Tables
        } else if TextureWorker::runs(kind) {
            Self::Textures
        } else if NavigationWorker::runs(kind) {
            Self::Navigation
        } else {
            Self::Terrain
        }
    }
}

pub struct Worker {
    terrain: TerrainWorker,
    navigation: NavigationWorker,
    tables: TableWorker,
    textures: TextureWorker,
    held: Vec<Held>,
}

pub fn register_standard_layers() {
    for kinds in [
        engine_terrain::expr::KINDS,
        engine_layer_noise::KINDS,
        engine_layer_stamps::KINDS,
        engine_layer_heightmap::KINDS,
        engine_layer_paths::KINDS,
        engine_layer_erosion::KINDS,
        engine_layer_hydrology::KINDS,
        engine_layer_relief::KINDS,
    ] {
        engine_terrain::layer::register_kinds(kinds);
    }
}

impl Default for Worker {
    fn default() -> Self {
        register_standard_layers();
        Self {
            terrain: TerrainWorker::new(),
            navigation: NavigationWorker::new(),
            tables: TableWorker::new(),
            textures: TextureWorker::new(),
            held: Vec::new(),
        }
    }
}

impl Worker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn share(&mut self, key: &str, bytes: impl Into<Arc<[u8]>>) -> Result<(), String> {
        self.terrain.share(key, bytes)
    }

    pub fn unshare(&mut self, key: &str) {
        self.terrain.unshare(key);
    }

    pub fn clear(&mut self) {
        self.terrain = TerrainWorker::new();
        self.navigation = NavigationWorker::new();
        self.tables = TableWorker::new();
        self.textures = TextureWorker::new();
        self.held.clear();
    }

    pub fn hold(&mut self, id: u64, kind: &str, input: Vec<u8>, priority: u32) {
        self.held.push(Held {
            id,
            kind: kind.to_owned(),
            input,
            priority,
            started: false,
        });
    }

    pub fn busy(&self) -> bool {
        !self.held.is_empty()
    }

    pub fn cancel(&mut self, id: u64) {
        self.held.retain(|job| job.id != id);
        self.terrain.cancel(id);
        self.navigation.cancel(id);
        self.tables.cancel(id);
        self.textures.cancel(id);
    }

    pub fn slice(&mut self) -> Option<Report> {
        let at =
            (0..self.held.len()).min_by_key(|&at| (self.held[at].priority, self.held[at].id))?;
        let job = &mut self.held[at];
        let id = job.id;
        #[cfg(not(target_arch = "wasm32"))]
        if job.kind == engine_lodbuild::KIND {
            let input = std::mem::take(&mut job.input);
            self.held.remove(at);
            return Some(match engine_lodbuild::run(&input) {
                Ok(output) => Report::Done(id, output),
                Err(error) => Report::Failed(id, error),
            });
        }
        let runner = Runner::of(&job.kind);
        let (terrain, navigation) = (&mut self.terrain, &mut self.navigation);
        let (tables, textures) = (&mut self.tables, &mut self.textures);
        let stepped = (|| {
            if !job.started {
                match runner {
                    Runner::Tables => tables.start(id, &job.kind, &job.input)?,
                    Runner::Textures => textures.start(id, &job.kind, &job.input)?,
                    Runner::Navigation => {
                        navigation.start(id, &job.kind, &job.input, &*terrain, &*terrain)?
                    }
                    Runner::Terrain => terrain.start(id, &job.kind, &job.input)?,
                }
                job.started = true;
            }
            let start = clock::now_ms();
            loop {
                let output = match runner {
                    Runner::Tables => tables.step(id, STEP_UNITS)?,
                    Runner::Textures => textures.step(id, STEP_UNITS)?,
                    Runner::Navigation => navigation.step(id, STEP_UNITS)?,
                    Runner::Terrain => terrain.step(id, STEP_UNITS)?,
                };
                if let Some(output) = output {
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
            Ok(None) => None,
            Err(error) => {
                self.held.remove(at);
                self.terrain.cancel(id);
                self.navigation.cancel(id);
                self.tables.cancel(id);
                self.textures.cancel(id);
                Some(Report::Failed(id, error))
            }
        }
    }

    pub fn run(&mut self, kind: &str, input: &[u8]) -> Result<Vec<u8>, String> {
        #[cfg(not(target_arch = "wasm32"))]
        if kind == engine_lodbuild::KIND {
            return engine_lodbuild::run(input);
        }
        match engine_tables::run(kind, input).or_else(|| engine_textures::run(kind, input)) {
            Some(output) => output,
            None if NavigationWorker::runs(kind) => {
                self.navigation
                    .run(kind, input, &self.terrain, &self.terrain)
            }
            None => self.terrain.run(kind, input),
        }
    }

    pub fn start(&mut self, id: u64, kind: &str, input: &[u8]) -> Result<(), String> {
        match Runner::of(kind) {
            Runner::Tables => self.tables.start(id, kind, input),
            Runner::Textures => self.textures.start(id, kind, input),
            Runner::Navigation => {
                self.navigation
                    .start(id, kind, input, &self.terrain, &self.terrain)
            }
            Runner::Terrain => self.terrain.start(id, kind, input),
        }
    }

    pub fn step(&mut self, id: u64, budget: u64) -> Result<Option<Vec<u8>>, String> {
        if self.tables.holds(id) {
            self.tables.step(id, budget)
        } else if self.textures.holds(id) {
            self.textures.step(id, budget)
        } else if self.navigation.holds(id) {
            self.navigation.step(id, budget)
        } else {
            self.terrain.step(id, budget)
        }
    }
}

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

    #[wasm_bindgen]
    pub struct JobWorker {
        worker: super::Worker,
    }

    #[wasm_bindgen]
    impl JobWorker {
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

        pub fn slice(&mut self) -> JsValue {
            let Some(report) = self.worker.slice() else {
                return JsValue::UNDEFINED;
            };
            let object = js_sys::Object::new();
            let set = |field: &str, value: &JsValue| {
                let _ = js_sys::Reflect::set(&object, &JsValue::from_str(field), value);
            };
            let (kind, id) = match &report {
                Report::Done(id, output) => {
                    set("output", &js_sys::Uint8Array::from(output.as_slice()));
                    ("done", id)
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

        pub fn cancel(&mut self, id: f64) {
            self.worker.cancel(id as u64);
        }

        pub fn unpack(&self, stored: &[u8], raw: u32) -> Result<Vec<u8>, JsValue> {
            Ok(crate::unpack_bytes(stored, raw)?)
        }

        pub fn decode_image(&self, uri: &str, bytes: &[u8]) -> Result<JsValue, JsValue> {
            let image = crate::decode_image(uri, bytes)?;
            let object = js_sys::Object::new();
            let set = |field: &str, value: &JsValue| {
                let _ = js_sys::Reflect::set(&object, &JsValue::from_str(field), value);
            };
            set("width", &JsValue::from(image.width));
            set("height", &JsValue::from(image.height));
            set("pixels", &js_sys::Uint8Array::from(image.pixels.as_slice()));
            Ok(object.into())
        }
    }

    impl Default for JobWorker {
        fn default() -> Self {
            Self::new()
        }
    }

    #[wasm_bindgen]
    pub struct PackDirectory {
        index: crate::PackIndex,
    }

    #[wasm_bindgen]
    impl PackDirectory {
        #[wasm_bindgen(constructor)]
        pub fn new(header: &[u8], directory: &[u8]) -> Result<PackDirectory, JsValue> {
            Ok(crate::PackIndex::open(header, directory).map(|index| Self { index })?)
        }

        pub fn locate(header: &[u8]) -> Result<Vec<f64>, JsValue> {
            let (offset, stored, raw) = crate::pack_directory(header)?;
            Ok(vec![offset as f64, stored as f64, raw as f64])
        }

        pub fn find(&self, key: &str) -> Option<Vec<f64>> {
            self.index.find(key).map(|entry| {
                vec![
                    entry.offset as f64,
                    f64::from(entry.stored),
                    f64::from(entry.raw),
                ]
            })
        }

        pub fn unpack(&self, key: &str, stored: &[u8]) -> Result<Vec<u8>, JsValue> {
            let entry = self.index.find(key).ok_or_else(|| {
                engine_settings::Error::new(engine_settings::ErrorCode::InvalidPack, "$.key")
                    .expected(engine_settings::Expected::Present)
                    .got(key)
            })?;
            Ok(crate::unpack(&entry, stored)?)
        }

        pub fn records(&self) -> u32 {
            self.index.len() as u32
        }
    }

    #[wasm_bindgen]
    pub struct JobCacheIndex {
        cache: JobCache,
    }

    #[wasm_bindgen]
    impl JobCacheIndex {
        #[wasm_bindgen(constructor)]
        pub fn new(
            budget: f64,
            available: Option<f64>,
            verify: bool,
            index: Option<String>,
        ) -> Self {
            Self {
                cache: JobCache::open(
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

        pub fn counters(&self) -> Vec<f64> {
            let (hits, misses, bytes, mismatches) = self.cache.counters();
            vec![hits as f64, misses as f64, bytes as f64, mismatches as f64]
        }

        pub fn key(&self, kind: &str, identity: &[u8]) -> String {
            crate::record_key(kind, identity)
        }

        pub fn lists(&self, name: &str) -> bool {
            self.cache.lists(name)
        }

        pub fn holds(&self, key: &str) -> bool {
            self.cache.holds(key)
        }

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
