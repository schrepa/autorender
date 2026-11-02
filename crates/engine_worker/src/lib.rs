#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

use std::collections::BTreeMap;
use std::sync::Arc;

use engine_core::{BrickList, Job, JobCx, JobMemo, JobRef, RecordKey};
use engine_registry::{JobList, Shares};
use engine_settings::{Error, ErrorCode, Expected};

#[cfg(target_arch = "wasm32")]
mod browser;
mod cache;
mod image;
mod pack;

pub use cache::JobCache;
pub use image::{IMAGE_FORMATS, Image, ImageFormat, decode_image, image_format};
pub use pack::{
    LISTING_KIND, PACK_HEADER, PackEntry, PackIndex, key_from_hex, listing_bytes, listing_key,
    pack_bytes, pack_directory, read_listing, unpack, unpack_bytes,
};

pub const SLICE_MS: f64 = 8.0;
pub const STEP_UNITS: u64 = 4096;

pub fn job_list(list: impl FnOnce(BrickList) -> BrickList) -> Result<JobList, Error> {
    list(BrickList::default()).job_list()
}

fn hex_key(hex: &str, path: &str) -> Result<[u8; 32], Error> {
    key_from_hex(hex).ok_or_else(|| {
        Error::new(ErrorCode::InvalidId, path)
            .expected(Expected::Id)
            .got(hex)
    })
}

pub fn share_key(hex: &str) -> Result<RecordKey, Error> {
    hex_key(hex, "$.key").map(RecordKey::from_hash)
}

pub fn job_ref(name: &str, source: &str) -> Result<JobRef, Error> {
    Ok(JobRef::named(name, hex_key(source, "$.source")?))
}

#[derive(Debug, PartialEq)]
pub enum Report {
    Done(u64, Arc<[u8]>),
    Failed(u64, Error),
}

#[derive(Default)]
struct Shared {
    bytes: BTreeMap<[u8; 32], Arc<[u8]>>,
}

impl Shares for Shared {
    fn share(&self, key: [u8; 32]) -> Option<&[u8]> {
        self.bytes.get(&key).map(|bytes| &**bytes)
    }
}

struct Held {
    id: u64,
    kind: JobRef,
    input: Vec<u8>,
    priority: u32,
    job: Option<Box<dyn Job>>,
}

pub struct Worker {
    list: Arc<JobList>,
    clock: fn() -> f64,
    shares: Shared,
    memo: JobMemo,
    held: Vec<Held>,
    stepped: Vec<(u64, Box<dyn Job>)>,
}

impl Worker {
    pub fn new(list: Arc<JobList>, clock: fn() -> f64) -> Self {
        Self {
            list,
            clock,
            shares: Shared::default(),
            memo: JobMemo::default(),
            held: Vec::new(),
            stepped: Vec::new(),
        }
    }

    pub fn hash(&self) -> [u8; 32] {
        self.list.hash()
    }

    pub fn handshake(&self, hash: &[u8]) -> Result<(), Error> {
        if hash == self.list.hash().as_slice() {
            return Ok(());
        }
        Err(Error::new(ErrorCode::BuildMismatch, "$.hash")
            .expected(Expected::Matching)
            .got(pack::hex(hash)))
    }

    pub fn share(&mut self, key: RecordKey, bytes: impl Into<Arc<[u8]>>) {
        self.shares.bytes.insert(key.hash(), bytes.into());
    }

    pub fn unshare(&mut self, key: RecordKey) {
        self.shares.bytes.remove(&key.hash());
    }

    pub fn clear(&mut self) {
        self.shares.bytes.clear();
        self.memo = JobMemo::default();
        self.held.clear();
        self.stepped.clear();
    }

    pub fn hold(&mut self, id: u64, kind: JobRef, input: Vec<u8>, priority: u32) {
        self.held.push(Held {
            id,
            kind,
            input,
            priority,
            job: None,
        });
    }

    pub fn busy(&self) -> bool {
        !self.held.is_empty()
    }

    pub fn cancel(&mut self, id: u64) {
        self.held.retain(|held| held.id != id);
        self.stepped.retain(|(stepped, _)| *stepped != id);
    }

    pub fn slice(&mut self) -> Option<Report> {
        let at =
            (0..self.held.len()).min_by_key(|&at| (self.held[at].priority, self.held[at].id))?;
        let id = self.held[at].id;
        let stepped = {
            let mut cx = JobCx::new(&self.list, &self.shares, &mut self.memo);
            sliced(&mut self.held[at], &mut cx, self.clock)
        };
        match stepped {
            Ok(Some(output)) => {
                self.held.remove(at);
                Some(Report::Done(id, Arc::from(output)))
            }
            Ok(None) => None,
            Err(error) => {
                self.held.remove(at);
                Some(Report::Failed(id, error))
            }
        }
    }

    pub fn run(&mut self, kind: &JobRef, input: &[u8]) -> Result<Vec<u8>, Error> {
        let mut cx = JobCx::new(&self.list, &self.shares, &mut self.memo);
        let mut job = cx.start(kind, input)?;
        loop {
            if let Some(output) = job.step(u64::MAX, &mut cx)? {
                return Ok(output);
            }
        }
    }

    pub fn start(&mut self, id: u64, kind: &JobRef, input: &[u8]) -> Result<(), Error> {
        let job = {
            let mut cx = JobCx::new(&self.list, &self.shares, &mut self.memo);
            cx.start(kind, input)?
        };
        self.stepped.push((id, job));
        Ok(())
    }

    pub fn step(&mut self, id: u64, budget: u64) -> Result<Option<Vec<u8>>, Error> {
        let at = self
            .stepped
            .iter()
            .position(|(stepped, _)| *stepped == id)
            .ok_or_else(|| Error::new(ErrorCode::UnknownValue, "$.id").got(id))?;
        let stepped = {
            let mut cx = JobCx::new(&self.list, &self.shares, &mut self.memo);
            self.stepped[at].1.step(budget.max(1), &mut cx)
        };
        match stepped {
            Ok(None) => Ok(None),
            done @ (Ok(Some(_)) | Err(_)) => {
                self.stepped.remove(at);
                done
            }
        }
    }
}

fn sliced(
    held: &mut Held,
    cx: &mut JobCx<'_>,
    clock: fn() -> f64,
) -> Result<Option<Vec<u8>>, Error> {
    let mut job = match held.job.take() {
        Some(job) => job,
        None => cx.start(&held.kind, &held.input)?,
    };
    let start = clock();
    loop {
        if let Some(output) = job.step(STEP_UNITS, cx)? {
            return Ok(Some(output));
        }
        if clock() - start >= SLICE_MS {
            held.job = Some(job);
            return Ok(None);
        }
    }
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

#[doc(hidden)]
pub mod __private {
    #[cfg(target_arch = "wasm32")]
    pub use crate::browser::{image, report};
    pub use engine_core::BrickList;
    #[cfg(target_arch = "wasm32")]
    pub use engine_panic::install;
    #[cfg(target_arch = "wasm32")]
    pub use engine_settings::js_error;
    #[cfg(target_arch = "wasm32")]
    pub use js_sys::wasm_bindgen::{self, JsValue, prelude::wasm_bindgen as bindgen};
}

#[macro_export]
macro_rules! export_worker {
    ($list:path) => {
        $crate::__export_worker!($list);
    };
}

#[cfg(not(target_arch = "wasm32"))]
#[doc(hidden)]
#[macro_export]
macro_rules! __export_worker {
    ($list:path) => {
        const _: fn($crate::__private::BrickList) -> $crate::__private::BrickList =
            |list| $list(list);
    };
}

#[cfg(target_arch = "wasm32")]
#[doc(hidden)]
#[macro_export]
macro_rules! __export_worker {
    ($list:path) => {
        #[$crate::__private::bindgen(wasm_bindgen = $crate::__private::wasm_bindgen)]
        extern "C" {
            #[wasm_bindgen(js_namespace = performance, js_name = now)]
            fn performance_now() -> f64;
        }

        #[$crate::__private::bindgen(wasm_bindgen = $crate::__private::wasm_bindgen)]
        pub struct JobWorker {
            worker: $crate::Worker,
        }

        #[$crate::__private::bindgen(wasm_bindgen = $crate::__private::wasm_bindgen)]
        impl JobWorker {
            #[wasm_bindgen(constructor)]
            pub fn new(hash: &[u8]) -> Result<JobWorker, $crate::__private::JsValue> {
                $crate::__private::install();
                let list =
                    $crate::job_list(|list| $list(list)).map_err($crate::__private::js_error)?;
                let worker = $crate::Worker::new(::std::sync::Arc::new(list), performance_now);
                worker
                    .handshake(hash)
                    .map_err($crate::__private::js_error)?;
                Ok(Self { worker })
            }

            pub fn start() {
                $crate::__private::install();
            }

            pub fn module() -> $crate::__private::JsValue {
                $crate::__private::wasm_bindgen::module()
            }

            pub fn share(
                &mut self,
                key: &str,
                bytes: &[u8],
            ) -> Result<(), $crate::__private::JsValue> {
                let key = $crate::share_key(key).map_err($crate::__private::js_error)?;
                self.worker.share(key, bytes);
                Ok(())
            }

            pub fn unshare(&mut self, key: &str) {
                if let Ok(key) = $crate::share_key(key) {
                    self.worker.unshare(key);
                }
            }

            pub fn clear(&mut self) {
                self.worker.clear();
            }

            pub fn hold(
                &mut self,
                id: f64,
                kind: &str,
                source: &str,
                input: Vec<u8>,
                priority: u32,
            ) -> Result<(), $crate::__private::JsValue> {
                let kind = $crate::job_ref(kind, source).map_err($crate::__private::js_error)?;
                self.worker.hold(id as u64, kind, input, priority);
                Ok(())
            }

            pub fn busy(&self) -> bool {
                self.worker.busy()
            }

            pub fn slice(&mut self) -> $crate::__private::JsValue {
                $crate::__private::report(self.worker.slice())
            }

            pub fn cancel(&mut self, id: f64) {
                self.worker.cancel(id as u64);
            }

            pub fn unpack(stored: &[u8], raw: u32) -> Result<Vec<u8>, $crate::__private::JsValue> {
                $crate::unpack_bytes(stored, raw).map_err($crate::__private::js_error)
            }

            pub fn decode_image(
                uri: &str,
                bytes: &[u8],
            ) -> Result<$crate::__private::JsValue, $crate::__private::JsValue> {
                let image =
                    $crate::decode_image(uri, bytes).map_err($crate::__private::js_error)?;
                Ok($crate::__private::image(&image))
            }
        }

        #[$crate::__private::bindgen(wasm_bindgen = $crate::__private::wasm_bindgen)]
        pub struct PackDirectory {
            index: $crate::PackIndex,
        }

        #[$crate::__private::bindgen(wasm_bindgen = $crate::__private::wasm_bindgen)]
        impl PackDirectory {
            #[wasm_bindgen(constructor)]
            pub fn new(
                header: &[u8],
                directory: &[u8],
            ) -> Result<PackDirectory, $crate::__private::JsValue> {
                let index = $crate::PackIndex::open(header, directory)
                    .map_err($crate::__private::js_error)?;
                Ok(Self { index })
            }

            pub fn locate(header: &[u8]) -> Result<Vec<f64>, $crate::__private::JsValue> {
                let (offset, stored, raw) =
                    $crate::pack_directory(header).map_err($crate::__private::js_error)?;
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

            pub fn unpack(
                &self,
                key: &str,
                stored: &[u8],
            ) -> Result<Vec<u8>, $crate::__private::JsValue> {
                let entry = self
                    .index
                    .find(key)
                    .ok_or_else(|| $crate::__private::js_error($crate::__pack_missing(key)))?;
                $crate::unpack(&entry, stored).map_err($crate::__private::js_error)
            }

            pub fn records(&self) -> u32 {
                self.index.len() as u32
            }
        }

        #[$crate::__private::bindgen(wasm_bindgen = $crate::__private::wasm_bindgen)]
        pub struct JobCacheIndex {
            cache: $crate::JobCache,
        }

        #[$crate::__private::bindgen(wasm_bindgen = $crate::__private::wasm_bindgen)]
        impl JobCacheIndex {
            #[wasm_bindgen(constructor)]
            pub fn new(
                budget: f64,
                available: Option<f64>,
                verify: bool,
                index: Option<String>,
            ) -> Self {
                Self {
                    cache: $crate::JobCache::open(
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
    };
}

#[doc(hidden)]
pub fn __pack_missing(key: &str) -> Error {
    Error::new(ErrorCode::InvalidPack, "$.key")
        .expected(Expected::Present)
        .got(key)
}
