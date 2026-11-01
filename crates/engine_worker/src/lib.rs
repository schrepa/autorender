#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

use std::any::TypeId;
use std::collections::BTreeMap;
use std::sync::Arc;

use engine_core::{Brick, Job, JobCx, JobKind, Listing, PlaceDecl};
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

#[derive(Default)]
pub struct JobListing {
    bricks: Vec<(TypeId, &'static [&'static JobKind])>,
}

impl Listing for JobListing {
    fn with<B: Brick>(mut self) -> Self {
        let brick = TypeId::of::<B>();
        if !self.bricks.iter().any(|(listed, _)| *listed == brick) {
            self.bricks.push((brick, B::JOBS));
        }
        self
    }

    fn without<B: Brick>(mut self) -> Self {
        let brick = TypeId::of::<B>();
        self.bricks.retain(|(listed, _)| *listed != brick);
        self
    }

    fn places(self, _: &'static [PlaceDecl]) -> Self {
        self
    }
}

impl JobListing {
    pub fn list(list: impl FnOnce(Self) -> Self) -> Result<JobList, Error> {
        let listed = list(Self::default());
        let kinds: Vec<&'static JobKind> = listed
            .bricks
            .iter()
            .flat_map(|(_, jobs)| jobs.iter().copied())
            .collect();
        engine_core::job_list(&kinds)
    }
}

pub fn share_key(hex: &str) -> Result<[u8; 32], Error> {
    key_from_hex(hex).ok_or_else(|| {
        Error::new(ErrorCode::InvalidId, "$.key")
            .expected(Expected::Id)
            .got(hex)
    })
}

#[derive(Debug, PartialEq)]
pub enum Report {
    Done(u64, Vec<u8>),
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
    kind: Box<str>,
    input: Vec<u8>,
    priority: u32,
    job: Option<Box<dyn Job>>,
}

pub struct Worker {
    list: Arc<JobList>,
    clock: fn() -> f64,
    shares: Shared,
    held: Vec<Held>,
    stepped: Vec<(u64, Box<dyn Job>)>,
}

impl Worker {
    pub fn new(list: Arc<JobList>, clock: fn() -> f64) -> Self {
        Self {
            list,
            clock,
            shares: Shared::default(),
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

    pub fn share(&mut self, key: [u8; 32], bytes: impl Into<Arc<[u8]>>) {
        self.shares.bytes.insert(key, bytes.into());
    }

    pub fn unshare(&mut self, key: [u8; 32]) {
        self.shares.bytes.remove(&key);
    }

    pub fn clear(&mut self) {
        self.shares.bytes.clear();
        self.held.clear();
        self.stepped.clear();
    }

    pub fn hold(&mut self, id: u64, kind: &str, input: Vec<u8>, priority: u32) {
        self.held.push(Held {
            id,
            kind: kind.into(),
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
            let mut cx = JobCx::new(&self.list, &self.shares);
            sliced(&mut self.held[at], &self.list, &mut cx, self.clock)
        };
        match stepped {
            Ok(Some(output)) => {
                self.held.remove(at);
                Some(Report::Done(id, output))
            }
            Ok(None) => None,
            Err(error) => {
                self.held.remove(at);
                Some(Report::Failed(id, error))
            }
        }
    }

    pub fn run(&self, kind: &str, input: &[u8]) -> Result<Vec<u8>, Error> {
        let mut cx = JobCx::new(&self.list, &self.shares);
        let mut job = begin(&self.list, kind, input, &mut cx)?;
        loop {
            if let Some(output) = job.step(u64::MAX, &mut cx)? {
                return Ok(output);
            }
        }
    }

    pub fn start(&mut self, id: u64, kind: &str, input: &[u8]) -> Result<(), Error> {
        let job = {
            let mut cx = JobCx::new(&self.list, &self.shares);
            begin(&self.list, kind, input, &mut cx)?
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
            let mut cx = JobCx::new(&self.list, &self.shares);
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

fn begin(
    list: &JobList,
    kind: &str,
    input: &[u8],
    cx: &mut JobCx<'_>,
) -> Result<Box<dyn Job>, Error> {
    let listed = list
        .named(kind)
        .ok_or_else(|| Error::new(ErrorCode::UnknownValue, "$.kind").got(kind))?;
    cx.start(&listed, input)
}

fn sliced(
    held: &mut Held,
    list: &JobList,
    cx: &mut JobCx<'_>,
    clock: fn() -> f64,
) -> Result<Option<Vec<u8>>, Error> {
    let mut job = match held.job.take() {
        Some(job) => job,
        None => begin(list, &held.kind, &held.input, cx)?,
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
    #[cfg(target_arch = "wasm32")]
    pub use engine_panic::install;
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
        const _: fn($crate::JobListing) -> $crate::JobListing = |list| $list(list);
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
                let list = $crate::JobListing::list(|list| $list(list))?;
                let worker = $crate::Worker::new(::std::sync::Arc::new(list), performance_now);
                worker.handshake(hash)?;
                Ok(Self { worker })
            }

            pub fn share(
                &mut self,
                key: &str,
                bytes: &[u8],
            ) -> Result<(), $crate::__private::JsValue> {
                self.worker.share($crate::share_key(key)?, bytes);
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

            pub fn hold(&mut self, id: f64, kind: &str, input: Vec<u8>, priority: u32) {
                self.worker.hold(id as u64, kind, input, priority);
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

            pub fn unpack(
                &self,
                stored: &[u8],
                raw: u32,
            ) -> Result<Vec<u8>, $crate::__private::JsValue> {
                Ok($crate::unpack_bytes(stored, raw)?)
            }

            pub fn decode_image(
                &self,
                uri: &str,
                bytes: &[u8],
            ) -> Result<$crate::__private::JsValue, $crate::__private::JsValue> {
                let image = $crate::decode_image(uri, bytes)?;
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
                let index = $crate::PackIndex::open(header, directory)?;
                Ok(Self { index })
            }

            pub fn locate(header: &[u8]) -> Result<Vec<f64>, $crate::__private::JsValue> {
                let (offset, stored, raw) = $crate::pack_directory(header)?;
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
                    .ok_or_else(|| $crate::__pack_missing(key))?;
                Ok($crate::unpack(&entry, stored)?)
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
