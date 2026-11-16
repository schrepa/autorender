#![expect(
    unsafe_code,
    reason = "opens the driver's pipeline cache from a file it wrote"
)]

use std::hash::{DefaultHasher, Hasher};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{SyncSender, TrySendError};
use std::thread::JoinHandle;

const SEAL_BYTES: usize = size_of::<u64>();

pub(crate) struct KeptCache {
    pub(crate) cache: wgpu::PipelineCache,
    place: Place,
    asks: Option<SyncSender<()>>,
    saver: Option<JoinHandle<()>>,
}

#[derive(Clone)]
struct Place {
    file: PathBuf,
    key: String,
    most: u64,
}

impl Drop for KeptCache {
    fn drop(&mut self) {
        self.asks = None;
        if let Some(saver) = self.saver.take() {
            let _ = saver.join();
        }
    }
}

fn seal(key: &str, data: &[u8]) -> u64 {
    let mut hasher = DefaultHasher::new();
    hasher.write(key.as_bytes());
    hasher.write(data);
    hasher.finish()
}

fn unsealed(place: &Place, mut bytes: Vec<u8>) -> Option<Vec<u8>> {
    if bytes.len() as u64 > place.most {
        return None;
    }
    let at = bytes.len().checked_sub(SEAL_BYTES)?;
    let sealed = u64::from_le_bytes(bytes[at..].try_into().ok()?);
    bytes.truncate(at);
    (seal(&place.key, &bytes) == sealed).then_some(bytes)
}

pub(super) fn open(device: &wgpu::Device, adapter: &wgpu::AdapterInfo) -> Option<KeptCache> {
    if !device.features().contains(wgpu::Features::PIPELINE_CACHE) {
        return None;
    }
    let (folder, most) = crate::platform::pipeline_cache_folder()?;
    let key = wgpu::util::pipeline_cache_key(adapter)?;
    let place = Place {
        file: folder.join(&key),
        key,
        most,
    };
    let data = std::fs::read(&place.file)
        .ok()
        .and_then(|bytes| unsealed(&place, bytes));
    // SAFETY: the data passed its seal, so it holds exactly the bytes `write` took
    // from `PipelineCache::get_data` for this adapter's key; wgpu checks the header
    // against the adapter and driver, and `fallback` starts an empty cache when the
    // data is refused.
    let cache = unsafe {
        device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
            label: Some("engine/pipeline-cache"),
            data: data.as_deref(),
            fallback: true,
        })
    };
    let (asks, asked) = std::sync::mpsc::sync_channel(1);
    let saver = std::thread::Builder::new()
        .name("engine_pipeline_cache_save".to_owned())
        .spawn({
            let (cache, place) = (cache.clone(), place.clone());
            move || {
                while asked.recv().is_ok() {
                    write(&cache, &place);
                }
            }
        })
        .ok();
    Some(KeptCache {
        cache,
        place,
        asks: saver.is_some().then_some(asks),
        saver,
    })
}

pub(super) fn save(kept: &KeptCache) {
    match kept.asks.as_ref().map(|asks| asks.try_send(())) {
        Some(Ok(()) | Err(TrySendError::Full(()))) => {}
        Some(Err(TrySendError::Disconnected(()))) | None => write(&kept.cache, &kept.place),
    }
}

fn write(cache: &wgpu::PipelineCache, place: &Place) {
    let Some(mut data) = cache.get_data() else {
        return;
    };
    if (data.len() + SEAL_BYTES) as u64 > place.most {
        return;
    }
    let sealed = seal(&place.key, &data);
    data.extend_from_slice(&sealed.to_le_bytes());
    let file: &Path = &place.file;
    let partial = file.with_extension("partial");
    let saved = file
        .parent()
        .map_or(Ok(()), std::fs::create_dir_all)
        .and_then(|()| std::fs::write(&partial, &data))
        .and_then(|()| std::fs::rename(&partial, file));
    if saved.is_err() {
        let _ = std::fs::remove_file(&partial);
    }
}
