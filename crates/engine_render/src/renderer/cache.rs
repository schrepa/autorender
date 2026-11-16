#![expect(
    unsafe_code,
    reason = "opens the driver's pipeline cache from a file it wrote"
)]

use std::path::{Path, PathBuf};
use std::sync::mpsc::{SyncSender, TrySendError};
use std::thread::JoinHandle;

pub(crate) struct KeptCache {
    pub(crate) cache: wgpu::PipelineCache,
    file: PathBuf,
    asks: Option<SyncSender<()>>,
    saver: Option<JoinHandle<()>>,
}

impl Drop for KeptCache {
    fn drop(&mut self) {
        self.asks = None;
        if let Some(saver) = self.saver.take() {
            let _ = saver.join();
        }
    }
}

pub(super) fn open(device: &wgpu::Device, adapter: &wgpu::AdapterInfo) -> Option<KeptCache> {
    if !device.features().contains(wgpu::Features::PIPELINE_CACHE) {
        return None;
    }
    let file =
        crate::platform::pipeline_cache_folder()?.join(wgpu::util::pipeline_cache_key(adapter)?);
    let data = std::fs::read(&file).ok();
    // SAFETY: the file holds only bytes `write` took from `PipelineCache::get_data`
    // for this adapter's key; wgpu checks the header against the adapter and
    // driver, and `fallback` starts an empty cache when the data is refused.
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
            let (cache, file) = (cache.clone(), file.clone());
            move || {
                while asked.recv().is_ok() {
                    write(&cache, &file);
                }
            }
        })
        .ok();
    Some(KeptCache {
        cache,
        file,
        asks: saver.is_some().then_some(asks),
        saver,
    })
}

pub(super) fn save(kept: &KeptCache) {
    match kept.asks.as_ref().map(|asks| asks.try_send(())) {
        Some(Ok(()) | Err(TrySendError::Full(()))) => {}
        Some(Err(TrySendError::Disconnected(()))) | None => write(&kept.cache, &kept.file),
    }
}

fn write(cache: &wgpu::PipelineCache, file: &Path) {
    let Some(data) = cache.get_data() else {
        return;
    };
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
