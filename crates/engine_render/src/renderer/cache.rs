#![expect(unsafe_code, reason = "opens the driver's pipeline cache from a file it wrote")]

use std::path::PathBuf;

pub(crate) struct KeptCache {
    pub(crate) cache: wgpu::PipelineCache,
    file: PathBuf,
}

pub(super) fn open(device: &wgpu::Device, adapter: &wgpu::AdapterInfo) -> Option<KeptCache> {
    if !device.features().contains(wgpu::Features::PIPELINE_CACHE) {
        return None;
    }
    let file = crate::platform::pipeline_cache_folder()?
        .join(wgpu::util::pipeline_cache_key(adapter)?);
    let data = std::fs::read(&file).ok();
    // SAFETY: the file holds only bytes `save` took from `PipelineCache::get_data`
    // for this adapter's key; wgpu checks the header against the adapter and
    // driver, and `fallback` starts an empty cache when the data is refused.
    let cache = unsafe {
        device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
            label: Some("engine/pipeline-cache"),
            data: data.as_deref(),
            fallback: true,
        })
    };
    Some(KeptCache { cache, file })
}

pub(super) fn save(kept: &KeptCache) {
    let Some(data) = kept.cache.get_data() else {
        return;
    };
    let partial = kept.file.with_extension("partial");
    let saved = kept
        .file
        .parent()
        .map_or(Ok(()), std::fs::create_dir_all)
        .and_then(|()| std::fs::write(&partial, &data))
        .and_then(|()| std::fs::rename(&partial, &kept.file));
    if saved.is_err() {
        let _ = std::fs::remove_file(&partial);
    }
}
