use super::*;

pub(super) fn device_limits(adapter: &wgpu::Limits) -> wgpu::Limits {
    let required = crate::pipelines::REQUIRED_LIMITS;
    wgpu::Limits {
        max_buffer_size: adapter.max_buffer_size,
        max_storage_buffer_binding_size: adapter.max_storage_buffer_binding_size,
        max_texture_dimension_2d: adapter.max_texture_dimension_2d,
        max_texture_dimension_3d: adapter.max_texture_dimension_3d,
        max_storage_buffers_per_shader_stage: adapter
            .max_storage_buffers_per_shader_stage
            .max(required.storage_buffers),
        max_bind_groups: required.bind_groups,
        max_uniform_buffers_per_shader_stage: required.uniform_buffers,
        max_sampled_textures_per_shader_stage: required.sampled_textures,
        max_samplers_per_shader_stage: required.samplers,
        max_storage_textures_per_shader_stage: required.storage_textures,
        max_dynamic_uniform_buffers_per_pipeline_layout: required.dynamic_uniform_buffers,
        max_dynamic_storage_buffers_per_pipeline_layout: required.dynamic_storage_buffers,
        max_uniform_buffer_binding_size: engine_place::UNIFORM_BINDING_BYTES,
        max_texture_array_layers: adapter.max_texture_array_layers,
        max_color_attachments: adapter.max_color_attachments,
        max_color_attachment_bytes_per_sample: adapter.max_color_attachment_bytes_per_sample,
        ..wgpu::Limits::downlevel_defaults()
    }
}

pub(crate) struct Opened {
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    features: wgpu::Features,
}

pub(super) type Opening =
    std::pin::Pin<Box<dyn std::future::Future<Output = Result<Opened, Error>>>>;

fn features_of(adapter: &wgpu::Adapter) -> wgpu::Features {
    let compression = adapter.features()
        & (wgpu::Features::TEXTURE_COMPRESSION_BC
            | wgpu::Features::TEXTURE_COMPRESSION_ETC2
            | wgpu::Features::TEXTURE_COMPRESSION_ASTC);
    let core = wgpu::Features::INDIRECT_FIRST_INSTANCE | wgpu::Features::TIMESTAMP_QUERY;
    let features = compression | (adapter.features() & core);
    #[cfg(not(target_arch = "wasm32"))]
    let features = features | (adapter.features() & wgpu::Features::PIPELINE_CACHE);
    #[cfg(not(feature = "profiling"))]
    let required = features;
    #[cfg(feature = "profiling")]
    let required = features | (adapter.features() & crate::profiling::FEATURES);
    required
}

async fn opened(adapter: wgpu::Adapter) -> Result<Opened, Error> {
    let features = features_of(&adapter);
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("engine/device"),
            required_features: features,
            required_limits: device_limits(&adapter.limits()),
            ..Default::default()
        })
        .await
        .map_err(|error| failed_with("$.device", error))?;
    Ok(Opened {
        adapter,
        device,
        queue,
        features,
    })
}

pub(super) async fn open(
    instance: wgpu::Instance,
    power_preference: wgpu::PowerPreference,
) -> Result<Opened, Error> {
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference,
            compatible_surface: None,
            force_fallback_adapter: false,
            ..Default::default()
        })
        .await
        .map_err(|error| failed_with("$.adapter", error))?;
    opened(adapter).await
}

fn watch(device: &wgpu::Device, health: &Arc<Mutex<health::Reports>>) {
    let lost = Arc::clone(health);
    device.set_device_lost_callback(move |reason, message| {
        if let Ok(mut reports) = lost.lock() {
            reports.lost(reason, &message);
        }
    });
    let uncaptured = Arc::clone(health);
    device.on_uncaptured_error(Arc::new(move |error: wgpu::Error| {
        if let Ok(mut reports) = uncaptured.lock() {
            reports.uncaptured(&error);
        }
    }));
}

pub struct StartingRenderer {
    renderer: Renderer,
    validation: std::pin::Pin<Box<dyn std::future::Future<Output = Option<wgpu::Error>>>>,
}

impl StartingRenderer {
    pub fn finish(self) -> Renderer {
        let mut renderer = self.renderer;
        super::validate_also(&mut renderer.installing, self.validation);
        renderer
    }
}

impl Renderer {
    pub(super) fn adopt_device(&mut self, opened: Opened) -> Result<(), Error> {
        let Opened {
            adapter,
            device,
            queue,
            features,
        } = opened;
        if !self.core.display.supports(&adapter) {
            return Err(failed("$.surface").expected(Expected::Supported));
        }
        watch(&device, &self.core.health);
        let queue = crate::gpu_calls::Queue::new(queue);
        let period = queue.get_timestamp_period();
        self.surface = None;
        self.run = None;
        self.installing = None;
        self.plans = PlanCache::default();
        self.core.display.configure(&device, period);
        self.core.profiler = GpuProfiler::new(features, f64::from(period));
        self.core.ledger = crate::pipelines::Ledger::default();
        self.core.limits = super::install::device_limits(
            &device.limits(),
            device.features(),
            self.core.limits.cores,
        );
        self.core.adapter = crate::AdapterInfo::of(&adapter.get_info());
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.core.pipeline_cache = super::cache::open(&device, &adapter.get_info());
        }
        self.assets = Some(new_assets(&device));
        self.core.device = device;
        self.core.queue = queue;
        Ok(())
    }

    pub async fn start(target: RenderTarget) -> Result<StartingRenderer, Error> {
        let instance = target.instance();
        let surface = target
            .surface(&instance)
            .map_err(|error| failed_with("$.surface", error))?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: target.power_preference(),
                compatible_surface: surface.as_ref(),
                force_fallback_adapter: false,
                ..Default::default()
            })
            .await
            .map_err(|error| failed_with("$.adapter", error))?;
        let adapter_info = crate::AdapterInfo::of(&adapter.get_info());
        let Opened {
            adapter,
            device,
            queue: wgpu_queue,
            features: required_features,
        } = opened(adapter).await?;
        let queue = crate::gpu_calls::Queue::new(wgpu_queue);
        let health = Arc::new(Mutex::new(health::Reports::default()));
        watch(&device, &health);
        let limits = super::install::device_limits(&device.limits(), device.features(), 0);

        let mut display = Display::new(
            instance,
            surface,
            target,
            &adapter,
            device.limits().max_texture_dimension_2d,
        )?;
        let scope = crate::gpu_calls::validation_scope(&device);
        display.configure(&device, queue.get_timestamp_period());
        let assets = new_assets(&device);
        let validation = Box::pin(scope.pop());
        let profiler = GpuProfiler::new(required_features, f64::from(queue.get_timestamp_period()));
        let ledger = crate::pipelines::Ledger::default();
        #[cfg(not(target_arch = "wasm32"))]
        let pipeline_cache = super::cache::open(&device, &adapter.get_info());
        let core = Core {
            device,
            queue,
            adapter: adapter_info,
            display,
            health,
            profiler,
            ledger,
            limits,
            #[cfg(not(target_arch = "wasm32"))]
            pipeline_cache,
        };
        let renderer = Self {
            surface: None,
            hidden: super::frame::Hidden::default(),
            picture: super::frame::Picture::default(),
            core,
            run: None,
            assets: Some(assets),
            plans: PlanCache::default(),
            passes: &[],
            spans: Vec::new(),
            installing: None,
            recovering: None,
            #[cfg(all(feature = "dev", not(target_arch = "wasm32")))]
            watched: super::reload::Watched::default(),
        };
        crate::platform::yield_now().await;
        Ok(StartingRenderer {
            renderer,
            validation,
        })
    }
}
