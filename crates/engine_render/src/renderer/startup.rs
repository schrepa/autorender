use super::*;

pub struct StartingRenderer {
    renderer: Renderer,
    validation: std::pin::Pin<Box<dyn std::future::Future<Output = Option<wgpu::Error>>>>,
}

impl StartingRenderer {
    pub async fn finish(self) -> Result<Renderer, Error> {
        if let Some(error) = self.validation.await {
            return Err(failed_with("$.pipelines", error));
        }
        Ok(self.renderer)
    }
}

impl Renderer {
    pub async fn new(target: RenderTarget) -> Result<Self, Error> {
        Self::start(target).await?.finish().await
    }

    #[expect(
        clippy::too_many_lines,
        reason = "longer than the line ceiling; it may only shrink"
    )]
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
        let adapter_limits = adapter.limits();
        let required = crate::pipeline_layouts::REQUIRED_LIMITS;
        let limits = wgpu::Limits {
            max_buffer_size: adapter_limits.max_buffer_size,
            max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
            max_texture_dimension_2d: adapter_limits.max_texture_dimension_2d.min(8_192),
            max_storage_buffers_per_shader_stage: adapter_limits
                .max_storage_buffers_per_shader_stage
                .max(required.storage_buffers),
            max_bind_groups: required.bind_groups,
            max_uniform_buffers_per_shader_stage: required.uniform_buffers,
            max_sampled_textures_per_shader_stage: required.sampled_textures,
            max_samplers_per_shader_stage: required.samplers,
            max_storage_textures_per_shader_stage: required.storage_textures,
            max_dynamic_uniform_buffers_per_pipeline_layout: required.dynamic_uniform_buffers,
            max_dynamic_storage_buffers_per_pipeline_layout: required.dynamic_storage_buffers,
            max_uniform_buffer_binding_size: crate::pipeline_layouts::UNIFORM_BINDING_BYTES,
            max_texture_array_layers: adapter_limits.max_texture_array_layers,
            ..wgpu::Limits::downlevel_defaults()
        };
        let compression = adapter.features()
            & (wgpu::Features::TEXTURE_COMPRESSION_BC
                | wgpu::Features::TEXTURE_COMPRESSION_ETC2
                | wgpu::Features::TEXTURE_COMPRESSION_ASTC);
        let features = compression
            | (adapter.features()
                & (wgpu::Features::INDIRECT_FIRST_INSTANCE
                    | wgpu::Features::RG11B10UFLOAT_RENDERABLE
                    | wgpu::Features::TIMESTAMP_QUERY));
        #[cfg(not(feature = "profiling"))]
        let required_features = features;
        #[cfg(feature = "profiling")]
        let required_features = features | (adapter.features() & crate::profiling::FEATURES);
        let (device, wgpu_queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("engine/device"),
                required_features,
                required_limits: limits,
                ..Default::default()
            })
            .await
            .map_err(|error| failed_with("$.device", error))?;
        let queue = crate::gpu_calls::Queue::new(wgpu_queue);

        let health = Arc::new(Mutex::new(health::Reports::default()));
        let lost = Arc::clone(&health);
        device.set_device_lost_callback(move |reason, message| {
            if let Ok(mut reports) = lost.lock() {
                reports.lost(reason, &message);
            }
        });
        let uncaptured = Arc::clone(&health);
        device.on_uncaptured_error(Arc::new(move |error: wgpu::Error| {
            if let Ok(mut reports) = uncaptured.lock() {
                reports.uncaptured(&error);
            }
        }));

        let device_limits = device.limits();
        let mut display = Display::new(
            instance,
            surface,
            target,
            &adapter,
            (
                device_limits.max_texture_dimension_2d,
                device_limits.max_texture_array_layers,
            ),
        )?;
        let scope = crate::gpu_calls::validation_scope(&device);
        display.configure(&device);
        let assets = new_assets(&device, &queue);
        let validation = Box::pin(scope.pop());
        let profiler = GpuProfiler::new(required_features, f64::from(queue.get_timestamp_period()));
        let pipeline_counts = crate::pipelines::PipelineCounts::new(&device, display.format());
        let core = Core {
            device,
            queue,
            adapter: adapter_info,
            #[cfg(not(target_arch = "wasm32"))]
            debugger_capture: false,
            display,
            health,
            profiler,
            pipeline_counts,
        };
        let renderer = Self {
            core,
            run: None,
            assets: Some(assets),
            plans: PlanCache::default(),
            passes: &[],
            spans: Vec::new(),
            #[cfg(feature = "profiling")]
            prelude_owners: Vec::new(),
            installing: None,
            previous: None,
            seconds: None,
            ran: Vec::new(),
            part_names: &[],
            linked: engine_core::LinkedStats::default(),
        };
        crate::platform::yield_now().await;
        Ok(StartingRenderer {
            renderer,
            validation,
        })
    }
}
