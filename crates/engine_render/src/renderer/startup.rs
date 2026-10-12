//! Starting a renderer: the adapter, device and limits it asks for, the
//! callbacks recording its failures, and every pass it creates up front,
//! while the browser compiles and validates its startup pipelines.

use super::*;

/// A renderer whose startup pipelines the browser may still be compiling
/// and validating: build the world meanwhile, then [`Self::finish`].
pub struct StartingRenderer {
    renderer: Renderer,
    validation: std::pin::Pin<Box<dyn std::future::Future<Output = Option<wgpu::Error>>>>,
}

impl StartingRenderer {
    /// The renderer, once its startup pipelines passed validation.
    pub async fn finish(self) -> Result<Renderer, RenderError> {
        if let Some(error) = self.validation.await {
            return Err(render_error("WebGPU pipeline initialization failed", error));
        }
        Ok(self.renderer)
    }
}

impl Renderer {
    /// Create an engine-owned GPU context presenting to `target`: a canvas
    /// in a secure context, or a native window.
    ///
    /// Returns an actionable error when no compatible WebGPU adapter exists,
    /// device creation fails, or pipeline validation fails. Ordinary builds require
    /// no optional GPU features. Profiling builds request timestamps when supported;
    /// there is deliberately no WebGL fallback in this milestone.
    pub async fn new(target: RenderTarget) -> Result<Self, RenderError> {
        Self::start(target).await?.finish().await
    }

    /// [`Self::new`] without waiting for its startup pipelines: the browser
    /// compiles and validates them while the caller builds its world, the
    /// usual startup cost, until [`StartingRenderer::finish`].
    #[expect(
        clippy::too_many_lines,
        reason = "longer than the line ceiling; it may only shrink"
    )]
    pub async fn start(target: RenderTarget) -> Result<StartingRenderer, RenderError> {
        let instance = target.instance();
        let surface = target
            .surface(&instance)
            .map_err(|error| render_error("Could not create the GPU surface", error))?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: target.power_preference(),
                compatible_surface: surface.as_ref(),
                force_fallback_adapter: false,
                ..Default::default()
            })
            .await
            .map_err(|error| render_error("No WebGPU adapter is available. Use a WebGPU-capable browser on HTTPS or localhost", error))?;
        let adapter_info = crate::AdapterInfo::of(&adapter.get_info());
        let adapter_limits = adapter.limits();
        // Buffers as large as the adapter allows: content, not constants,
        // sizes them.
        let limits = wgpu::Limits {
            max_buffer_size: adapter_limits.max_buffer_size,
            max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
            max_texture_dimension_2d: adapter_limits.max_texture_dimension_2d.min(8_192),
            // GPU presentation's cull pass binds eight, the WebGPU default;
            // the visibility buffer's resolve up to twelve
            // where the adapter allows them.
            max_storage_buffers_per_shader_stage: adapter_limits
                .max_storage_buffers_per_shader_stage
                .max(8),
            // Terrain tiles, impostors and reflection probes
            // take a layer each.
            max_texture_array_layers: adapter_limits.max_texture_array_layers,
            ..wgpu::Limits::downlevel_defaults()
        };
        // Block-compressed textures in every format the
        // adapter samples; hosts transcode to one of them.
        let compression = adapter.features()
            & (wgpu::Features::TEXTURE_COMPRESSION_BC
                | wgpu::Features::TEXTURE_COMPRESSION_ETC2
                | wgpu::Features::TEXTURE_COMPRESSION_ASTC);
        // Indirect draws name their visible-ID regions as first instances
        // where the adapter allows it, so passes bind each ID buffer once.
        // Bloom keeps its levels in 32 bits where the device
        // renders them.
        let features = compression
            | (adapter.features()
                & (wgpu::Features::INDIRECT_FIRST_INSTANCE
                    | wgpu::Features::RG11B10UFLOAT_RENDERABLE));
        #[cfg(not(feature = "profiling"))]
        let required_features = features;
        #[cfg(feature = "profiling")]
        let required_features = features | (adapter.features() & crate::profiling::FEATURES);
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("engine/device"),
                required_features,
                required_limits: limits,
                ..Default::default()
            })
            .await
            .map_err(|error| render_error("Could not initialize the WebGPU device", error))?;

        let failure = Arc::new(Mutex::new(None));
        let lost_failure = Arc::clone(&failure);
        device.set_device_lost_callback(move |reason, message| {
            if let Ok(mut slot) = lost_failure.lock() {
                *slot = Some(format!(
                    "GPU device lost ({reason:?}): {message}. Reload to recreate the engine."
                ));
            }
        });
        let uncaptured_failure = Arc::clone(&failure);
        device.on_uncaptured_error(Arc::new(move |error: wgpu::Error| {
            if let Ok(mut slot) = uncaptured_failure.lock() {
                *slot = Some(format!(
                    "WebGPU rendering failed: {error}. Reload to recreate the engine."
                ));
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
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        display.configure(&device);
        let depth = DepthTarget::new(&device, display.size());
        let lighting_buffer = crate::buffers::uniform(
            &device,
            "engine/lighting-uniform",
            size_of::<LightingUniform>() as u64,
        );
        let materials = MaterialStore::new(&device, &queue);
        let shadows = ShadowPass::new(&device, materials.layout());
        let clouds = crate::clouds::CloudPass::new(&device, &queue);
        let sky = crate::sky::SkyPass::new(&device);
        let environment = Environment::new(
            &device,
            &lighting_buffer,
            engine_core::ReflectionProbeSettings::default().resolution,
            clouds.dome(),
            sky.bindings(),
        );
        let probes = ProbePass::new(&device, size_of::<CameraUniform>() as u64);
        let terrain = crate::terrain::TerrainPass::new(&device, &queue);
        let ground_cover = crate::ground_cover::GroundCoverPass::new(&device);
        let water = crate::water::WaterPass::new(&device);
        let scatter = crate::scatter::ScatterPass::new(&device);
        let occupancy = crate::occupancy::OccupancyPass::new(&device);
        let camera = CameraBindings::new(
            &device,
            &queue,
            lighting_buffer,
            &camera::GroupSources {
                shadows: &shadows,
                environment: &environment,
                probes: &probes,
                horizons: &terrain.horizons,
                occupancy: &occupancy,
            },
        );
        let virtual_draws = VirtualDraws::new(&device);
        let visibility = VisibilityDraws::new(&device);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("engine/meshes"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let instances = Instances::new(&device);
        let mesh_pipelines = MeshPipelines::new(&device);
        let impostor_pipelines =
            ImpostorPipelines::new(&device, materials.layout(), &mesh_pipelines.empty_layout);
        let impostor_atlases = (
            ImpostorAtlas::new(&device, AtlasShape::SPHERE),
            ImpostorAtlas::new(&device, AtlasShape::PLACED),
        );
        let gpu_presentation = GpuPresentationPass::new(&device);
        // Popping takes effect now; validation resolves once they compiled.
        let validation = Box::pin(scope.pop());

        let profiler = GpuProfiler::new(required_features, f64::from(queue.get_timestamp_period()));
        let mesh_arena = MeshArena::new(
            &device,
            [
                size_of::<Vertex>(),
                size_of::<ExtraVertex>(),
                size_of::<SkinVertex>(),
            ]
            .map(|bytes| bytes as u64),
        );
        let renderer = Self {
            device,
            queue,
            adapter: adapter_info,
            #[cfg(not(target_arch = "wasm32"))]
            debugger_capture: false,
            display,
            depth,
            terrain_ids: ScreenTarget::default(),
            meshes: BTreeMap::new(),
            mesh_arena,
            mesh_layout: 0,
            vertex_staging: Vec::new(),
            materials,
            plan: BatchPlan::with_instance_limit(instances.limit),
            draw_order: DrawOrder::default(),
            structure_keys: StructureKeys::default(),
            instances,
            uploads: UploadCache::default(),
            mesh_shader: shader,
            camera,
            mesh_pipelines,
            pipeline_counts: PipelineCounts::default(),
            gpu_presentation,
            culling: Culling { pyramid: None },
            shadows,
            lights: None,
            environment,
            probes,
            clouds,
            sky,
            virtual_draws,
            visibility,
            impostors: ImpostorDraws {
                atlas: impostor_atlases.0,
                placed: impostor_atlases.1,
                pipelines: impostor_pipelines,
                records: Vec::new(),
                dropped: 0,
            },
            terrain,
            ground_cover,
            water,
            scatter,
            scatter_meshes: (BTreeSet::new(), 0),
            scatter_draws: scatter_draws::ScatterDraws::default(),
            switch_fades: frame::SwitchFades::default(),
            occupancy,
            last_seconds: None,
            post: None,
            temporal: None,
            transparent: None,
            particles: None,
            decals: None,
            graph: Default::default(),
            gpu_frame: false,
            failure,
            profiler,
        };
        crate::platform::yield_now().await;
        Ok(StartingRenderer {
            renderer,
            validation,
        })
    }
}
