use super::*;
use crate::platform::{GpuRequest, SurfaceForm};
use engine_core::{ErrorCode, Present};

pub(crate) struct Opened {
    adapter: wgpu::Adapter,
    pub(super) device: wgpu::Device,
    pub(super) queue: wgpu::Queue,
    features: wgpu::Features,
}

pub(super) type Opening =
    std::pin::Pin<Box<dyn std::future::Future<Output = Result<Opened, Error>>>>;

fn own_features(offered: wgpu::Features) -> wgpu::Features {
    let own = wgpu::Features::TIMESTAMP_QUERY;
    #[cfg(not(target_arch = "wasm32"))]
    let own = own | wgpu::Features::PIPELINE_CACHE;
    #[cfg(feature = "profiling")]
    let own = own | crate::profiling::FEATURES;
    offered & own
}

fn granted(
    request: &GpuRequest,
    adapter: &wgpu::Adapter,
) -> Result<(wgpu::Features, wgpu::Limits), Error> {
    let (offered, most) = (adapter.features(), adapter.limits());
    let missing = request.need.features.difference(offered);
    if !missing.is_empty() {
        let path = "$.gpu.need.features";
        return Err(Error::new(ErrorCode::UnsupportedFeature, path).got(format!("{missing:?}")));
    }
    let mut short = None;
    (request.need.limits).check_limits_with_fail_fn(&most, true, |name, _, offered| {
        short = Some((name, offered));
    });
    if let Some((name, offered)) = short {
        let path = format!("$.gpu.need.limits.{name}");
        return Err(Error::new(ErrorCode::UnsupportedFeature, path).got(offered));
    }
    let limits = (request.want.limits.clone())
        .or_worse_values_from(&most)
        .or_better_values_from(&request.need.limits)
        .or_better_values_from(&crate::pipelines::floor());
    let features = request.need.features | (request.want.features & offered);
    Ok((features | own_features(offered), limits))
}

async fn opened(adapter: wgpu::Adapter, request: &GpuRequest) -> Result<Opened, Error> {
    let (features, limits) = granted(request, &adapter)?;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("engine/device"),
            required_features: features,
            required_limits: limits,
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

pub(super) async fn open(instance: wgpu::Instance, request: GpuRequest) -> Result<Opened, Error> {
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: request.power,
            compatible_surface: None,
            force_fallback_adapter: false,
            ..Default::default()
        })
        .await
        .map_err(|error| failed_with("$.adapter", error))?;
    opened(adapter, &request).await
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
        self.core.display.configure(&device, period);
        self.core.profiler = GpuProfiler::new(features, f64::from(period));
        self.core.ledger = crate::pipelines::Ledger::default();
        self.core.limits = super::install::device_limits(&device.limits());
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

    pub async fn start(
        target: RenderTarget,
        gpu: GpuRequest,
        form: SurfaceForm,
        present: Present,
    ) -> Result<StartingRenderer, Error> {
        let instance = target.instance();
        let surface = target
            .surface(&instance)
            .map_err(|error| failed_with("$.surface", error))?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: gpu.power,
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
        } = opened(adapter, &gpu).await?;
        let queue = crate::gpu_calls::Queue::new(wgpu_queue);
        let health = Arc::new(Mutex::new(health::Reports::default()));
        watch(&device, &health);
        let limits = super::install::device_limits(&device.limits());

        let (mut display, notice) = Display::new(
            instance,
            surface,
            target,
            &adapter,
            device.limits().max_texture_dimension_2d,
            (form, present),
        )?;
        if let (Some(notice), Ok(mut reports)) = (notice, health.lock()) {
            reports.notice(notice);
        }
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
            request: gpu,
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
