use std::fmt;

use engine_core::{Error, ErrorCode, Expected};

pub fn failed(path: &str) -> Error {
    Error::new(ErrorCode::RenderFailed, path)
}

pub fn failed_with(path: &str, error: impl fmt::Display) -> Error {
    failed(path).got(error.to_string())
}

#[cfg(not(target_arch = "wasm32"))]
pub trait NativeWindow:
    wgpu::rwh::HasWindowHandle + wgpu::rwh::HasDisplayHandle + fmt::Debug + Send + Sync + 'static
{
}

#[cfg(not(target_arch = "wasm32"))]
impl<T> NativeWindow for T where
    T: wgpu::rwh::HasWindowHandle
        + wgpu::rwh::HasDisplayHandle
        + fmt::Debug
        + Send
        + Sync
        + 'static
{
}

#[derive(Clone, Debug)]
pub enum RenderTarget {
    #[cfg(target_arch = "wasm32")]
    Canvas(web_sys::HtmlCanvasElement),
    #[cfg(not(target_arch = "wasm32"))]
    Window {
        window: std::sync::Arc<dyn NativeWindow>,
        size: (u32, u32),
        gpu: GpuPreference,
        vsync: bool,
    },
    #[cfg(not(target_arch = "wasm32"))]
    Offscreen {
        size: (u32, u32),
        gpu: GpuPreference,
    },
}

#[cfg(not(target_arch = "wasm32"))]
fn native(mut descriptor: wgpu::InstanceDescriptor) -> wgpu::InstanceDescriptor {
    let dxc = dxc();
    let browsers = if cfg!(windows) && dxc.is_some() {
        wgpu::Backends::DX12
    } else if cfg!(target_vendor = "apple") {
        wgpu::Backends::METAL
    } else {
        wgpu::Backends::VULKAN
    };
    descriptor.backends = browsers.with_env();
    if let Some(path) = dxc {
        descriptor.backend_options.dx12.shader_compiler = wgpu::Dx12Compiler::DynamicDxc {
            dxc_path: path.to_string_lossy().into_owned(),
        };
    }
    descriptor
}

#[cfg(not(target_arch = "wasm32"))]
static NATIVE_START: std::sync::OnceLock<fn(wgpu::InstanceDescriptor) -> wgpu::Instance> =
    std::sync::OnceLock::new();

#[cfg(not(target_arch = "wasm32"))]
pub fn start_native_instances_with(start: fn(wgpu::InstanceDescriptor) -> wgpu::Instance) -> bool {
    NATIVE_START.set(start).is_ok()
}

#[cfg(not(target_arch = "wasm32"))]
fn start_native(descriptor: wgpu::InstanceDescriptor) -> wgpu::Instance {
    let start = NATIVE_START.get().copied().unwrap_or(wgpu::Instance::new);
    start(native(descriptor))
}

#[cfg(not(target_arch = "wasm32"))]
fn dxc() -> Option<std::path::PathBuf> {
    if !cfg!(windows) {
        return None;
    }
    let beside = std::env::current_exe()
        .ok()
        .and_then(|program| program.parent().map(std::path::Path::to_path_buf));
    let path = std::env::var_os("PATH");
    beside
        .into_iter()
        .chain(path.iter().flat_map(std::env::split_paths))
        .map(|folder| folder.join("dxcompiler.dll"))
        .find(|file| file.is_file())
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GpuPreference {
    #[default]
    HighPerformance,
    LowPower,
}

impl RenderTarget {
    pub fn instance(&self) -> wgpu::Instance {
        match self {
            #[cfg(target_arch = "wasm32")]
            Self::Canvas(_) => wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::BROWSER_WEBGPU,
                ..wgpu::InstanceDescriptor::new_without_display_handle()
            }),
            #[cfg(not(target_arch = "wasm32"))]
            Self::Window { window, .. } => start_native(
                wgpu::InstanceDescriptor::new_with_display_handle(Box::new(window.clone())),
            ),
            #[cfg(not(target_arch = "wasm32"))]
            Self::Offscreen { .. } => {
                start_native(wgpu::InstanceDescriptor::new_without_display_handle())
            }
        }
    }

    pub fn power_preference(&self) -> wgpu::PowerPreference {
        match self {
            #[cfg(target_arch = "wasm32")]
            Self::Canvas(_) => wgpu::PowerPreference::HighPerformance,
            #[cfg(not(target_arch = "wasm32"))]
            Self::Window { gpu, .. } | Self::Offscreen { gpu, .. } => match gpu {
                GpuPreference::HighPerformance => wgpu::PowerPreference::HighPerformance,
                GpuPreference::LowPower => wgpu::PowerPreference::LowPower,
            },
        }
    }

    pub fn surface(
        &self,
        instance: &wgpu::Instance,
    ) -> Result<Option<wgpu::Surface<'static>>, wgpu::CreateSurfaceError> {
        match self {
            #[cfg(target_arch = "wasm32")]
            Self::Canvas(canvas) => instance
                .create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
                .map(Some),
            #[cfg(not(target_arch = "wasm32"))]
            Self::Window { window, .. } => instance
                .create_surface(wgpu::SurfaceTarget::Window(Box::new(window.clone())))
                .map(Some),
            #[cfg(not(target_arch = "wasm32"))]
            Self::Offscreen { .. } => Ok(None),
        }
    }

    pub fn present_mode(&self) -> wgpu::PresentMode {
        match self {
            #[cfg(not(target_arch = "wasm32"))]
            Self::Window { vsync: false, .. } => wgpu::PresentMode::AutoNoVsync,
            _ => wgpu::PresentMode::Fifo,
        }
    }

    pub fn size(&self) -> (u32, u32) {
        match self {
            #[cfg(target_arch = "wasm32")]
            Self::Canvas(canvas) => (canvas.width(), canvas.height()),
            #[cfg(not(target_arch = "wasm32"))]
            Self::Window { size, .. } | Self::Offscreen { size, .. } => *size,
        }
    }

    pub fn set_size(&mut self, (width, height): (u32, u32)) {
        match self {
            #[cfg(target_arch = "wasm32")]
            Self::Canvas(canvas) => {
                canvas.set_width(width);
                canvas.set_height(height);
            }
            #[cfg(not(target_arch = "wasm32"))]
            Self::Window { size, .. } | Self::Offscreen { size, .. } => *size = (width, height),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AdapterInfo {
    pub name: String,
    pub vendor: u32,
    pub device: u32,
    pub device_type: String,
    pub driver: String,
    pub driver_info: String,
    pub backend: String,
}

impl AdapterInfo {
    pub fn of(info: &wgpu::AdapterInfo) -> Self {
        Self {
            name: info.name.clone(),
            vendor: info.vendor,
            device: info.device,
            device_type: format!("{:?}", info.device_type),
            driver: info.driver.clone(),
            driver_info: info.driver_info.clone(),
            backend: format!("{:?}", info.backend),
        }
    }
}

#[cfg(target_arch = "wasm32")]
pub fn map_callback(
    callback: impl FnOnce(Result<(), wgpu::BufferAsyncError>) + 'static,
) -> impl FnOnce(Result<(), wgpu::BufferAsyncError>) + 'static {
    callback
}

#[cfg(not(target_arch = "wasm32"))]
type Parked = Box<dyn FnOnce(Result<(), wgpu::BufferAsyncError>)>;

#[cfg(not(target_arch = "wasm32"))]
thread_local! {
    static PARKED: std::cell::RefCell<(u64, engine_core::collections::HashMap<u64, Parked>)> =
        std::cell::RefCell::new((0, engine_core::collections::HashMap::default()));
}

#[cfg(not(target_arch = "wasm32"))]
type Arrived = (
    std::thread::ThreadId,
    u64,
    Result<(), wgpu::BufferAsyncError>,
);

#[cfg(not(target_arch = "wasm32"))]
static ARRIVED: std::sync::Mutex<Vec<Arrived>> = std::sync::Mutex::new(Vec::new());

#[cfg(not(target_arch = "wasm32"))]
pub fn map_callback(
    callback: impl FnOnce(Result<(), wgpu::BufferAsyncError>) + 'static,
) -> impl FnOnce(Result<(), wgpu::BufferAsyncError>) + Send + 'static {
    let key = PARKED.with(|parked| {
        let (next, store) = &mut *parked.borrow_mut();
        *next += 1;
        store.insert(*next, Box::new(callback));
        *next
    });
    let parked_on = std::thread::current().id();
    move |result| {
        if std::thread::current().id() == parked_on {
            run_parked(key, result);
        } else if let Ok(mut arrived) = ARRIVED.lock() {
            arrived.push((parked_on, key, result));
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn run_parked(key: u64, result: Result<(), wgpu::BufferAsyncError>) {
    let callback = PARKED.with(|parked| parked.borrow_mut().1.remove(&key));
    if let Some(callback) = callback {
        callback(result);
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn run_arrived_maps() {
    let this = std::thread::current().id();
    loop {
        let next = ARRIVED.lock().ok().and_then(|mut arrived| {
            let at = arrived.iter().position(|&(thread, _, _)| thread == this)?;
            Some(arrived.remove(at))
        });
        let Some((_, key, result)) = next else {
            return;
        };
        run_parked(key, result);
    }
}

pub async fn yield_now() {
    #[cfg(target_arch = "wasm32")]
    {
        use wasm_bindgen::JsValue;
        let promise = js_sys::Promise::new(&mut |resolve, _| {
            let scheduled = web_sys::window()
                .is_some_and(|window| window.set_timeout_with_callback(&resolve).is_ok());
            if !scheduled {
                let _ = resolve.call0(&JsValue::NULL);
            }
        });
        let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
    }
}

pub fn ready<F: std::future::Future>(future: F) -> Option<F::Output> {
    let mut future = std::pin::pin!(future);
    let mut context = std::task::Context::from_waker(std::task::Waker::noop());
    match future.as_mut().poll(&mut context) {
        std::task::Poll::Ready(output) => Some(output),
        std::task::Poll::Pending => None,
    }
}

pub fn poll(device: &wgpu::Device) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = device.poll(wgpu::PollType::Poll);
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = device;
    }
}

pub fn read_range(
    url: &str,
    range: std::ops::Range<u64>,
    arrive: impl FnOnce(Result<Vec<u8>, Error>) + 'static,
) {
    let unread = |path: &str| Error::new(ErrorCode::SourceIo, path).expected(Expected::Readable);
    #[cfg(target_arch = "wasm32")]
    {
        use wasm_bindgen::JsCast;
        let request = (|| {
            let init = web_sys::RequestInit::new();
            init.set_method("GET");
            let headers = web_sys::Headers::new().ok()?;
            headers
                .set("Range", &format!("bytes={}-{}", range.start, range.end - 1))
                .ok()?;
            init.set_headers(&headers);
            web_sys::Request::new_with_str_and_init(url, &init).ok()
        })();
        let promise = request
            .zip(web_sys::window())
            .map(|(request, window)| window.fetch_with_request(&request));
        let url = url.to_owned();
        wasm_bindgen_futures::spawn_local(async move {
            let failed = |error: wasm_bindgen::JsValue| {
                unread("$.url").got(error.as_string().unwrap_or_else(|| url.clone()))
            };
            let bytes = async {
                let promise = promise.ok_or_else(|| unread("$.url").got(&url))?;
                let response: web_sys::Response = wasm_bindgen_futures::JsFuture::from(promise)
                    .await
                    .map_err(failed)?
                    .dyn_into()
                    .map_err(|_| unread("$.url").got(&url))?;
                if !response.ok() {
                    return Err(Error::new(ErrorCode::SourceIo, "$.status")
                        .range(200.0, 299.0)
                        .got(response.status()));
                }
                let whole = response.status() == 200;
                let buffer = response.array_buffer().map_err(failed)?;
                let buffer = wasm_bindgen_futures::JsFuture::from(buffer)
                    .await
                    .map_err(failed)?;
                let bytes = js_sys::Uint8Array::new(&buffer).to_vec();
                if whole {
                    let length = bytes.len();
                    bytes
                        .get(range.start as usize..range.end as usize)
                        .map(<[u8]>::to_vec)
                        .ok_or_else(|| {
                            Error::new(ErrorCode::SourceIo, "$.range")
                                .at_most(length as u64)
                                .got([range.start, range.end])
                        })
                } else {
                    Ok(bytes)
                }
            }
            .await;
            arrive(bytes);
        });
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        use std::io::{Read, Seek, SeekFrom};
        let bytes = (|| {
            let mut file = std::fs::File::open(url)?;
            file.seek(SeekFrom::Start(range.start))?;
            let mut bytes = vec![0; (range.end - range.start) as usize];
            file.read_exact(&mut bytes)?;
            Ok::<_, std::io::Error>(bytes)
        })();
        arrive(bytes.map_err(|_| unread("$.url").got(url)));
    }
}
