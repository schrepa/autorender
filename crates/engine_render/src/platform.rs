//! What differs between the platforms the renderer runs on, the browser's
//! WebGPU and native GPU interfaces: where frames present (a canvas, a
//! window or an offscreen texture), how GPU instances start, how map
//! callbacks run, how mapped buffers are written, how the renderer yields
//! while pipelines compile, and how files are read by range.

use std::fmt;

/// A renderer's failure, with what it was doing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RenderError(pub String);

impl fmt::Display for RenderError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for RenderError {}

/// The error of `context` failing with `error`.
pub(crate) fn render_error(context: &str, error: impl fmt::Display) -> RenderError {
    RenderError(format!("{context}: {error}"))
}

/// A native window frames present to: what windowing libraries' windows
/// are.
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

/// Where a renderer presents.
#[derive(Clone, Debug)]
pub enum RenderTarget {
    /// A page's canvas, sized by the page.
    #[cfg(target_arch = "wasm32")]
    Canvas(web_sys::HtmlCanvasElement),
    /// A native window, `size` physical pixels, on the `gpu` preferred;
    /// frames wait for the display's refresh (`vsync`), or show as they
    /// finish, as browsers' measured without it do.
    #[cfg(not(target_arch = "wasm32"))]
    Window {
        window: std::sync::Arc<dyn NativeWindow>,
        size: (u32, u32),
        gpu: GpuPreference,
        vsync: bool,
    },
    /// An offscreen texture of `size` physical pixels, which frames draw
    /// into without presenting, on the `gpu` preferred: for tools that run
    /// without a window.
    #[cfg(not(target_arch = "wasm32"))]
    Offscreen {
        size: (u32, u32),
        gpu: GpuPreference,
    },
}

/// `descriptor` for the GPU interface native renderers draw through: those
/// wgpu's `WGPU_BACKEND` names (`vulkan`, `dx12`, `metal`), else the one
/// browsers use on the platform. That is D3D12 on Windows where DXC
/// (`dxcompiler.dll`, which browsers ship) is beside the program or on the
/// path, as the engine's shaders need it, else Vulkan; Metal on Apple's;
/// Vulkan elsewhere.
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

/// How native renderers start their GPU instances in place of wgpu's own
/// start, where a tool must see the GPU interface's calls (`engine_native`'s
/// hardware counters do).
#[cfg(not(target_arch = "wasm32"))]
static NATIVE_START: std::sync::OnceLock<fn(wgpu::InstanceDescriptor) -> wgpu::Instance> =
    std::sync::OnceLock::new();

/// Starts native renderers' GPU instances with `start`, given the
/// descriptor wgpu's own start would take, from now on; set before the
/// first renderer starts, once per process. Whether it was the first set.
#[cfg(not(target_arch = "wasm32"))]
pub fn start_native_instances_with(start: fn(wgpu::InstanceDescriptor) -> wgpu::Instance) -> bool {
    NATIVE_START.set(start).is_ok()
}

#[cfg(not(target_arch = "wasm32"))]
fn start_native(descriptor: wgpu::InstanceDescriptor) -> wgpu::Instance {
    let start = NATIVE_START.get().copied().unwrap_or(wgpu::Instance::new);
    start(native(descriptor))
}

/// DXC's library beside the program or on the path, on Windows.
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

/// The GPU a native renderer asks for where a machine has an integrated and
/// a discrete one.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GpuPreference {
    /// The discrete GPU, as browsers are asked for.
    #[default]
    HighPerformance,
    /// The integrated GPU.
    LowPower,
}

impl RenderTarget {
    /// The GPU instance presenting to it.
    pub(crate) fn instance(&self) -> wgpu::Instance {
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

    /// The GPU to ask its instance for.
    pub(crate) fn power_preference(&self) -> wgpu::PowerPreference {
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

    /// A surface of `instance` presenting to it; none for an offscreen
    /// texture.
    pub(crate) fn surface(
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

    /// How its frames present.
    pub(crate) fn present_mode(&self) -> wgpu::PresentMode {
        match self {
            #[cfg(not(target_arch = "wasm32"))]
            Self::Window { vsync: false, .. } => wgpu::PresentMode::AutoNoVsync,
            _ => wgpu::PresentMode::Fifo,
        }
    }

    /// Its size in physical pixels.
    pub(crate) fn size(&self) -> (u32, u32) {
        match self {
            #[cfg(target_arch = "wasm32")]
            Self::Canvas(canvas) => (canvas.width(), canvas.height()),
            #[cfg(not(target_arch = "wasm32"))]
            Self::Window { size, .. } | Self::Offscreen { size, .. } => *size,
        }
    }

    /// Takes size `(width, height)`: the canvas's drawing buffer; a native
    /// window's size is its windowing library's.
    pub(crate) fn set_size(&mut self, (width, height): (u32, u32)) {
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

/// The GPU a renderer draws on, as its interface names it: browsers name
/// little of it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AdapterInfo {
    pub name: String,
    /// PCI vendor and device IDs, where known.
    pub vendor: u32,
    pub device: u32,
    /// Integrated, discrete, virtual or software.
    pub device_type: String,
    pub driver: String,
    pub driver_info: String,
    /// DX12, Vulkan, Metal, or the browser's WebGPU.
    pub backend: String,
}

impl AdapterInfo {
    pub(crate) fn of(info: &wgpu::AdapterInfo) -> Self {
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

/// A map callback the platform's GPU interface accepts: the browser's runs
/// on its own thread, native ones must be sendable, so there the callback
/// waits in this thread's store and the interface holds its key; native
/// interfaces run map callbacks while the renderer polls, on its thread.
#[cfg(target_arch = "wasm32")]
pub(crate) fn map_callback(
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
pub(crate) fn map_callback(
    callback: impl FnOnce(Result<(), wgpu::BufferAsyncError>) + 'static,
) -> impl FnOnce(Result<(), wgpu::BufferAsyncError>) + Send + 'static {
    let key = PARKED.with(|parked| {
        let (next, store) = &mut *parked.borrow_mut();
        *next += 1;
        store.insert(*next, Box::new(callback));
        *next
    });
    move |result| {
        let callback = PARKED.with(|parked| parked.borrow_mut().1.remove(&key));
        if let Some(callback) = callback {
            callback(result);
        }
    }
}

/// Writes each of `writes` (bytes at an offset) into mapped `slice`;
/// whether it was mapped. The browser writes through its own view of the
/// mapping, which copies the bytes once.
pub(crate) fn write_mapped<'a>(
    slice: wgpu::BufferSlice<'_>,
    writes: impl IntoIterator<Item = (usize, &'a [u8])>,
) -> bool {
    #[cfg(target_arch = "wasm32")]
    {
        let Ok(view) = slice.get_mapped_range() else {
            return false;
        };
        let target = view.as_uint8array();
        for (offset, bytes) in writes {
            target
                .subarray(offset as u32, (offset + bytes.len()) as u32)
                .copy_from(bytes);
        }
        true
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let Ok(mut view) = slice.get_mapped_range_mut() else {
            return false;
        };
        for (offset, bytes) in writes {
            view.slice(offset..offset + bytes.len())
                .copy_from_slice(bytes);
        }
        true
    }
}

/// Lets the platform run: the browser one task, so commands issued before
/// reach its GPU process before the caller's next synchronous work.
pub(crate) async fn yield_now() {
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

/// Runs the GPU interface's finished work, its map callbacks among it:
/// the browser does so between tasks, native renderers once a frame.
pub(crate) fn poll(device: &wgpu::Device) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = device.poll(wgpu::PollType::Poll);
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = device;
    }
}

/// Reads bytes `range` of the file `url` names and hands them, or why the
/// read failed, to `arrive` once read: the browser fetches them, native
/// hosts read the file at that path.
pub fn read_range(
    url: &str,
    range: std::ops::Range<u64>,
    arrive: impl FnOnce(Result<Vec<u8>, String>) + 'static,
) {
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
            // A JavaScript failure as text.
            let reason = |what: &str, error: wasm_bindgen::JsValue| {
                format!(
                    "{what} {url}: {}",
                    error.as_string().unwrap_or_else(|| format!("{error:?}"))
                )
            };
            let bytes = async {
                let promise = promise.ok_or_else(|| format!("cannot request {url}"))?;
                let response: web_sys::Response = wasm_bindgen_futures::JsFuture::from(promise)
                    .await
                    .map_err(|error| reason("cannot fetch", error))?
                    .dyn_into()
                    .map_err(|_| format!("fetching {url} answered no response"))?;
                if !response.ok() {
                    return Err(format!("{url} answered HTTP {}", response.status()));
                }
                // A server ignoring ranges sends the whole file, which is cut
                // to the range.
                let whole = response.status() == 200;
                let buffer = response
                    .array_buffer()
                    .map_err(|error| reason("cannot read", error))?;
                let buffer = wasm_bindgen_futures::JsFuture::from(buffer)
                    .await
                    .map_err(|error| reason("cannot read", error))?;
                let bytes = js_sys::Uint8Array::new(&buffer).to_vec();
                if whole {
                    let length = bytes.len();
                    bytes
                        .get(range.start as usize..range.end as usize)
                        .map(<[u8]>::to_vec)
                        .ok_or_else(|| {
                            format!(
                                "{url} holds {length} bytes, not bytes {}..{}",
                                range.start, range.end
                            )
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
        arrive(bytes.map_err(|error| {
            format!(
                "cannot read bytes {}..{} of {url}: {error}",
                range.start, range.end
            )
        }));
    }
}
