//! What differs between the platforms the renderer runs on, the browser's
//! WebGPU and native GPU interfaces: where frames present, how GPU
//! instances start, how map callbacks run, how mapped buffers are written,
//! how the renderer yields while pipelines compile, and how files are read
//! by range.

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
    /// A native window, `size` physical pixels.
    #[cfg(not(target_arch = "wasm32"))]
    Window {
        window: std::sync::Arc<dyn NativeWindow>,
        size: (u32, u32),
    },
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
            Self::Window { window, .. } => wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::PRIMARY,
                ..wgpu::InstanceDescriptor::new_with_display_handle(Box::new(window.clone()))
            }),
        }
    }

    /// A surface of `instance` presenting to it.
    pub(crate) fn surface(
        &self,
        instance: &wgpu::Instance,
    ) -> Result<wgpu::Surface<'static>, wgpu::CreateSurfaceError> {
        match self {
            #[cfg(target_arch = "wasm32")]
            Self::Canvas(canvas) => {
                instance.create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
            }
            #[cfg(not(target_arch = "wasm32"))]
            Self::Window { window, .. } => {
                instance.create_surface(wgpu::SurfaceTarget::Window(Box::new(window.clone())))
            }
        }
    }

    /// Its size in physical pixels.
    pub(crate) fn size(&self) -> (u32, u32) {
        match self {
            #[cfg(target_arch = "wasm32")]
            Self::Canvas(canvas) => (canvas.width(), canvas.height()),
            #[cfg(not(target_arch = "wasm32"))]
            Self::Window { size, .. } => *size,
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
            Self::Window { size, .. } => *size = (width, height),
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
    static PARKED: std::cell::RefCell<(u64, std::collections::HashMap<u64, Parked>)> =
        std::cell::RefCell::new((0, std::collections::HashMap::new()));
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

/// Reads bytes `range` of the file `url` names and hands them, or the
/// failure, to `arrive` once read: the browser fetches them, native
/// hosts read the file at that path.
pub fn read_range(
    url: &str,
    range: std::ops::Range<u64>,
    arrive: impl FnOnce(Result<Vec<u8>, ()>) + 'static,
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
        wasm_bindgen_futures::spawn_local(async move {
            let bytes = async {
                let response: web_sys::Response =
                    wasm_bindgen_futures::JsFuture::from(promise.ok_or(())?)
                        .await
                        .map_err(|_| ())?
                        .dyn_into()
                        .map_err(|_| ())?;
                if !response.ok() {
                    return Err(());
                }
                // A server ignoring ranges sends the whole file, which is cut
                // to the range.
                let whole = response.status() == 200;
                let buffer =
                    wasm_bindgen_futures::JsFuture::from(response.array_buffer().map_err(|_| ())?)
                        .await
                        .map_err(|_| ())?;
                let bytes = js_sys::Uint8Array::new(&buffer).to_vec();
                if whole {
                    bytes
                        .get(range.start as usize..range.end as usize)
                        .map(<[u8]>::to_vec)
                        .ok_or(())
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
            let mut file = std::fs::File::open(url).map_err(|_| ())?;
            file.seek(SeekFrom::Start(range.start)).map_err(|_| ())?;
            let mut bytes = vec![0; (range.end - range.start) as usize];
            file.read_exact(&mut bytes).map_err(|_| ())?;
            Ok(bytes)
        })();
        arrive(bytes);
    }
}
