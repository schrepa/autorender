//! RenderDoc on Windows, through its in-application API: when the program
//! runs under RenderDoc (`renderdoccmd capture -w PROGRAM ...`, or launched
//! from its UI), the GPU work between [`RenderDoc::begin`] and
//! [`RenderDoc::end`] is captured into an `.rdc` file. wgpu reaches
//! RenderDoc only in builds with debug assertions; this reaches it in any.

#![cfg_attr(
    windows,
    expect(unsafe_code, reason = "calls RenderDoc's capture functions")
)]

use std::path::Path;

/// RenderDoc's API, in a program running under it.
pub struct RenderDoc {
    #[cfg(windows)]
    api: *const Api,
}

// The leading entries of `RENDERDOC_API_1_1_2`, as `renderdoc_app.h`
// orders them; this module calls the capture path template, start and end.
#[cfg(windows)]
#[repr(C)]
struct Api {
    leading: [usize; 11],
    set_capture_file_path_template: unsafe extern "C" fn(*const u8),
    following: [usize; 7],
    start_frame_capture: unsafe extern "C" fn(*mut u8, *mut u8),
    is_frame_capturing: unsafe extern "C" fn() -> u32,
    end_frame_capture: unsafe extern "C" fn(*mut u8, *mut u8) -> u32,
}

// `eRENDERDOC_API_Version_1_1_2`.
#[cfg(windows)]
const VERSION_1_1_2: u32 = 10102;

#[cfg(windows)]
type GetApi = unsafe extern "C" fn(u32, *mut *const Api) -> i32;

#[cfg(windows)]
#[link(name = "kernel32")]
unsafe extern "system" {
    fn GetModuleHandleW(name: *const u16) -> isize;
    fn GetProcAddress(module: isize, name: *const u8) -> Option<unsafe extern "system" fn()>;
}

impl RenderDoc {
    /// RenderDoc's API where the program runs under RenderDoc; `None`
    /// otherwise (and off Windows).
    pub fn attached() -> Option<Self> {
        #[cfg(windows)]
        {
            let name: Vec<u16> = "renderdoc.dll\0".encode_utf16().collect();
            // SAFETY: a NUL-terminated name; RenderDoc's library stays
            // loaded in the program it runs, and its `RENDERDOC_GetAPI`
            // fills the pointer with its API table of that version.
            unsafe {
                let module = GetModuleHandleW(name.as_ptr());
                if module == 0 {
                    return None;
                }
                let get = GetProcAddress(module, c"RENDERDOC_GetAPI".as_ptr().cast())?;
                let get = std::mem::transmute::<unsafe extern "system" fn(), GetApi>(get);
                let mut api = std::ptr::null();
                (get(VERSION_1_1_2, &mut api) == 1 && !api.is_null()).then_some(Self { api })
            }
        }
        #[cfg(not(windows))]
        {
            None
        }
    }

    /// Starts capturing the GPU work submitted from now, into a file named
    /// after `path` (RenderDoc adds `_capture.rdc`).
    pub fn begin(&self, path: &Path) -> Result<(), String> {
        #[cfg(windows)]
        {
            let template =
                std::ffi::CString::new(path.with_extension("").to_string_lossy().as_bytes())
                    .map_err(|_| "the capture path holds a NUL".to_owned())?;
            // SAFETY: the API table RenderDoc filled; RenderDoc copies the
            // template; null device and window capture its only device.
            unsafe {
                ((*self.api).set_capture_file_path_template)(template.as_ptr().cast());
                ((*self.api).start_frame_capture)(std::ptr::null_mut(), std::ptr::null_mut());
                if ((*self.api).is_frame_capturing)() == 0 {
                    return Err("RenderDoc did not start its capture".into());
                }
            }
            Ok(())
        }
        #[cfg(not(windows))]
        {
            let _ = path;
            Err("RenderDoc's API is reached on Windows".into())
        }
    }

    /// Ends the capture, keeping its file.
    pub fn end(&self) -> Result<(), String> {
        #[cfg(windows)]
        {
            // SAFETY: ends the capture begun above on the same device.
            let ended = unsafe {
                ((*self.api).end_frame_capture)(std::ptr::null_mut(), std::ptr::null_mut())
            };
            if ended == 0 {
                return Err("RenderDoc could not end its capture".into());
            }
            Ok(())
        }
        #[cfg(not(windows))]
        {
            Err("RenderDoc's API is reached on Windows".into())
        }
    }
}
