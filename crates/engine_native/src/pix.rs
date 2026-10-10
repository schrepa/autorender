//! PIX on Windows: its GPU capturer, loaded into the program before the
//! D3D12 device exists, captures the GPU work between [`Pix::begin`] and
//! [`Pix::end`] into a `.wpix` file, as PIX's `PIXBeginCapture2` and
//! `PIXEndCapture` do. Unsafe code, as `renderdoc.rs` holds: loading the
//! library and calling its two exports.

use std::path::Path;

/// PIX's GPU capturer, loaded.
pub struct Pix {
    #[cfg(windows)]
    begin: BeginCapture,
    #[cfg(windows)]
    end: EndCapture,
}

// `PIX_CAPTURE_GPU`'s parameters: the capture's file, then room for the
// largest member of PIX's parameter union.
#[cfg(windows)]
#[repr(C)]
struct Parameters {
    file_name: *const u16,
    rest: [u32; 14],
}

#[cfg(windows)]
type BeginCapture = unsafe extern "system" fn(*const Parameters) -> i32;
#[cfg(windows)]
type EndCapture = unsafe extern "system" fn(i32) -> i32;

#[cfg(windows)]
#[allow(unsafe_code)]
#[link(name = "kernel32")]
unsafe extern "system" {
    fn LoadLibraryW(name: *const u16) -> isize;
    fn GetProcAddress(module: isize, name: *const u8) -> Option<unsafe extern "system" fn()>;
}

impl Pix {
    /// PIX's GPU capturer from its newest installed version, loaded into
    /// the program; `None` where PIX is not installed (and off Windows).
    /// Load it before the game's D3D12 device exists.
    pub fn load() -> Option<Self> {
        #[cfg(windows)]
        {
            let mut versions: Vec<_> = std::fs::read_dir(r"C:\Program Files\Microsoft PIX")
                .ok()?
                .flatten()
                .map(|entry| entry.path().join("WinPixGpuCapturer.dll"))
                .filter(|path| path.is_file())
                .collect();
            versions.sort();
            let library = wide(versions.last()?.as_os_str());
            // SAFETY: a NUL-terminated path; the library stays loaded for
            // the program's life, as PIX's capturer must.
            #[allow(unsafe_code)]
            let module = unsafe { LoadLibraryW(library.as_ptr()) };
            if module == 0 {
                return None;
            }
            // SAFETY: NUL-terminated export names of the module loaded
            // above; their signatures are PIX's `BeginProgrammaticGpuCapture`
            // and `EndProgrammaticGpuCapture`.
            #[allow(unsafe_code)]
            unsafe {
                let begin = GetProcAddress(module, c"BeginProgrammaticGpuCapture".as_ptr().cast())?;
                let end = GetProcAddress(module, c"EndProgrammaticGpuCapture".as_ptr().cast())?;
                Some(Self {
                    begin: std::mem::transmute::<unsafe extern "system" fn(), BeginCapture>(begin),
                    end: std::mem::transmute::<unsafe extern "system" fn(), EndCapture>(end),
                })
            }
        }
        #[cfg(not(windows))]
        {
            None
        }
    }

    /// Starts capturing the GPU work submitted from now into `path`.
    pub fn begin(&self, path: &Path) -> Result<(), String> {
        #[cfg(windows)]
        {
            let file = wide(path.as_os_str());
            let parameters = Parameters {
                file_name: file.as_ptr(),
                rest: [0; 14],
            };
            // SAFETY: PIX reads the parameters, and copies the file name,
            // during the call.
            #[allow(unsafe_code)]
            let result = unsafe { (self.begin)(&parameters) };
            checked("start", result)
        }
        #[cfg(not(windows))]
        {
            let _ = path;
            Err("PIX runs on Windows".into())
        }
    }

    /// Ends the capture, keeping its file.
    pub fn end(&self) -> Result<(), String> {
        #[cfg(windows)]
        {
            // SAFETY: ends the capture begun above; FALSE keeps it.
            #[allow(unsafe_code)]
            let result = unsafe { (self.end)(0) };
            checked("end", result)
        }
        #[cfg(not(windows))]
        {
            Err("PIX runs on Windows".into())
        }
    }
}

/// `result` of PIX's `what`, an `HRESULT`.
#[cfg(windows)]
fn checked(what: &str, result: i32) -> Result<(), String> {
    if result < 0 {
        Err(format!(
            "PIX could not {what} its capture: {:#010x}",
            result as u32
        ))
    } else {
        Ok(())
    }
}

/// `text` as a NUL-terminated UTF-16 string.
#[cfg(windows)]
fn wide(text: &std::ffi::OsStr) -> Vec<u16> {
    use std::os::windows::ffi::OsStrExt;
    text.encode_wide().chain([0]).collect()
}
