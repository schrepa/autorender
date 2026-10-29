#![cfg_attr(
    windows,
    expect(
        unsafe_code,
        reason = "loads PIX's capturer and calls its capture functions"
    )
)]

use std::path::Path;

pub struct Pix {
    #[cfg(windows)]
    begin: BeginCapture,
    #[cfg(windows)]
    end: EndCapture,
}

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
#[link(name = "kernel32")]
unsafe extern "system" {
    fn LoadLibraryW(name: *const u16) -> isize;
    fn GetProcAddress(module: isize, name: *const u8) -> Option<unsafe extern "system" fn()>;
}

impl Pix {
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
            let module = unsafe { LoadLibraryW(library.as_ptr()) };
            if module == 0 {
                return None;
            }
            // SAFETY: NUL-terminated export names of the module loaded
            // above; their signatures are PIX's `BeginProgrammaticGpuCapture`
            // and `EndProgrammaticGpuCapture`.
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

    pub fn begin(&self, path: &Path) -> Result<(), engine::Error> {
        #[cfg(windows)]
        {
            let file = wide(path.as_os_str());
            let parameters = Parameters {
                file_name: file.as_ptr(),
                rest: [0; 14],
            };
            // SAFETY: PIX reads the parameters, and copies the file name,
            // during the call.
            let result = unsafe { (self.begin)(&parameters) };
            checked("$.start", result)
        }
        #[cfg(not(windows))]
        {
            let _ = path;
            Err(unsupported())
        }
    }

    pub fn end(&self) -> Result<(), engine::Error> {
        #[cfg(windows)]
        {
            // SAFETY: ends the capture begun above; FALSE keeps it.
            let result = unsafe { (self.end)(0) };
            checked("$.end", result)
        }
        #[cfg(not(windows))]
        {
            Err(unsupported())
        }
    }
}

#[cfg(not(windows))]
fn unsupported() -> engine::Error {
    engine::Error::new(engine::ErrorCode::PixUnavailable, "$").expected(engine::Expected::Supported)
}

#[cfg(windows)]
fn checked(path: &str, result: i32) -> Result<(), engine::Error> {
    if result < 0 {
        Err(engine::Error::new(engine::ErrorCode::CaptureFailed, path)
            .got(format!("{:#010x}", result as u32)))
    } else {
        Ok(())
    }
}

#[cfg(windows)]
fn wide(text: &std::ffi::OsStr) -> Vec<u16> {
    use std::os::windows::ffi::OsStrExt;
    text.encode_wide().chain([0]).collect()
}
