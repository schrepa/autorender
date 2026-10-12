//! Windows: the sampler suspends the thread, reads its registers and
//! unwinds it with the system's unwind tables, then resumes it. Opening a
//! thread of one's own process for that needs no privilege.

#![expect(
    unsafe_code,
    reason = "suspends threads, reads their registers and unwinds them"
)]

use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
use windows_sys::Win32::System::Diagnostics::Debug::{CONTEXT, GetThreadContext};
use windows_sys::Win32::System::Threading::{
    GetCurrentThreadId, GetCurrentThreadStackLimits, OpenThread, ResumeThread, SuspendThread,
    THREAD_GET_CONTEXT, THREAD_QUERY_LIMITED_INFORMATION, THREAD_SUSPEND_RESUME,
};
use windows_sys::Win32::System::WindowsProgramming::QueryThreadCycleTime;

pub(super) const METHOD: Option<&str> =
    Some("suspend, read registers, unwind with the system's tables");

#[derive(Clone, Copy)]
pub(super) struct Id(u32);

/// A thread opened for sampling.
pub(super) struct Open(HANDLE);

impl Drop for Open {
    fn drop(&mut self) {
        // SAFETY: the handle `open` returned, closed once.
        unsafe {
            CloseHandle(self.0);
        }
    }
}

// SAFETY: a thread handle works from any thread of its process, and the
// sampler only reads the handle's thread through system calls.
unsafe impl Send for Open {}

pub(super) fn current() -> Option<(Id, (u64, u64))> {
    let (mut low, mut high) = (0usize, 0usize);
    // SAFETY: plain calls about the calling thread.
    let id = unsafe {
        GetCurrentThreadStackLimits(&mut low, &mut high);
        GetCurrentThreadId()
    };
    Some((Id(id), (low as u64, high as u64)))
}

pub(super) fn open(Id(id): Id) -> Option<Open> {
    // SAFETY: opens a thread of this process by its ID.
    let handle = unsafe {
        OpenThread(
            THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_LIMITED_INFORMATION,
            0,
            id,
        )
    };
    (!handle.is_null()).then_some(Open(handle))
}

pub(super) fn cpu_ns(Open(handle): &Open) -> Option<u64> {
    let mut cycles = 0u64;
    // SAFETY: a thread handle opened above.
    let read = unsafe { QueryThreadCycleTime(*handle, &mut cycles) };
    (read != 0).then(|| (cycles as f64 * nanos_per_cycle()) as u64)
}

/// Nanoseconds per cycle of the counter thread CPU time counts in, measured
/// once against the clock while this thread spins.
fn nanos_per_cycle() -> f64 {
    static SCALE: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *SCALE.get_or_init(|| {
        let cycles = || {
            let mut cycles = 0u64;
            // SAFETY: the pseudo-handle of the calling thread.
            unsafe {
                QueryThreadCycleTime(
                    windows_sys::Win32::System::Threading::GetCurrentThread(),
                    &mut cycles,
                );
            }
            cycles
        };
        let (start, first) = (std::time::Instant::now(), cycles());
        while start.elapsed() < std::time::Duration::from_millis(5) {
            std::hint::spin_loop();
        }
        let spun = cycles().saturating_sub(first).max(1);
        start.elapsed().as_nanos() as f64 / spun as f64
    })
}

pub(super) fn sample(Open(handle): &Open, bounds: (u64, u64), frames: &mut [u64]) -> usize {
    // SAFETY: the thread stays suspended while its registers are read and
    // its stack unwound, reading only within its stack's bounds and the
    // loaded modules' unwind tables; nothing here allocates or locks what
    // the stopped thread could hold, but the system's function table lock.
    unsafe {
        if SuspendThread(*handle) == u32::MAX {
            return 0;
        }
        let mut context: CONTEXT = std::mem::zeroed();
        context.ContextFlags = FULL;
        let depth = if GetThreadContext(*handle, &mut context) != 0 {
            unwind(&mut context, bounds, frames)
        } else {
            0
        };
        ResumeThread(*handle);
        depth
    }
}

#[cfg(target_arch = "x86_64")]
const FULL: u32 = windows_sys::Win32::System::Diagnostics::Debug::CONTEXT_FULL_AMD64;
#[cfg(target_arch = "aarch64")]
const FULL: u32 = windows_sys::Win32::System::Diagnostics::Debug::CONTEXT_FULL_ARM64;

#[cfg(target_arch = "x86_64")]
unsafe fn unwind(context: &mut CONTEXT, (low, high): (u64, u64), frames: &mut [u64]) -> usize {
    use windows_sys::Win32::System::Diagnostics::Debug::{
        RtlLookupFunctionEntry, RtlVirtualUnwind, UNW_FLAG_NHANDLER,
    };
    let mut depth = 0;
    while depth < frames.len() && context.Rip != 0 {
        if context.Rsp < low || context.Rsp.saturating_add(8) > high {
            break;
        }
        frames[depth] = context.Rip;
        depth += 1;
        let mut base = 0u64;
        // SAFETY: looks up unwind data by address; reads only the stack
        // within its bounds.
        unsafe {
            let entry = RtlLookupFunctionEntry(context.Rip, &mut base, std::ptr::null_mut());
            if entry.is_null() {
                // A leaf function keeps its return address at the stack's
                // top.
                context.Rip = *(context.Rsp as *const u64);
                context.Rsp += 8;
            } else {
                let mut handler = std::ptr::null_mut();
                let mut frame = 0u64;
                RtlVirtualUnwind(
                    UNW_FLAG_NHANDLER,
                    base,
                    context.Rip,
                    entry,
                    context,
                    &mut handler,
                    &mut frame,
                    std::ptr::null_mut(),
                );
            }
        }
    }
    depth
}

/// Windows on Arm keeps frame records: the frame pointer chain.
#[cfg(target_arch = "aarch64")]
unsafe fn unwind(context: &mut CONTEXT, (low, high): (u64, u64), frames: &mut [u64]) -> usize {
    // SAFETY: the register union's named view of a full context.
    let (pc, fp) = unsafe { (context.Pc, context.Anonymous.Anonymous.Fp) };
    let mut depth = 0;
    if frames.is_empty() || pc == 0 {
        return 0;
    }
    frames[0] = pc;
    depth += 1;
    let mut fp = fp;
    while depth < frames.len() && fp >= low && fp.saturating_add(16) <= high && fp.is_multiple_of(8)
    {
        // SAFETY: both words lie within the stack's bounds, checked above.
        let (next, ret) = unsafe { (*(fp as *const u64), *((fp + 8) as *const u64)) };
        if ret == 0 || next <= fp {
            break;
        }
        frames[depth] = ret;
        depth += 1;
        fp = next;
    }
    depth
}
