//! Linux (and other Unix systems with POSIX threads, Apple's aside): the
//! sampler interrupts the thread with `SIGPROF`, whose handler reads the
//! interrupted registers and follows the frame pointer chain into a shared
//! slot, which the sampler then copies. Signalling a thread of one's own
//! process needs no privilege, unlike `perf_event_open` under Ubuntu's or
//! Debian's defaults.

#![expect(unsafe_code, reason = "signals threads and reads their registers")]

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

pub(super) const METHOD: Option<&str> =
    Some("signal the thread, follow frame pointers from its registers");

#[derive(Clone, Copy)]
pub(super) struct Id(libc::pthread_t);

// SAFETY: a pthread handle names a thread for signalling and clocks, which
// any thread of the process may use; the sampler uses it for nothing else.
unsafe impl Send for Id {}

pub(super) struct Open(libc::pthread_t);

const MAX: usize = super::MAX_FRAMES;

// The slot one sample at a time goes through: requested by the sampler,
// written by the handler, read back by the sampler.
const IDLE: u32 = 0;
const REQUESTED: u32 = 1;
const WRITING: u32 = 2;
const WRITTEN: u32 = 3;
static STATE: AtomicU32 = AtomicU32::new(IDLE);
static LOW: AtomicU64 = AtomicU64::new(0);
static HIGH: AtomicU64 = AtomicU64::new(0);
static DEPTH: AtomicUsize = AtomicUsize::new(0);
static FRAMES: [AtomicU64; MAX] = [const { AtomicU64::new(0) }; MAX];

pub(super) fn current() -> Option<(Id, (u64, u64))> {
    // SAFETY: queries the calling thread's own attributes.
    unsafe {
        let thread = libc::pthread_self();
        let mut attributes: libc::pthread_attr_t = std::mem::zeroed();
        if libc::pthread_getattr_np(thread, &mut attributes) != 0 {
            return None;
        }
        let (mut base, mut size) = (std::ptr::null_mut(), 0usize);
        let read = libc::pthread_attr_getstack(&attributes, &mut base, &mut size);
        libc::pthread_attr_destroy(&mut attributes);
        (read == 0).then(|| (Id(thread), (base as u64, base as u64 + size as u64)))
    }
}

pub(super) fn open(Id(thread): Id) -> Option<Open> {
    install()?;
    Some(Open(thread))
}

pub(super) fn cpu_ns(Open(thread): &Open) -> Option<u64> {
    // SAFETY: the clock of a live thread of this process.
    unsafe {
        let mut clock = 0;
        if libc::pthread_getcpuclockid(*thread, &mut clock) != 0 {
            return None;
        }
        let mut time: libc::timespec = std::mem::zeroed();
        (libc::clock_gettime(clock, &mut time) == 0)
            .then(|| time.tv_sec as u64 * 1_000_000_000 + time.tv_nsec as u64)
    }
}

pub(super) fn sample(Open(thread): &Open, (low, high): (u64, u64), frames: &mut [u64]) -> usize {
    LOW.store(low, Ordering::Relaxed);
    HIGH.store(high, Ordering::Relaxed);
    DEPTH.store(0, Ordering::Relaxed);
    STATE.store(REQUESTED, Ordering::Release);
    // SAFETY: signals a thread of this process; the handler is installed.
    if unsafe { libc::pthread_kill(*thread, libc::SIGPROF) } != 0 {
        STATE.store(IDLE, Ordering::Release);
        return 0;
    }
    let start = Instant::now();
    loop {
        match STATE.load(Ordering::Acquire) {
            WRITTEN => break,
            REQUESTED if start.elapsed() > Duration::from_millis(20) => {
                // The thread did not run its handler in time: withdraw the
                // request, unless the handler took it meanwhile.
                if STATE
                    .compare_exchange(REQUESTED, IDLE, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    return 0;
                }
            }
            _ => std::thread::yield_now(),
        }
    }
    let depth = DEPTH.load(Ordering::Relaxed).min(frames.len());
    for (frame, slot) in frames.iter_mut().zip(&FRAMES).take(depth) {
        *frame = slot.load(Ordering::Relaxed);
    }
    STATE.store(IDLE, Ordering::Release);
    depth
}

/// Installs the `SIGPROF` handler once; `None` if it could not be.
fn install() -> Option<()> {
    static INSTALLED: OnceLock<bool> = OnceLock::new();
    INSTALLED
        .get_or_init(|| {
            // SAFETY: installs a handler that only reads registers and the
            // interrupted thread's stack, and writes atomics.
            unsafe {
                let mut action: libc::sigaction = std::mem::zeroed();
                action.sa_sigaction = handler as *const () as usize;
                action.sa_flags = libc::SA_SIGINFO | libc::SA_RESTART;
                libc::sigemptyset(&mut action.sa_mask);
                libc::sigaction(libc::SIGPROF, &action, std::ptr::null_mut()) == 0
            }
        })
        .then_some(())
}

extern "C" fn handler(_: libc::c_int, _: *mut libc::siginfo_t, context: *mut libc::c_void) {
    if STATE
        .compare_exchange(REQUESTED, WRITING, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return;
    }
    // SAFETY: the kernel passes the interrupted context.
    let (pc, fp) = unsafe { registers(context.cast()) };
    let mut frames = [0u64; MAX];
    let bounds = (LOW.load(Ordering::Relaxed), HIGH.load(Ordering::Relaxed));
    let depth = super::walk_frame_pointers(pc, fp, bounds, &mut frames);
    for (slot, frame) in FRAMES.iter().zip(&frames[..depth]) {
        slot.store(*frame, Ordering::Relaxed);
    }
    DEPTH.store(depth, Ordering::Relaxed);
    STATE.store(WRITTEN, Ordering::Release);
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
unsafe fn registers(context: *const libc::ucontext_t) -> (u64, u64) {
    // SAFETY: the interrupted thread's saved registers.
    unsafe {
        let registers = &(*context).uc_mcontext.gregs;
        (
            registers[libc::REG_RIP as usize] as u64,
            registers[libc::REG_RBP as usize] as u64,
        )
    }
}

#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
unsafe fn registers(context: *const libc::ucontext_t) -> (u64, u64) {
    // SAFETY: the interrupted thread's saved registers; x29 is the frame
    // pointer.
    unsafe {
        let machine = &(*context).uc_mcontext;
        (machine.pc, machine.regs[29])
    }
}

/// Other Unix systems: no stack, the sample is the thread's absence.
#[cfg(not(all(
    target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")
)))]
unsafe fn registers(_: *const libc::ucontext_t) -> (u64, u64) {
    (0, 0)
}
