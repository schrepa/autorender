//! macOS: the sampler suspends the thread through its Mach port, reads its
//! registers and follows the frame pointer chain Apple's ABIs keep, then
//! resumes it. A process may suspend its own threads without privilege.

#![expect(unsafe_code, reason = "suspends threads and reads their registers")]

use mach2::kern_return::KERN_SUCCESS;
use mach2::thread_act::{thread_get_state, thread_resume, thread_suspend};

pub(super) const METHOD: Option<&str> = Some("suspend, read registers, follow frame pointers");

#[derive(Clone, Copy)]
pub(super) struct Id(mach2::port::mach_port_t);

// SAFETY: a Mach port name of this process's thread, which any thread of the
// process may suspend, resume and read.
unsafe impl Send for Id {}

pub(super) struct Open(mach2::port::mach_port_t);

pub(super) fn current() -> Option<(Id, (u64, u64))> {
    // SAFETY: queries the calling thread's own port and stack.
    unsafe {
        let thread = libc::pthread_self();
        let port = libc::pthread_mach_thread_np(thread);
        let high = libc::pthread_get_stackaddr_np(thread) as u64;
        let size = libc::pthread_get_stacksize_np(thread) as u64;
        Some((Id(port), (high.saturating_sub(size), high)))
    }
}

pub(super) fn open(Id(port): Id) -> Option<Open> {
    Some(Open(port))
}

pub(super) fn cpu_ns(_: &Open) -> Option<u64> {
    None
}

pub(super) fn sample(Open(port): &Open, bounds: (u64, u64), frames: &mut [u64]) -> usize {
    // SAFETY: the thread stays suspended while its registers are read and
    // its frame records followed within its stack's bounds; nothing here
    // allocates or locks.
    unsafe {
        if thread_suspend(*port) != KERN_SUCCESS {
            return 0;
        }
        let depth = match registers(*port) {
            Some((pc, fp)) => super::walk_frame_pointers(pc, fp, bounds, frames),
            None => 0,
        };
        thread_resume(*port);
        depth
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn registers(port: mach2::port::mach_port_t) -> Option<(u64, u64)> {
    use mach2::structs::arm_thread_state64_t;
    use mach2::thread_status::ARM_THREAD_STATE64;
    let mut state = arm_thread_state64_t::new();
    let mut count = arm_thread_state64_t::count();
    // SAFETY: a suspended thread's general registers into a state of
    // their flavor's size.
    let read = unsafe {
        thread_get_state(
            port,
            ARM_THREAD_STATE64,
            (&raw mut state).cast(),
            &mut count,
        )
    };
    (read == KERN_SUCCESS).then_some((state.__pc, state.__fp))
}

#[cfg(target_arch = "x86_64")]
unsafe fn registers(port: mach2::port::mach_port_t) -> Option<(u64, u64)> {
    use mach2::structs::x86_thread_state64_t;
    use mach2::thread_status::x86_THREAD_STATE64;
    let mut state = x86_thread_state64_t::new();
    let mut count = x86_thread_state64_t::count();
    // SAFETY: a suspended thread's general registers into a state of
    // their flavor's size.
    let read = unsafe {
        thread_get_state(
            port,
            x86_THREAD_STATE64,
            (&raw mut state).cast(),
            &mut count,
        )
    };
    (read == KERN_SUCCESS).then_some((state.__rip, state.__rbp))
}
