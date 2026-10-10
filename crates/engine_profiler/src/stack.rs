//! How much of its stack the calling thread used at most, where the
//! operating system tells: Windows commits a thread's stack pages as it
//! first reaches them and keeps them, so the committed part is the deepest
//! the thread went. Native hosts report their main thread's, which has a
//! megabyte on Windows, so that growing startup frames show before they
//! overflow it. Unsafe code, as `pix.rs` holds: two system calls.

/// The calling thread's deepest stack use in bytes; `None` off Windows.
pub fn peak() -> Option<usize> {
    #[cfg(windows)]
    {
        windows::peak()
    }
    #[cfg(not(windows))]
    {
        None
    }
}

#[cfg(windows)]
mod windows {
    // `MEMORY_BASIC_INFORMATION` on 64-bit Windows.
    #[repr(C)]
    struct Region {
        base: usize,
        allocation_base: usize,
        allocation_protect: u32,
        partition: u16,
        size: usize,
        state: u32,
        protect: u32,
        kind: u32,
    }

    const MEM_COMMIT: u32 = 0x1000;

    #[allow(unsafe_code)]
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentThreadStackLimits(low: *mut usize, high: *mut usize);
        fn VirtualQuery(
            address: *const core::ffi::c_void,
            region: *mut Region,
            size: usize,
        ) -> usize;
    }

    /// The span from the stack's top down to its lowest committed page.
    #[allow(unsafe_code)]
    pub(super) fn peak() -> Option<usize> {
        let (mut low, mut high) = (0, 0);
        // SAFETY: two out parameters.
        unsafe { GetCurrentThreadStackLimits(&mut low, &mut high) };
        let mut address = low;
        while address < high {
            // SAFETY: an all-zero region is a valid value to fill.
            let mut region: Region = unsafe { std::mem::zeroed() };
            // SAFETY: queries the calling thread's own stack reservation
            // into `region`.
            let filled = unsafe {
                VirtualQuery(
                    address as *const _,
                    &mut region,
                    std::mem::size_of::<Region>(),
                )
            };
            if filled == 0 || region.size == 0 {
                return None;
            }
            if region.state == MEM_COMMIT {
                return Some(high - region.base);
            }
            address = region.base + region.size;
        }
        None
    }
}
