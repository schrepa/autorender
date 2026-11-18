#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Cause(pub u32);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Carried(#[cfg(feature = "dev")] Cause);

#[cfg(feature = "dev")]
mod traced {
    use std::cell::Cell;
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering::Relaxed};

    use super::{Carried, Cause};
    use crate::Fact;

    static TRACING: AtomicBool = AtomicBool::new(false);
    static NEXT: AtomicU32 = AtomicU32::new(1);

    thread_local! {
        static CURRENT: Cell<u32> = const { Cell::new(0) };
    }

    pub fn cause() -> Cause {
        Cause(CURRENT.with(Cell::get))
    }

    pub fn carry() -> Carried {
        Carried(cause())
    }

    impl Carried {
        pub fn cause(self) -> Cause {
            self.0
        }
    }

    pub fn caused<R>(cause: Cause, work: impl FnOnce() -> R) -> R {
        let saved = CURRENT.with(|current| current.replace(cause.0));
        let done = work();
        CURRENT.with(|current| current.set(saved));
        done
    }

    pub fn adopt(cause: Cause) {
        if cause.0 != 0 {
            CURRENT.with(|current| current.set(cause.0));
        }
    }

    pub fn trace(on: bool) {
        TRACING.store(on, Relaxed);
    }

    pub fn tracing() -> bool {
        TRACING.load(Relaxed)
    }

    pub fn open(origin: &'static str, what: &str) -> Cause {
        open_noting(origin, what, &[])
    }

    pub fn open_noting(
        origin: &'static str,
        what: &str,
        noted: &[(&'static str, Fact<'_>)],
    ) -> Cause {
        if !tracing() {
            return Cause(0);
        }
        let opened = Cause(NEXT.fetch_add(1, Relaxed));
        let facts: Vec<(&'static str, Fact<'_>)> =
            std::iter::once(("what", Fact::Text(what))).chain(noted.iter().copied()).collect();
        caused(opened, || crate::event("cause", origin, &facts));
        opened
    }
}
#[cfg(feature = "dev")]
pub use traced::*;

#[cfg(not(feature = "dev"))]
mod untraced {
    use super::{Carried, Cause};
    use crate::Fact;

    #[inline(always)]
    pub fn carry() -> Carried {
        Carried()
    }

    impl Carried {
        #[inline(always)]
        pub fn cause(self) -> Cause {
            Cause(0)
        }
    }

    #[inline(always)]
    pub fn cause() -> Cause {
        Cause(0)
    }
    #[inline(always)]
    pub fn caused<R>(_: Cause, work: impl FnOnce() -> R) -> R {
        work()
    }
    #[inline(always)]
    pub fn adopt(_: Cause) {}
    #[inline(always)]
    pub fn trace(_: bool) {}
    #[inline(always)]
    pub fn tracing() -> bool {
        false
    }
    #[inline(always)]
    pub fn open(_: &'static str, _: &str) -> Cause {
        Cause(0)
    }
    #[inline(always)]
    pub fn open_noting(_: &'static str, _: &str, _: &[(&'static str, Fact<'_>)]) -> Cause {
        Cause(0)
    }
}
#[cfg(not(feature = "dev"))]
pub use untraced::*;
