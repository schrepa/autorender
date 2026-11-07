use crate::profiling::GpuProfiler;

pub(crate) struct RawDraw<'e> {
    encoding: &'e mut crate::encode::Run,
    part: u32,
    timer: &'e GpuProfiler,
}

impl<'e> RawDraw<'e> {
    pub(crate) fn new(
        encoding: &'e mut crate::encode::Run,
        part: u32,
        timer: &'e GpuProfiler,
    ) -> Self {
        Self {
            encoding,
            part,
            timer,
        }
    }

    pub(crate) fn reborrow(&mut self) -> RawDraw<'_> {
        RawDraw {
            encoding: &mut *self.encoding,
            part: self.part,
            timer: self.timer,
        }
    }

    pub(crate) fn member(&mut self, part: u32) -> RawDraw<'_> {
        RawDraw {
            encoding: &mut *self.encoding,
            part,
            timer: self.timer,
        }
    }

    pub(crate) fn part(&self) -> u32 {
        self.part
    }

    pub(crate) fn timer(&self) -> &'e GpuProfiler {
        self.timer
    }

    pub(crate) fn run(&mut self) -> &mut crate::encode::Run {
        self.encoding
    }
}
