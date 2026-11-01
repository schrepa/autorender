use std::any::Any;

use engine_core::DrawPoint;

pub(crate) type DrawCall<D> =
    Box<dyn Fn(&dyn Any, &<D as DrawPoint>::Args, &mut RawDraw<'_>) -> u32 + Send + Sync>;

pub(crate) struct DrawMember<D: DrawPoint> {
    pub(crate) part: u32,
    pub(crate) layer: D::Layer,
    pub(crate) call: DrawCall<D>,
}

pub(crate) struct RawDraw<'e> {
    encoding: &'e mut crate::encode::Run,
    part: u32,
}

impl<'e> RawDraw<'e> {
    pub(crate) fn new(encoding: &'e mut crate::encode::Run, part: u32) -> Self {
        Self { encoding, part }
    }

    pub(crate) fn reborrow(&mut self) -> RawDraw<'_> {
        RawDraw {
            encoding: &mut *self.encoding,
            part: self.part,
        }
    }

    pub(crate) fn member(&mut self, part: u32) -> RawDraw<'_> {
        RawDraw {
            encoding: &mut *self.encoding,
            part,
        }
    }

    pub(crate) fn part(&self) -> u32 {
        self.part
    }

    pub(crate) fn run(&mut self) -> &mut crate::encode::Run {
        self.encoding
    }
}
