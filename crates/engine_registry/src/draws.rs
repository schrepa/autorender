use std::any::Any;

use engine_place::DrawPoint;

pub struct DrawMember<D: DrawPoint> {
    pub part: u32,
    pub layer: D::Layer,
    pub call: Box<dyn Fn(&dyn Any, &D::Args, &mut RawDraw<'_>) -> u32 + Send + Sync>,
}

pub struct RawDraw<'e> {
    encoding: &'e mut (dyn Any + 'static),
    part: u32,
}

impl<'e> RawDraw<'e> {
    pub fn new(encoding: &'e mut (dyn Any + 'static), part: u32) -> Self {
        Self { encoding, part }
    }

    pub fn reborrow(&mut self) -> RawDraw<'_> {
        RawDraw {
            encoding: &mut *self.encoding,
            part: self.part,
        }
    }

    pub fn member(&mut self, part: u32) -> RawDraw<'_> {
        RawDraw {
            encoding: &mut *self.encoding,
            part,
        }
    }

    pub fn part(&self) -> u32 {
        self.part
    }

    pub fn encoding<T: Any>(&mut self) -> Option<&mut T> {
        self.encoding.downcast_mut::<T>()
    }
}
