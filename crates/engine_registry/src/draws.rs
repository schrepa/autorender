use std::any::Any;
use std::marker::PhantomData;

use engine_place::DrawPoint;

pub struct DrawMember<D: DrawPoint> {
    pub part: u32,
    pub layer: D::Layer,
    pub call: Box<dyn Fn(&dyn Any, &D::Args, &mut RawDraw<'_>) -> u32 + Send + Sync>,
}

#[derive(Default)]
pub struct InstallCore {
    pub part: u32,
}

#[derive(Default)]
pub struct FrameCore {
    pub frame: u64,
}

#[derive(Default)]
pub struct EncodeCore {
    pub pass: u32,
}

pub struct RawDraw<'e> {
    pub pass: PhantomData<&'e mut ()>,
}
