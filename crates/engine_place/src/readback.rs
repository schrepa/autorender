use bevy_ecs::component::Mutable;
use bevy_ecs::resource::Resource;

#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct FrameId(pub u64);

impl FrameId {
    #[must_use]
    pub fn next(self) -> Self {
        Self(self.0.saturating_add(1))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Readback<T> {
    pub value: T,
    pub written: FrameId,
}

pub trait Readable: Resource<Mutability = Mutable> + Default {
    fn arrived(&mut self, tag: u32, read: Readback<&[u8]>);

    fn lost(&mut self, _tag: u32) {}
}
