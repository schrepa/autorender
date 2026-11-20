use std::any::Any;
use std::sync::Arc;

use engine_core::{Bricks, DrawKind, PassEntry, PassName};
use engine_registry::RecordExt;

use crate::registration;
use crate::{Draw, GpuPart, Scene};

pub trait RenderBricks {
    fn part<P: GpuPart>(&mut self) -> &mut Self;
    fn pass<N: PassName>(&mut self) -> &mut Self;
    fn draws<P: GpuPart>(
        &mut self,
        kind: DrawKind,
        draw: fn(&P, &mut Draw<'_, P, Scene>),
    ) -> &mut Self;
    fn generated(&mut self, wgsl: &'static [&'static str]) -> &mut Self;
    fn entry<T: Any + Send + Sync>(&mut self, entry: T) -> &mut Self;
}

impl RenderBricks for Bricks<'_> {
    fn part<P: GpuPart>(&mut self) -> &mut Self {
        self.entry(registration::PartRecord::of::<P>())
    }

    fn pass<N: PassName>(&mut self) -> &mut Self {
        let record = self.record();
        if !record
            .render_entries::<PassEntry>()
            .any(|entry| entry.name == N::NAME)
        {
            record.render.push(Arc::new(PassEntry::of::<N>()));
        }
        self
    }

    fn draws<P: GpuPart>(
        &mut self,
        kind: DrawKind,
        draw: fn(&P, &mut Draw<'_, P, Scene>),
    ) -> &mut Self {
        self.part::<P>()
            .entry(registration::KindDraw::of::<P>(kind, draw))
    }

    fn generated(&mut self, wgsl: &'static [&'static str]) -> &mut Self {
        self.entry(registration::Generated(wgsl))
    }

    fn entry<T: Any + Send + Sync>(&mut self, entry: T) -> &mut Self {
        self.record().render.push(Arc::new(entry));
        self
    }
}
