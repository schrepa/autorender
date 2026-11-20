use std::ops::Range;

use engine_core::{Asset, Material, MeshFile, PassBit};

use crate::DrawOrder;
use crate::drawn::{Batches, KindBatch};
use crate::profiling::GpuProfiler;

#[derive(Clone, Copy, Debug)]
pub struct BatchOf<'a> {
    pub geometry: Option<Asset<MeshFile>>,
    pub material: Option<Asset<Material>>,
    pub instances: &'a Range<u32>,
}

impl<'a> BatchOf<'a> {
    pub(crate) fn of(batch: &'a KindBatch) -> Self {
        Self {
            geometry: batch.geometry,
            material: batch.material,
            instances: &batch.instances,
        }
    }
}

pub(crate) struct RawDraw<'e> {
    encoding: &'e mut crate::encode::Run,
    part: u32,
    timer: &'e GpuProfiler,
    batches: Option<(&'e mut Batches, Option<PassBit>)>,
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
            batches: None,
        }
    }

    pub(crate) fn reborrow(&mut self) -> RawDraw<'_> {
        RawDraw {
            encoding: &mut *self.encoding,
            part: self.part,
            timer: self.timer,
            batches: (self.batches.as_mut()).map(|(batches, pass)| (&mut **batches, *pass)),
        }
    }

    pub(crate) fn member<'m>(
        &'m mut self,
        part: u32,
        batches: (&'m mut Batches, Option<PassBit>),
    ) -> RawDraw<'m> {
        RawDraw {
            encoding: &mut *self.encoding,
            part,
            timer: self.timer,
            batches: Some(batches),
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

    pub(crate) fn batches(&mut self, (handle, vertices, order): (u32, Range<u32>, DrawOrder)) {
        let Some((batches, pass)) = self.batches.as_mut() else {
            return;
        };
        let drawn = (&mut **batches, *pass);
        (self.encoding).batches((self.part, self.timer), (handle, vertices, order), drawn);
    }
}
