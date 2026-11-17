use super::*;
use engine_core::gpu_struct;
use engine_render::{Allocation, Bind, Buffer, GpuPart, Install, Lifetime, Prepare};

gpu_struct! {
    pub struct SurfaceEntry {
        pub base: u32,
        pub gpu: u32,
        pub levels: u32,
        pub image_spacing: f32,
        pub offset: [f32; 2],
        pub blend: f32,
        pub faded: u32,
        pub fade: [f32; 4],
    }
}

#[derive(Resource, Default)]
pub struct SurfacePrograms {
    words: usize,
    entries: Vec<SurfaceEntry>,
    sources: Vec<(Option<Arc<SurfaceData>>, bool)>,
    pieces: Vec<(Arc<SurfaceData>, u32)>,
}

impl SurfacePrograms {
    pub fn program_bytes(&self) -> u64 {
        (4 * self.words.max(1)) as u64
    }

    pub fn materials(&self) -> &[SurfaceEntry] {
        &self.entries
    }

    pub fn material_bytes(&self) -> u64 {
        (std::mem::size_of::<SurfaceEntry>() * self.entries.len().max(1)) as u64
    }

    pub(super) fn made_from(&self, terrains: &Terrains) -> bool {
        surfaces(terrains)
            .map(|(data, drawn)| (data.map(Arc::as_ptr), drawn))
            .eq(self
                .sources
                .iter()
                .map(|(data, drawn)| (data.as_ref().map(Arc::as_ptr), *drawn)))
    }

    pub(super) fn of(terrains: &Terrains) -> Self {
        let mut programs = Self::default();
        for (data, drawn) in surfaces(terrains) {
            programs.sources.push((data.cloned(), drawn));
            let Some(data) = data else {
                if drawn {
                    programs.entries.push(SurfaceEntry::default());
                }
                continue;
            };
            let shared = (programs.pieces.iter())
                .find(|(held, _)| Arc::ptr_eq(held, data) || held.words == data.words);
            let base = match shared {
                Some(&(_, base)) => base,
                None => {
                    let base = programs.words as u32;
                    programs.words += data.words.len();
                    programs.pieces.push((Arc::clone(data), base));
                    base
                }
            };
            if drawn {
                let (image_spacing, levels) = data.hydrology().unwrap_or((0.0, 0));
                let (fade, blend) = data.fade().unwrap_or_default();
                programs.entries.push(SurfaceEntry {
                    base,
                    gpu: 1,
                    levels: levels as u32,
                    image_spacing,
                    offset: data.offset(),
                    blend,
                    faded: u32::from(data.fade().is_some()),
                    fade,
                });
            }
        }
        programs
    }
}

fn surfaces(terrains: &Terrains) -> impl Iterator<Item = (Option<&Arc<SurfaceData>>, bool)> {
    terrains
        .entries
        .iter()
        .map(|state| (state.programmed(), state.drawn))
}

pub(crate) struct SurfaceProgramGpu {
    program: Buffer<Self>,
    materials: Buffer<Self>,
    written: Vec<(Arc<SurfaceData>, u32)>,
}

impl GpuPart for SurfaceProgramGpu {
    type Reads = Res<'static, SurfacePrograms>;

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        use crate::shaders::terrain::faces::surface as face;
        let (program, materials) = (
            cx.buffer("surface_program", Lifetime::Kept),
            cx.buffer("surface_materials", Lifetime::Kept),
        );
        cx.frame(face::SURFACE_PROGRAM, Bind::Buffer(program))?;
        cx.frame(face::SURFACE_MATERIALS, Bind::Buffer(materials))?;
        Ok(Self {
            program,
            materials,
            written: Vec::new(),
        })
    }

    fn prepare(
        &mut self,
        programs: SystemParamItem<'_, '_, Self::Reads>,
        cx: &mut Prepare<'_, Self>,
    ) {
        let program = cx.allocate_buffer(self.program, programs.program_bytes());
        let materials = cx.allocate_buffer(self.materials, programs.material_bytes());
        if program == Allocation::Fresh {
            self.written.clear();
        }
        for (data, base) in &programs.pieces {
            let held = self
                .written
                .iter()
                .any(|(held, at)| at == base && Arc::ptr_eq(held, data));
            if !held {
                cx.write(self.program, u64::from(*base) * 4, &data.words);
            }
        }
        self.written.clone_from(&programs.pieces);
        if programs.is_changed() || materials == Allocation::Fresh {
            cx.write(self.materials, 0, programs.materials());
        }
    }
}
