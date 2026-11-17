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
        pub held: [u32; 2],
    }
}

#[derive(Resource, Default)]
pub struct SurfacePrograms {
    program: Vec<u32>,
    entries: Vec<SurfaceEntry>,
    sources: Vec<Option<Arc<SurfaceData>>>,
}

impl SurfacePrograms {
    pub fn program_bytes(&self) -> u64 {
        (4 * self.program.len().max(1)) as u64
    }

    pub fn materials(&self) -> &[SurfaceEntry] {
        &self.entries
    }

    pub fn material_bytes(&self) -> u64 {
        (std::mem::size_of::<SurfaceEntry>() * self.entries.len().max(1)) as u64
    }

    fn pieces(&self) -> impl Iterator<Item = (Option<&Arc<SurfaceData>>, u32)> + '_ {
        self.sources
            .iter()
            .zip(&self.entries)
            .map(|(data, entry)| (data.as_ref(), entry.base))
    }

    pub(super) fn made_from(&self, terrains: &Terrains) -> bool {
        drawn_surfaces(terrains)
            .map(|data| data.map(Arc::as_ptr))
            .eq(self
                .sources
                .iter()
                .map(|data| data.as_ref().map(Arc::as_ptr)))
    }

    pub(super) fn of(terrains: &Terrains) -> Self {
        let mut programs = Self::default();
        for data in drawn_surfaces(terrains) {
            programs.sources.push(data.cloned());
            let entry = match data {
                Some(data) => {
                    let base = programs.program.len() as u32;
                    programs.program.extend_from_slice(&data.words);
                    let (image_spacing, levels) = data.hydrology().unwrap_or((0.0, 0));
                    SurfaceEntry {
                        base,
                        gpu: 1,
                        levels: levels as u32,
                        image_spacing,
                        offset: data.offset(),
                        held: [0; 2],
                    }
                }
                None => SurfaceEntry::default(),
            };
            programs.entries.push(entry);
        }
        programs
    }
}

fn drawn_surfaces(terrains: &Terrains) -> impl Iterator<Item = Option<&Arc<SurfaceData>>> {
    terrains
        .entries
        .iter()
        .filter(|state| state.drawn)
        .map(TerrainState::programmed)
}

pub(crate) struct SurfaceProgramGpu {
    program: Buffer<Self>,
    materials: Buffer<Self>,
    written: Vec<(Option<Arc<SurfaceData>>, u32)>,
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
        for (index, (data, base)) in programs.pieces().enumerate() {
            let Some(data) = data else {
                continue;
            };
            let held = self.written.get(index).is_some_and(|(held, at)| {
                *at == base && held.as_ref().is_some_and(|held| Arc::ptr_eq(held, data))
            });
            if !held {
                cx.write(self.program, u64::from(base) * 4, &data.words);
            }
        }
        self.written = programs
            .pieces()
            .map(|(data, base)| (data.cloned(), base))
            .collect();
        if programs.is_changed() || materials == Allocation::Fresh {
            cx.write(self.materials, 0, programs.materials());
        }
    }
}
