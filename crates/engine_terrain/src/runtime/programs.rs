use super::*;
use engine_core::gpu_struct;
use engine_render::{Bind, Buffer, GpuPart, Install, Lifetime, Prepare};

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
}

impl SurfacePrograms {
    pub fn program(&self) -> &[u32] {
        &self.program
    }

    pub fn program_bytes(&self) -> u64 {
        (4 * self.program.len().max(1)) as u64
    }

    pub fn materials(&self) -> &[SurfaceEntry] {
        &self.entries
    }

    pub fn material_bytes(&self) -> u64 {
        (std::mem::size_of::<SurfaceEntry>() * self.entries.len().max(1)) as u64
    }

    fn of(terrains: &Terrains) -> Self {
        let mut programs = Self::default();
        for data in drawn_surfaces(terrains) {
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
        .map(|state| {
            (!state.materials)
                .then(|| state.surface.as_ref().and_then(|(_, data)| data.as_ref()))
                .flatten()
        })
}

pub(crate) fn publish_programs(
    terrains: Res<Terrains>,
    mut programs: ResMut<SurfacePrograms>,
    mut published: Local<Option<Vec<Option<Arc<SurfaceData>>>>>,
) {
    let unchanged = published.as_ref().is_some_and(|held| {
        held.len() == drawn_surfaces(&terrains).count()
            && held
                .iter()
                .zip(drawn_surfaces(&terrains))
                .all(|(held, now)| match (held, now) {
                    (Some(held), Some(now)) => Arc::ptr_eq(held, now),
                    (None, None) => true,
                    _ => false,
                })
    });
    if unchanged {
        return;
    }
    let made = SurfacePrograms::of(&terrains);
    if made.program != programs.program || made.entries != programs.entries {
        *programs = made;
    }
    *published = Some(
        drawn_surfaces(&terrains)
            .map(|data| data.cloned())
            .collect(),
    );
}

pub(crate) struct SurfaceProgramGpu {
    program: Buffer<Self>,
    materials: Buffer<Self>,
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
        Ok(Self { program, materials })
    }

    fn prepare(
        &mut self,
        programs: SystemParamItem<'_, '_, Self::Reads>,
        cx: &mut Prepare<'_, Self>,
    ) {
        if !programs.is_changed() {
            return;
        }
        cx.allocate_buffer(self.program, programs.program_bytes());
        cx.write(self.program, 0, programs.program());
        cx.allocate_buffer(self.materials, programs.material_bytes());
        cx.write(self.materials, 0, programs.materials());
    }
}
