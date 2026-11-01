use engine_render::renderer::*;

pub struct CasterDraw {
    pub index_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub lods: Vec<(u32, u32, f32)>,
}

impl CasterDraw {
    pub fn level_within(&self, error: f32) -> u8 {
        self.lods
            .iter()
            .take_while(|&&(_, _, lod_error)| lod_error <= error)
            .count() as u8
    }

    pub fn range(&self, level: u8) -> (u32, u32, i32) {
        match level.checked_sub(1) {
            Some(lod) => {
                let (first_index, index_count, _) = self.lods[lod as usize];
                (index_count, first_index, self.base_vertex)
            }
            None => (self.index_count, self.first_index, self.base_vertex),
        }
    }
}

impl Core {
    pub fn views_settled(&self) -> bool {
        self.gpu_presentation.stillness_settled()
    }

    pub fn view_still_frames(&self) -> (&wgpu::Buffer, u64) {
        self.gpu_presentation.still_frames()
    }

    pub fn caster_draw(&self, mesh: MeshHandle, levels: bool) -> CasterDraw {
        self.meshes.get(&mesh).map_or(
            CasterDraw {
                index_count: 0,
                first_index: 0,
                base_vertex: 0,
                lods: Vec::new(),
            },
            |mesh| {
                let draw = mesh.draw(&self.mesh_arena);
                let lods = if !levels {
                    Vec::new()
                } else {
                    mesh.lods
                        .iter()
                        .enumerate()
                        .map(|(level, &(offset, count, error))| {
                            let (offset, count) = mesh.caster_range(level + 1, (offset, count));
                            (draw.first_index + offset, count, error)
                        })
                        .collect()
                };
                let (offset, count) = mesh.caster_range(0, (0, draw.index_count));
                CasterDraw {
                    index_count: count,
                    first_index: draw.first_index + offset,
                    base_vertex: draw.base_vertex,
                    lods,
                }
            },
        )
    }
}
