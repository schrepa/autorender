use std::hash::{DefaultHasher, Hash, Hasher};

#[cfg(any(feature = "dev", feature = "profiling"))]
use serde_json::{Value, json};

#[derive(Default)]
pub(crate) struct Layouts {
    pub(crate) groups: Vec<(Vec<wgpu::BindGroupLayoutEntry>, wgpu::BindGroupLayout)>,
    pub(crate) pipelines: Vec<(Vec<u32>, wgpu::PipelineLayout, u32)>,
}

impl Layouts {
    pub(crate) fn group(
        &mut self,
        device: &wgpu::Device,
        entries: Vec<wgpu::BindGroupLayoutEntry>,
    ) -> wgpu::BindGroupLayout {
        if let Some((_, known)) = self.groups.iter().find(|(known, _)| *known == entries) {
            return known.clone();
        }
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("engine/group"),
            entries: &entries,
        });
        self.groups.push((entries, layout.clone()));
        layout
    }

    #[expect(
        clippy::disallowed_methods,
        reason = "the machine lays out every pipeline it builds, once per distinct layout"
    )]
    pub(crate) fn pipeline(
        &mut self,
        device: &wgpu::Device,
        groups: &[&wgpu::BindGroupLayout],
    ) -> (wgpu::PipelineLayout, u64) {
        let ids: Vec<u32> = (groups.iter())
            .map(|&layout| {
                (self.groups.iter())
                    .position(|(_, known)| known == layout)
                    .map_or(u32::MAX, |at| at as u32)
            })
            .collect();
        let mut hasher = DefaultHasher::new();
        for &id in &ids {
            self.groups.get(id as usize).map(|(entries, _)| entries).hash(&mut hasher);
        }
        let key = hasher.finish();
        if let Some(found) = self.pipelines.iter_mut().find(|(known, ..)| *known == ids) {
            found.2 += 1;
            return (found.1.clone(), key);
        }
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("engine/part-pipeline"),
            bind_group_layouts: &groups.iter().map(|&layout| Some(layout)).collect::<Vec<_>>(),
            immediate_size: 0,
        });
        self.pipelines.push((ids, layout.clone(), 1));
        (layout, key)
    }

    #[cfg(any(feature = "dev", feature = "profiling"))]
    pub(crate) fn facts(&self) -> Value {
        let groups: Vec<Value> = (self.groups.iter().enumerate())
            .map(|(at, (entries, _))| {
                let shared: u32 = (self.pipelines.iter())
                    .filter(|(ids, ..)| ids.contains(&(at as u32)))
                    .map(|(.., shared)| *shared)
                    .sum();
                json!({"entries": entries, "shared": shared})
            })
            .collect();
        let pipelines: Vec<Value> = (self.pipelines.iter())
            .map(|(ids, _, shared)| json!({"groups": ids, "shared": shared}))
            .collect();
        json!({"bind_group_layouts": groups, "pipeline_layouts": pipelines})
    }
}

#[cfg(any(feature = "dev", feature = "profiling"))]
pub(crate) fn layout_names(world: &mut bevy_ecs::world::World) {
    use engine_registry::{Kind, NameEntry, register, unavailable};
    register(
        world,
        NameEntry::read("gpu.layouts", Kind::State, |reach, _| {
            Ok(reach.part::<crate::Renderer>().map_or_else(
                || unavailable("no_renderer"),
                |renderer| {
                    renderer
                        .layouts()
                        .map_or_else(|| unavailable("not_installed"), Layouts::facts)
                },
            ))
        }),
    );
}
