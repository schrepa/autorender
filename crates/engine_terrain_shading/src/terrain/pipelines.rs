use super::atlas::TileAtlas;
use super::patches::{PatchDraws, patch_layout};
use super::tables::TerrainTables;
use super::water_blocks::WaterBlocks;
use super::*;
use crate::shaders::entry::terrain as draw;

type GroupKey = (u64, u64, u64, u64, u64, u64, u64, u64, u64);

const SHADOW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth16Unorm;

pub(super) struct GroupSources<'a> {
    pub(super) atlas: &'a TileAtlas,
    pub(super) tables: &'a TerrainTables,
    pub(super) patches: &'a PatchDraws,
    pub(super) surface: &'a SurfaceTrees,
    pub(super) maps: &'a TerrainMaps,
    pub(super) sky: &'a SkyMaps,
    pub(super) macro_maps: &'a MacroMaps,
    pub(super) water: &'a WaterBlocks,
}

pub(crate) struct TerrainPipelines {
    pub(super) layout: engine_render::pipelines::GroupLayout,
    group: Cached<GroupKey, wgpu::BindGroup>,
    surface_group: Cached<GroupKey, wgpu::BindGroup>,
    pipelines: HashMap<(Targets, bool), wgpu::RenderPipeline>,
    pub(crate) deferred_ids: HashMap<bool, wgpu::RenderPipeline>,
    pub(crate) resolves: HashMap<Targets, wgpu::RenderPipeline>,
    pub(crate) resolve_layout: engine_render::pipelines::GroupLayout,
    view_pipelines: HashMap<bool, wgpu::RenderPipeline>,
    shadow: Option<wgpu::RenderPipeline>,
    shading: Vec<(&'static str, f64)>,
}

impl TerrainPipelines {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        Self {
            layout: engine_render::pipelines::layout(device, &TERRAIN),
            group: Cached::default(),
            surface_group: Cached::default(),
            pipelines: HashMap::default(),
            deferred_ids: HashMap::default(),
            resolves: HashMap::default(),
            resolve_layout: engine_render::pipelines::layout(
                device,
                &crate::bindings::TERRAIN_RESOLVE,
            ),
            view_pipelines: HashMap::default(),
            shadow: None,
            shading: Vec::new(),
        }
    }

    pub(crate) fn specialize(&mut self, shading: Vec<(&'static str, f64)>) {
        if self.shading != shading {
            self.shading = shading;
            self.pipelines.clear();
            self.resolves.clear();
            self.view_pipelines.clear();
        }
    }

    pub(super) fn group(&self) -> Option<&wgpu::BindGroup> {
        self.group.get()
    }

    pub(super) fn surface_group(&self) -> Option<&wgpu::BindGroup> {
        self.surface_group.get()
    }

    pub(super) fn ensure_group(&mut self, device: &wgpu::Device, sources: &GroupSources<'_>) {
        let (Some((heights, normals, control)), Some((ground_errors, water_errors))) =
            (sources.atlas.views(), sources.atlas.errors())
        else {
            return;
        };
        let (tables, maps, sky) = (sources.tables, sources.maps, sources.sky);
        let leaves = sources.surface.leaves();
        let key = (
            sources.atlas.generation,
            tables.uniform_generation,
            tables.holes_generation,
            tables.directory_generation,
            maps.views_generation,
            sky.generation(),
            sources.macro_maps.generation(),
            leaves.generation(),
            sources.water.generation,
        );
        let build =
            |label: &str, material_maps: &[wgpu::TextureView; 2], surface_leaves: &wgpu::Buffer| {
                engine_render::pipelines::group(
                    device,
                    label,
                    &TERRAIN,
                    &self.layout,
                    &[
                        ("terrain_heights", engine_render::pipelines::view(heights)),
                        (
                            "terrain_view",
                            wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &sources.patches.views,
                                offset: 0,
                                size: wgpu::BufferSize::new(size_of::<ViewUniform>() as u64),
                            }),
                        ),
                        (
                            "terrain",
                            wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &tables.uniforms,
                                offset: 0,
                                size: wgpu::BufferSize::new(size_of::<TerrainUniform>() as u64),
                            }),
                        ),
                        engine_render::buffers::entry("terrain_holes", &tables.holes),
                        ("terrain_normals", engine_render::pipelines::view(normals)),
                        (
                            "terrain_sky_sampler",
                            engine_render::pipelines::sampler(&sources.atlas.sky_sampler),
                        ),
                        engine_render::buffers::entry("terrain_directory", &tables.directory),
                        ("terrain_control", engine_render::pipelines::view(control)),
                        (
                            "terrain_palette",
                            wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &tables.palette,
                                offset: 0,
                                size: None,
                            }),
                        ),
                        ("terrain_albedo", engine_render::pipelines::view(&material_maps[0])),
                        (
                            "terrain_material_data",
                            engine_render::pipelines::view(&material_maps[1]),
                        ),
                        (
                            "terrain_material_sampler",
                            engine_render::pipelines::sampler(&maps.sampler),
                        ),
                        ("terrain_sky", engine_render::pipelines::view(sky.view())),
                        (
                            "terrain_macro",
                            engine_render::pipelines::view(sources.macro_maps.view()),
                        ),
                        (
                            "terrain_control_sampler",
                            engine_render::pipelines::sampler(&sources.atlas.control_sampler),
                        ),
                        engine_render::buffers::entry("terrain_surface_leaves", surface_leaves),
                        (
                            "water_surface",
                            engine_render::pipelines::view(sources.water.surface_view()),
                        ),
                        (
                            "water_info",
                            engine_render::pipelines::view(sources.water.info_view()),
                        ),
                        ("terrain_errors", engine_render::pipelines::view(ground_errors)),
                        ("water_errors", engine_render::pipelines::view(water_errors)),
                    ],
                )
            };
        if self.group.stale(&key) {
            let group = build("engine/terrain-group", &maps.views, leaves);
            let surface = build(
                "engine/terrain-surface-group",
                &maps.views,
                &sources.surface.stand_in,
            );
            self.group.set(key, group);
            self.surface_group.set(key, surface);
        }
    }

    pub(crate) fn pipeline(&self, targets: Targets, holes: bool) -> Option<&wgpu::RenderPipeline> {
        self.pipelines.get(&(targets, holes))
    }

    pub(crate) fn view_pipeline(&self, holes: bool) -> Option<&wgpu::RenderPipeline> {
        self.view_pipelines.get(&holes)
    }

    pub(crate) fn shadow_pipeline(&self) -> Option<&wgpu::RenderPipeline> {
        self.shadow.as_ref()
    }

    pub(crate) fn ensure_pipeline(
        &mut self,
        device: &wgpu::Device,
        (targets, holes): (Targets, bool),
        canvas: wgpu::TextureFormat,
        depth: wgpu::TextureFormat,
    ) {
        if self.pipelines.contains_key(&(targets, holes)) {
            return;
        }
        let layout = pipeline_layout(device);
        let (vertex, fragment) = crate::entries::terrain(Some(targets));
        let color_targets = targets.color_targets(canvas);
        let pipeline = geometry(
            device,
            &layout,
            (vertex, fragment, holes, Some(&self.shading[..])),
            &color_targets,
            depth,
            wgpu::FrontFace::Ccw,
        );
        self.pipelines.insert((targets, holes), pipeline);
        self.ensure_deferred(device, (targets, holes), depth);
    }

    fn ensure_deferred(
        &mut self,
        device: &wgpu::Device,
        (targets, holes): (Targets, bool),
        depth: wgpu::TextureFormat,
    ) {
        if let std::collections::hash_map::Entry::Vacant(slot) = self.deferred_ids.entry(holes) {
            let layout = engine_render::pipelines::layout_of(
                device,
                &crate::pipeline_layouts::TERRAIN_PIPELINES,
            );
            let ids = [Some(wgpu::ColorTargetState::from(
                wgpu::TextureFormat::Rg32Uint,
            ))];
            let pipeline = geometry(
                device,
                &layout,
                (draw::vs_terrain, draw::fs_terrain_ids, holes, None),
                &ids,
                depth,
                wgpu::FrontFace::Ccw,
            );
            slot.insert(pipeline);
        }
        let fragment = match targets {
            Targets::Post => draw::fs_terrain_resolve_post,
            Targets::Temporal => draw::fs_terrain_resolve_temporal,
            Targets::Canvas | Targets::Debug => return,
        };
        if self.resolves.contains_key(&targets) {
            return;
        }
        let layout = engine_render::pipelines::layout_of(
            device,
            &crate::pipeline_layouts::LIT_TERRAIN_RESOLVE_PIPELINES,
        );
        let color_targets = targets.color_targets(wgpu::TextureFormat::Bgra8Unorm);
        let pipeline = engine_render::pipelines::fullscreen(
            device,
            &engine_render::pipelines::Fullscreen {
                label: "engine/terrain-resolve",
                layout: &layout,
                vertex: draw::vs_fullscreen,
                fragment,
                constants: &self.shading,
                targets: &color_targets,
                depth: Some(wgpu::DepthStencilState {
                    format: depth,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Always),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
            },
        );
        self.resolves.insert(targets, pipeline);
    }

    pub(crate) fn ensure_view_pipeline(
        &mut self,
        device: &wgpu::Device,
        (holes, mirrored): (bool, bool),
        format: wgpu::TextureFormat,
        depth: wgpu::TextureFormat,
    ) {
        if self.view_pipelines.contains_key(&holes) {
            return;
        }
        let layout = pipeline_layout(device);
        let (vertex, fragment) = crate::entries::terrain(None);
        let front_face = if mirrored {
            wgpu::FrontFace::Cw
        } else {
            wgpu::FrontFace::Ccw
        };
        let pipeline = geometry(
            device,
            &layout,
            (vertex, fragment, holes, Some(&self.shading[..])),
            &[Some(wgpu::ColorTargetState::from(format))],
            depth,
            front_face,
        );
        self.view_pipelines.insert(holes, pipeline);
    }

    pub(crate) fn ensure_shadow_pipeline(&mut self, device: &wgpu::Device) {
        if self.shadow.is_some() {
            return;
        }
        let layout = engine_render::pipelines::layout_of(
            device,
            &crate::pipeline_layouts::TERRAIN_CASTER_PIPELINES,
        );
        self.shadow = Some(engine_render::pipelines::geometry(
            device,
            &engine_render::pipelines::Geometry {
                label: "engine/terrain-shadow",
                layout: &layout,
                vertex_constants: &[],
                vertex: crate::entries::TERRAIN_SHADOW_VERTEX,
                buffers: &[Some(patch_layout())],
                primitive: wgpu::PrimitiveState::default(),
                depth: Some(wgpu::DepthStencilState {
                    format: SHADOW_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: Default::default(),
                    bias: wgpu::DepthBiasState {
                        constant: 2,
                        slope_scale: 2.5,
                        clamp: 0.0,
                    },
                }),
                fragment: None,
            },
        ));
    }
}

fn pipeline_layout(device: &wgpu::Device) -> wgpu::PipelineLayout {
    engine_render::pipelines::layout_of(device, &crate::pipeline_layouts::LIT_TERRAIN_PIPELINES)
}

fn geometry(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    (vertex, fragment, holes, shading): (
        engine_render::shaders::Entry,
        engine_render::shaders::Entry,
        bool,
        Option<&[(&'static str, f64)]>,
    ),
    targets: &[Option<wgpu::ColorTargetState>],
    depth: wgpu::TextureFormat,
    front_face: wgpu::FrontFace,
) -> wgpu::RenderPipeline {
    let holes: &[(&str, f64)] = if holes {
        &[("TERRAIN_HOLES", 1.0)]
    } else {
        &[]
    };
    let constants: Vec<(&str, f64)> = shading.map_or_else(
        || holes.to_vec(),
        |shading| holes.iter().chain(shading).copied().collect(),
    );
    let constants = &constants[..];
    engine_render::pipelines::geometry(
        device,
        &engine_render::pipelines::Geometry {
            label: "engine/terrain",
            layout,
            vertex_constants: &[],
            vertex,
            buffers: &[],
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth: Some(wgpu::DepthStencilState {
                format: depth,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Greater),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            fragment: Some(engine_render::pipelines::Fragment {
                entry: fragment,
                constants,
                targets,
            }),
        },
    )
}
