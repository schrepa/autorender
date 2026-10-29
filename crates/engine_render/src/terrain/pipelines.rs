use super::atlas::TileAtlas;
use super::patches::{PatchDraws, patch_layout};
use super::tables::TerrainTables;
use super::water_blocks::WaterBlocks;
use super::*;
use crate::scene_shading::SceneShading;
use crate::shaders::entry::meshes;

type GroupKey = (u64, u64, u64, u64, u64, u64, u64, u64, u64);

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
    pub(super) layout: crate::pipelines::GroupLayout,
    group: Cached<GroupKey, wgpu::BindGroup>,
    surface_group: Cached<GroupKey, wgpu::BindGroup>,
    pipelines: HashMap<(Targets, bool, bool), wgpu::RenderPipeline>,
    pub(crate) deferred_ids: HashMap<bool, wgpu::RenderPipeline>,
    pub(crate) resolves: HashMap<(Targets, bool), wgpu::RenderPipeline>,
    pub(crate) resolve_layout: crate::pipelines::GroupLayout,
    probe_pipelines: HashMap<(bool, bool), wgpu::RenderPipeline>,
    shadow: Option<wgpu::RenderPipeline>,
    shading: SceneShading,
}

impl TerrainPipelines {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        Self {
            layout: crate::pipelines::layout(device, &TERRAIN),
            group: Cached::default(),
            surface_group: Cached::default(),
            pipelines: HashMap::default(),
            deferred_ids: HashMap::default(),
            resolves: HashMap::default(),
            resolve_layout: crate::pipelines::layout(device, &crate::bindings::TERRAIN_RESOLVE),
            probe_pipelines: HashMap::default(),
            shadow: None,
            shading: SceneShading::ALL,
        }
    }

    pub(crate) fn specialize(&mut self, shading: SceneShading) {
        if self.shading != shading {
            self.shading = shading;
            self.pipelines.clear();
            self.resolves.clear();
            self.probe_pipelines.clear();
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
                crate::pipelines::group(
                    device,
                    label,
                    &TERRAIN,
                    &self.layout,
                    &[
                        ("terrain_heights", crate::pipelines::view(heights)),
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
                        crate::buffers::entry("terrain_holes", &tables.holes),
                        ("terrain_normals", crate::pipelines::view(normals)),
                        (
                            "terrain_sky_sampler",
                            crate::pipelines::sampler(&sources.atlas.sky_sampler),
                        ),
                        crate::buffers::entry("terrain_directory", &tables.directory),
                        ("terrain_control", crate::pipelines::view(control)),
                        (
                            "terrain_palette",
                            wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &tables.palette,
                                offset: 0,
                                size: None,
                            }),
                        ),
                        ("terrain_albedo", crate::pipelines::view(&material_maps[0])),
                        (
                            "terrain_material_data",
                            crate::pipelines::view(&material_maps[1]),
                        ),
                        (
                            "terrain_material_sampler",
                            crate::pipelines::sampler(&maps.sampler),
                        ),
                        ("terrain_sky", crate::pipelines::view(sky.view())),
                        (
                            "terrain_macro",
                            crate::pipelines::view(sources.macro_maps.view()),
                        ),
                        (
                            "terrain_control_sampler",
                            crate::pipelines::sampler(&sources.atlas.control_sampler),
                        ),
                        crate::buffers::entry("terrain_surface_leaves", surface_leaves),
                        (
                            "water_surface",
                            crate::pipelines::view(sources.water.surface_view()),
                        ),
                        (
                            "water_info",
                            crate::pipelines::view(sources.water.info_view()),
                        ),
                        ("terrain_errors", crate::pipelines::view(ground_errors)),
                        ("water_errors", crate::pipelines::view(water_errors)),
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

    pub(crate) fn pipeline(
        &self,
        targets: Targets,
        lit: bool,
        holes: bool,
    ) -> Option<&wgpu::RenderPipeline> {
        self.pipelines.get(&(targets, lit, holes))
    }

    pub(crate) fn probe_pipeline(&self, lit: bool, holes: bool) -> Option<&wgpu::RenderPipeline> {
        self.probe_pipelines.get(&(lit, holes))
    }

    pub(crate) fn shadow_pipeline(&self) -> Option<&wgpu::RenderPipeline> {
        self.shadow.as_ref()
    }

    pub(crate) fn ensure_pipeline(
        &mut self,
        device: &wgpu::Device,
        (targets, lit, holes): (Targets, bool, bool),
        canvas: wgpu::TextureFormat,
        depth: wgpu::TextureFormat,
    ) {
        if self.pipelines.contains_key(&(targets, lit, holes)) {
            return;
        }
        let layout = pipeline_layout(device, lit);
        let (vertex, fragment) = crate::entries::terrain(Some(targets), lit);
        let color_targets = targets.color_targets(canvas);
        let pipeline = geometry(
            device,
            &layout,
            (vertex, fragment, holes, Some(self.shading)),
            &color_targets,
            depth,
            wgpu::FrontFace::Ccw,
        );
        self.pipelines.insert((targets, lit, holes), pipeline);
        self.ensure_deferred(device, (targets, lit, holes), depth);
    }

    fn ensure_deferred(
        &mut self,
        device: &wgpu::Device,
        (targets, lit, holes): (Targets, bool, bool),
        depth: wgpu::TextureFormat,
    ) {
        if !self.deferred_ids.contains_key(&holes) {
            let layout =
                crate::pipelines::layout_of(device, &crate::pipeline_layouts::TERRAIN_PIPELINES);
            let ids = [Some(wgpu::ColorTargetState::from(
                wgpu::TextureFormat::Rg32Uint,
            ))];
            let pipeline = geometry(
                device,
                &layout,
                (meshes::vs_terrain, meshes::fs_terrain_ids, holes, None),
                &ids,
                depth,
                wgpu::FrontFace::Ccw,
            );
            self.deferred_ids.insert(holes, pipeline);
        }
        let fragment = match (targets, lit) {
            (Targets::Post, false) => meshes::fs_terrain_resolve_post,
            (Targets::Post, true) => meshes::fs_terrain_resolve_post_lit,
            (Targets::Temporal, false) => meshes::fs_terrain_resolve_temporal,
            (Targets::Temporal, true) => meshes::fs_terrain_resolve_temporal_lit,
            (Targets::Canvas | Targets::Debug, _) => return,
        };
        if self.resolves.contains_key(&(targets, lit)) {
            return;
        }
        let layout = crate::pipelines::layout_of(
            device,
            if lit {
                &crate::pipeline_layouts::LIT_TERRAIN_RESOLVE_PIPELINES
            } else {
                &crate::pipeline_layouts::TERRAIN_RESOLVE_PIPELINES
            },
        );
        let color_targets = targets.color_targets(wgpu::TextureFormat::Bgra8Unorm);
        let pipeline = crate::pipelines::fullscreen(
            device,
            &crate::pipelines::Fullscreen {
                label: "engine/terrain-resolve",
                layout: &layout,
                vertex: meshes::vs_fullscreen,
                fragment,
                constants: &self.shading.constants(),
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
        self.resolves.insert((targets, lit), pipeline);
    }

    pub(crate) fn ensure_probe_pipeline(
        &mut self,
        device: &wgpu::Device,
        (lit, holes): (bool, bool),
        format: wgpu::TextureFormat,
        depth: wgpu::TextureFormat,
    ) {
        if self.probe_pipelines.contains_key(&(lit, holes)) {
            return;
        }
        let layout = pipeline_layout(device, lit);
        let (vertex, fragment) = crate::entries::terrain(None, lit);
        let pipeline = geometry(
            device,
            &layout,
            (vertex, fragment, holes, Some(self.shading)),
            &[Some(wgpu::ColorTargetState::from(format))],
            depth,
            wgpu::FrontFace::Cw,
        );
        self.probe_pipelines.insert((lit, holes), pipeline);
    }

    pub(crate) fn ensure_shadow_pipeline(&mut self, device: &wgpu::Device) {
        if self.shadow.is_some() {
            return;
        }
        let layout =
            crate::pipelines::layout_of(device, &crate::pipeline_layouts::TERRAIN_CASTER_PIPELINES);
        self.shadow = Some(crate::pipelines::geometry(
            device,
            &crate::pipelines::Geometry {
                label: "engine/terrain-shadow",
                layout: &layout,
                vertex_constants: &[],
                vertex: crate::entries::TERRAIN_SHADOW_VERTEX,
                buffers: &[Some(patch_layout())],
                primitive: wgpu::PrimitiveState::default(),
                depth: Some(wgpu::DepthStencilState {
                    format: crate::shadows::SHADOW_FORMAT,
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

fn pipeline_layout(device: &wgpu::Device, lit: bool) -> wgpu::PipelineLayout {
    crate::pipelines::layout_of(
        device,
        if lit {
            &crate::pipeline_layouts::LIT_TERRAIN_PIPELINES
        } else {
            &crate::pipeline_layouts::TERRAIN_PIPELINES
        },
    )
}

fn geometry(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    (vertex, fragment, holes, shading): (
        crate::shaders::Entry,
        crate::shaders::Entry,
        bool,
        Option<SceneShading>,
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
    let constants = shading.map_or_else(|| holes.to_vec(), |shading| shading.with(holes));
    let constants = &constants[..];
    crate::pipelines::geometry(
        device,
        &crate::pipelines::Geometry {
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
            fragment: Some(crate::pipelines::Fragment {
                entry: fragment,
                constants,
                targets,
            }),
        },
    )
}
