//! Terrain's pipelines and the terrain group: its subdivisions' leaves in
//! the opaque passes (forward, or deferred as visibility and resolves),
//! reflection probe faces and the shadow maps drawn every frame, the kept
//! levels' patches, and the group binding the arrays (the tiles' error
//! pyramids among them, which the subdivisions' passes read), tables, views,
//! leaves, material, sky and macro color maps, deformation's map and the
//! tiles' water's surfaces they read.

use super::atlas::TileAtlas;
use super::patches::{PatchDraws, patch_layout};
use super::tables::TerrainTables;
use super::water_blocks::WaterBlocks;
use super::*;
use crate::scene_shading::SceneShading;
use crate::shaders::entry::meshes;

/// The generations of what the terrain group binds: the atlas, uniforms,
/// hole bits, directory, material maps, sky maps, macro color maps,
/// deformation map, the subdivisions' leaves and the water blocks' arrays.
type GroupKey = (u64, u64, u64, u64, u64, u64, u64, u64, u64, u64);

/// What the terrain group binds besides the pipelines' own: the arrays,
/// tables, views, the subdivisions' leaves, material, sky and macro color
/// maps, deformation's map and the tiles' water's surfaces.
pub(super) struct GroupSources<'a> {
    pub(super) atlas: &'a TileAtlas,
    pub(super) tables: &'a TerrainTables,
    pub(super) patches: &'a PatchDraws,
    pub(super) surface: &'a SurfaceTrees,
    pub(super) maps: &'a TerrainMaps,
    pub(super) sky: &'a SkyMaps,
    pub(super) macro_maps: &'a MacroMaps,
    pub(super) deformation: &'a crate::deformation::DeformationPass,
    pub(super) water: &'a WaterBlocks,
}

pub(crate) struct TerrainPipelines {
    pub(super) layout: crate::pipelines::GroupLayout,
    group: Cached<GroupKey, wgpu::BindGroup>,
    // The group the subdivisions' passes bind, leaving out the leaves they
    // write.
    surface_group: Cached<GroupKey, wgpu::BindGroup>,
    pipelines: HashMap<(Targets, bool, bool), wgpu::RenderPipeline>,
    // Deferred shading: the visibility pipelines (by
    // holes), the resolves (by targets and lighting) and the layout of the
    // resolves' own group.
    pub(crate) deferred_ids: HashMap<bool, wgpu::RenderPipeline>,
    pub(crate) resolves: HashMap<(Targets, bool), wgpu::RenderPipeline>,
    pub(crate) resolve_layout: crate::pipelines::GroupLayout,
    probe_pipelines: HashMap<(bool, bool), wgpu::RenderPipeline>,
    // The shadow clipmap's depth pipeline, of the patches its regions take.
    shadow: Option<wgpu::RenderPipeline>,
    // The scene features the shading pipelines are specialized for.
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

    /// Specializes the pipelines shading terrain for `shading`'s scene
    /// features: those made for others are dropped, to be created anew.
    pub(crate) fn specialize(&mut self, shading: SceneShading) {
        if self.shading != shading {
            self.shading = shading;
            self.pipelines.clear();
            self.resolves.clear();
            self.probe_pipelines.clear();
        }
    }

    /// The terrain group, once tiles arrived.
    pub(super) fn group(&self) -> Option<&wgpu::BindGroup> {
        self.group.get()
    }

    /// The group the subdivisions' passes bind, once tiles arrived.
    pub(super) fn surface_group(&self) -> Option<&wgpu::BindGroup> {
        self.surface_group.get()
    }

    /// Rebuilds the terrain group and the subdivisions', when what
    /// `sources` binds was replaced.
    pub(super) fn ensure_group(&mut self, device: &wgpu::Device, sources: &GroupSources<'_>) {
        let (Some((heights, normals, control)), Some((ground_errors, water_errors))) =
            (sources.atlas.views(), sources.atlas.errors())
        else {
            return;
        };
        let (tables, maps, sky, deformation) = (
            sources.tables,
            sources.maps,
            sources.sky,
            sources.deformation,
        );
        let leaves = sources.surface.leaves();
        let key = (
            sources.atlas.generation,
            tables.uniform_generation,
            tables.holes_generation,
            tables.directory_generation,
            maps.views_generation,
            sky.generation(),
            sources.macro_maps.generation(),
            deformation.map().1,
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
                        crate::buffers::entry("terrain_deformation", deformation.map().0),
                        (
                            "terrain_control_sampler",
                            crate::pipelines::sampler(&sources.atlas.control_sampler),
                        ),
                        crate::buffers::entry("terrain_surface_leaves", surface_leaves),
                        (
                            "water_surface",
                            crate::pipelines::view(sources.water.surface_view()),
                        ),
                        ("terrain_errors", crate::pipelines::view(ground_errors)),
                        ("water_errors", crate::pipelines::view(water_errors)),
                    ],
                )
            };
        if self.group.stale(&key) {
            let group = build("engine/terrain-group", &maps.views, leaves);
            // The subdivisions' passes bind a stand-in for the leaves they
            // write.
            let surface = build(
                "engine/terrain-surface-group",
                &maps.views,
                &sources.surface.stand_in,
            );
            self.group.set(key, group);
            self.surface_group.set(key, surface);
        }
    }

    /// The opaque pipeline into `targets`, shading local lights when `lit`,
    /// of patches with `holes` or without.
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

    /// The shadow clipmap's depth pipeline, of the patches its regions
    /// take.
    pub(crate) fn shadow_pipeline(&self) -> Option<&wgpu::RenderPipeline> {
        self.shadow.as_ref()
    }

    /// Creates the opaque pipeline into `targets` (`lit` or not) unless it
    /// exists: `camera` and `lights` are the main passes' first and lights
    /// groups, `empty` the group standing in for the poses.
    #[expect(clippy::too_many_arguments, reason = "a pass's disjoint borrows")]
    pub(crate) fn ensure_pipeline(
        &mut self,
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
        (camera, empty, lights): (
            &crate::pipelines::GroupLayout,
            &crate::pipelines::GroupLayout,
            Option<&crate::pipelines::GroupLayout>,
        ),
        (targets, lit, holes): (Targets, bool, bool),
        canvas: wgpu::TextureFormat,
        depth: wgpu::TextureFormat,
        camera_terrain: &crate::pipelines::GroupLayout,
    ) {
        if self.pipelines.contains_key(&(targets, lit, holes)) {
            return;
        }
        let Some(layout) = pipeline_layout(device, camera, &self.layout, empty, lights, lit) else {
            return;
        };
        let (vertex, fragment) = crate::entries::terrain(Some(targets), lit);
        let color_targets = targets.color_targets(canvas);
        let pipeline = geometry(
            device,
            shader,
            &layout,
            (vertex, fragment, holes, Some(self.shading)),
            &color_targets,
            depth,
            wgpu::FrontFace::Ccw,
        );
        self.pipelines.insert((targets, lit, holes), pipeline);
        self.ensure_deferred(
            device,
            shader,
            (camera, lights),
            (targets, lit, holes),
            depth,
            camera_terrain,
        );
    }

    /// Creates the visibility pipeline of patches with `holes` or without
    /// and the resolve into `targets` (`lit` or not) of deferred shading
    /// unless they exist; canvas and debug targets shade forward.
    fn ensure_deferred(
        &mut self,
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
        (camera, lights): (
            &crate::pipelines::GroupLayout,
            Option<&crate::pipelines::GroupLayout>,
        ),
        (targets, lit, holes): (Targets, bool, bool),
        depth: wgpu::TextureFormat,
        camera_terrain: &crate::pipelines::GroupLayout,
    ) {
        if !self.deferred_ids.contains_key(&holes) {
            let layout = crate::pipelines::pipeline_layout(
                device,
                &crate::pipeline_layouts::TERRAIN_PIPELINES,
                &[camera, &self.layout],
            );
            let ids = [Some(wgpu::ColorTargetState::from(
                wgpu::TextureFormat::Rg32Uint,
            ))];
            let pipeline = geometry(
                device,
                shader,
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
        let groups = (camera_terrain, &self.layout, &self.resolve_layout);
        let layout = if lit {
            let Some(lights) = lights else {
                return;
            };
            crate::pipelines::pipeline_layout(
                device,
                &crate::pipeline_layouts::LIT_TERRAIN_RESOLVE_PIPELINES,
                &[groups.0, groups.1, groups.2, lights],
            )
        } else {
            crate::pipelines::pipeline_layout(
                device,
                &crate::pipeline_layouts::TERRAIN_RESOLVE_PIPELINES,
                &[groups.0, groups.1, groups.2],
            )
        };
        let color_targets = targets.color_targets(wgpu::TextureFormat::Bgra8Unorm);
        let pipeline = crate::pipelines::fullscreen(
            device,
            &crate::pipelines::Fullscreen {
                label: "engine/terrain-resolve",
                layout: &layout,
                module: shader,
                vertex: meshes::vs_fullscreen,
                fragment,
                constants: &self.shading.constants(),
                targets: &color_targets,
                depth: None,
            },
        );
        self.resolves.insert((targets, lit), pipeline);
    }

    /// Creates the reflection probe faces' pipeline (`lit` or not) unless it
    /// exists; faces mirror the world, so front faces turn clockwise.
    pub(crate) fn ensure_probe_pipeline(
        &mut self,
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
        (camera, empty, lights): (
            &crate::pipelines::GroupLayout,
            &crate::pipelines::GroupLayout,
            Option<&crate::pipelines::GroupLayout>,
        ),
        (lit, holes): (bool, bool),
        format: wgpu::TextureFormat,
        depth: wgpu::TextureFormat,
    ) {
        if self.probe_pipelines.contains_key(&(lit, holes)) {
            return;
        }
        let Some(layout) = pipeline_layout(device, camera, &self.layout, empty, lights, lit) else {
            return;
        };
        let (vertex, fragment) = crate::entries::terrain(None, lit);
        let pipeline = geometry(
            device,
            shader,
            &layout,
            (vertex, fragment, holes, Some(self.shading)),
            &[Some(wgpu::ColorTargetState::from(format))],
            depth,
            wgpu::FrontFace::Cw,
        );
        self.probe_pipelines.insert((lit, holes), pipeline);
    }

    /// Creates the shadow clipmap's depth pipeline unless it exists.
    pub(crate) fn ensure_shadow_pipeline(&mut self, device: &wgpu::Device) {
        if self.shadow.is_some() {
            return;
        }
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("engine/terrain-shadow"),
            source: wgpu::ShaderSource::Wgsl(crate::gpu_data::TERRAIN_SHADOW_SHADER.into()),
        });
        let view_layout = crate::pipelines::layout(device, &SHADOW_VIEW);
        let layout = crate::pipelines::pipeline_layout(
            device,
            &crate::pipeline_layouts::TERRAIN_CASTER_PIPELINES,
            &[&view_layout, &self.layout],
        );
        self.shadow = Some(crate::pipelines::geometry(
            device,
            &crate::pipelines::Geometry {
                label: "engine/terrain-shadow",
                layout: &layout,
                module: &module,
                vertex_constants: &[],
                vertex: crate::entries::TERRAIN_SHADOW_VERTEX,
                buffers: &[Some(patch_layout())],
                // Both faces cast, as meshes' casters do.
                primitive: wgpu::PrimitiveState::default(),
                depth: Some(wgpu::DepthStencilState {
                    format: crate::shadows::SHADOW_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: Default::default(),
                    // Terrain is seen at grazing angles over texels
                    // meters wide: its depth also slides back by its
                    // slope, so the facets of a texel never shadow the
                    // surface they make.
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

/// The pipeline layout of terrain's mesh passes: the camera, terrain, and
/// for `lit` pipelines an empty group and the lights.
fn pipeline_layout(
    device: &wgpu::Device,
    camera: &crate::pipelines::GroupLayout,
    terrain: &crate::pipelines::GroupLayout,
    empty: &crate::pipelines::GroupLayout,
    lights: Option<&crate::pipelines::GroupLayout>,
    lit: bool,
) -> Option<wgpu::PipelineLayout> {
    Some(if lit {
        crate::pipelines::pipeline_layout(
            device,
            &crate::pipeline_layouts::LIT_TERRAIN_PIPELINES,
            &[camera, terrain, empty, lights?],
        )
    } else {
        crate::pipelines::pipeline_layout(
            device,
            &crate::pipeline_layouts::TERRAIN_PIPELINES,
            &[camera, terrain],
        )
    })
}

/// A pipeline drawing a subdivision's leaves, cutting holes or not,
/// through `vertex` and `fragment`, which shades a scene of the features
/// given (pipelines shading nothing take none); leaves read no vertex
/// buffers, only their instance and the lattice vertex their meshlet's
/// indices name (`surface::SurfaceDraw`).
fn geometry(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
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
            module: shader,
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
