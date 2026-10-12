//! Terrain's pipelines and the terrain group: patches in the opaque passes
//! (forward, or deferred as visibility and resolves), reflection probe
//! faces and shadow maps, and the group binding the arrays, tables, views,
//! material and sky maps and deformation's map they read.

use super::atlas::TileAtlas;
use super::patches::{PatchDraws, patch_layout};
use super::tables::TerrainTables;
use super::*;
use crate::shaders::entry::meshes;

/// The generations of what the terrain group binds: the atlas, uniforms,
/// hole bits, directory, material maps, sky maps and deformation map.
type GroupKey = (u64, u64, u64, u64, u64, u64, u64);

/// What the terrain group binds besides the pipelines' own: the arrays,
/// tables, views, material and sky maps and deformation's map.
pub(super) struct GroupSources<'a> {
    pub(super) atlas: &'a TileAtlas,
    pub(super) tables: &'a TerrainTables,
    pub(super) patches: &'a PatchDraws,
    pub(super) maps: &'a TerrainMaps,
    pub(super) sky: &'a SkyMaps,
    pub(super) deformation: &'a crate::deformation::DeformationPass,
}

pub(crate) struct TerrainPipelines {
    pub(super) layout: crate::pipelines::GroupLayout,
    group: Cached<GroupKey, wgpu::BindGroup>,
    // The group the runtime virtual texture's bakes bind,
    // its material maps' views leaving out the sheets they draw into.
    bake_group: Cached<GroupKey, wgpu::BindGroup>,
    pipelines: HashMap<(Targets, bool, bool), wgpu::RenderPipeline>,
    // Deferred shading: the visibility pipelines (by
    // holes), the resolves (by targets and lighting) and the layout of the
    // resolves' own group.
    pub(crate) deferred_ids: HashMap<bool, wgpu::RenderPipeline>,
    pub(crate) resolves: HashMap<(Targets, bool), wgpu::RenderPipeline>,
    pub(crate) resolve_layout: crate::pipelines::GroupLayout,
    probe_pipelines: HashMap<(bool, bool), wgpu::RenderPipeline>,
    shadow: Option<wgpu::RenderPipeline>,
}

impl TerrainPipelines {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        Self {
            layout: crate::pipelines::layout(device, &TERRAIN),
            group: Cached::default(),
            bake_group: Cached::default(),
            pipelines: HashMap::default(),
            deferred_ids: HashMap::default(),
            resolves: HashMap::default(),
            resolve_layout: crate::pipelines::layout(device, &crate::bindings::TERRAIN_RESOLVE),
            probe_pipelines: HashMap::default(),
            shadow: None,
        }
    }

    /// The terrain group, once tiles arrived.
    pub(super) fn group(&self) -> Option<&wgpu::BindGroup> {
        self.group.get()
    }

    /// The group the virtual texture's bakes bind, once tiles arrived.
    pub(super) fn bake_group(&self) -> Option<&wgpu::BindGroup> {
        self.bake_group.get()
    }

    /// Rebuilds the terrain group, and the bakes', when what `sources`
    /// binds was replaced.
    pub(super) fn ensure_group(&mut self, device: &wgpu::Device, sources: &GroupSources<'_>) {
        let Some((heights, normals, control)) = sources.atlas.views() else {
            return;
        };
        let (tables, maps, sky, deformation) = (
            sources.tables,
            sources.maps,
            sources.sky,
            sources.deformation,
        );
        let key = (
            sources.atlas.generation,
            tables.uniform_generation,
            tables.holes_generation,
            tables.directory_generation,
            maps.views_generation,
            sky.generation(),
            deformation.map().1,
        );
        let build = |label: &str, material_maps: &[wgpu::TextureView; 3]| {
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
                        "terrain_normal_sampler",
                        crate::pipelines::sampler(&sources.atlas.normal_sampler),
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
                        "terrain_material_normals",
                        crate::pipelines::view(&material_maps[1]),
                    ),
                    ("terrain_orm", crate::pipelines::view(&material_maps[2])),
                    (
                        "terrain_material_sampler",
                        crate::pipelines::sampler(&maps.sampler),
                    ),
                    ("terrain_sky", crate::pipelines::view(sky.view())),
                    crate::buffers::entry("terrain_deformation", deformation.map().0),
                    (
                        "terrain_control_sampler",
                        crate::pipelines::sampler(&sources.atlas.control_sampler),
                    ),
                ],
            )
        };
        if self.group.stale(&key) {
            let group = build("engine/terrain-group", &maps.views);
            // The runtime virtual texture's bakes read the materials' layers
            // alone while they draw into its sheets.
            let bake = build("engine/terrain-bake-group", &maps.material_views);
            self.group.set(key, group);
            self.bake_group.set(key, bake);
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
            (vertex, fragment, holes),
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
    /// unless they exist; canvas targets shade forward.
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
                (meshes::vs_terrain, meshes::fs_terrain_ids, holes),
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
            (Targets::Canvas, _) => return,
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
                constants: &[],
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
            (vertex, fragment, holes),
            &[Some(wgpu::ColorTargetState::from(format))],
            depth,
            wgpu::FrontFace::Cw,
        );
        self.probe_pipelines.insert((lit, holes), pipeline);
    }

    /// Creates the shadow maps' depth pipeline unless it exists.
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
        let pipeline = crate::pipelines::geometry(
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
                    // Terrain is seen at grazing angles over texels meters
                    // wide: its depth also slides back by its slope, so the
                    // facets of a texel never shadow the surface they make.
                    bias: wgpu::DepthBiasState {
                        constant: 2,
                        slope_scale: 2.5,
                        clamp: 0.0,
                    },
                }),
                fragment: None,
            },
        );
        self.shadow = Some(pipeline);
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

fn geometry(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    (vertex, fragment, holes): (crate::shaders::Entry, crate::shaders::Entry, bool),
    targets: &[Option<wgpu::ColorTargetState>],
    depth: wgpu::TextureFormat,
    front_face: wgpu::FrontFace,
) -> wgpu::RenderPipeline {
    let constants: &[(&str, f64)] = if holes {
        &[("TERRAIN_HOLES", 1.0)]
    } else {
        &[]
    };
    crate::pipelines::geometry(
        device,
        &crate::pipelines::Geometry {
            label: "engine/terrain",
            layout,
            module: shader,
            vertex_constants: &[],
            vertex,
            buffers: &[Some(patch_layout())],
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
