//! The visibility buffer: which camera bins it resolves, and
//! the pipelines and groups of its pass and resolves.

use super::*;
use crate::bindings::{VISIBILITY_PASS, VISIBILITY_RESOLVE};
use crate::cached::GroupCache;
use crate::entries;

/// The visibility buffer, created with the first frame resolving through
/// it: its passes' group layouts, pipelines (the visibility pass's single-
/// and double-sided; resolves per pass kind and material variant over their
/// layouts, unlit and lit) and groups (the visibility pass's per current
/// pose buffer, the resolves' per current and previous one); and which of
/// this frame's camera bins it resolves.
pub(super) struct VisibilityDraws {
    pub(super) buffer: Option<VisibilityBuffer>,
    pub(super) layout: crate::pipelines::GroupLayout,
    pub(super) resolve_layout: crate::pipelines::GroupLayout,
    pub(super) pipelines: Option<[wgpu::RenderPipeline; 2]>,
    pub(super) resolve_layouts: [Option<wgpu::PipelineLayout>; 2],
    pub(super) resolve_pipelines: HashMap<(PassKind, Variant), wgpu::RenderPipeline>,
    pub(super) pass_groups: [GroupCache<5>; 2],
    pub(super) resolve_groups: [GroupCache<10>; 4],
    pub(super) bins: Vec<bool>,
}

impl VisibilityDraws {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        // Its pass poses clusters from the arena, and its resolves read
        // them, the selected clusters and the target in fragments, the tile
        // lists in vertices, and each draw's bin through a dynamic offset.
        let layout = crate::pipelines::layout(device, &VISIBILITY_PASS);
        let resolve_layout = crate::pipelines::layout(device, &VISIBILITY_RESOLVE);
        Self {
            buffer: None,
            layout,
            resolve_layout,
            pipelines: None,
            resolve_layouts: [None, None],
            resolve_pipelines: HashMap::default(),
            pass_groups: Default::default(),
            resolve_groups: Default::default(),
            bins: Vec::new(),
        }
    }
}

impl Renderer {
    /// Readies the visibility buffer for this frame's camera
    /// bins and marks those it resolves: every bin but those of alpha-masked
    /// materials, whose coverage its pass does not test. False when it is
    /// off, resolves no bin, or the device binds too few storage buffers or
    /// too small ones for `kind`'s resolves; the bins then draw forward.
    pub(super) fn prepare_visibility(&mut self, snapshot: &RenderSnapshot, kind: PassKind) -> bool {
        self.visibility.bins.clear();
        let lit = kind.lit;
        // The storage buffers its resolves bind in fragments: poses, colors,
        // previous poses, the arena's meshes, clusters and pages, the
        // selected clusters, and the camera's reflection probe grid and
        // cloud shadows; lit frames add the local lights' three.
        let needed = if lit {
            &crate::pipeline_layouts::LIT_RESOLVE_PIPELINES
        } else {
            &crate::pipeline_layouts::RESOLVE_PIPELINES
        }
        .limits
        .storage_buffers;
        if !snapshot.environment.virtual_geometry.visibility_buffer
            || self.device.limits().max_storage_buffers_per_shader_stage < needed
            || self
                .virtual_draws
                .bins
                .iter()
                .all(|(_, variant)| variant.alpha_mask())
        {
            return false;
        }
        let bins = self.virtual_draws.bins.len();
        let bases = [Some(0), self.virtual_draws.phase_two].map(|first| {
            (0..bins)
                .map(|bin| first.map_or(0, |first| self.virtual_draws.pass.bin_base(first + bin)))
                .collect::<Vec<u32>>()
        });
        let size = self.display.size();
        let visibility = self
            .visibility
            .buffer
            .get_or_insert_with(|| VisibilityBuffer::new(&self.device));
        if !visibility.prepare(&self.device, &self.queue, size, [&bases[0], &bases[1]]) {
            return false;
        }
        self.visibility.bins.extend(
            self.virtual_draws
                .bins
                .iter()
                .map(|(_, variant)| !variant.alpha_mask()),
        );
        true
    }

    /// The visibility pass's pipelines and `kind`'s resolves
    /// for this frame's resolved bins' material variants.
    pub(super) fn ensure_visibility_pipelines(&mut self, kind: PassKind) {
        if self.visibility.pipelines.is_none() {
            let layout = crate::pipelines::pipeline_layout(
                &self.device,
                &crate::pipeline_layouts::VISIBILITY_PIPELINES,
                &[
                    &self.camera.layout,
                    &self.mesh_pipelines.empty_layout,
                    &self.visibility.layout,
                ],
            );
            let pipeline = |double_sided: bool| {
                crate::pipelines::geometry(
                    &self.device,
                    &crate::pipelines::Geometry {
                        label: "engine/visibility",
                        layout: &layout,
                        module: &self.mesh_shader,
                        vertex_constants: &[],
                        vertex: entries::VISIBILITY_VERTEX,
                        buffers: &[Some(wgpu::VertexBufferLayout {
                            array_stride: crate::virtual_geometry::VISIBLE_BYTES,
                            step_mode: wgpu::VertexStepMode::Instance,
                            attributes: &VIRTUAL_ATTRIBUTES,
                        })],
                        primitive: wgpu::PrimitiveState {
                            topology: wgpu::PrimitiveTopology::TriangleList,
                            cull_mode: (!double_sided).then_some(wgpu::Face::Back),
                            ..Default::default()
                        },
                        depth: Some(wgpu::DepthStencilState {
                            format: DEPTH_FORMAT,
                            depth_write_enabled: Some(true),
                            depth_compare: Some(wgpu::CompareFunction::Greater),
                            stencil: Default::default(),
                            bias: Default::default(),
                        }),
                        fragment: Some(crate::pipelines::Fragment {
                            entry: entries::VISIBILITY_FRAGMENT,
                            constants: &[],
                            targets: &[Some(ID_FORMAT.into())],
                        }),
                    },
                )
            };
            self.visibility.pipelines = Some([pipeline(false), pipeline(true)]);
        }
        let lit = kind.lit;
        if self.visibility.resolve_layouts[usize::from(lit)].is_none() {
            let lights = self.lights.as_ref().filter(|_| lit);
            let groups = (
                &self.camera.layout,
                self.materials.layout(),
                &self.visibility.resolve_layout,
            );
            self.visibility.resolve_layouts[usize::from(lit)] = Some(match lights {
                Some(lights) => crate::pipelines::pipeline_layout(
                    &self.device,
                    &crate::pipeline_layouts::LIT_RESOLVE_PIPELINES,
                    &[groups.0, groups.1, groups.2, lights.fragment_layout()],
                ),
                None => crate::pipelines::pipeline_layout(
                    &self.device,
                    &crate::pipeline_layouts::RESOLVE_PIPELINES,
                    &[groups.0, groups.1, groups.2],
                ),
            });
        }
        let missing: Vec<Variant> = self
            .virtual_draws
            .bins
            .iter()
            .zip(&self.visibility.bins)
            .filter(|&(_, &resolved)| resolved)
            .map(|((_, variant), _)| *variant)
            .filter(|variant| {
                !self
                    .visibility
                    .resolve_pipelines
                    .contains_key(&(kind, *variant))
            })
            .collect();
        for variant in missing {
            let pipeline = self.create_resolve(kind, variant, lit);
            self.visibility
                .resolve_pipelines
                .insert((kind, variant), pipeline);
        }
    }

    /// A resolve of the visibility buffer: `vs_resolve`'s
    /// tiles with `kind`'s resolving fragment stage and targets, testing no
    /// depth.
    pub(super) fn create_resolve(
        &self,
        kind: PassKind,
        variant: Variant,
        lit: bool,
    ) -> wgpu::RenderPipeline {
        let layout = self.visibility.resolve_layouts[usize::from(lit)]
            .as_ref()
            .expect("created above");
        let VirtualStages {
            vertex,
            fragment,
            targets,
            constants,
        } = self.virtual_stages(kind, variant, true);
        crate::pipelines::geometry(
            &self.device,
            &crate::pipelines::Geometry {
                label: "engine/visibility-resolve",
                layout,
                module: &self.mesh_shader,
                vertex_constants: &[],
                vertex,
                buffers: &[],
                primitive: wgpu::PrimitiveState::default(),
                depth: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(false),
                    depth_compare: Some(wgpu::CompareFunction::Always),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                fragment: Some(crate::pipelines::Fragment {
                    entry: fragment,
                    constants: &constants,
                    targets: &targets,
                }),
            },
        )
    }

    /// The visibility buffer's groups: its pass's per current
    /// pose buffer, and the resolves' per current and previous one, as
    /// `virtual_pose_groups`.
    pub(super) fn ensure_visibility_groups(&mut self) {
        let (Some(([meshes, clusters, pages], arena)), Some(visibility)) = (
            self.virtual_draws.pass.arena(),
            self.visibility.buffer.as_ref(),
        ) else {
            return;
        };
        let (visible, selection) = self.virtual_draws.pass.visible();
        let buffers = self.instances.pose_bounds();
        let colors = self.instances.colors().bound();
        let (draws, tiles, _) = visibility.buffers();
        let draw = Bound::of(
            draws,
            wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: draws,
                offset: 0,
                size: wgpu::BufferSize::new(16),
            }),
        );
        // The arena and the selection count their buffers' replacements
        // themselves.
        let arena = [meshes, clusters, pages]
            .map(|buffer| Bound::counted(arena, buffer.as_entire_binding()));
        let visible = Bound::counted(selection, visible.as_entire_binding());
        for (current, group) in self.visibility.pass_groups.iter_mut().enumerate() {
            let [meshes, clusters, pages] = arena.clone();
            group.bind(
                &self.device,
                "engine/visibility-group",
                (&VISIBILITY_PASS, &self.visibility.layout),
                [
                    ("poses", buffers[current].clone()),
                    ("virtual_meshes", meshes),
                    ("virtual_clusters", clusters),
                    ("virtual_pages", pages),
                    ("visibility_draw", draw.clone()),
                ],
            );
        }
        for (index, group) in self.visibility.resolve_groups.iter_mut().enumerate() {
            let (current, previous) = (index / 2, index % 2);
            let [meshes, clusters, pages] = arena.clone();
            group.bind(
                &self.device,
                "engine/visibility-resolve-group",
                (&VISIBILITY_RESOLVE, &self.visibility.resolve_layout),
                [
                    ("poses", buffers[current].clone()),
                    ("colors", colors.clone()),
                    ("previous_poses", buffers[previous].clone()),
                    ("virtual_meshes", meshes),
                    ("virtual_clusters", clusters),
                    ("virtual_pages", pages),
                    ("visibility_draw", draw.clone()),
                    ("visible_clusters", visible.clone()),
                    ("visibility_ids", visibility.ids_view().bound()),
                    ("resolve_tiles", tiles.bound()),
                ],
            );
        }
    }
}
