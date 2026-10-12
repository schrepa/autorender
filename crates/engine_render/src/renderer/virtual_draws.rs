//! Virtual geometry draws: the frame's camera bins and shadow
//! views, the selection's bindings, and the pipelines and pose groups
//! drawing the selected clusters.

use super::*;
use crate::bindings::VIRTUAL_POSES;
use crate::cached::GroupCache;
use crate::entries;

/// Virtual geometry's draws: the arena and cluster
/// selection; pipelines per pass kind and material variant over their
/// layouts (unlit, lit) and pose groups (poses, colors, previous poses and
/// the arena); this frame's camera bins (material and variant), the plan
/// batches whose clusters draw instead of their fallback meshes, each
/// shadow map's bin and view offset, and the first of the camera bins'
/// phase-two twins; the depth pyramid generation and projection of the
/// last frame that drew two phases.
pub(super) struct VirtualDraws {
    pub(super) pass: VirtualGeometryPass,
    pub(super) pipelines: HashMap<(PassKind, Variant), wgpu::RenderPipeline>,
    pub(super) pose_layout: crate::pipelines::GroupLayout,
    pub(super) layouts: [Option<wgpu::PipelineLayout>; 2],
    pub(super) pose_groups: [GroupCache<6>; 4],
    pub(super) bins: Vec<(Option<MaterialHandle>, Variant)>,
    pub(super) drawn: Vec<bool>,
    pub(super) shadow_bins: Vec<(usize, usize, u32)>,
    pub(super) phase_two: Option<usize>,
    pub(super) previous: Option<(u64, engine_core::glam::Mat4)>,
}

impl VirtualDraws {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        // Poses, colors, previous poses and the arena's meshes, clusters and
        // pages.
        let pose_layout = crate::pipelines::layout(device, &VIRTUAL_POSES);
        Self {
            pass: VirtualGeometryPass::new(device),
            pipelines: HashMap::default(),
            pose_layout,
            layouts: [None, None],
            pose_groups: Default::default(),
            bins: Vec::new(),
            drawn: Vec::new(),
            shadow_bins: Vec::new(),
            phase_two: None,
            previous: None,
        }
    }
}

impl Renderer {
    /// Plans this frame's virtual geometry: the camera's
    /// batches of virtual meshes and their bins, each shadow map's virtual
    /// casters and bin, the views, the camera's occlusion and sub-pixel
    /// tests (with `rasterized`, the projection the frame rasterizes, and
    /// the depth pyramid's levels when occlusion culling runs), and the
    /// bindings selection reads (this frame's poses, GPU-compacted visible
    /// IDs or CPU-prepared slots, the shadow maps' casters and the depth
    /// pyramid). Returns whether any batch draws clusters.
    #[expect(
        clippy::too_many_lines,
        reason = "longer than the line ceiling; it may only shrink"
    )]
    pub(super) fn prepare_virtual(
        &mut self,
        snapshot: &RenderSnapshot,
        gpu_mode: bool,
        odd: bool,
        (rasterized, pyramid_levels): (engine_core::glam::Mat4, Option<u32>),
    ) -> bool {
        self.virtual_draws.bins.clear();
        self.virtual_draws.drawn.clear();
        self.virtual_draws
            .drawn
            .resize(self.plan.batches.len(), false);
        self.virtual_draws.shadow_bins.clear();
        self.virtual_draws.phase_two = None;
        // Frames drawing two phases retest the camera's
        // clusters the last such frame's pyramid hid, if it is this one's.
        let two_phases = gpu_mode && self.gpu_presentation.occlusion();
        let pyramid = self
            .culling
            .pyramid
            .as_ref()
            .and_then(DepthPyramid::view)
            .map(|view| (&**view, view.generation()))
            .filter(|_| two_phases);
        let previous = self
            .virtual_draws
            .previous
            .take()
            .and_then(|(generation, projection)| {
                pyramid
                    .is_some_and(|(_, current)| current == generation)
                    .then_some(projection)
            });
        self.virtual_draws.previous = pyramid.map(|(_, generation)| (generation, rasterized));
        if !snapshot.environment.virtual_geometry.enabled {
            return false;
        }
        let mut batches = Vec::new();
        let mut max_instances = 0;
        for (index, batch) in self.plan.batches.iter().enumerate() {
            if batch.transparent {
                continue;
            }
            let Some(mesh) = self.virtual_draws.pass.mesh_index(batch.mesh) else {
                continue;
            };
            let (_, variant) = self.materials.binding(batch.material);
            let bin = match self
                .virtual_draws
                .bins
                .iter()
                .position(|(material, _)| *material == batch.material)
            {
                Some(bin) => bin,
                None => {
                    self.virtual_draws.bins.push((batch.material, variant));
                    self.virtual_draws.bins.len() - 1
                }
            };
            let count = batch.instances.end - batch.instances.start;
            max_instances = max_instances.max(count);
            batches.push(VirtualBatch {
                start: batch.instances.start,
                count: (!gpu_mode).then_some(count),
                draw: index as u32,
                mesh,
                bin: bin as u32,
                double_sided: variant.double_sided(),
                view: 0,
            });
            self.virtual_draws.drawn[index] = true;
        }
        if batches.is_empty() {
            return false;
        }
        let view_projection = snapshot.view_projection;
        let mut camera_planes = [[0.0, 0.0, 0.0, 1.0]; crate::shadows::VIRTUAL_SHADOW_PLANES];
        camera_planes[..6].copy_from_slice(&crate::virtual_geometry::planes(view_projection));
        let camera = VirtualView {
            planes: camera_planes,
            eye: snapshot.camera_position,
            min_distance: VIRTUAL_NEAREST,
            // The projection's y scale: its second row's length.
            projection: view_projection.row(1).truncate().length() * 0.5,
            threshold: snapshot.environment.virtual_geometry.pixel_error
                / self.display.size().height.max(1) as f32,
            unseen_error: 0.0,
            cone: Cone::Eye,
        };
        // Shadow maps drawn every frame select their virtual casters'
        // clusters as the camera sees them, from where their
        // receivers start, and where the camera sees nothing within as many
        // of the map's texels as pixels, into a bin each. Closed meshes cast
        // the clusters facing the sun, which hide the rest; others cast both
        // faces.
        let mut views = vec![camera];
        let mut shadow_batches = Vec::new();
        let mut shadow_instances = 0;
        let (_, _, id_capacity, _) = self.shadows.caster_buffers();
        let shadow_groups = self.shadows.groups();
        let group_count = shadow_groups.len() as u32;
        for shadow_view in self.shadows.virtual_views() {
            if views.len() == MAX_VIEWS {
                break;
            }
            let bin = self.virtual_draws.bins.len() + self.virtual_draws.shadow_bins.len();
            let view = views.len() as u32;
            let mut any = false;
            for (index, group) in shadow_groups.iter().enumerate() {
                if !group.virtual_geometry {
                    continue;
                }
                let Some(mesh) = self.virtual_draws.pass.mesh_index(group.mesh) else {
                    continue;
                };
                shadow_batches.push(VirtualBatch {
                    start: shadow_view.map as u32 * id_capacity + group.start,
                    count: None,
                    draw: shadow_view.map as u32 * group_count + index as u32,
                    mesh,
                    bin: bin as u32,
                    double_sided: !self.virtual_draws.pass.closed(mesh),
                    view,
                });
                shadow_instances = shadow_instances.max(group.capacity);
                any = true;
            }
            if any {
                self.virtual_draws.shadow_bins.push((
                    shadow_view.map,
                    bin,
                    shadow_view.view_offset,
                ));
                views.push(match shadow_view.error {
                    // Errors project from beyond any distance: within
                    // `error` everywhere, whatever the camera sees.
                    Some(error) => VirtualView {
                        planes: shadow_view.planes,
                        min_distance: KEPT_DISTANCE,
                        threshold: error * camera.projection / KEPT_DISTANCE,
                        unseen_error: 0.0,
                        cone: Cone::Along(shadow_view.direction),
                        ..camera
                    },
                    None => VirtualView {
                        planes: shadow_view.planes,
                        min_distance: shadow_view.min_distance.unwrap_or(VIRTUAL_NEAREST),
                        unseen_error: snapshot.environment.virtual_geometry.pixel_error
                            * shadow_view.texel,
                        cone: Cone::Along(shadow_view.direction),
                        ..camera
                    },
                });
            }
        }
        // Phase two's twins of the camera's bins follow the shadow maps'.
        let mut bins = self.virtual_draws.bins.len() + self.virtual_draws.shadow_bins.len();
        if pyramid.is_some() {
            self.virtual_draws.phase_two = Some(bins);
            bins += self.virtual_draws.bins.len();
        }
        let tests = CameraTests {
            view_projection: rasterized,
            viewport: (
                self.display.size().width,
                self.display.size().height,
                pyramid_levels.unwrap_or(0),
            ),
            previous,
            phase_two: self.virtual_draws.phase_two.map(|bin| bin as u32),
            subpixel: gpu_mode && self.gpu_presentation.culls_subpixel(),
        };
        let camera = if gpu_mode {
            let (ids, draws, generation) = self.gpu_presentation.visible_sources();
            IdBuffers {
                ids,
                draws,
                key: generation * 2,
            }
        } else {
            let slots = self.instances.slots();
            IdBuffers {
                ids: slots,
                draws: slots,
                key: self.instances.slot_capacity as u64 * 2 + 1,
            }
        };
        let (shadow_ids, shadow_args, _, shadow_key) = self.shadows.caster_buffers();
        self.virtual_draws.pass.prepare(
            &self.device,
            &self.queue,
            &SelectionFrame {
                batches: [&batches, &shadow_batches],
                bins,
                views: &views,
                tests,
                budget: snapshot.environment.virtual_geometry.max_resident_pages,
                poses: self.instances.frame_poses(usize::from(odd)),
                camera,
                shadows: (!shadow_batches.is_empty()).then_some(IdBuffers {
                    ids: shadow_ids,
                    draws: shadow_args,
                    key: shadow_key,
                }),
                max_instances: [max_instances, shadow_instances],
                shadow_views: self.shadows.view_buffer(),
                pyramid,
            },
        );
        true
    }

    /// Pose groups of virtual geometry pipelines: each pair
    /// of this frame's and the previous frame's pose buffers, with the
    /// colors and the arena.
    pub(super) fn ensure_virtual_pose_groups(&mut self) {
        let Some(([meshes, clusters, pages], arena)) = self.virtual_draws.pass.arena() else {
            return;
        };
        let buffers = self.instances.pose_bounds();
        let colors = self.instances.colors().bound();
        // The arena counts its buffers' replacements itself.
        let arena = [meshes, clusters, pages]
            .map(|buffer| Bound::counted(arena, buffer.as_entire_binding()));
        for (index, group) in self.virtual_draws.pose_groups.iter_mut().enumerate() {
            let (current, previous) = (index / 2, index % 2);
            let [meshes, clusters, pages] = arena.clone();
            group.bind(
                &self.device,
                "engine/virtual-pose-group",
                (&VIRTUAL_POSES, &self.virtual_draws.pose_layout),
                [
                    ("poses", buffers[current].clone()),
                    ("colors", colors.clone()),
                    ("previous_poses", buffers[previous].clone()),
                    ("virtual_meshes", meshes),
                    ("virtual_clusters", clusters),
                    ("virtual_pages", pages),
                ],
            );
        }
    }

    /// The virtual geometry pipelines of `kind` for this
    /// frame's bins' material variants.
    pub(super) fn ensure_virtual_pipelines(&mut self, kind: PassKind) {
        let lit = kind.lit;
        if self.virtual_draws.layouts[usize::from(lit)].is_none() {
            let lights = self.lights.as_ref().filter(|_| lit);
            let groups = (
                &self.camera.layout,
                self.materials.layout(),
                &self.virtual_draws.pose_layout,
            );
            self.virtual_draws.layouts[usize::from(lit)] = Some(match lights {
                Some(lights) => crate::pipelines::pipeline_layout(
                    &self.device,
                    &crate::pipeline_layouts::LIT_VIRTUAL_MESH_PIPELINES,
                    &[groups.0, groups.1, groups.2, lights.fragment_layout()],
                ),
                None => crate::pipelines::pipeline_layout(
                    &self.device,
                    &crate::pipeline_layouts::VIRTUAL_MESH_PIPELINES,
                    &[groups.0, groups.1, groups.2],
                ),
            });
        }
        let missing: Vec<Variant> = self
            .virtual_draws
            .bins
            .iter()
            .map(|(_, variant)| *variant)
            .filter(|variant| !self.virtual_draws.pipelines.contains_key(&(kind, *variant)))
            .collect();
        for variant in missing {
            let pipeline = self.create_virtual(kind, variant, lit);
            self.virtual_draws
                .pipelines
                .insert((kind, variant), pipeline);
        }
    }

    /// A virtual geometry pipeline: `vs_virtual` (or its
    /// temporal twin) with `kind`'s fragment stage and targets.
    pub(super) fn create_virtual(
        &self,
        kind: PassKind,
        variant: Variant,
        lit: bool,
    ) -> wgpu::RenderPipeline {
        let layout = self.virtual_draws.layouts[usize::from(lit)]
            .as_ref()
            .expect("created above");
        let VirtualStages {
            vertex,
            fragment,
            targets,
            constants,
        } = self.virtual_stages(kind, variant, false);
        let targets = &targets[..];
        crate::pipelines::geometry(
            &self.device,
            &crate::pipelines::Geometry {
                label: "engine/virtual-meshes",
                layout,
                module: &self.mesh_shader,
                vertex_constants: &[],
                vertex,
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: crate::virtual_geometry::VISIBLE_BYTES,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &VIRTUAL_ATTRIBUTES,
                })],
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: (!variant.double_sided()).then_some(wgpu::Face::Back),
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
                    entry: fragment,
                    constants: &constants,
                    targets,
                }),
            },
        )
    }

    /// The stages of `kind`'s virtual geometry pipelines for material
    /// `variant`: forward or resolving the visibility buffer,
    /// with the targets and constants of `kind`.
    pub(super) fn virtual_stages(
        &self,
        kind: PassKind,
        variant: Variant,
        resolve: bool,
    ) -> VirtualStages {
        let targets = kind
            .targets()
            .expect("virtual geometry draws in opaque passes");
        let canvas = self.display.format();
        let (vertex, fragment) = if resolve {
            (
                entries::RESOLVE_VERTEX,
                entries::resolve_fragment(targets, kind.lit),
            )
        } else {
            (entries::virtual_vertex(targets), kind.fragment_entry(false))
        };
        VirtualStages {
            vertex,
            fragment,
            targets: targets.color_targets(canvas),
            constants: variant.constants().to_vec(),
        }
    }
}

/// A virtual geometry pipeline's entry points, color targets and constants.
pub(super) struct VirtualStages {
    pub(super) vertex: crate::shaders::Entry,
    pub(super) fragment: crate::shaders::Entry,
    pub(super) targets: Vec<Option<wgpu::ColorTargetState>>,
    pub(super) constants: Vec<(&'static str, f64)>,
}

/// The smallest distance virtual geometry's errors project from:
/// a guard for the eye's own position.
pub(super) const VIRTUAL_NEAREST: f32 = 0.05;
/// The distance a kept map's view projects every error
/// from, beyond any scene's: errors then compare in world units.
const KEPT_DISTANCE: f32 = 1.0e7;

/// Selected clusters of virtual geometry: visible ID and
/// cluster, one entry per instance.
pub(super) const VIRTUAL_ATTRIBUTES: [wgpu::VertexAttribute; 2] =
    wgpu::vertex_attr_array![2 => Uint32, 13 => Uint32];
