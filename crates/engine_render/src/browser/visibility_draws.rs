//! The visibility buffer (decision 045): which camera bins it resolves, and
//! the pipelines and groups of its pass and resolves.

use super::*;

/// What the visibility buffer's groups bind: the pose buffers' generation
/// and whether alternate poses exist, and the generations of the virtual
/// arena, the selected clusters and the visibility buffer's target and
/// buffers.
type VisibilityGroupsKey = (u64, bool, u64, u64, u64);

/// The visibility buffer, created with the first frame resolving through
/// it: its passes' group layouts, pipelines (the visibility pass's single-
/// and double-sided; resolves per pass kind and material variant over their
/// layouts, unlit and lit) and groups (the visibility pass's per current
/// pose buffer, the resolves' per current and previous one); and which of
/// this frame's camera bins it resolves.
pub(super) struct VisibilityDraws {
    pub(super) buffer: Option<VisibilityBuffer>,
    pub(super) layout: wgpu::BindGroupLayout,
    pub(super) resolve_layout: wgpu::BindGroupLayout,
    pub(super) pipelines: Option<[wgpu::RenderPipeline; 2]>,
    pub(super) resolve_layouts: [Option<wgpu::PipelineLayout>; 2],
    pub(super) resolve_pipelines: HashMap<(PassKind, Variant), wgpu::RenderPipeline>,
    pub(super) groups: Cached<VisibilityGroupsKey, ([wgpu::BindGroup; 2], [wgpu::BindGroup; 4])>,
    pub(super) bins: Vec<bool>,
}

impl VisibilityDraws {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        // Its pass poses clusters from the arena, and its resolves read
        // them, the selected clusters and the target in fragments, the tile
        // lists in vertices, and each draw's bin through a dynamic offset.
        let (vertex, fragment) = (wgpu::ShaderStages::VERTEX, wgpu::ShaderStages::FRAGMENT);
        let draw_entry = |visibility| wgpu::BindGroupLayoutEntry {
            binding: 8,
            visibility,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: wgpu::BufferSize::new(16),
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("engine/visibility-layout"),
            entries: &[
                storage_entry(0, size_of::<PoseData>() as u64, vertex),
                storage_entry(5, 80, vertex),
                storage_entry(6, 48, vertex),
                storage_entry(7, 16, vertex),
                draw_entry(vertex),
            ],
        });
        let resolve_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("engine/visibility-resolve-layout"),
            entries: &[
                storage_entry(0, size_of::<PoseData>() as u64, fragment),
                storage_entry(1, size_of::<ColorData>() as u64, fragment),
                storage_entry(2, size_of::<PoseData>() as u64, fragment),
                storage_entry(5, 80, fragment),
                storage_entry(6, 48, fragment),
                storage_entry(7, 16, fragment),
                draw_entry(vertex | fragment),
                storage_entry(9, 8, fragment),
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: vertex | fragment,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                storage_entry(11, 4, vertex),
            ],
        });
        Self {
            buffer: None,
            layout,
            resolve_layout,
            pipelines: None,
            resolve_layouts: [None, None],
            resolve_pipelines: HashMap::new(),
            groups: Cached::default(),
            bins: Vec::new(),
        }
    }
}

impl Renderer {
    /// Readies the visibility buffer (decision 045) for this frame's camera
    /// bins and marks those it resolves: every bin but those of alpha-masked
    /// materials, whose coverage its pass does not test. False when it is
    /// off, resolves no bin, or the device binds too few storage buffers or
    /// too small ones for `kind`'s resolves; the bins then draw forward.
    pub(super) fn prepare_visibility(&mut self, snapshot: &RenderSnapshot, kind: PassKind) -> bool {
        self.visibility.bins.clear();
        let lit = kind.lit;
        let needed = if lit {
            RESOLVE_LIT_STORAGE_BUFFERS
        } else {
            RESOLVE_STORAGE_BUFFERS
        };
        if !snapshot.virtual_geometry.visibility_buffer
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

    /// The visibility pass's pipelines and `kind`'s resolves (decision 045)
    /// for this frame's resolved bins' material variants.
    pub(super) fn ensure_visibility_pipelines(&mut self, kind: PassKind) {
        if self.visibility.pipelines.is_none() {
            let layout = self
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("engine/visibility-pipeline-layout"),
                    bind_group_layouts: &[
                        Some(&self.camera.layout),
                        Some(&self.mesh_pipelines.empty_layout),
                        Some(&self.visibility.layout),
                    ],
                    immediate_size: 0,
                });
            let pipeline = |double_sided: bool| {
                crate::pipelines::render(
                    &self.device,
                    &wgpu::RenderPipelineDescriptor {
                        label: Some("engine/visibility"),
                        layout: Some(&layout),
                        vertex: wgpu::VertexState {
                            module: &self.mesh_shader,
                            entry_point: Some("vs_visibility"),
                            compilation_options: Default::default(),
                            buffers: &[Some(wgpu::VertexBufferLayout {
                                array_stride: crate::virtual_geometry::VISIBLE_BYTES,
                                step_mode: wgpu::VertexStepMode::Instance,
                                attributes: &VIRTUAL_ATTRIBUTES,
                            })],
                        },
                        primitive: wgpu::PrimitiveState {
                            topology: wgpu::PrimitiveTopology::TriangleList,
                            cull_mode: (!double_sided).then_some(wgpu::Face::Back),
                            ..Default::default()
                        },
                        depth_stencil: Some(wgpu::DepthStencilState {
                            format: DEPTH_FORMAT,
                            depth_write_enabled: Some(true),
                            depth_compare: Some(wgpu::CompareFunction::Greater),
                            stencil: Default::default(),
                            bias: Default::default(),
                        }),
                        multisample: wgpu::MultisampleState::default(),
                        fragment: Some(wgpu::FragmentState {
                            module: &self.mesh_shader,
                            entry_point: Some("fs_visibility"),
                            compilation_options: Default::default(),
                            targets: &[Some(ID_FORMAT.into())],
                        }),
                        multiview_mask: None,
                        cache: None,
                    },
                )
            };
            self.visibility.pipelines = Some([pipeline(false), pipeline(true)]);
        }
        let lit = kind.lit;
        if self.visibility.resolve_layouts[usize::from(lit)].is_none() {
            let lights = self.lights.as_ref().filter(|_| lit);
            let mut layouts = vec![
                Some(&self.camera.layout),
                Some(self.materials.layout()),
                Some(&self.visibility.resolve_layout),
            ];
            if let Some(lights) = lights {
                layouts.push(Some(lights.fragment_layout()));
            }
            self.visibility.resolve_layouts[usize::from(lit)] = Some(
                self.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("engine/visibility-resolve-pipeline-layout"),
                        bind_group_layouts: &layouts,
                        immediate_size: 0,
                    }),
            );
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

    /// A resolve of the visibility buffer (decision 045): `vs_resolve`'s
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
        crate::pipelines::render(
            &self.device,
            &wgpu::RenderPipelineDescriptor {
                label: Some("engine/visibility-resolve"),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module: &self.mesh_shader,
                    entry_point: Some(vertex),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(false),
                    depth_compare: Some(wgpu::CompareFunction::Always),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &self.mesh_shader,
                    entry_point: Some(&fragment),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &constants,
                        ..Default::default()
                    },
                    targets: &targets,
                }),
                multiview_mask: None,
                cache: None,
            },
        )
    }

    /// The visibility buffer's groups (decision 045): its pass's per current
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
        let key = (
            self.instances.generation,
            self.instances.alternate().is_some(),
            arena,
            selection,
            visibility.generation(),
        );
        if !self.visibility.groups.stale(&key) {
            return;
        }
        let buffers = self.instances.pose_buffers();
        let colors = self.instances.colors();
        let (draws, tiles, _) = visibility.buffers();
        let draw = wgpu::BindGroupEntry {
            binding: 8,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: draws,
                offset: 0,
                size: wgpu::BufferSize::new(16),
            }),
        };
        let arena = [meshes, clusters, pages];
        let pass_group = |current: usize| {
            let [poses, meshes, clusters, pages] = arena_entries(buffers[current], arena);
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("engine/visibility-group"),
                layout: &self.visibility.layout,
                entries: &[poses, meshes, clusters, pages, draw.clone()],
            })
        };
        let resolve_group = |current: usize, previous: usize| {
            let [poses, meshes, clusters, pages] = arena_entries(buffers[current], arena);
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("engine/visibility-resolve-group"),
                layout: &self.visibility.resolve_layout,
                entries: &[
                    poses,
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: colors.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buffers[previous].as_entire_binding(),
                    },
                    meshes,
                    clusters,
                    pages,
                    draw.clone(),
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: visible.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 10,
                        resource: wgpu::BindingResource::TextureView(visibility.ids_view()),
                    },
                    wgpu::BindGroupEntry {
                        binding: 11,
                        resource: tiles.as_entire_binding(),
                    },
                ],
            })
        };
        let pass_groups = [pass_group(0), pass_group(1)];
        let resolve_groups = [
            resolve_group(0, 0),
            resolve_group(0, 1),
            resolve_group(1, 0),
            resolve_group(1, 1),
        ];
        self.visibility
            .groups
            .set(key, (pass_groups, resolve_groups));
    }
}

/// Group entries of slot poses and the virtual arena's meshes, clusters and
/// pages (decision 044), at the bindings `meshes.wgsl` declares.
fn arena_entries<'a>(
    poses: &'a wgpu::Buffer,
    [meshes, clusters, pages]: [&'a wgpu::Buffer; 3],
) -> [wgpu::BindGroupEntry<'a>; 4] {
    [
        wgpu::BindGroupEntry {
            binding: 0,
            resource: poses.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 5,
            resource: meshes.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 6,
            resource: clusters.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 7,
            resource: pages.as_entire_binding(),
        },
    ]
}
