use super::*;

pub(crate) struct Core {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: crate::gpu_calls::Queue,
    pub(crate) adapter: crate::AdapterInfo,
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) debugger_capture: bool,
    pub(crate) display: Display,
    pub(crate) depth: DepthTarget,
    pub(crate) meshes: BTreeMap<MeshHandle, GpuMesh>,
    pub(crate) mesh_arena: MeshArena,
    pub(crate) mesh_layout: u64,
    pub(crate) mesh_uploads: MeshUploads,
    pub(crate) materials: MaterialStore,
    pub(crate) plan: BatchPlan,
    pub(crate) draw_order: DrawOrder,
    pub(crate) structure_keys: StructureKeys,
    pub(crate) instances: Instances,
    pub(crate) uploads: UploadCache,
    pub(crate) camera: CameraBindings,
    pub(crate) mesh_pipelines: MeshPipelines,
    pub(crate) pipeline_counts: PipelineCounts,
    pub(crate) gpu_presentation: GpuPresentationPass,
    pub(crate) culling: Culling,
    pub(crate) impostors: ImpostorDraws,
    pub(crate) switch_fades: frame::SwitchFades,
    pub(crate) last_seconds: Option<f32>,
    pub(crate) gpu_frame: bool,
    pub(crate) health: Arc<Mutex<health::Reports>>,
    pub(crate) profiler: GpuProfiler,
    pub(crate) shared: crate::bindings::SharedGroups,
}

impl Core {
    pub(crate) fn structure_key(&self) -> StructureKey {
        (
            self.plan.generation(),
            self.mesh_layout,
            self.materials.generation(),
        )
    }

    pub(crate) fn mesh_draw(&self, mesh: MeshHandle) -> Option<MeshDraw<'_>> {
        self.meshes
            .get(&mesh)
            .map(|mesh| mesh.draw(&self.mesh_arena))
    }

    pub(crate) fn material_group(&self, material: MaterialHandle) -> Option<&wgpu::BindGroup> {
        self.materials.group(material)
    }

    pub(crate) fn shared_group(&self, group: &str, without: &[&str]) -> &wgpu::BindGroup {
        self.camera.group(group, without)
    }

    pub(crate) fn probe_group(&self, face: usize, without: &[&str]) -> &wgpu::BindGroup {
        self.camera.probe_group(face, without)
    }

    pub(crate) fn frame_poses(&self, parity: usize) -> FramePoses<'_> {
        self.instances.frame_poses(parity)
    }

    pub(crate) fn render_pixels(&self) -> (u32, u32) {
        self.display.region.render.pixels()
    }

    pub(crate) fn canvas_format(&self) -> wgpu::TextureFormat {
        self.display.format()
    }

    pub(crate) fn forget_temporal(&mut self) {
        let current = |&(kind, _): &(PassKind, Variant)| kind.targets() != Some(Targets::Temporal);
        self.mesh_pipelines.variants.retain(|key, _| current(key));
        self.impostors.pipelines.forget_pipelines(Targets::Temporal);
        self.instances.release_alternate();
    }
}
