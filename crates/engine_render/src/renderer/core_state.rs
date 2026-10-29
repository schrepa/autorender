use super::*;
use crate::entries::MeshEntries;

pub struct Core {
    pub device: wgpu::Device,
    pub queue: crate::gpu_calls::Queue,
    pub adapter: crate::AdapterInfo,
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) debugger_capture: bool,
    pub display: Display,
    pub depth: DepthTarget,
    pub meshes: BTreeMap<MeshHandle, GpuMesh>,
    pub mesh_arena: MeshArena,
    pub mesh_layout: u64,
    pub(crate) mesh_uploads: MeshUploads,
    pub materials: MaterialStore,
    pub plan: BatchPlan,
    pub draw_order: DrawOrder,
    pub(crate) structure_keys: StructureKeys,
    pub instances: Instances,
    pub(crate) uploads: UploadCache,
    pub camera: CameraBindings,
    pub mesh_pipelines: MeshPipelines,
    pub mesh_entries: Option<&'static MeshEntries>,
    pub(crate) pipeline_counts: PipelineCounts,
    pub gpu_presentation: GpuPresentationPass,
    pub(crate) culling: Culling,
    pub impostors: ImpostorDraws,
    pub switch_fades: frame::SwitchFades,
    pub(crate) last_seconds: Option<f32>,
    pub(crate) gpu_frame: bool,
    pub(crate) health: Arc<Mutex<health::Reports>>,
    pub profiler: GpuProfiler,
    pub shared: crate::bindings::SharedGroups,
    pub previous: PreviousFrame,
    pub debug: crate::debug_targets::DebugFrame,
    pub mesh_entries: Option<&'static crate::entries::MeshEntries>,
    pub warm: Vec<SceneShading>,
    pub views: Option<(wgpu::TextureFormat, bool)>,
}

impl Core {
    pub(crate) fn structure_key(&self) -> StructureKey {
        (
            self.plan.generation(),
            self.mesh_layout,
            self.materials.generation(),
        )
    }

    pub fn mesh_draw(&self, mesh: MeshHandle) -> Option<MeshDraw<'_>> {
        self.meshes
            .get(&mesh)
            .map(|mesh| mesh.draw(&self.mesh_arena))
    }

    pub fn material_group(&self, material: MaterialHandle) -> Option<&wgpu::BindGroup> {
        self.materials.group(material)
    }

    pub fn shared_group(&self, group: &str, without: &[&str]) -> &wgpu::BindGroup {
        self.camera.group(group, without)
    }

    pub fn frame_poses(&self, parity: usize) -> FramePoses<'_> {
        self.instances.frame_poses(parity)
    }

    pub fn render_pixels(&self) -> (u32, u32) {
        self.display.region.render.pixels()
    }

    pub fn canvas_format(&self) -> wgpu::TextureFormat {
        self.display.format()
    }

    pub(crate) fn forget_temporal(&mut self) {
        self.mesh_pipelines.forget(Some(Targets::Temporal));
        self.impostors.pipelines.forget_pipelines(Targets::Temporal);
        self.instances.release_alternate();
    }
}
