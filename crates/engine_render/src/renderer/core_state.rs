use super::*;
use crate::entries::MeshEntries;

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
    pub(crate) mesh_entries: Option<&'static MeshEntries>,
    pub(crate) pipeline_counts: PipelineCounts,
    pub(crate) gpu_presentation: GpuPresentationPass,
    pub(crate) culling: Culling,
    pub(crate) impostors: ImpostorDraws,
    pub(crate) switch_fades: frame::SwitchFades,
    pub(crate) last_seconds: Option<f32>,
    pub(crate) gpu_frame: bool,
    pub(crate) health: Arc<Mutex<health::Reports>>,
    pub(crate) profiler: GpuProfiler,
    pub(crate) previous: PreviousFrame,
    pub(crate) debug: crate::debug_targets::DebugFrame,
    pub(crate) views: Option<(wgpu::TextureFormat, bool)>,
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

    pub(crate) fn frame_poses(&self, parity: usize) -> FramePoses<'_> {
        self.instances.frame_poses(parity)
    }

    pub(crate) fn render_pixels(&self) -> (u32, u32) {
        self.display.region.render.pixels()
    }

    pub(crate) fn canvas_format(&self) -> wgpu::TextureFormat {
        self.display.format()
    }

    pub(crate) fn frame_resources(&mut self, kind: FrameKind, views: bool) {
        if kind.debug {
            self.debug.prepare(&self.device, self.display.size());
        } else {
            self.debug.release();
        }
        if !views {
            self.mesh_pipelines.release_view();
            self.camera.release_views();
            self.gpu_presentation.release_views();
        }
    }

    pub(crate) fn forget_temporal(&mut self) {
        self.mesh_pipelines.forget(Some(Targets::Temporal));
        self.impostors.pipelines.forget_pipelines(Targets::Temporal);
        self.instances.release_alternate();
    }
}
