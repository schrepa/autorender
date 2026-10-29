use std::cell::Cell;
use std::mem::offset_of;
use std::rc::Rc;

use engine_core::{AutoExposure, RenderSnapshot, RenderStats};

use crate::bindings::EXPOSURE;
use crate::buffers::{ReadbackRing, storage, uniform};
use crate::cached::{Bound, GroupCache, Versioned};
use crate::gpu_layouts::constants::{
    ADAPTATION_MOST, ADAPTATION_POWER, EXPOSURE_BINS, EXPOSURE_GRID, EXPOSURE_GROUP, NOON_LIGHT,
};
use crate::gpu_layouts::exposure::{ExposureState, ExposureUniform};
use crate::gpu_layouts::scene::LightingUniform;
use crate::graph::{
    FrameGraph, GraphTable, Lifetime, PassDecl, PassId, ResourceDecl, ResourceId, Stage,
};
use crate::lazy::Lazy;
use crate::parts::{
    Create, Encode, GpuPart, PartDecl, PartEntry, Prelude, PreludeStage, Step, UniformsOut, View,
};
use crate::post::PostPass;
use crate::profiling::{GpuProfiler, Span};
use crate::shaders::entry::exposure;

const STATE_BYTES: u64 = size_of::<ExposureState>() as u64;
const UNIFORM_BYTES: u64 = size_of::<ExposureUniform>() as u64;
const BIN_BYTES: u64 = EXPOSURE_BINS as u64 * 4;

const DUSK: f32 = 0.069_756;
const NIGHT: f32 = 0.241_922;

pub(crate) static PART: PartEntry = PartEntry {
    decl: PartDecl {
        name: "exposure",
        brick: "post_effects",
        passes: &[PassDecl {
            name: "exposure",
            stage: Stage::Exposure,
        }],
        resources: &[ResourceDecl {
            name: "exposure",
            lifetime: Lifetime::Kept,
        }],
        hooks: &[],
        contributes: &[],
        effects: &[],
        labels: &["engine/exposure"],
        lazy: true,
        faces: &[],
        needs: &[],
        piece: None,
    },
    create,
};

fn create(_: &Create<'_>) -> Box<dyn GpuPart> {
    Box::new(ExposurePass::default())
}

#[derive(Clone, Copy, Default)]
struct Ids {
    pass: Option<PassId>,
    scene: Option<ResourceId>,
    depth: Option<ResourceId>,
    exposure: Option<ResourceId>,
}

#[derive(Default)]
pub(crate) struct ExposurePass {
    meter: Lazy<Meter>,
    frame: Option<ExposureFrame>,
    ids: Ids,
}

struct Meter {
    meter: wgpu::ComputePipeline,
    adapt: wgpu::ComputePipeline,
    layout: crate::pipelines::GroupLayout,
    uniform: wgpu::Buffer,
    bins: wgpu::Buffer,
    state: wgpu::Buffer,
    group: GroupCache<6>,
    written: Option<ExposureUniform>,
    fresh: bool,
    last_seconds: Option<f64>,
    last_camera: [u32; 16],
    showing: bool,
    readbacks: ReadbackRing<()>,
    read: Rc<Cell<Option<[f32; 3]>>>,
}

#[derive(Clone, Copy, Debug)]
struct ExposureFrame;

impl Meter {
    fn new(device: &wgpu::Device) -> Self {
        let layout = crate::pipelines::layout(device, &EXPOSURE);
        let pipeline_layout =
            crate::pipelines::layout_of(device, &crate::pipeline_layouts::EXPOSURE_PIPELINES);
        let pipeline = |entry| crate::pipelines::compute_entry(device, &pipeline_layout, entry);
        let copied = wgpu::BufferUsages::COPY_SRC;
        Self {
            meter: pipeline(exposure::meter_exposure),
            adapt: pipeline(exposure::adapt_exposure),
            layout,
            uniform: uniform(device, "engine/exposure-uniform", UNIFORM_BYTES),
            bins: storage(
                device,
                "engine/exposure-bins",
                BIN_BYTES,
                wgpu::BufferUsages::empty(),
            ),
            state: storage(device, "engine/exposure-state", STATE_BYTES, copied),
            group: GroupCache::default(),
            written: None,
            fresh: true,
            last_seconds: None,
            last_camera: [0; 16],
            showing: false,
            readbacks: ReadbackRing::new("engine/exposure-readback"),
            read: Rc::default(),
        }
    }

    fn prepare(
        &mut self,
        (device, queue): (&wgpu::Device, &crate::gpu_calls::Queue),
        snapshot: &RenderSnapshot,
        (scene, depth, drawn): (Bound<'_>, &Versioned<wgpu::TextureView>, (u32, u32)),
        lighting: &wgpu::Buffer,
    ) -> ExposureFrame {
        let settings: &AutoExposure = &snapshot.environment.auto_exposure;
        let authored = snapshot.environment.atmosphere.exposure;
        if self.fresh {
            let state = ExposureState {
                exposure: authored,
                stops: 0.0,
                metered: 0.0,
                held: 0.0,
                view: 0.0,
                day: 0.0,
            };
            queue.write_buffer(&self.state, 0, bytemuck::bytes_of(&state));
            self.fresh = false;
        }
        let seconds = snapshot.presented_seconds;
        let camera = snapshot.view_projection.to_cols_array().map(f32::to_bits);
        let moved = std::mem::replace(&mut self.last_camera, camera) != camera;
        let elapsed = self.last_seconds.replace(seconds).map_or(0.0, |last| {
            engine_core::math::max((seconds - last) as f32, 0.0)
        });
        let elapsed = if elapsed <= 0.0 && moved {
            -1.0
        } else {
            elapsed
        };
        let sky = snapshot.environment.sky.sanitized();
        let noon = if sky.enabled {
            sky.sun_intensity * NOON_LIGHT
        } else {
            0.0
        };
        let below = -snapshot
            .environment
            .atmosphere
            .sun_direction
            .normalize_or_zero()
            .y;
        let night = engine_core::math::clamp((below - DUSK) / (NIGHT - DUSK), 0.0, 1.0);
        let night = night * night * (3.0 - 2.0 * night);
        let most = ADAPTATION_MOST.log2();
        let most = most + (settings.night - most) * night;
        let values = ExposureUniform {
            frame: [authored, elapsed, settings.strength, settings.compensation],
            reach: [
                settings.brighten * (1.0 - night),
                settings.darken,
                settings.to_darker,
                settings.to_brighter,
            ],
            meter: [
                settings.skip_dark,
                settings.skip_bright,
                settings.center,
                settings.sky,
            ],
            day: [noon, sky.adaptation * ADAPTATION_POWER, most, 0.0],
            drawn: [drawn.0 as f32, drawn.1 as f32, 0.0, 0.0],
        };
        if self.written != Some(values) {
            queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&values));
            self.written = Some(values);
        }
        self.group.bind(
            device,
            "engine/exposure-group",
            (&EXPOSURE, &self.layout),
            [
                ("settings", Bound::fixed(self.uniform.as_entire_binding())),
                ("scene_color", scene),
                ("scene_depth", depth.bound()),
                ("bins", Bound::fixed(self.bins.as_entire_binding())),
                ("state", Bound::fixed(self.state.as_entire_binding())),
                ("lighting", Bound::fixed(lighting.as_entire_binding())),
            ],
        );
        ExposureFrame
    }

    fn show(&mut self, encoder: &mut wgpu::CommandEncoder, lighting: &wgpu::Buffer) {
        self.showing = true;
        let exposure = offset_of!(LightingUniform, horizon) + 12;
        encoder.copy_buffer_to_buffer(
            &self.state,
            offset_of!(ExposureState, exposure) as u64,
            lighting,
            exposure as u64,
            4,
        );
    }

    fn stop_showing(&mut self) -> bool {
        std::mem::take(&mut self.showing)
    }

    fn capacity_bytes(&self) -> u64 {
        UNIFORM_BYTES + BIN_BYTES + STATE_BYTES
    }

    fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &GpuProfiler,
    ) {
        let Some(group) = self.group.get() else {
            return;
        };
        encoder.clear_buffer(&self.bins, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("engine/exposure"),
                timestamp_writes: profiler.span("auto_exposure").map(Span::compute),
            });
            pass.set_bind_group(0, group, &[]);
            pass.set_pipeline(&self.meter);
            let groups = EXPOSURE_GRID / EXPOSURE_GROUP;
            pass.dispatch_workgroups(groups, groups, 1);
            pass.set_pipeline(&self.adapt);
            pass.dispatch_workgroups(1, 1, 1);
        }
        self.readbacks
            .record(device, encoder, &self.state, STATE_BYTES, ());
    }

    fn after_submit(&mut self) {
        let read = Rc::clone(&self.read);
        self.readbacks.after_submit(move |words, ()| {
            read.set(Some([1, 2, 4].map(|word| f32::from_bits(words.get(word)))));
        });
    }
}

impl GpuPart for ExposurePass {
    fn install(&mut self, table: &GraphTable) {
        self.ids = Ids {
            pass: table.pass_named("exposure"),
            scene: table.resource("scene"),
            depth: table.resource("depth"),
            exposure: table.resource("exposure"),
        };
    }

    fn uniforms(
        &mut self,
        cx: &mut Step<'_>,
        _lighting: &mut LightingUniform,
        out: &mut UniformsOut,
    ) {
        if !cx.frame.snapshot.environment.auto_exposure.enabled
            && self.meter.get_mut().is_some_and(Meter::stop_showing)
        {
            out.forget_lighting = true;
        }
    }

    fn prepare(&mut self, cx: &mut Step<'_>, prelude: &mut Prelude) {
        let snapshot = cx.frame.snapshot;
        let post = cx.parts.get::<PostPass>();
        let needs =
            post.is_some_and(PostPass::active) && snapshot.environment.auto_exposure.enabled;
        if !needs {
            self.frame = None;
            return;
        }
        let core = &*cx.core;
        let (device, queue) = (&core.device, &core.queue);
        let lighting = core.camera.lighting_buffer();
        let drawn = core.display.region.render.pixels();
        self.meter
            .create(device, "engine/exposure", || Meter::new(device));
        self.frame = self.meter.get_mut().and_then(|meter| {
            let targets = (post?.scene()?, &core.depth.view, drawn);
            let frame = meter.prepare((device, queue), snapshot, targets, lighting);
            let encoder = prelude.encoder(
                device,
                PreludeStage::Inputs,
                "pre_inputs",
                "engine/exposure-copy",
            );
            meter.show(encoder, lighting);
            Some(frame)
        });
    }

    fn declare(&self, cx: &View<'_>, graph: &mut FrameGraph) {
        let (Some(_), Some(pass)) = (self.frame, self.ids.pass) else {
            return;
        };
        let _ = graph
            .add(cx.table, pass)
            .read(self.ids.scene)
            .read(self.ids.depth)
            .write(self.ids.exposure);
    }

    fn encode(
        &mut self,
        _pass: usize,
        cx: &mut Encode<'_>,
        encoder: &mut wgpu::CommandEncoder,
    ) -> u32 {
        if let Some(meter) = self.meter.get() {
            meter.encode(&cx.core.device, encoder, &cx.core.profiler);
        }
        0
    }

    fn after_submit(&mut self) {
        if self.frame.is_none() {
            return;
        }
        if let Some(meter) = self.meter.get_mut() {
            meter.after_submit();
        }
    }

    fn capacity_bytes(&self) -> u64 {
        self.meter.get().map_or(0, Meter::capacity_bytes)
    }

    fn stats(&self, _cx: &View<'_>, stats: &mut RenderStats) {
        let [stops, metered, view] = self
            .meter
            .get()
            .and_then(|meter| meter.read.get())
            .filter(|_| self.frame.is_some())
            .unwrap_or_default();
        let hundredths = |stops: f32| (stops * 100.0).round() as i32;
        stats.sky.exposure = engine_core::ExposureStats {
            active: self.frame.is_some(),
            stops: hundredths(stops),
            metered: hundredths(metered),
            view: hundredths(view),
        };
    }

    fn poll(&mut self) -> bool {
        self.meter.poll()
    }

    fn disable(&mut self) -> bool {
        self.meter.disable()
    }

    fn disabled(&self) -> bool {
        self.meter.disabled()
    }

    #[cfg(feature = "dev")]
    fn inject(&mut self) -> bool {
        self.meter.inject()
    }

    #[cfg(feature = "dev")]
    fn encode_fault(&self, encoder: &mut wgpu::CommandEncoder) {
        self.meter.encode_fault(encoder);
    }
}
