use std::cell::Cell;
use std::mem::offset_of;
use std::rc::Rc;

use engine_core::{Grading, RenderSnapshot, RenderStats, Sky, Sunlight};
use engine_faces::display::MeteredExposure;
use engine_faces::environment::{ADAPTATION_MOST, ADAPTATION_POWER, NOON_LIGHT};
use engine_render::buffers::{ReadbackRing, storage, uniform};
use engine_render::cached::{Bound, GroupCache, Versioned};
use engine_render::graph::{
    FrameGraph, GraphTable, Lifetime, PassDecl, PassId, ResourceDecl, ResourceId, Stage,
};
use engine_render::lazy::Lazy;
use engine_render::parts::{
    Begin, BeginOut, Create, Encode, GpuPart, PartDecl, PartEntry, Prelude, Step, View,
};
use engine_render::profiling::{GpuProfiler, Span};

use crate::AutoExposure;
use crate::bindings::EXPOSURE;
use crate::gpu_layouts::{
    EXPOSURE_BINS, EXPOSURE_GRID, EXPOSURE_GROUP, ExposureState, ExposureUniform,
};
use crate::shaders::entry::exposure;

const STATE_BYTES: u64 = size_of::<ExposureState>() as u64;
const UNIFORM_BYTES: u64 = size_of::<ExposureUniform>() as u64;
const BIN_BYTES: u64 = EXPOSURE_BINS as u64 * 4;

const DUSK: f32 = 0.069_756;
const NIGHT: f32 = 0.241_922;

pub(crate) static PART: PartEntry = PartEntry {
    decl: PartDecl {
        name: "exposure",
        brick: "auto_exposure",
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
        faces: &[engine_render::offer_face!(ExposurePass => dyn MeteredExposure)],
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
    enabled: bool,
    shows: bool,
    ids: Ids,
}

struct Meter {
    meter: wgpu::ComputePipeline,
    adapt: wgpu::ComputePipeline,
    layout: engine_render::pipelines::GroupLayout,
    uniform: wgpu::Buffer,
    bins: wgpu::Buffer,
    state: wgpu::Buffer,
    group: GroupCache<6>,
    written: Option<ExposureUniform>,
    fresh: bool,
    last_seconds: Option<f64>,
    last_camera: [u32; 16],
    readbacks: ReadbackRing<()>,
    read: Rc<Cell<Option<[f32; 3]>>>,
}

#[derive(Clone, Copy, Debug)]
struct ExposureFrame;

impl Meter {
    fn new(device: &wgpu::Device) -> Self {
        let layout = engine_render::pipelines::layout(device, &EXPOSURE);
        let pipeline_layout = engine_render::pipelines::layout_of(
            device,
            &crate::pipeline_layouts::EXPOSURE_PIPELINES,
        );
        let pipeline =
            |entry| engine_render::pipelines::compute_entry(device, &pipeline_layout, entry);
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
            readbacks: ReadbackRing::new("engine/exposure-readback"),
            read: Rc::default(),
        }
    }

    fn start(&mut self, queue: &engine_render::gpu_calls::Queue, authored: f32) {
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
    }

    fn prepare(
        &mut self,
        (device, queue): (&wgpu::Device, &engine_render::gpu_calls::Queue),
        snapshot: &RenderSnapshot,
        (scene, depth, drawn): (Bound<'_>, &Versioned<wgpu::TextureView>, (u32, u32)),
        pieces: Bound<'_>,
    ) -> ExposureFrame {
        let settings = snapshot.section::<AutoExposure>();
        let authored = snapshot.section::<Grading>().values().exposure;
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
        let sky = snapshot.section::<Sky>();
        let noon = if sky.enabled {
            sky.sun_intensity * NOON_LIGHT
        } else {
            0.0
        };
        let below = -snapshot
            .section::<Sunlight>()
            .values()
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
                ("pieces", pieces),
            ],
        );
        ExposureFrame
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
        if let Some(meter) = self.meter.get_mut() {
            meter.group.clear();
        }
    }

    fn begin(&mut self, cx: &mut Begin<'_>, _out: &mut BeginOut) {
        let snapshot = cx.snapshot;
        self.enabled = snapshot.section::<AutoExposure>().enabled;
        if self.enabled {
            let device = &cx.core.device;
            self.meter
                .create(device, "engine/exposure", || Meter::new(device));
        } else {
            self.meter.release();
        }
        let post = cx.parts.post().is_some_and(|post| !post.disabled());
        self.shows = self.enabled && post && self.meter.get().is_some();
        if self.shows
            && let Some(meter) = self.meter.get_mut()
        {
            meter.start(
                &cx.core.queue,
                snapshot.section::<Grading>().values().exposure,
            );
        }
    }

    fn prepare(&mut self, cx: &mut Step<'_>, _prelude: &mut Prelude) {
        if !self.enabled || cx.frame.kind.debug {
            self.frame = None;
            return;
        }
        let snapshot = cx.frame.snapshot;
        let post = cx.parts.post();
        let core = &*cx.core;
        let (device, queue) = (&core.device, &core.queue);
        let pieces = core.pieces_bound();
        let drawn = core.display.region.render.pixels();
        self.frame = self.meter.get_mut().and_then(|meter| {
            let targets = (post?.scene()?, &core.depth.view, drawn);
            Some(meter.prepare((device, queue), snapshot, targets, pieces))
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

impl MeteredExposure for ExposurePass {
    fn metered(&self) -> Option<(&wgpu::Buffer, u64)> {
        let meter = self.meter.get().filter(|_| self.shows)?;
        Some((&meter.state, offset_of!(ExposureState, exposure) as u64))
    }
}
