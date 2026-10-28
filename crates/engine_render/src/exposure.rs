use std::cell::Cell;
use std::mem::offset_of;
use std::rc::Rc;

use engine_core::{AutoExposure, RenderSnapshot};

use crate::bindings::EXPOSURE;
use crate::buffers::{ReadbackRing, storage, uniform};
use crate::cached::{Bound, GroupCache, Versioned};
use crate::gpu_layouts::constants::{
    ADAPTATION_MOST, ADAPTATION_POWER, EXPOSURE_BINS, EXPOSURE_GRID, EXPOSURE_GROUP, NOON_LIGHT,
};
use crate::gpu_layouts::exposure::{ExposureState, ExposureUniform};
use crate::gpu_layouts::scene::LightingUniform;
use crate::profiling::Span;
use crate::render_features::{Encode, FeaturePass, RenderFeature};
use crate::render_graph::{FrameShape, Node, Pass, RenderGraph, Resource};
use crate::shaders::entry::exposure;

const STATE_BYTES: u64 = size_of::<ExposureState>() as u64;
const UNIFORM_BYTES: u64 = size_of::<ExposureUniform>() as u64;
const BIN_BYTES: u64 = EXPOSURE_BINS as u64 * 4;

const DUSK: f32 = 0.069_756;
const NIGHT: f32 = 0.241_922;

pub(crate) struct ExposurePass {
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
pub(crate) struct ExposureFrame;

impl ExposurePass {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("engine/exposure.wgsl"),
            source: wgpu::ShaderSource::Wgsl(crate::gpu_data::EXPOSURE_SHADER.into()),
        });
        let layout = crate::pipelines::layout(device, &EXPOSURE);
        let pipeline_layout = crate::pipelines::pipeline_layout(
            device,
            &crate::pipeline_layouts::EXPOSURE_PIPELINES,
            &[&layout],
        );
        let pipeline =
            |entry| crate::pipelines::compute_entry(device, &pipeline_layout, &module, entry);
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

    pub(crate) fn prepare(
        &mut self,
        (device, queue): (&wgpu::Device, &wgpu::Queue),
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

    pub(crate) fn show(&mut self, encoder: &mut wgpu::CommandEncoder, lighting: &wgpu::Buffer) {
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

    pub(crate) fn stop_showing(&mut self) -> bool {
        std::mem::take(&mut self.showing)
    }

    pub(crate) fn capacity_bytes(&self) -> u64 {
        UNIFORM_BYTES + BIN_BYTES + STATE_BYTES
    }
}

impl RenderFeature for ExposurePass {
    type Frame = ExposureFrame;
    type Stats = engine_core::ExposureStats;

    const LABELS: &'static [&'static str] = &["engine/exposure"];

    fn declare(_frame: &ExposureFrame, _shape: &FrameShape, graph: &mut RenderGraph) {
        graph
            .add(Pass::Feature(FeaturePass::Exposure))
            .read(Resource::Scene)
            .read(Resource::Depth)
            .write(Resource::Exposure);
    }

    fn encode(
        &self,
        _frame: &ExposureFrame,
        _node: &Node,
        encode: &Encode<'_>,
        encoder: &mut wgpu::CommandEncoder,
    ) -> u32 {
        let Some(group) = self.group.get() else {
            return 0;
        };
        encoder.clear_buffer(&self.bins, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("engine/exposure"),
                timestamp_writes: encode.profiler.span("auto_exposure").map(Span::compute),
            });
            pass.set_bind_group(0, group, &[]);
            pass.set_pipeline(&self.meter);
            let groups = EXPOSURE_GRID / EXPOSURE_GROUP;
            pass.dispatch_workgroups(groups, groups, 1);
            pass.set_pipeline(&self.adapt);
            pass.dispatch_workgroups(1, 1, 1);
        }
        self.readbacks
            .record(encode.device, encoder, &self.state, STATE_BYTES, ());
        0
    }

    fn after_submit(&mut self, frame: Option<&ExposureFrame>) {
        if frame.is_none() {
            return;
        }
        let read = Rc::clone(&self.read);
        self.readbacks.after_submit(move |words, ()| {
            read.set(Some([1, 2, 4].map(|word| f32::from_bits(words.get(word)))));
        });
    }

    fn capacity_bytes(&self) -> u64 {
        ExposurePass::capacity_bytes(self)
    }

    fn stats(
        this: Option<&Self>,
        frame: Option<&ExposureFrame>,
        _snapshot: &RenderSnapshot,
    ) -> engine_core::ExposureStats {
        let [stops, metered, view] = this
            .and_then(|this| this.read.get())
            .filter(|_| frame.is_some())
            .unwrap_or_default();
        let hundredths = |stops: f32| (stops * 100.0).round() as i32;
        engine_core::ExposureStats {
            active: frame.is_some(),
            stops: hundredths(stops),
            metered: hundredths(metered),
            view: hundredths(view),
        }
    }
}
