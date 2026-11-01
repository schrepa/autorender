use bevy_ecs::prelude::*;
use engine_core::glam::{Mat4, Vec4};
use engine_core::math::max;
use engine_core::{Error, ErrorCode, Joined, Operation, Readback, View};
use engine_render::debug::{
    DEBUG_CASTS, DEBUG_FADING, DEBUG_KIND_CARD, DEBUG_KIND_COVER, DEBUG_KIND_FALLS,
    DEBUG_KIND_MESH, DEBUG_KIND_PLACED, DEBUG_KIND_PLACED_CARD, DEBUG_KIND_SKY, DEBUG_KIND_TERRAIN,
    DEBUG_KIND_WATER, DEBUG_LEVEL_SHIFT,
};
use engine_render::{
    AfterRead, Bind, Buffer, ComputePipeline, Count, DebugModes, Encode, GpuPart, Install,
    Lifetime, Pass, Prepare, ViewTarget,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::DebugView;
use crate::gpu::ProbedTargets;

const TARGETS: u32 = 0;
const SCENE: u32 = 1;
const PIXEL_BYTES: u64 = 16;
const SCENE_BYTES: u64 = 16;
const TARGET_BYTES: u64 = size_of::<ProbedTargets>() as u64;

type Decode = fn([u32; 4], [f32; 4]) -> Value;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Stage {
    #[default]
    Idle,
    Armed,
    Debug,
    Shaded,
    Waiting,
}

#[derive(Resource, Default)]
pub(crate) struct Probe {
    stage: Stage,
    pixel: (u32, u32),
    armed_size: (u32, u32),
    size: (u32, u32),
    view_projection: Mat4,
    mode: Option<(&'static str, Decode)>,
    targets: Option<ProbedTargets>,
    scene: Option<u32>,
    result: Option<Value>,
}

impl Probe {
    pub(crate) fn armed(&self) -> bool {
        self.stage != Stage::Idle
    }

    pub(crate) fn forces(&self) -> Option<bool> {
        match self.stage {
            Stage::Debug => Some(true),
            Stage::Shaded => Some(false),
            Stage::Idle | Stage::Armed | Stage::Waiting => None,
        }
    }

    fn probed(&self, targets: ProbedTargets, scene: u32) -> PixelProbe {
        let [record, object, detail, normal] = targets.ids;
        let kind = drawn_kind(record & 0xf);
        let (x, y) = self.pixel;
        let depth = if self.size == self.armed_size {
            targets.depth
        } else {
            0.0
        };
        let fragments = targets.count;
        let (width, height) = (self.size.0.max(1) as f32, self.size.1.max(1) as f32);
        let clip = Vec4::new(
            (x as f32 + 0.5) / width * 2.0 - 1.0,
            1.0 - (y as f32 + 0.5) / height * 2.0,
            depth,
            1.0,
        );
        let point = self.view_projection.inverse() * clip;
        let texel = scene.to_le_bytes();
        let with_object = matches!(
            kind,
            Some(DrawnKind::Mesh | DrawnKind::Placed | DrawnKind::Card | DrawnKind::PlacedCard)
        );
        let with_detail = matches!(kind, Some(DrawnKind::Card | DrawnKind::PlacedCard));
        let [first, second] = targets.terrain;
        PixelProbe {
            pixel: [x, y],
            kind,
            level: (record >> DEBUG_LEVEL_SHIFT) & 0xf,
            fading: record & DEBUG_FADING != 0,
            casts_shadows: record & DEBUG_CASTS != 0,
            object: with_object.then_some(object),
            detail: with_detail.then_some(detail),
            normal: (kind.is_some() && kind != Some(DrawnKind::Sky))
                .then(|| unpacked_normal(normal)),
            fragments: max(fragments.round(), 0.0) as u32,
            depth,
            world: (depth > 0.0 && point.w != 0.0).then(|| (point.truncate() / point.w).to_array()),
            shaded: [texel[0], texel[1], texel[2]],
            mode: self.mode.map(|(name, _)| name),
            decoded: self.mode.map_or(Value::Null, |(_, decode)| {
                decode(
                    [record, object, detail, first],
                    [f32::from_bits(second), fragments, depth, 0.0],
                )
            }),
        }
    }
}

impl Readback for Probe {
    fn arrived(&mut self, _: u64, tag: u32, bytes: &[u8]) {
        if matches!(self.stage, Stage::Idle | Stage::Armed) {
            return;
        }
        match tag {
            TARGETS => {
                self.targets = bytes
                    .get(..TARGET_BYTES as usize)
                    .map(bytemuck::pod_read_unaligned);
            }
            _ => {
                self.scene = bytes
                    .get(..4)
                    .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]));
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum DrawnKind {
    Mesh,
    Placed,
    Card,
    PlacedCard,
    #[serde(rename = "terrain")]
    Ground,
    #[serde(rename = "cover")]
    Blades,
    #[serde(rename = "water")]
    Lakes,
    Falls,
    Sky,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct PixelProbe {
    pixel: [u32; 2],
    kind: Option<DrawnKind>,
    level: u32,
    fading: bool,
    casts_shadows: bool,
    object: Option<u32>,
    detail: Option<u32>,
    normal: Option<[f32; 3]>,
    fragments: u32,
    depth: f32,
    world: Option<[f32; 3]>,
    shaded: [u8; 3],
    mode: Option<&'static str>,
    decoded: Value,
}

fn unpacked_normal(packed: u32) -> [f32; 3] {
    let snorm = |bits: u32| max(f32::from(bits as u16 as i16) / 32767.0, -1.0);
    let (x, y) = (snorm(packed & 0xffff), snorm(packed >> 16));
    let z = 1.0 - x.abs() - y.abs();
    let fold = max(-z, 0.0);
    let x = x - if x >= 0.0 { fold } else { -fold };
    let y = y - if y >= 0.0 { fold } else { -fold };
    let length = max((x * x + y * y + z * z).sqrt(), 1.0e-6);
    [x / length, y / length, z / length]
}

fn drawn_kind(kind: u32) -> Option<DrawnKind> {
    Some(match kind {
        DEBUG_KIND_MESH => DrawnKind::Mesh,
        DEBUG_KIND_PLACED => DrawnKind::Placed,
        DEBUG_KIND_CARD => DrawnKind::Card,
        DEBUG_KIND_PLACED_CARD => DrawnKind::PlacedCard,
        DEBUG_KIND_TERRAIN => DrawnKind::Ground,
        DEBUG_KIND_COVER => DrawnKind::Blades,
        DEBUG_KIND_WATER => DrawnKind::Lakes,
        DEBUG_KIND_FALLS => DrawnKind::Falls,
        DEBUG_KIND_SKY => DrawnKind::Sky,
        _ => return None,
    })
}

pub(crate) fn advance(
    view: Res<View>,
    settings: Res<DebugView>,
    modes: Joined<DebugModes>,
    mut probe: ResMut<Probe>,
) {
    let stage = probe.stage;
    match stage {
        Stage::Idle => {}
        Stage::Armed => {
            let size = view.render_size();
            let mode = if settings.showing() {
                modes
                    .iter()
                    .find(|mode| mode.name == settings.view.as_str())
                    .map(|mode| (mode.name, mode.decode))
            } else {
                None
            };
            let probe = &mut *probe;
            probe.stage = Stage::Debug;
            probe.size = (size.x, size.y);
            probe.view_projection = view.view_projection();
            probe.mode = mode;
        }
        Stage::Debug => probe.stage = Stage::Shaded,
        Stage::Shaded => probe.stage = Stage::Waiting,
        Stage::Waiting => {
            let (Some(targets), Some(scene)) = (probe.targets, probe.scene) else {
                return;
            };
            let result = serde_json::to_value(probe.probed(targets, scene)).unwrap_or(Value::Null);
            let probe = &mut *probe;
            probe.stage = Stage::Idle;
            probe.result = Some(result);
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PixelRequest {
    x: u32,
    y: u32,
}

pub(crate) struct ProbePixel;

impl Operation for ProbePixel {
    const NAME: &'static str = "probe_pixel";
    type Params = (Res<'static, View>, ResMut<'static, Probe>);
    type Request = PixelRequest;

    fn run(
        (view, mut probe): SystemParamItem<'_, '_, Self::Params>,
        request: PixelRequest,
    ) -> Result<Value, Error> {
        let size = view.render_size();
        for (path, value, size) in [("$.x", request.x, size.x), ("$.y", request.y, size.y)] {
            if value >= size {
                return Err(Error::new(ErrorCode::OutOfRange, path)
                    .range(0.0, f64::from(size.saturating_sub(1)))
                    .got(value));
            }
        }
        *probe = Probe {
            stage: Stage::Armed,
            pixel: (request.x, request.y),
            armed_size: (size.x, size.y),
            ..Probe::default()
        };
        Ok(json!({ "pixel": [request.x, request.y] }))
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ProbedRequest {}

pub(crate) struct ProbedPixel;

impl Operation for ProbedPixel {
    const NAME: &'static str = "probed_pixel";
    type Params = Res<'static, Probe>;
    type Request = ProbedRequest;

    fn run(probe: SystemParamItem<'_, '_, Self::Params>, _: ProbedRequest) -> Result<Value, Error> {
        Ok(json!({ "probing": probe.armed(), "probe": probe.result }))
    }
}

pub(crate) struct ProbeGpu {
    pixel: Buffer<Self>,
    targets: Buffer<Self>,
    scene: Buffer<Self>,
    read_targets: Pass<Self>,
    read_scene: Pass<Self>,
    reading_targets: ComputePipeline<Self>,
    reading_scene: ComputePipeline<Self>,
}

impl GpuPart for ProbeGpu {
    type Reads = Res<'static, Probe>;

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        use crate::shaders::debug_view::{probe_scene, probe_targets};
        let pixel = cx.buffer("debug_view_probe_pixel", Lifetime::Kept);
        let targets = cx.buffer("debug_view_probe_targets", Lifetime::Kept);
        let scene = cx.buffer("debug_view_probe_scene", Lifetime::Kept);
        let read_targets = cx.pass("debug_view_probe_targets").id();
        let read_scene = cx.pass("debug_view_probe_scene").id();
        let reading_targets = cx.compute_pipeline(
            read_targets,
            probe_targets::CS_PROBE_TARGETS,
            &[
                (probe_targets::PROBE_PIXEL, Bind::Buffer(pixel)),
                (probe_targets::PROBED, Bind::Buffer(targets)),
                (probe_targets::PROBE_IDS, Bind::Target(ViewTarget::Debug(0))),
                (probe_targets::PROBE_COUNT, Bind::Target(ViewTarget::Debug(1))),
                (probe_targets::PROBE_TERRAIN, Bind::Target(ViewTarget::Debug(2))),
                (probe_targets::PROBE_DEPTH, Bind::Target(ViewTarget::Depth)),
            ],
        )?;
        let reading_scene = cx.compute_pipeline(
            read_scene,
            probe_scene::CS_PROBE_SCENE,
            &[
                (probe_scene::PROBE_PIXEL, Bind::Buffer(pixel)),
                (probe_scene::SHADED, Bind::Buffer(scene)),
                (probe_scene::PROBE_SCENE, Bind::Target(ViewTarget::Scene)),
            ],
        )?;
        Ok(Self {
            pixel,
            targets,
            scene,
            read_targets,
            read_scene,
            reading_targets,
            reading_scene,
        })
    }

    fn prepare(&mut self, probe: SystemParamItem<'_, '_, Self::Reads>, cx: &mut Prepare<'_, Self>) {
        let (pass, buffer, tag, bytes) = match probe.stage {
            Stage::Debug => (self.read_targets, self.targets, TARGETS, TARGET_BYTES),
            Stage::Shaded => (self.read_scene, self.scene, SCENE, SCENE_BYTES),
            Stage::Idle | Stage::Armed | Stage::Waiting => return,
        };
        cx.allocate_buffer(self.pixel, PIXEL_BYTES);
        cx.allocate_buffer(buffer, bytes);
        cx.write(self.pixel, 0, &[probe.pixel.0, probe.pixel.1, 0, 0]);
        cx.run(pass);
        let _ = cx.read_back::<Probe>(tag, buffer, 0..bytes, AfterRead::Keep);
    }

    fn encode(&mut self, pass: Pass<Self>, cx: &mut Encode<'_, Self>) -> u32 {
        let pipeline = if pass == self.read_targets {
            &self.reading_targets
        } else {
            &self.reading_scene
        };
        cx.dispatch(pipeline, Count::Groups([1, 1, 1]))
    }
}
