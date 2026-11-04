use bevy_ecs::prelude::*;
use engine_core::glam::{Mat4, Vec4};
use engine_core::math::{clamp, half, max, pow};
use engine_core::{Error, ErrorCode, Joined, Operation, Readback, View};
use engine_places::meshes::Instances;
use engine_render::debug::{
    DEBUG_CASTS, DEBUG_FADING, DEBUG_KIND_CARD, DEBUG_KIND_COVER, DEBUG_KIND_FALLS,
    DEBUG_KIND_MESH, DEBUG_KIND_PLACED, DEBUG_KIND_PLACED_CARD, DEBUG_KIND_SKY, DEBUG_KIND_TERRAIN,
    DEBUG_KIND_WATER, DEBUG_LEVEL_SHIFT,
};
use engine_render::{
    AfterRead, Buffer, Copied, CopyRegion, DebugModes, DebugShown, GpuPart, Install, Lifetime,
    Prepare, ViewTarget, wgpu,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

const IDS: u64 = 0;
const COUNT: u64 = 256;
const SCENE: u64 = 512;
const CANVAS: u64 = 768;
const TERRAIN: u64 = 1024;
const DEPTH: u64 = 1280;
const ARMED: u64 = 1284;
const CANVAS_ORDER: u64 = 1288;
const PROBED_BYTES: u64 = 1292;
const IN_ORDER: u32 = 1;
const BLUE_FIRST: u32 = 2;
const ROW_ALIGNMENT: u32 = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;

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
    armed: u32,
    pixel: (u32, u32),
    armed_size: (u32, u32),
    depth_row: u32,
    size: (u32, u32),
    view_projection: Mat4,
    bytes: Option<Vec<u8>>,
}

impl Probe {
    pub(crate) fn forces(&self) -> Option<bool> {
        match self.stage {
            Stage::Debug => Some(true),
            Stage::Shaded => Some(false),
            Stage::Idle | Stage::Armed | Stage::Waiting => None,
        }
    }

    pub(crate) fn coming(&self) -> Option<bool> {
        match self.stage {
            Stage::Armed => Some(true),
            Stage::Debug => Some(false),
            Stage::Idle | Stage::Shaded | Stage::Waiting => None,
        }
    }

    fn probing(&self) -> bool {
        self.stage != Stage::Idle && self.bytes.is_none()
    }

    fn result(
        &self,
        bytes: &[u8],
        modes: &Joined<'_, DebugModes>,
        instances: Option<&Instances>,
    ) -> Value {
        let record = word(bytes, IDS);
        let kind = drawn_kind(record & 0xf);
        let object = word(bytes, IDS + 4);
        let detail = word(bytes, IDS + 8);
        let depth = f32::from_bits(word(bytes, DEPTH));
        let fragments = half(word(bytes, COUNT) as u16);
        let (width, height) = (self.size.0.max(1) as f32, self.size.1.max(1) as f32);
        let clip = Vec4::new(
            (self.pixel.0 as f32 + 0.5) / width * 2.0 - 1.0,
            1.0 - (self.pixel.1 as f32 + 0.5) / height * 2.0,
            depth,
            1.0,
        );
        let point = self.view_projection.inverse() * clip;
        let color = |offset: u64, blue_first: bool| {
            let texel = word(bytes, offset).to_le_bytes();
            if blue_first {
                [texel[2], texel[1], texel[0]]
            } else {
                [texel[0], texel[1], texel[2]]
            }
        };
        let canvas = word(bytes, CANVAS_ORDER);
        let with_object = matches!(
            kind,
            Some(DrawnKind::Mesh | DrawnKind::Placed | DrawnKind::Card | DrawnKind::PlacedCard)
        );
        let with_detail = matches!(kind, Some(DrawnKind::Card | DrawnKind::PlacedCard));
        let entity = instances
            .filter(|_| matches!(kind, Some(DrawnKind::Mesh | DrawnKind::Card)))
            .and_then(|instances| {
                instances
                    .slots
                    .iter()
                    .find_map(|(&entity, &slot)| (slot == object).then_some(entity))
            });
        let probe = PixelProbe {
            pixel: [self.pixel.0, self.pixel.1],
            kind,
            level: (record >> DEBUG_LEVEL_SHIFT) & 0xf,
            fading: record & DEBUG_FADING != 0,
            casts_shadows: record & DEBUG_CASTS != 0,
            object: with_object.then_some(object),
            detail: with_detail.then_some(detail),
            normal: (kind.is_some() && kind != Some(DrawnKind::Sky))
                .then(|| unpacked_normal(word(bytes, IDS + 12))),
            fragments: max(fragments.round(), 0.0) as u32,
            depth,
            world: (depth > 0.0 && point.w != 0.0).then(|| (point.truncate() / point.w).to_array()),
            shaded: Some(scene_texel(bytes, SCENE)),
            displayed: (canvas != 0).then(|| color(CANVAS, canvas == BLUE_FIRST)),
        };
        let Ok(Value::Object(mut fields)) = serde_json::to_value(probe) else {
            return Value::Null;
        };
        let ids = [record, object, detail, word(bytes, TERRAIN)];
        let words = [
            f32::from_bits(word(bytes, TERRAIN + 4)),
            fragments,
            depth,
            0.0,
        ];
        for mode in modes.all() {
            if let Value::Object(decoded) = (mode.decode)(ids, words) {
                for (name, value) in decoded {
                    fields.entry(name).or_insert(value);
                }
            }
        }
        if let Some(entity) = entity {
            fields.insert(String::from("entity"), Value::from(entity.to_bits()));
        }
        Value::Object(fields)
    }
}

impl Readback for Probe {
    fn arrived(&mut self, _: u64, _: u32, bytes: &[u8]) {
        if self.stage != Stage::Idle && word(bytes, ARMED) == self.armed {
            self.bytes = Some(bytes.to_vec());
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

#[derive(Debug, Serialize)]
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
    shaded: Option<[u8; 3]>,
    displayed: Option<[u8; 3]>,
}

fn word(bytes: &[u8], offset: u64) -> u32 {
    let start = offset as usize;
    bytes.get(start..start + 4).map_or(0, |word| {
        u32::from_le_bytes([word[0], word[1], word[2], word[3]])
    })
}

fn scene_texel(bytes: &[u8], offset: u64) -> [u8; 3] {
    let (first, second) = (word(bytes, offset), word(bytes, offset + 4));
    [first & 0xffff, first >> 16, second & 0xffff].map(|bits| encoded(half(bits as u16)))
}

fn encoded(light: f32) -> u8 {
    let light = clamp(light, 0.0, 1.0);
    let shown = if light <= 0.003_130_8 {
        light * 12.92
    } else {
        1.055 * pow(light, 1.0 / 2.4) - 0.055
    };
    (shown * 255.0).round() as u8
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

pub(crate) fn advance(view: Res<View>, mut probe: ResMut<Probe>) {
    let stage = probe.stage;
    match stage {
        Stage::Idle => {}
        Stage::Armed => {
            let size = view.display_size();
            let probe = &mut *probe;
            probe.stage = Stage::Debug;
            probe.size = (size.x, size.y);
            probe.view_projection = view.view_projection();
        }
        Stage::Debug => probe.stage = Stage::Shaded,
        Stage::Shaded | Stage::Waiting => {
            let arrived = probe.bytes.is_some();
            probe.stage = if arrived { Stage::Idle } else { Stage::Waiting };
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
    type Params = (Res<'static, View>, ResMut<'static, Probe>, ResMut<'static, DebugShown>);
    type Request = PixelRequest;

    fn run(
        (view, mut probe, mut shown): SystemParamItem<'_, '_, Self::Params>,
        request: PixelRequest,
    ) -> Result<Value, Error> {
        let size = view.display_size();
        for (path, value, size) in [("$.x", request.x, size.x), ("$.y", request.y, size.y)] {
            if value >= size {
                return Err(Error::new(ErrorCode::OutOfRange, path)
                    .range(0.0, f64::from(size.saturating_sub(1)))
                    .got(value));
            }
        }
        let armed = probe.armed.wrapping_add(1);
        *probe = Probe {
            stage: Stage::Armed,
            armed,
            pixel: (request.x, request.y),
            armed_size: (size.x, size.y),
            depth_row: (size.x * 4).next_multiple_of(ROW_ALIGNMENT),
            ..Probe::default()
        };
        shown.next_debug = true;
        shown.next_whole = true;
        Ok(json!({ "pixel": [request.x, request.y] }))
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ProbedRequest {}

pub(crate) struct ProbedPixel;

impl Operation for ProbedPixel {
    const NAME: &'static str = "probed_pixel";
    type Params = (
        Res<'static, Probe>,
        Joined<'static, DebugModes>,
        Option<Res<'static, Instances>>,
    );
    type Request = ProbedRequest;

    fn run(
        (probe, modes, instances): SystemParamItem<'_, '_, Self::Params>,
        _: ProbedRequest,
    ) -> Result<Value, Error> {
        let result = probe
            .bytes
            .as_deref()
            .map(|bytes| probe.result(bytes, &modes, instances.as_deref()));
        Ok(json!({ "probing": probe.probing(), "probe": result }))
    }
}

fn texel(at: wgpu::Origin3d, offset: u64) -> CopyRegion {
    CopyRegion::ToBuffer {
        from: at,
        to: wgpu::TexelCopyBufferLayout {
            offset,
            bytes_per_row: None,
            rows_per_image: None,
        },
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    }
}

pub(crate) struct ProbeGpu {
    probed: Buffer<Self>,
    rows: Buffer<Self>,
    unread: Option<u32>,
}

impl GpuPart for ProbeGpu {
    type Reads = (Res<'static, Probe>, Res<'static, View>);
    const NAME: &'static str = "debug_view";
    const LAZY: bool = true;
    const LABELS: &'static [&'static str] = &["engine/debug-view"];

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        Ok(Self {
            probed: cx.buffer("debug_view_probe", Lifetime::Kept),
            rows: cx.buffer("debug_view_probe_depth", Lifetime::Kept),
            unread: None,
        })
    }

    fn prepare(
        &mut self,
        (probe, view): SystemParamItem<'_, '_, Self::Reads>,
        cx: &mut Prepare<'_, Self>,
    ) {
        let size = view.display_size();
        let at = wgpu::Origin3d {
            x: probe.pixel.0.min(size.x.saturating_sub(1)),
            y: probe.pixel.1.min(size.y.saturating_sub(1)),
            z: 0,
        };
        match probe.stage {
            Stage::Debug => {
                cx.allocate_buffer(self.probed, PROBED_BYTES);
                cx.write(self.probed, ARMED, &[probe.armed]);
                for (debug, offset) in [(0, IDS), (1, COUNT), (2, TERRAIN)] {
                    cx.copy_at_end(
                        Copied::Target(ViewTarget::Debug(debug)),
                        Copied::Buffer(self.probed),
                        &[texel(at, offset)],
                    );
                }
                if (size.x, size.y) == probe.armed_size {
                    let (width, height) = probe.armed_size;
                    let row = u64::from(probe.depth_row);
                    cx.allocate_buffer(self.rows, row * u64::from(height));
                    cx.copy_at_end(
                        Copied::Target(ViewTarget::Depth),
                        Copied::Buffer(self.rows),
                        &[CopyRegion::ToBuffer {
                            from: wgpu::Origin3d::ZERO,
                            to: wgpu::TexelCopyBufferLayout {
                                offset: 0,
                                bytes_per_row: Some(probe.depth_row),
                                rows_per_image: Some(height),
                            },
                            size: wgpu::Extent3d {
                                width,
                                height,
                                depth_or_array_layers: 1,
                            },
                        }],
                    );
                    cx.copy_at_end(
                        Copied::Buffer(self.rows),
                        Copied::Buffer(self.probed),
                        &[CopyRegion::Bytes {
                            from: u64::from(at.y) * row + u64::from(at.x) * 4,
                            to: DEPTH,
                            size: 4,
                        }],
                    );
                } else {
                    cx.write(self.probed, DEPTH, &[0_u32]);
                }
            }
            Stage::Shaded => {
                cx.allocate_buffer(self.probed, PROBED_BYTES);
                cx.allocate_buffer(self.rows, 0);
                cx.copy_at_end(
                    Copied::Target(ViewTarget::Scene),
                    Copied::Buffer(self.probed),
                    &[texel(at, SCENE)],
                );
                let order = cx.canvas().map_or(0, |format| {
                    if matches!(
                        format,
                        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
                    ) {
                        BLUE_FIRST
                    } else {
                        IN_ORDER
                    }
                });
                if order != 0 {
                    cx.copy_at_end(
                        Copied::Canvas,
                        Copied::Buffer(self.probed),
                        &[texel(at, CANVAS)],
                    );
                }
                cx.write(self.probed, CANVAS_ORDER, &[order]);
                self.unread = Some(probe.armed);
            }
            Stage::Idle | Stage::Armed | Stage::Waiting => {}
        }
        if matches!(probe.stage, Stage::Shaded | Stage::Waiting)
            && self.unread == Some(probe.armed)
            && cx.read_back::<Probe>(0, self.probed, 0..PROBED_BYTES, AfterRead::Keep)
        {
            self.unread = None;
        }
    }
}
