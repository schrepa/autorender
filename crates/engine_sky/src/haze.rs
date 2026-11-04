use engine_core::Error;
use engine_render::{
    Access, Attach, Bind, Buffer, DepthUse, Encode, FULLSCREEN, Install, Lifetime, Own, Pass,
    Pipeline, Prepare, RenderPipeline, Shape, Texture, ViewTarget, wgpu,
};
use glam::{Mat4, UVec2, Vec3};

use crate::gpu_layouts::HazeView;
use crate::sky::{SkyGpu, inverse_or_identity};

const HAZE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

#[derive(Clone, Copy)]
struct History {
    view_projection: Mat4,
    eye: Vec3,
    drawn: (u32, u32),
}

pub(crate) struct HazeFrame {
    pub(crate) view_projection: Mat4,
    pub(crate) eye: Vec3,
    pub(crate) display: UVec2,
    pub(crate) render: UVec2,
}

#[derive(Default)]
pub(crate) struct HazePlan {
    runs: bool,
    size: Option<UVec2>,
    drawn: (u32, u32),
    history: Option<History>,
    kept: Option<History>,
    frames: u32,
    uniform: HazeView,
}

impl HazePlan {
    pub(crate) fn runs(&self) -> bool {
        self.runs
    }

    pub(crate) fn submitted(&mut self) {
        self.history = self.kept.take();
        if self.history.is_some() {
            self.frames = self.frames.wrapping_add(1);
        }
    }

    pub(crate) fn prepare(&mut self, runs: bool, frame: &HazeFrame) -> [f32; 4] {
        self.runs = false;
        if !runs {
            self.release();
            return [0.0; 4];
        }
        let size = frame.display;
        if self.size != Some(size) {
            self.size = Some(size);
            self.history = None;
        }
        self.drawn = (frame.render.x, frame.render.y);
        self.kept = Some(History {
            view_projection: frame.view_projection,
            eye: frame.eye,
            drawn: self.drawn,
        });
        let history = self.history;
        let turn = (self.frames % 1024) as f32 * 0.618_034;
        let turn = turn - turn.floor();
        let (width, height) = (frame.render.x as f32, frame.render.y as f32);
        let view_projection = frame.view_projection;
        let focal = view_projection.row(1).truncate().length();
        self.uniform = HazeView {
            inverse_view_projection: inverse_or_identity(view_projection).to_cols_array_2d(),
            eye: frame.eye.extend(focal * height * 0.5).to_array(),
            size: [width, height, 1.0 / width, 1.0 / height],
            previous_view_projection: history
                .map_or(view_projection, |last| last.view_projection)
                .to_cols_array_2d(),
            previous_eye: history
                .map_or(frame.eye, |last| last.eye)
                .extend(0.0)
                .to_array(),
        };
        self.runs = true;
        history.map_or([turn, 0.0, 0.0, 1.0], |last| {
            [turn, last.drawn.0 as f32, last.drawn.1 as f32, 1.0]
        })
    }

    fn release(&mut self) {
        self.size = None;
        self.history = None;
    }
}

pub(crate) struct Haze {
    marched: Texture<SkyGpu>,
    hazed: Texture<SkyGpu>,
    history_texture: Texture<SkyGpu>,
    uniform: Buffer<SkyGpu>,
    march_pass: Pass<SkyGpu>,
    filter_pass: Pass<SkyGpu>,
    marching: Pipeline<SkyGpu, Own>,
    filtering: Pipeline<SkyGpu, Own>,
    drawn: (u32, u32),
    written: Option<HazeView>,
}

impl Haze {
    pub(crate) fn install(cx: &mut Install<'_, SkyGpu>) -> Result<Self, Error> {
        use crate::shaders::sky::{haze, haze_filter, haze_march, haze_view};
        let marched = cx.texture("haze_march", HAZE_FORMAT, Lifetime::Frame);
        let hazed = cx.texture("haze", HAZE_FORMAT, Lifetime::Kept);
        let history_texture = cx.texture("haze_history", HAZE_FORMAT, Lifetime::Previous);
        let uniform = cx.buffer("haze_view", Lifetime::Kept);
        cx.frame(haze::HAZE, Bind::Texture(hazed))?;
        let march_pass = cx
            .pass("haze")
            .label("engine/post-haze")
            .span("haze")
            .color(marched, Access::Clear)
            .id();
        let filter_pass = cx
            .pass("haze_filter")
            .after(march_pass)
            .label("engine/post-haze-filter")
            .span("haze")
            .color(hazed, Access::Load)
            .color(history_texture, Access::Load)
            .id();
        let fullscreen = |fragment| RenderPipeline {
            vertex: FULLSCREEN,
            fragment: Some(fragment),
            debug: None,
            vertices: &[],
            primitive: wgpu::PrimitiveState::default(),
            depth: DepthUse::None,
            blend: None,
            writes: wgpu::ColorWrites::ALL,
        };
        let viewed = || {
            [
                (haze_view::POST, Bind::Buffer(uniform)),
                (haze_view::DEPTH_TEXTURE, Bind::Target(ViewTarget::Depth)),
            ]
        };
        let marching =
            cx.render_pipeline(march_pass, &fullscreen(haze_march::FS_HAZE), &viewed())?;
        let mut filtered = Vec::from(viewed());
        filtered.extend([
            (haze_filter::MARCHED, Bind::Texture(marched)),
            (haze_filter::HAZE_HISTORY, Bind::Texture(history_texture)),
        ]);
        let filtering =
            cx.render_pipeline(filter_pass, &fullscreen(haze_filter::FS_HAZE_FILTER), &filtered)?;
        Ok(Self {
            marched,
            hazed,
            history_texture,
            uniform,
            march_pass,
            filter_pass,
            marching,
            filtering,
            drawn: (0, 0),
            written: None,
        })
    }

    pub(crate) fn prepare(&mut self, cx: &mut Prepare<'_, SkyGpu>, plan: &HazePlan) {
        let Some(size) = plan.size.filter(|_| plan.runs) else {
            self.written = None;
            return;
        };
        let extent = wgpu::Extent3d {
            width: size.x,
            height: size.y,
            depth_or_array_layers: 1,
        };
        for texture in [self.marched, self.hazed, self.history_texture] {
            cx.allocate(texture, extent, 1);
        }
        cx.allocate_buffer(self.uniform, size_of::<HazeView>() as u64);
        self.drawn = plan.drawn;
        if self.written != Some(plan.uniform) {
            cx.write(self.uniform, 0, &[plan.uniform]);
            self.written = Some(plan.uniform);
        }
        cx.run(self.march_pass);
        cx.run(self.filter_pass);
    }

    pub(crate) fn owns(&self, pass: Pass<SkyGpu>) -> bool {
        pass == self.march_pass || pass == self.filter_pass
    }

    pub(crate) fn encode(&self, pass: Pass<SkyGpu>, cx: &mut Encode<'_, SkyGpu>) -> u32 {
        let pipeline = if pass == self.march_pass {
            &self.marching
        } else {
            &self.filtering
        };
        let mut draw = cx.render(Attach::Declared, None);
        draw.within(self.drawn);
        draw.draw(
            pipeline,
            Shape::Pulled {
                vertices: 0..3,
                instances: 0..1,
            },
        );
        0
    }
}
