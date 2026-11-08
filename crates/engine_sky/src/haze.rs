use bevy_ecs::prelude::Resource;
use engine_core::{Error, Readable, Readback};
use engine_render::{
    Access, Attach, Bind, DepthUse, Encode, FULLSCREEN, Install, Lifetime, Own, Pass,
    Pipeline, Prepare, RenderPipeline, Shape, TargetName, Texture, wgpu,
};
use glam::UVec2;

use crate::sky::SkyGpu;

const HAZE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

#[derive(Resource, Default)]
pub(crate) struct HazeSteps {
    missed: bool,
}

impl HazeSteps {
    pub(crate) fn missed(&mut self) -> bool {
        std::mem::take(&mut self.missed)
    }
}

impl Readable for HazeSteps {
    fn arrived(&mut self, _: u32, read: Readback<&[u8]>) {
        self.missed |= read.value.first().is_none_or(|&ran| ran == 0);
    }
}

pub(crate) struct HazeFrame {
    pub(crate) display: UVec2,
    pub(crate) render: UVec2,
    pub(crate) history: bool,
    pub(crate) frame: u64,
}

#[derive(Default)]
pub(crate) struct HazePlan {
    runs: bool,
    size: UVec2,
    drawn: (u32, u32),
    kept: bool,
}

impl HazePlan {
    pub(crate) fn runs(&self) -> bool {
        self.runs
    }

    pub(crate) fn prepare(&mut self, (runs, missed): (bool, bool), frame: &HazeFrame) -> [f32; 4] {
        let history = std::mem::take(&mut self.kept) && !missed && frame.history;
        self.runs = false;
        if !runs {
            return [0.0; 4];
        }
        self.size = frame.display;
        self.drawn = (frame.render.x, frame.render.y);
        self.kept = true;
        let turn = (frame.frame % 1024) as f32 * 0.618_034;
        let turn = turn - turn.floor();
        self.runs = true;
        if history {
            [turn, frame.render.x as f32, frame.render.y as f32, 1.0]
        } else {
            [turn, 0.0, 0.0, 1.0]
        }
    }
}

pub(crate) struct Haze {
    marched: Texture<SkyGpu>,
    hazed: Texture<SkyGpu>,
    history_texture: Texture<SkyGpu>,
    march_pass: Pass<SkyGpu>,
    filter_pass: Pass<SkyGpu>,
    marching: Pipeline<SkyGpu, Own>,
    filtering: Pipeline<SkyGpu, Own>,
    drawn: (u32, u32),
    stepped: bool,
}

impl Haze {
    pub(crate) fn install(cx: &mut Install<'_, SkyGpu>) -> Result<Self, Error> {
        use crate::shaders::sky::{haze, haze_filter, haze_march, haze_view};
        let marched = cx.texture("haze_march", HAZE_FORMAT, Lifetime::Frame);
        let hazed = cx.texture("haze", HAZE_FORMAT, Lifetime::Kept);
        let history_texture = cx.texture("haze_history", HAZE_FORMAT, Lifetime::Previous);
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
            vertices: &[],
            primitive: wgpu::PrimitiveState::default(),
            depth: DepthUse::None,
            blend: None,
            writes: wgpu::ColorWrites::ALL,
        };
        let viewed = || [(haze_view::DEPTH_TEXTURE, Bind::Target(TargetName::DEPTH))];
        let marching =
            cx.render_pipeline(march_pass, &fullscreen(haze_march::FS_HAZE), &viewed())?;
        let mut filtered = Vec::from(viewed());
        filtered.extend([
            (haze_filter::MARCHED, Bind::Texture(marched)),
            (haze_filter::HAZE_HISTORY, Bind::Texture(history_texture)),
        ]);
        let filtering = cx.render_pipeline(
            filter_pass,
            &fullscreen(haze_filter::FS_HAZE_FILTER),
            &filtered,
        )?;
        Ok(Self {
            marched,
            hazed,
            history_texture,
            march_pass,
            filter_pass,
            marching,
            filtering,
            drawn: (0, 0),
            stepped: false,
        })
    }

    pub(crate) fn prepare(&mut self, cx: &mut Prepare<'_, SkyGpu>, plan: &HazePlan) {
        if std::mem::take(&mut self.stepped) {
            cx.deliver::<HazeSteps>(0, vec![u8::from(cx.ran(self.filter_pass))]);
        }
        if !plan.runs {
            return;
        }
        let size = plan.size;
        let extent = wgpu::Extent3d {
            width: size.x,
            height: size.y,
            depth_or_array_layers: 1,
        };
        for texture in [self.marched, self.hazed, self.history_texture] {
            cx.allocate(texture, extent, 1);
        }
        self.drawn = plan.drawn;
        cx.run(self.march_pass);
        cx.run(self.filter_pass);
        self.stepped = true;
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
        let mut draw = cx.render(Attach::Declared);
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
