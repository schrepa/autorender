use bevy_ecs::prelude::*;
use engine_core::{Error, gpu_consts, gpu_struct};
use engine_places::ground::{GroundTiles, TerrainEntry, TerrainTiles};
use engine_render::{
    Attach, Bind, Buffer, DepthUse, Encode, GpuPart, Install, Layer, Lifetime, Own, Pass, Pipeline,
    Prepare, RenderPipeline, Shape, wgpu,
};
use glam::Vec2;

use crate::Pond;

gpu_consts! {
    pub(crate) const POND_MOST: u32 = 16;
}

gpu_struct! {
    pub(crate) struct PondShape {
        pub area: [f32; 4],
        pub surface: [f32; 4],
        pub terrain: [f32; 4],
        pub ground: [u32; 4],
    }
}

const COVERING: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
        operation: wgpu::BlendOperation::Add,
    },
    alpha: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::Zero,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    },
};

pub(crate) struct PondGpu {
    pass: Pass<Self>,
    ponds: Buffer<Self>,
    pipeline: Pipeline<Self, Own>,
    drawn: u32,
}

impl GpuPart for PondGpu {
    type Reads = (
        Query<'static, 'static, &'static Pond>,
        Res<'static, TerrainTiles>,
        Res<'static, GroundTiles>,
    );

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        use crate::shaders::pond::surface;
        let ponds = cx.buffer("ponds", Lifetime::Kept);
        let pass = cx.pass("pond").at(Layer::Media).id();
        let pipeline = cx.render_pipeline(
            pass,
            &RenderPipeline {
                vertex: surface::VS_POND,
                fragment: Some(surface::FS_POND),
                debug: None,
                vertices: &[],
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth: DepthUse::Write {
                    bias: wgpu::DepthBiasState::default(),
                },
                blend: Some(COVERING),
                writes: wgpu::ColorWrites::ALL,
            },
            &[(surface::PONDS, Bind::Buffer(ponds))],
        )?;
        Ok(Self {
            pass,
            ponds,
            pipeline,
            drawn: 0,
        })
    }

    fn prepare(
        &mut self,
        (ponds, tiles, ground): SystemParamItem<'_, '_, Self::Reads>,
        cx: &mut Prepare<'_, Self>,
    ) {
        self.drawn = 0;
        if ponds.is_empty() || ground.layers == 0 {
            return;
        }
        let (shapes, count) = shapes(ponds.iter(), &tiles.terrains, &ground);
        cx.allocate_buffer(self.ponds, std::mem::size_of_val(&shapes) as u64);
        cx.write(self.ponds, 0, &shapes);
        self.drawn = count;
        cx.run(self.pass);
    }

    fn encode(&mut self, _pass: Pass<Self>, cx: &mut Encode<'_, Self>) -> u32 {
        let mut draw = cx.render(Attach::Declared, None);
        draw.draw(
            &self.pipeline,
            Shape::Pulled {
                vertices: 0..6,
                instances: 0..self.drawn,
            },
        );
        1
    }
}

fn shapes<'p>(
    ponds: impl Iterator<Item = &'p Pond>,
    terrains: &[TerrainEntry],
    ground: &GroundTiles,
) -> ([PondShape; POND_MOST as usize], u32) {
    let mut shapes = [PondShape::default(); POND_MOST as usize];
    let mut count = 0;
    for (shape, pond) in shapes.iter_mut().zip(ponds) {
        let (low, high) = (pond.low(), pond.high());
        shape.area = [low.x, low.y, high.x, high.y];
        shape.surface = [pond.surface, 0.0, 0.0, 0.0];
        if let Some(terrain) = under(terrains, pond)
            && let Some(&(_, index)) = ground
                .tables
                .iter()
                .find(|(entity, _)| *entity == terrain.terrain)
        {
            let origin = terrain.origin;
            shape.terrain = [origin.x, origin.y, origin.z, terrain.pyramid.spacing(0)];
            shape.ground = [index, terrain.pyramid.levels(), 1, 0];
        }
        count += 1;
    }
    (shapes, count)
}

fn under<'t>(terrains: &'t [TerrainEntry], pond: &Pond) -> Option<&'t TerrainEntry> {
    terrains.iter().find(|terrain| {
        let local = Vec2::from(pond.center) - Vec2::new(terrain.origin.x, terrain.origin.z);
        let extent = terrain.pyramid.extent();
        !terrain.surroundings
            && local.x >= 0.0
            && local.y >= 0.0
            && local.x <= extent.x
            && local.y <= extent.y
    })
}
