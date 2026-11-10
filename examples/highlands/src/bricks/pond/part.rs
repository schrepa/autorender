use bevy_ecs::prelude::*;
use engine_core::{
    DrawKind, Error, KindDecl, PassName, PassPurpose, PassViews, SlotSet, Writes, gpu_consts,
    gpu_struct,
};
use engine_open_world_kit::ground::{GroundTiles, TerrainEntry, TerrainTiles};
use engine_render::{
    Allocation, AttachPoint, Bind, Buffer, Draw, FaceCull, Form, FormSide, GpuPart, Install,
    KindForm, Lifetime, OwnFragments, Prepare, RoleFragments, Scene, SceneDraw, Shape, TargetName,
    Varyings,
};
use glam::Vec2;

use crate::bricks::pond::Pond;
use crate::bricks::pond::shaders::pond::surface;

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

pub(crate) const POND_KIND: DrawKind = DrawKind("pond");

pub(crate) struct PondPass;

impl PassName for PondPass {
    const NAME: &'static str = "pond";
    const PASS: engine_core::Pass = engine_core::Pass {
        writes: Writes::LIT,
        reads: SlotSet::ALL,
        views: PassViews::Main,
        at: AttachPoint::LateOpaque,
        purpose: PassPurpose::Scene,
        takes: takes_pond,
    };
}

fn takes_pond(kind: &KindDecl) -> bool {
    kind.kind == POND_KIND
}

const POND_FRAGMENTS: OwnFragments = OwnFragments {
    placed: RoleFragments {
        lit: Some(surface::FS_POND),
        shown: Some(surface::FS_DEBUG_POND),
        ..RoleFragments::NONE
    },
    owned: RoleFragments::NONE,
    faces: engine_render::OwnFaces::Culled(FaceCull::None),
};

const POND_FORM: KindForm = KindForm {
    kind: POND_KIND,
    placed: FormSide {
        vertex: surface::VS_POND,
        streams: &[],
        depth_bias: FormSide::UNBIASED,
    },
    owned: None,
    varyings: Varyings::Own(&POND_FRAGMENTS),
};

pub(crate) struct PondGpu {
    ponds: Buffer<Self>,
    form: Form<Self>,
    drawn: u32,
}

impl PondGpu {
    pub(crate) fn draw(&self, _scene: &SceneDraw, draw: &mut Draw<'_, Self, Scene>) -> u32 {
        if self.drawn == 0 {
            return draw.skip("no pond is placed");
        }
        let Some(pipeline) = draw.form(&self.form) else {
            return draw.skip("the pond draws only in its own pass and the debug view");
        };
        draw.draw(
            &pipeline,
            Shape::Pulled {
                vertices: 0..6,
                instances: 0..self.drawn,
                material: None,
            },
        );
        1
    }
}

impl GpuPart for PondGpu {
    type Reads = (
        Query<'static, 'static, &'static Pond>,
        Res<'static, TerrainTiles>,
        Res<'static, GroundTiles>,
    );

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        let ponds = cx.buffer("ponds", Lifetime::Kept);
        let form = cx.form(
            &POND_FORM,
            &[
                (surface::PONDS, Bind::Buffer(ponds)),
                (surface::POND_SCENE, Bind::Target(TargetName::COLOR)),
            ],
        )?;
        Ok(Self {
            ponds,
            form,
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
        let (Allocation::Fresh | Allocation::Kept) =
            cx.allocate_buffer(self.ponds, std::mem::size_of_val(&shapes) as u64);
        cx.write(self.ponds, 0, &shapes);
        self.drawn = count;
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
