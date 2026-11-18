use engine_core::{Bricks, DrawKind, PassEntry, PassName};
use engine_registry::RecordExt;

use crate::registration;
use crate::{
    Draw, GpuPart, LightingModule, OpenSlot, PointInput, PointSet, Scene, SlotModule, SlotPrepare,
    SurfaceKind, SurfaceTerms,
};

pub trait RenderBricks {
    fn part<P: GpuPart>(&mut self) -> &mut Self;
    fn pass<N: PassName>(&mut self) -> &mut Self;
    fn draws<P: GpuPart>(
        &mut self,
        kind: DrawKind,
        draw: fn(&P, &mut Draw<'_, P, Scene>),
    ) -> &mut Self;
    fn generated(&mut self, wgsl: &'static [&'static str]) -> &mut Self;
    fn material_slot(&mut self, module: SlotModule) -> &mut Self;
    fn slot_prepare(&mut self, prepare: SlotPrepare) -> &mut Self;
    fn open_slot(&mut self, slot: OpenSlot) -> &mut Self;
    fn point_input(&mut self, input: PointInput) -> &mut Self;
    fn point_set(&mut self, set: PointSet) -> &mut Self;
    fn surface_terms(&mut self, terms: SurfaceTerms) -> &mut Self;
    fn lighting_model(&mut self, module: LightingModule) -> &mut Self;
    fn surface_kind(&mut self, kind: SurfaceKind) -> &mut Self;
}

impl RenderBricks for Bricks<'_> {
    fn part<P: GpuPart>(&mut self) -> &mut Self {
        self.record()
            .render
            .push(Box::new(registration::PartRecord::of::<P>()));
        self
    }

    fn pass<N: PassName>(&mut self) -> &mut Self {
        let record = self.record();
        if !record
            .render_entries::<PassEntry>()
            .any(|entry| entry.name == N::NAME)
        {
            record.render.push(Box::new(PassEntry::of::<N>()));
        }
        self
    }

    fn draws<P: GpuPart>(
        &mut self,
        kind: DrawKind,
        draw: fn(&P, &mut Draw<'_, P, Scene>),
    ) -> &mut Self {
        let record = self.record();
        record
            .render
            .push(Box::new(registration::PartRecord::of::<P>()));
        record
            .render
            .push(Box::new(registration::KindDraw::of::<P>(kind, draw)));
        self
    }

    fn generated(&mut self, wgsl: &'static [&'static str]) -> &mut Self {
        self.record()
            .render
            .push(Box::new(registration::Generated(wgsl)));
        self
    }

    fn material_slot(&mut self, module: SlotModule) -> &mut Self {
        if module.slot == engine_core::SurfaceSlot::Position.name() {
            self.record()
                .render
                .push(Box::new(engine_core::PositionReach {
                    module: module.name,
                    reach: module.reach,
                }));
        }
        self.record().render.push(Box::new(module));
        self
    }

    fn open_slot(&mut self, slot: OpenSlot) -> &mut Self {
        self.record().render.push(Box::new(slot));
        self
    }

    fn point_input(&mut self, input: PointInput) -> &mut Self {
        self.record().render.push(Box::new(input));
        self
    }

    fn point_set(&mut self, set: PointSet) -> &mut Self {
        self.record().render.push(Box::new(set));
        self
    }

    fn surface_terms(&mut self, terms: SurfaceTerms) -> &mut Self {
        self.record().render.push(Box::new(terms));
        self
    }

    fn lighting_model(&mut self, module: LightingModule) -> &mut Self {
        self.record().render.push(Box::new(module));
        self
    }

    fn slot_prepare(&mut self, prepare: SlotPrepare) -> &mut Self {
        self.record().render.push(Box::new(prepare));
        self
    }

    fn surface_kind(&mut self, kind: SurfaceKind) -> &mut Self {
        self.record().render.push(Box::new(kind));
        self
    }
}

pub(crate) struct Composite {
    pass: crate::Pass<Self>,
    pipeline: crate::Pipeline<Self, crate::Own>,
}

impl GpuPart for Composite {
    type Reads = ();
    const NAME: &'static str = "composite";

    fn install(cx: &mut crate::Install<'_, Self>) -> Result<Self, engine_core::Error> {
        let frame = crate::shaders::core_module("engine::post::frame");
        cx.frame(
            engine_core::Var::generated(frame, "scene"),
            crate::Bind::Target(engine_core::TargetName::COLOR),
        )?;
        cx.frame(
            engine_core::Var::generated(frame, "depth_texture"),
            crate::Bind::Target(engine_core::TargetName::DEPTH),
        )?;
        let pass = cx
            .pass("composite")
            .label("engine/post-composite")
            .span("post_composite")
            .target(engine_core::TargetName::CANVAS, crate::Access::Clear)
            .id();
        let pipeline = cx.render_pipeline(
            pass,
            &crate::RenderPipeline {
                vertex: crate::shaders::FULLSCREEN,
                fragment: Some(crate::shaders::FS_COMPOSITE),
                vertices: &[],
                primitive: wgpu::PrimitiveState::default(),
                depth: crate::DepthUse::None,
                stencil: wgpu::StencilState::default(),
                samples: 1,
                targets: &[(None, wgpu::ColorWrites::ALL)],
                constants: &[],
            },
            &[],
        )?;
        Ok(Self { pass, pipeline })
    }

    fn prepare(&mut self, (): (), _cx: &mut crate::Prepare<'_, Self>) {}

    fn view(&mut self, _view: &engine_core::ViewFacts, cx: &mut crate::Prepare<'_, Self>) {
        cx.run(self.pass);
    }

    fn encode(&mut self, _pass: crate::Pass<Self>, cx: &mut crate::Encode<'_, Self>) {
        cx.render(crate::Attach::Declared).draw(
            &self.pipeline,
            crate::Shape::Pulled {
                vertices: 0..3,
                instances: 0..1,
                material: None,
            },
        );
    }
}
