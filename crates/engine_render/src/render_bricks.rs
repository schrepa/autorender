use engine_core::{Bricks, DrawKind, PassEntry, PassName};
use engine_registry::RecordExt;

use crate::registration;
use crate::{
    Draw, GpuPart, LightingModule, OpenSlot, PointInput, PointSet, Scene, SceneDraw, SlotModule,
    SurfaceKind, SurfaceTerms, TargetDecl, wgpu,
};

pub trait RenderBricks {
    fn part<P: GpuPart>(&mut self) -> &mut Self;
    fn pass<N: PassName>(&mut self) -> &mut Self;
    fn applies<N: PassName>(&mut self, module: engine_core::Module) -> &mut Self;
    fn draws<P: GpuPart>(
        &mut self,
        kind: DrawKind,
        draw: fn(&P, &SceneDraw, &mut Draw<'_, P, Scene>) -> u32,
    ) -> &mut Self;
    fn view_target(&mut self, target: TargetDecl) -> &mut Self;
    fn device_feature(&mut self, feature: wgpu::Features) -> &mut Self;
    fn generated(&mut self, wgsl: &'static [&'static str]) -> &mut Self;
    fn material_slot(&mut self, module: SlotModule) -> &mut Self;
    fn open_slot(&mut self, slot: OpenSlot) -> &mut Self;
    fn point_input(&mut self, input: PointInput) -> &mut Self;
    fn point_set(&mut self, set: PointSet) -> &mut Self;
    fn surface_terms(&mut self, terms: SurfaceTerms) -> &mut Self;
    fn lighting_model(&mut self, module: LightingModule) -> &mut Self;
    fn surface_kind(&mut self, kind: SurfaceKind) -> &mut Self;
}

impl RenderBricks for Bricks<'_> {
    fn part<P: GpuPart>(&mut self) -> &mut Self {
        let record = self.record();
        let scope = record.scope();
        record
            .render
            .push(Box::new(registration::PartRecord::of::<P>(scope)));
        self
    }

    fn pass<N: PassName>(&mut self) -> &mut Self {
        let record = self.record();
        if !record
            .render_entries::<PassEntry>()
            .any(|entry| entry.name == N::NAME)
        {
            let scope = record.scope();
            record.render.push(Box::new(PassEntry::of::<N>()));
            record.render.push(Box::new(registration::PassScope {
                name: N::NAME,
                scope,
            }));
        }
        self
    }

    fn applies<N: PassName>(&mut self, module: engine_core::Module) -> &mut Self {
        self.pass::<N>();
        let record = self.record();
        let scope = record.scope();
        record.render.push(Box::new(registration::Applied {
            pass: PassEntry::of::<N>(),
            module,
            scope,
        }));
        self
    }

    fn draws<P: GpuPart>(
        &mut self,
        kind: DrawKind,
        draw: fn(&P, &SceneDraw, &mut Draw<'_, P, Scene>) -> u32,
    ) -> &mut Self {
        let record = self.record();
        let scope = record.scope();
        record
            .render
            .push(Box::new(registration::PartRecord::of::<P>(scope)));
        record
            .render
            .push(Box::new(registration::KindDraw::of::<P>(kind, draw)));
        self
    }

    fn view_target(&mut self, target: TargetDecl) -> &mut Self {
        self.record().render.push(Box::new(target));
        self
    }

    fn device_feature(&mut self, feature: wgpu::Features) -> &mut Self {
        self.record()
            .render
            .push(Box::new(registration::DeviceFeature(feature)));
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

    fn surface_kind(&mut self, kind: SurfaceKind) -> &mut Self {
        self.record().render.push(Box::new(kind));
        self
    }
}
