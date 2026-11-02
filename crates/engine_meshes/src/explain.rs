use bevy_ecs::prelude::*;
use bevy_ecs::system::SystemParamItem;
use engine_core::{Detail, Explain, ExplainAsk, ExplainRecord, FrameTime};
use engine_places::meshes::Instances;

use crate::layouts::LOD_STATE;
use crate::presentation::PresentationTotals;
use crate::settings::{Impostors, OcclusionCulling, SubpixelCulling};

const IMPOSTOR_LEVEL: u32 = 3;

type Drawn<'w> = (Res<'w, Instances>, Res<'w, PresentationTotals>, Res<'w, FrameTime>);

fn drawn(
    (instances, totals, time): &Drawn<'_>,
    ask: &ExplainAsk<'_>,
) -> Option<(String, u32, Option<u32>)> {
    let ExplainAsk::Entity { id, entity } = ask else {
        return None;
    };
    let slot = instances.drawn(*entity)?;
    let (seen, state) = totals.last()?.watched?;
    (seen == slot && totals.age(time.frame).is_some_and(|age| age <= 1))
        .then(|| ((*id).to_owned(), slot, (state & LOD_STATE != 0).then_some(state & 3)))
}

pub(crate) struct Culled;

impl Explain for Culled {
    type Params = (
        Drawn<'static>,
        Res<'static, OcclusionCulling>,
        Res<'static, SubpixelCulling>,
    );

    fn explain(
        (read, occlusion, subpixel): SystemParamItem<'_, '_, Self::Params>,
        ask: &ExplainAsk<'_>,
    ) -> Option<ExplainRecord> {
        let (id, slot, level) = drawn(&read, ask)?;
        if level.is_some() {
            return None;
        }
        Some(
            ExplainRecord::new("culled")
                .with("entity", &id)
                .with("slot", slot)
                .with("occlusion_culling", occlusion.enabled)
                .with("subpixel_culling", subpixel.enabled),
        )
    }
}

pub(crate) struct Level;

impl Explain for Level {
    type Params = (Drawn<'static>, Res<'static, Detail>, Res<'static, Impostors>);

    fn explain(
        (read, detail, impostors): SystemParamItem<'_, '_, Self::Params>,
        ask: &ExplainAsk<'_>,
    ) -> Option<ExplainRecord> {
        let (id, _, level) = drawn(&read, ask)?;
        let level = level?;
        Some(
            ExplainRecord::new("level")
                .with("entity", &id)
                .with("level", level)
                .with("impostor", level == IMPOSTOR_LEVEL)
                .with("pixel_error", detail.pixel_error)
                .with("impostor_pixels", impostors.pixels),
        )
    }
}
