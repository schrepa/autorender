use bevy_ecs::prelude::*;
use bevy_ecs::system::SystemParamItem;
use engine_core::{Detail, DrawnTable, Explain, ExplainAsk, ExplainRecord, FrameTime};

use crate::layouts::{LOD_LEVELS, LodFade};
use crate::presentation::PresentationTotals;
use crate::settings::{Impostors, OcclusionCulling, SubpixelCulling};

type Drawn<'w> = (Res<'w, DrawnTable>, Res<'w, PresentationTotals>, Res<'w, FrameTime>);

struct Watched {
    id: String,
    slot: u32,
    level: Option<u32>,
    age: Option<u64>,
}

fn drawn((table, totals, time): &Drawn<'_>, ask: &ExplainAsk<'_>) -> Option<Watched> {
    let ExplainAsk::Entity { id, entity } = ask else {
        return None;
    };
    let slot = table.slot(*entity)?;
    let (seen, word) = totals.last()?.watched?;
    (seen == slot).then(|| Watched {
        id: (*id).to_owned(),
        slot,
        level: LodFade::drawn_level(word),
        age: totals.age(time.frame),
    })
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
        let watched = drawn(&read, ask)?;
        if watched.level.is_some() {
            return None;
        }
        Some(
            ExplainRecord::new("culled")
                .with("entity", &watched.id)
                .with("slot", watched.slot)
                .with("read_back_age", watched.age)
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
        let watched = drawn(&read, ask)?;
        let level = watched.level?;
        Some(
            ExplainRecord::new("level")
                .with("entity", &watched.id)
                .with("level", level)
                .with("read_back_age", watched.age)
                .with("impostor", level == LOD_LEVELS)
                .with("pixel_error", detail.pixel_error)
                .with("impostor_pixels", impostors.pixels),
        )
    }
}
