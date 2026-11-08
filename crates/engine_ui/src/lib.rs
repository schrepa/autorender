#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod gpu;
mod interact;
mod operations;
mod smokes;
mod tree;

use std::collections::BTreeSet;

use bevy_ecs::prelude::*;
use engine_core::{
    Asks, Asset, AssetResidency, AssetUses, AttachPoint, Brick, Bricks, ComponentKind, Contexts,
    CursorMode, CursorRequests, Devices, Error, ErrorCode, GlobalTransforms, Input, PassName,
    PassViews, Runs, SceneComponents, SceneContext, SceneCx, Schedule, Views, Verbs,
    SlotSet, Smoke, Source, Spawner, Writes, gpu_struct, never, settings, stats,
};
use engine_places_rendering::screens::{ScreenAsk, Screens};
use engine_places_rendering::text::Text;
use engine_places_rendering::ui::{UiEvents, UiLength, UiNode, UiRoot, UiValues};
use engine_render::RenderBricks;
use serde_json::Value;

use gpu::UiGpu;
use tree::{Facts, Roots, UiState};

const GENERATED: &[&str] = &[<UiRect as engine_core::GpuStruct>::WGSL];
const ASKER: &str = "engine_ui";
const CONTEXT: &str = "ui";

settings! {
    #[section("ui")]
    #[serde(deny_unknown_fields)]
    pub struct UiSettings {
        pub enabled: bool = true => any,
        pub scale: f32 = 1.0 => range(0.25, 8.0),
        pub max_nodes: u32 = 4096 => range(1, 1_000_000),
        pub max_rects: u32 = 8192 => range(1, 1_000_000),
        pub events: u32 = 256 => at_least(1),
    }
}

stats! {
    #[section("ui_stats")]
    pub struct UiStats {
        pub roots: u32,
        pub nodes: u32,
        pub rects_drawn: u32,
        pub rects_cut: u32,
        pub layouts: u32,
        pub events: u32,
        pub missing_bindings: u32,
    }
}

gpu_struct! {
    pub struct UiRect {
        pub place: [f32; 4],
        pub rect: [f32; 4],
        pub color: [f32; 4],
        pub border: [f32; 4],
        pub shape: [f32; 4],
        pub uv: [f32; 4],
        pub clip: [f32; 4],
    }
}

pub(crate) struct UiPass;

impl PassName for UiPass {
    const NAME: &'static str = "ui";
    const PASS: engine_core::Pass = engine_core::Pass {
        writes: Writes::NOTHING,
        reads: SlotSet::NONE,
        views: PassViews::Main,
        at: AttachPoint::Overlay,
        takes: never,
    };
}

type Reading<'w, 's> = (
    Roots<'w, 's>,
    Res<'w, Text>,
    Res<'w, UiValues>,
    Res<'w, Views>,
    Res<'w, UiSettings>,
    (Option<Res<'w, SceneContext>>, Res<'w, GlobalTransforms>),
);

type Steering<'w, 's> = (
    ResMut<'w, Contexts>,
    ResMut<'w, CursorRequests>,
    (Res<'w, Screens>, Asks<'w, 's, ScreenAsk>),
    ResMut<'w, UiEvents>,
);

fn reaches(cursor: &CursorRequests) -> bool {
    cursor.applied() != CursorMode::Captured
}

fn check_size(value: f32, path: &str) -> Result<(), Error> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(Error::new(ErrorCode::InvalidNumber, path).got(value))
    }
}

fn check_length(length: UiLength, path: &str) -> Result<(), Error> {
    match length {
        UiLength::Auto => Ok(()),
        UiLength::Px(value) | UiLength::Percent(value) => check_size(value, path),
    }
}

fn check_node<'n>(node: &'n UiNode, path: &str, ids: &mut BTreeSet<&'n str>) -> Result<(), Error> {
    if let Some(id) = node.id.as_deref()
        && !ids.insert(id)
    {
        return Err(Error::new(ErrorCode::DuplicateId, format!("{path}.id")).got(id));
    }
    let sizes = [
        ("gap", node.gap),
        ("grow", node.grow),
        ("radius", node.radius),
        ("border_width", node.border_width),
        ("text_size", node.text_size),
        ("image_rect.u", node.image_rect.u),
        ("image_rect.v", node.image_rect.v),
        ("image_rect.width", node.image_rect.width),
        ("image_rect.height", node.image_rect.height),
    ];
    for (name, value) in sizes {
        check_size(value, &format!("{path}.{name}"))?;
    }
    check_length(node.width, &format!("{path}.width"))?;
    check_length(node.height, &format!("{path}.height"))?;
    for (at, child) in node.children.iter().enumerate() {
        check_node(child, &format!("{path}.children[{at}]"), ids)?;
    }
    Ok(())
}

fn check_root(root: &UiRoot, path: &str) -> Result<(), Error> {
    check_node(&root.root, &format!("{path}.root"), &mut BTreeSet::new())
}

fn prepare_root(mut root: UiRoot, cx: &SceneCx<'_>, path: &str) -> Result<UiRoot, Error> {
    if let Some(uri) = &root.font {
        root.bytes = cx.buffer(uri, &format!("{path}.font"))?;
    }
    Ok(root)
}

fn root_files(root: &UiRoot) -> Vec<String> {
    root.font.iter().cloned().collect()
}

fn root_schema() -> Value {
    serde_json::to_value(schemars::schema_for!(UiRoot)).unwrap_or(Value::Null)
}

fn follow(
    (roots, text, values, views, settings, (scene, globals)): Reading<'_, '_>,
    (input, devices): (Res<Input>, Res<Devices>),
    (mut contexts, mut cursor, (screens, mut asks), mut events): Steering<'_, '_>,
    mut state: ResMut<UiState>,
    mut stats: ResMut<UiStats>,
) {
    if settings.is_changed() || events.is_added() {
        events.resize(settings.events as usize);
    }
    if state.is_idle() && (roots.is_empty() || !settings.enabled) {
        return;
    }
    let reach = reaches(&cursor);
    let pointer = interact::Pointer::of(&devices, reach);
    if state.verbs.is_none() {
        let [Some(up), Some(down), Some(left), Some(right), Some(accept), Some(back)] =
            interact::VERBS.map(|verb| input.find(verb.name))
        else {
            return;
        };
        state.verbs = Some([up, down, left, right, accept, back]);
    }
    let menu = interact::Menu::of(&input, state.verbs);
    interact::run(
        &mut state,
        (&pointer, reach),
        &menu,
        (&screens, &mut asks),
        &mut events,
    );
    let facts = Facts::of(
        &text,
        &values,
        views.frame(),
        &settings,
        (scene.as_ref(), &globals),
        screens.top(),
    );
    state.sync(&roots, &facts);
    let wanted = state.menu;
    if wanted != state.wanting {
        state.wanting = wanted;
        state.context = wanted.then(|| contexts.ask(ASKER, CONTEXT, true));
        state.cursor = wanted.then(|| cursor.ask(ASKER, CursorMode::Free));
    }
    let over = interact::over(&state, &pointer);
    if events.pointer_over() != over {
        events.set_pointer_over(over);
    }
    stats.set_if_neq(state.stats());
}

fn present(
    (roots, text, values, views, settings, (scene, globals)): Reading<'_, '_>,
    screens: Res<Screens>,
    mut state: ResMut<UiState>,
    mut stats: ResMut<UiStats>,
) {
    if state.is_idle() && (roots.is_empty() || !settings.enabled) {
        return;
    }
    let facts = Facts::of(
        &text,
        &values,
        views.frame(),
        &settings,
        (scene.as_ref(), &globals),
        screens.top(),
    );
    state.sync(&roots, &facts);
    gpu::stage(&mut state, &text, settings.max_rects as usize);
    stats.set_if_neq(state.stats());
}

#[derive(Component)]
struct Images;

fn publish_images(
    state: Res<UiState>,
    residency: Res<AssetResidency>,
    own: Query<(Entity, &AssetUses), With<Images>>,
    mut spawner: Spawner<UiBrick>,
) {
    let mut wanted: Vec<Asset> = state.textures.iter().copied().map(Asset::Texture).collect();
    wanted.sort_unstable();
    wanted.dedup();
    match own.single() {
        Ok((entity, held)) => {
            if held.assets() != wanted.as_slice() {
                spawner.insert(entity, AssetUses::new(&residency, wanted));
            }
        }
        Err(_) => {
            if !wanted.is_empty() {
                spawner.spawn((Images, AssetUses::new(&residency, wanted)));
            }
        }
    }
}

pub struct UiBrick;

impl Brick for UiBrick {
    const SOURCE: &'static Source = &SOURCE;
    const SMOKES: &'static [Smoke] = &[
        Smoke {
            name: "ui_layout_press",
            runs: Runs::Headless,
            run: smokes::layout_press,
        },
        Smoke {
            name: "ui_drawn",
            runs: Runs::Gpu,
            run: smokes::drawn,
        },
    ];

    fn register(bricks: &mut Bricks<'_>) {
        for verb in interact::VERBS {
            bricks.join::<Verbs>(verb, None);
        }
        bricks
            .settings::<UiSettings>()
            .stats::<UiStats>()
            .resource::<UiState>()
            .fill::<UiValues>(None)
            .fill::<UiEvents>(None)
            .join::<SceneComponents>(
                ComponentKind::of::<UiRoot>("ui")
                    .check(check_root)
                    .prepare(prepare_root)
                    .files(root_files)
                    .schema(root_schema),
                None,
            )
            .systems(Schedule::Presentation, follow)
            .systems(Schedule::ViewSync, (present, publish_images))
            .operation::<operations::UiPress>()
            .observation::<operations::UiTree>()
            .generated(GENERATED)
            .pass::<UiPass>()
            .part::<UiGpu>();
    }
}
