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
    AssetUses, Brick, Bricks, ComponentKind, CursorMode, Error, ErrorCode, MenuInput, PointerInput,
    Runs, SceneComponents, SceneContext, SceneCx, Schedule, Screens, Smoke, Source, Spawner, View,
    gpu_struct, settings, stats,
};
use engine_places::text::Text;
use engine_places::ui::{UiEvents, UiLength, UiNode, UiRoot, UiValues};
use engine_render::{Layer, OverlayStack, OverlaySteps, RenderBricks};
use serde_json::{Value, json};

use gpu::UiGpu;
use tree::{Facts, Roots, UiState};

const GENERATED: &[&str] = &[<UiRect as engine_core::GpuStruct>::WGSL];

settings! {
    #[section("ui")]
    #[serde(deny_unknown_fields)]
    pub struct UiSettings {
        pub enabled: bool = true => any,
        pub scale: f32 = 1.0 => range(0.25, 8.0),
        pub max_nodes: u32 = 4096 => range(1, 1_000_000),
        pub max_rects: u32 = 8192 => range(1, 1_000_000),
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

type Reading<'w, 's> = (
    Roots<'w, 's>,
    Res<'w, Text>,
    Res<'w, UiValues>,
    Res<'w, View>,
    Res<'w, UiSettings>,
    Option<Res<'w, SceneContext>>,
);

type Steering<'w> = (
    ResMut<'w, MenuInput>,
    ResMut<'w, CursorMode>,
    ResMut<'w, Screens>,
    ResMut<'w, UiEvents>,
);

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

fn root_example() -> Value {
    json!({
        "screen": "menu",
        "menu": true,
        "root": {
            "kind": "panel",
            "absolute": { "left": 16.0, "top": 16.0 },
            "padding": [12.0, 12.0, 12.0, 12.0],
            "gap": 8.0,
            "color": [0.05, 0.05, 0.08, 0.85],
            "radius": 8.0,
            "children": [
                { "kind": "label", "text": "Menu", "text_size": 24.0 },
                {
                    "id": "start",
                    "kind": "button",
                    "text": "Start",
                    "padding": [12.0, 8.0, 12.0, 8.0],
                    "color": [0.2, 0.2, 0.3, 1.0],
                    "hover_color": [0.3, 0.3, 0.45, 1.0],
                    "press_color": [0.15, 0.15, 0.25, 1.0],
                    "focus_color": [0.3, 0.3, 0.45, 1.0],
                    "action": { "emit": "start" },
                },
            ],
        },
    })
}

fn follow(
    (roots, text, values, view, settings, scene): Reading<'_, '_>,
    pointer: Res<PointerInput>,
    (mut menu, mut cursor, mut screens, mut events): Steering<'_>,
    mut state: ResMut<UiState>,
    mut stats: ResMut<UiStats>,
) {
    if state.is_idle() && (roots.is_empty() || !settings.enabled) {
        return;
    }
    if !events.names().is_empty() {
        events.clear();
    }
    let pointer = PointerInput {
        inside: pointer.inside && *cursor != CursorMode::Captured,
        ..*pointer
    };
    interact::run(&mut state, &pointer, &menu, &mut screens, &mut events);
    let facts = Facts::of(
        &text,
        &values,
        &view,
        &settings,
        scene.as_ref(),
        screens.top(),
    );
    state.sync(&roots, &facts);
    let wanted = state.menu;
    if wanted != state.wanting {
        state.wanting = wanted;
        menu.wanted = wanted;
        if wanted {
            state.held_cursor = Some(*cursor);
            *cursor = CursorMode::Free;
        } else if let Some(previous) = state.held_cursor.take() {
            *cursor = previous;
        }
    }
    let over = interact::over(&state, &pointer);
    if events.pointer_over() != over {
        events.set_pointer_over(over);
    }
    stats.set_if_neq(state.stats());
}

fn present(
    (roots, text, values, view, settings, scene): Reading<'_, '_>,
    screens: Res<Screens>,
    mut state: ResMut<UiState>,
    mut stats: ResMut<UiStats>,
    mut stack: ResMut<OverlaySteps>,
) {
    if state.is_idle() && (roots.is_empty() || !settings.enabled) {
        return;
    }
    let facts = Facts::of(
        &text,
        &values,
        &view,
        &settings,
        scene.as_ref(),
        screens.top(),
    );
    state.sync(&roots, &facts);
    gpu::stage(&mut state, settings.max_rects as usize);
    state.queue_text(&text);
    stats.set_if_neq(state.stats());
    stack.set_if_neq(OverlaySteps {
        count: state.shown_roots,
    });
}

#[derive(Component)]
struct Images;

fn publish_images(
    state: Res<UiState>,
    own: Query<(Entity, &AssetUses), With<Images>>,
    mut spawner: Spawner<UiBrick>,
) {
    let used = || AssetUses {
        textures: state.textures.clone(),
        ..AssetUses::default()
    };
    match own.single() {
        Ok((entity, held)) => {
            if held.textures != state.textures {
                spawner.insert(entity, used());
            }
        }
        Err(_) => {
            if !state.textures.is_empty() {
                spawner.spawn((Images, used()));
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
        bricks
            .settings::<UiSettings>()
            .stats::<UiStats>()
            .resource::<UiState>()
            .fill::<UiValues>(None)
            .fill::<UiEvents>(None)
            .fill::<OverlaySteps>(None)
            .join::<SceneComponents>(
                ComponentKind::of::<UiRoot>("ui")
                    .check(check_root)
                    .prepare(prepare_root)
                    .files(root_files)
                    .schema(root_schema)
                    .example(root_example),
                None,
            )
            .systems(Schedule::Presentation, follow)
            .systems(Schedule::ViewSync, (present, publish_images))
            .operation::<operations::UiPress>()
            .observation::<operations::UiTree>()
            .generated(GENERATED)
            .draws::<UiGpu, OverlayStack>(Layer::Opaque, UiGpu::draw_rects);
    }
}
