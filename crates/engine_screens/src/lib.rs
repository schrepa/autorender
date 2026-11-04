#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod smoke;

use bevy_ecs::prelude::*;
use engine_core::{
    Brick, Bricks, ComponentKind, Error, ErrorCode, LevelProgress, LevelRequest, Operation, Runs,
    SceneComponents, Schedule, Screens, Smoke, Source, TimeControl,
};
use engine_places::screens::Screen;
use serde::Deserialize;
use serde_json::{Value, json};

pub struct ScreensBrick;

impl Brick for ScreensBrick {
    const SOURCE: &'static Source = &SOURCE;
    const SMOKES: &'static [Smoke] = &[Smoke {
        name: "screens",
        runs: Runs::Headless,
        run: smoke::run,
    }];

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .join::<SceneComponents>(
                ComponentKind::of::<Screen>("screen")
                    .schema(screen_schema)
                    .example(screen_example),
                None,
            )
            .systems(Schedule::Presentation, follow_top)
            .operation::<ScreenOperation>()
            .observation::<ScreenList>()
            .operation::<LevelOperation>()
            .observation::<LevelState>();
    }
}

fn screen_schema() -> Value {
    serde_json::to_value(schemars::schema_for!(Screen)).unwrap_or(Value::Null)
}

fn screen_example() -> Value {
    json!({"name": "menu", "pauses": true})
}

fn follow_top(screens: Res<Screens>, declared: Query<&Screen>, mut time: ResMut<TimeControl>) {
    if !screens.is_changed() || declared.is_empty() {
        return;
    }
    let top = screens.top();
    time.paused = declared
        .iter()
        .any(|screen| screen.pauses && top == Some(screen.name.as_str()));
}

fn known(declared: &Query<'_, '_, &Screen>, name: &str) -> Result<(), Error> {
    if declared.is_empty() || declared.iter().any(|screen| screen.name == name) {
        return Ok(());
    }
    Err(Error::new(ErrorCode::InvalidValue, "$.name")
        .got(name)
        .variants(declared.iter().map(|screen| screen.name.clone())))
}

fn stack_view(screens: &Screens) -> Value {
    json!({"stack": screens.stack(), "top": screens.top()})
}

#[derive(Deserialize)]
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum ScreenRequest {
    Go { name: String },
    Push { name: String },
    Pop,
}

pub(crate) struct ScreenOperation;

impl Operation for ScreenOperation {
    const NAME: &'static str = "screen";
    type Params = (
        ResMut<'static, Screens>,
        Query<'static, 'static, &'static Screen>,
    );
    type Request = ScreenRequest;

    fn run(
        (mut screens, declared): SystemParamItem<'_, '_, Self::Params>,
        request: ScreenRequest,
    ) -> Result<Value, Error> {
        match request {
            ScreenRequest::Go { name } => {
                known(&declared, &name)?;
                screens.go(name);
            }
            ScreenRequest::Push { name } => {
                known(&declared, &name)?;
                screens.push(name);
            }
            ScreenRequest::Pop => {
                screens.pop();
            }
        }
        Ok(stack_view(&screens))
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NoRequest {}

pub(crate) struct ScreenList;

impl Operation for ScreenList {
    const NAME: &'static str = "screens";
    type Params = (
        Res<'static, Screens>,
        Query<'static, 'static, &'static Screen>,
    );
    type Request = NoRequest;

    fn run(
        (screens, declared): SystemParamItem<'_, '_, Self::Params>,
        _: NoRequest,
    ) -> Result<Value, Error> {
        let mut listed: Vec<&Screen> = declared.iter().collect();
        listed.sort_by(|a, b| a.name.cmp(&b.name));
        let mut view = stack_view(&screens);
        view["declared"] = json!(listed);
        Ok(view)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct LevelAsk {
    path: String,
}

pub(crate) struct LevelOperation;

impl Operation for LevelOperation {
    const NAME: &'static str = "level";
    type Params = ResMut<'static, LevelRequest>;
    type Request = LevelAsk;

    fn run(
        mut level: SystemParamItem<'_, '_, Self::Params>,
        request: LevelAsk,
    ) -> Result<Value, Error> {
        level.path = Some(request.path);
        Ok(json!({"path": level.path}))
    }
}

pub(crate) struct LevelState;

impl Operation for LevelState {
    const NAME: &'static str = "level_progress";
    type Params = Res<'static, LevelProgress>;
    type Request = NoRequest;

    fn run(progress: SystemParamItem<'_, '_, Self::Params>, _: NoRequest) -> Result<Value, Error> {
        Ok(json!(*progress))
    }
}
