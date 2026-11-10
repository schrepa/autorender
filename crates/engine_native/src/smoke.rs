use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use engine::bevy_ecs::world::World;
use engine::bricks::BrickList;
use engine::smokes::{Build, GpuSession, LIST, SmokeEntry, SmokeHost, SmokeRecord, listed};
use engine::{Engine, Error, ErrorCode, Game, GameEntry};
use serde_json::{Value, json};

use crate::dev::DevSession;
use crate::{HostOptions, Offscreen};

const SIZE: (u32, u32) = (960, 540);

pub fn run<G: Game>(
    options: &HostOptions,
    names: &str,
    entries: &[SmokeEntry],
    (bricks, page): (&BrickList, &BrickList),
    build: Build<'_, G>,
) -> ExitCode {
    if names == LIST {
        for entry in entries {
            println!("{}", listed(entry));
        }
        return ExitCode::SUCCESS;
    }
    crate::job_cache::keep_pipeline_cache();
    let Some(limit_ms) = options.limit_ms else {
        return crate::exit(Err(engine::Error::new(
            engine::ErrorCode::InvalidArguments,
            "--limit-ms",
        )
        .expected(engine::Expected::Present)));
    };
    let ran = if options.relay {
        let mut relay = Relay {
            page: page
                .listed()
                .iter()
                .map(|listed| listed.package())
                .collect(),
        };
        run_on(&mut relay, names, limit_ms, entries, bricks, build)
    } else {
        let mut native = Native {
            options,
            game: std::marker::PhantomData,
        };
        run_on(&mut native, names, limit_ms, entries, bricks, build)
    };
    match ran {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(3),
        Err(error) => {
            eprintln!("{}", serde_json::to_value(&error).unwrap_or(Value::Null));
            ExitCode::from(2)
        }
    }
}

fn run_on<G: Game, H: SmokeHost<G>>(
    host: &mut H,
    names: &str,
    limit_ms: u64,
    entries: &[SmokeEntry],
    bricks: &BrickList,
    build: Build<'_, G>,
) -> engine::Result<bool> {
    pollster::block_on(engine::smokes::run(
        host,
        names,
        limit_ms as f64,
        entries,
        bricks,
        build,
        &mut |record: &SmokeRecord| println!("{}", record.line()),
    ))
}

pub fn of_game<G: Game>(options: &HostOptions, names: &str, game: GameEntry<G>) -> ExitCode {
    let options = &options.for_game(&game);
    let list = match options.listed(&game) {
        Ok(list) => list,
        Err(error) => return crate::exit(Err(error)),
    };
    let entries = match engine::smokes::registered(&list) {
        Ok(entries) => entries,
        Err(error) => return crate::exit(Err(error)),
    };
    let build = |engine: &mut Engine| {
        engine::game::built(&game, engine, engine::scene::SceneFiles::default())
    };
    let page = (game.bricks)(BrickList::default());
    run::<G>(options, names, &entries, (&list, &page), &build)
}

struct Native<'a, G: Game> {
    options: &'a HostOptions,
    game: std::marker::PhantomData<fn() -> G>,
}

impl<G: Game> Native<'_, G> {
    fn launch(&self, bricks: BrickList, build: Build<'_, G>) -> engine::Result<DevSession<G>> {
        let size = self.options.size.unwrap_or(SIZE);
        let mut game = Offscreen::launch(
            size,
            self.options.gpu,
            &self.options.jobs,
            move |_| bricks,
            build,
        )?;
        self.options.settings.apply(game.host_mut())?;
        Ok(DevSession::new(game, self.artifacts()))
    }

    fn artifacts(&self) -> PathBuf {
        self.options
            .dev_artifacts
            .clone()
            .unwrap_or_else(|| std::env::temp_dir().join("engine-smoke-captures"))
    }
}

impl<G: Game> SmokeHost<G> for Native<'_, G> {
    type Gpu = DevSession<G>;

    fn open(
        &mut self,
        bricks: BrickList,
        build: Build<'_, G>,
    ) -> impl Future<Output = engine::Result<DevSession<G>>> {
        let opened = self.launch(bricks, build);
        if let Err(error) = &opened {
            eprintln!("{}", json!({"event": "gpu_unavailable", "error": error}));
        }
        std::future::ready(opened)
    }
}

struct Relay {
    page: Vec<&'static str>,
}

impl Relay {
    fn launch(&self, bricks: BrickList) -> engine::Result<Relayed> {
        let without: Vec<&str> = self
            .page
            .iter()
            .copied()
            .filter(|package| !bricks.contains(package))
            .collect();
        let routes = engine::engine(move |_| bricks)?;
        answered(
            &exchange(&json!({"relay": "open", "without": without})),
            "$.relay.open",
        )?;
        Ok(Relayed { routes })
    }
}

impl<G: Game> SmokeHost<G> for Relay {
    type Gpu = Relayed;

    fn open(
        &mut self,
        bricks: BrickList,
        _: Build<'_, G>,
    ) -> impl Future<Output = engine::Result<Relayed>> {
        std::future::ready(self.launch(bricks))
    }
}

pub struct Relayed {
    routes: Engine,
}

impl GpuSession for Relayed {
    fn world(&self) -> &World {
        self.routes.app.world()
    }

    fn send(&mut self, line: &str, _: &dyn Fn() -> Option<Value>) -> Value {
        let command = serde_json::from_str::<Value>(line).unwrap_or(Value::Null);
        exchange(&json!({"relay": "request", "command": command}))
    }

    fn play(&mut self, document: &str) -> engine::Result<()> {
        answered(
            &exchange(&json!({"relay": "source", "document": document})),
            "$.relay.source",
        )
    }
}

fn exchange(message: &Value) -> Value {
    let error = Error::new(ErrorCode::SessionClosed, "$.relay");
    let closed = || json!({"ok": false, "error": error});
    let mut out = std::io::stdout().lock();
    if writeln!(out, "{message}")
        .and_then(|()| out.flush())
        .is_err()
    {
        return closed();
    }
    drop(out);
    let mut line = String::new();
    match std::io::stdin().lock().read_line(&mut line) {
        Ok(0) | Err(_) => closed(),
        Ok(_) => serde_json::from_str(&line).unwrap_or_else(|_| closed()),
    }
}

fn answered(reply: &Value, path: &str) -> engine::Result<()> {
    if reply["ok"] == true {
        return Ok(());
    }
    Err(Error::new(ErrorCode::RuntimeUnavailable, path).got(reply["error"].clone()))
}
