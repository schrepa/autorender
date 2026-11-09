use std::path::PathBuf;
use std::process::ExitCode;

use engine::bricks::BrickList;
use engine::smokes::{Build, LIST, SmokeEntry, SmokeHost, SmokeRecord, listed};
use engine::{Engine, Game};
use serde_json::{Value, json};

use crate::dev::DevSession;
use crate::{HostOptions, Offscreen};

const SIZE: (u32, u32) = (960, 540);

pub type Attach<'a, G> = &'a dyn Fn(DevSession<G>) -> DevSession<G>;

pub fn run<G: Game>(
    options: &HostOptions,
    names: &str,
    entries: &[SmokeEntry],
    bricks: &BrickList,
    build: Build<'_, G>,
    attach: Attach<'_, G>,
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
    let ran = pollster::block_on(engine::smokes::run(
        &mut Native { options, attach },
        names,
        limit_ms as f64,
        entries,
        bricks,
        build,
        &mut |record: &SmokeRecord| println!("{}", record.line()),
    ));
    match ran {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(3),
        Err(error) => {
            eprintln!("{}", serde_json::to_value(&error).unwrap_or(Value::Null));
            ExitCode::from(2)
        }
    }
}

pub fn of_game<G: Game, E: Into<engine::Error>>(
    options: &HostOptions,
    names: &str,
    bricks: &dyn Fn(BrickList) -> BrickList,
    build: &dyn Fn(&mut Engine) -> Result<G, E>,
) -> ExitCode {
    let list = bricks(BrickList::default());
    let entries = match engine::smokes::registered(&list) {
        Ok(entries) => entries,
        Err(error) => return crate::exit(Err(error)),
    };
    let build = |_: Option<&str>, engine: &mut Engine| build(engine).map_err(Into::into);
    run::<G>(options, names, &entries, &list, &build, &|session| session)
}

struct Native<'a, G: Game> {
    options: &'a HostOptions,
    attach: Attach<'a, G>,
}

impl<G: Game> Native<'_, G> {
    fn launch(&self, bricks: BrickList, build: Build<'_, G>) -> engine::Result<DevSession<G>> {
        let size = self.options.size.unwrap_or(SIZE);
        let mut game = Offscreen::launch(
            size,
            self.options.gpu,
            &self.options.jobs,
            move |_| bricks,
            |engine| build(None, engine),
        )?;
        self.options.settings.apply(game.host_mut())?;
        Ok((self.attach)(DevSession::new(game, self.artifacts())))
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
