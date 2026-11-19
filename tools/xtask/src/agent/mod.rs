mod browser;
#[cfg(windows)]
mod job;
mod native;
mod session;
mod socket;

use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::sync::mpsc::channel;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use engine_command_line::{Command, CommandOption, commands_text};
use serde_json::{Value, json};

use crate::process::{Client, Project, clients, native_folders};
use crate::serve::free_listener;
use crate::pipeline;
use session::{Event, Host, Run, Session, Settings};

const DEADLINE_MS: u64 = 15_000;
const REPLY_LIMIT: usize = 65_536;
const RUNS: &str = "artifacts/runs";

pub fn commands() -> Vec<Command> {
    let run = || CommandOption::taking("--run", "NAME");
    vec![
        Command::new(
            "agent start",
            [
                CommandOption::taking("--host", "native|browser").required(),
                CommandOption::taking("--game", "NAME").required(),
                CommandOption::taking("--build", "dev|profiling").defaults_to("dev"),
                CommandOption::flag("--window"),
                CommandOption::taking("--size", "WxH").in_unit("px"),
                CommandOption::taking("--run", "NAME").defaults_to("<game>-<host>-<utc time>"),
                CommandOption::taking("--script", "FILE"),
                CommandOption::taking("--chrome", "PATH"),
                CommandOption::taking("--deadline-ms", "MS")
                    .defaults_to(DEADLINE_MS)
                    .in_unit("ms"),
                CommandOption::taking("--reply-limit", "N")
                    .defaults_to(REPLY_LIMIT)
                    .in_unit("bytes"),
                CommandOption::taking("--", "GAME_ARGS..."),
            ],
        ),
        Command::new("agent send", [run(), CommandOption::taking("LINE", "JSON...").required()]),
        Command::new("agent script", [run(), CommandOption::taking("FILE", "PATH").required()]),
        Command::new("agent help", []),
    ]
}

pub fn agent(root: &Path, arguments: &[String]) -> Result<u8, String> {
    let (command, rest) = arguments.split_first().map_or(("help", &[][..]), |(command, rest)| (command.as_str(), rest));
    match command {
        "start" => start(root, rest),
        "send" => {
            let (run, lines) = run_option(rest)?;
            send(root, run.as_deref(), lines.into_iter().enumerate().map(|(index, line)| (index + 1, line)))
        }
        "script" => {
            let (run, files) = run_option(rest)?;
            let [file] = files.as_slice() else {
                return Err("usage: cargo xtask agent script FILE [--run NAME]".into());
            };
            send(root, run.as_deref(), script(Path::new(file))?)
        }
        "help" | "--help" => {
            println!("{}", commands_text(&commands()));
            println!("names: cargo xtask agent send '{{\"verb\":\"describe\"}}'");
            Ok(0)
        }
        other => Err(format!("unknown agent command {other:?}; run `cargo xtask agent help`")),
    }
}

struct Start {
    host: String,
    game: String,
    profiling: bool,
    window: bool,
    size: Option<(u32, u32)>,
    run: Option<String>,
    script: Option<PathBuf>,
    chrome: Option<PathBuf>,
    deadline: Duration,
    reply_limit: usize,
    arguments: Vec<String>,
}

fn options(arguments: &[String]) -> Result<Start, String> {
    let mut start = Start {
        host: String::new(),
        game: String::new(),
        profiling: false,
        window: false,
        size: None,
        run: None,
        script: None,
        chrome: None,
        deadline: Duration::from_millis(DEADLINE_MS),
        reply_limit: REPLY_LIMIT,
        arguments: Vec::new(),
    };
    let mut given = arguments.iter();
    while let Some(option) = given.next() {
        if option == "--" {
            start.arguments = given.cloned().collect();
            break;
        }
        if option == "--window" {
            start.window = true;
            continue;
        }
        let value = given
            .next()
            .ok_or_else(|| format!("{option} takes a value; run `cargo xtask agent help`"))?
            .clone();
        let number = |what: &str| value.parse::<u64>().map_err(|_| format!("{option} takes {what}, got {value:?}"));
        match option.as_str() {
            "--host" if value == "native" || value == "browser" => start.host = value,
            "--game" => start.game = value,
            "--build" if value == "dev" || value == "profiling" => start.profiling = value == "profiling",
            "--size" => {
                start.size = value
                    .split_once('x')
                    .and_then(|(width, height)| Some((width.parse().ok()?, height.parse().ok()?)))
                    .filter(|&(width, height)| width > 0 && height > 0);
                if start.size.is_none() {
                    return Err(format!("--size takes WxH, got {value:?}"));
                }
            }
            "--run" => start.run = Some(value),
            "--script" => start.script = Some(value.into()),
            "--chrome" => start.chrome = Some(value.into()),
            "--deadline-ms" => start.deadline = Duration::from_millis(number("milliseconds")?),
            "--reply-limit" => start.reply_limit = usize::try_from(number("bytes")?).map_err(|error| error.to_string())?,
            _ => return Err(format!("{option} {value} is not an agent start option; run `cargo xtask agent help`")),
        }
    }
    if start.host.is_empty() || start.game.is_empty() {
        return Err("agent start needs --host native|browser and --game NAME".into());
    }
    if start.host == "native" && !start.window && start.size.is_none() {
        return Err("a headless native game needs --size WxH (or pass --window)".into());
    }
    Ok(start)
}

fn start(root: &Path, arguments: &[String]) -> Result<u8, String> {
    let start = options(arguments)?;
    #[cfg(windows)]
    job::contain()?;
    let natives = timed("games", || native_projects(root))?;
    let native = natives.iter().find(|(project, game)| {
        [Some(&project.package), Some(&project.native), game.as_ref()].contains(&Some(&start.game))
    });
    let page = clients().iter().find(|client| {
        [Some(&client.game), Some(&client.package), client.native.as_ref()].contains(&Some(&start.game))
    });
    let startable: Vec<&str> = if start.host == "native" {
        natives.iter().map(|(project, _)| project.package.as_str()).collect()
    } else {
        clients().iter().map(|client| client.game.as_str()).collect()
    };
    match (start.host.as_str(), native, page) {
        ("native", None, _) | ("browser", None, None) => {
            return Err(format!(
                "--game {} names nothing that can start on the {} host; these can: {}",
                start.game,
                start.host,
                startable.join(", ")
            ));
        }
        ("browser", Some((project, _)), None) => {
            return Err(format!(
                "--game {} ({}) declares no page in [package.metadata.engine], so it starts on the native host only",
                start.game, project.package
            ));
        }
        _ => {}
    }
    let name = start.run.clone().unwrap_or_else(|| {
        let [year, month, day, hour, minute, second] = calendar(SystemTime::now());
        format!("{}-{}-{year:04}{month:02}{day:02}-{hour:02}{minute:02}{second:02}", start.game, start.host)
    });
    let folder = run_folder(root, &name)?;
    let run = Run::create(folder.clone())?;
    let settings = Settings {
        deadline: start.deadline,
        reply_limit: start.reply_limit,
    };
    let started = match (native, page) {
        (Some((project, _)), _) if start.host == "native" => {
            native_session(root, project, &start, &name, run, settings)
        }
        (_, Some(client)) => browser_session(root, client, &start, run, settings),
        _ => Err("no game was chosen".to_owned()),
    };
    let mut session = started.map_err(|error| {
        format!(
            "{error}; {} holds the host lines for diagnosis, and a later start may reuse --run {name}",
            folder.join("host.jsonl").display()
        )
    })?;
    let listener = free_listener()?;
    let port = listener.local_addr().map_err(|error| error.to_string())?.port();
    let mut facts = json!({
        "port": port,
        "pid": std::process::id(),
        "host": start.host,
        "build": if start.profiling { "profiling" } else { "dev" },
        "window": start.window,
        "game": start.game,
        "run": name,
        "folder": folder,
        "started": utc(SystemTime::now()),
    });
    if let (Some(facts), Some(more)) = (facts.as_object_mut(), session.facts.as_object()) {
        facts.extend(more.clone());
    }
    session.facts = facts;
    let file = folder.join("session.json");
    fs::write(&file, session.facts.to_string()).map_err(|error| format!("cannot write {}: {error}", file.display()))?;
    let sender = session.sender.clone();
    thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            if sender.send(Event::Connected(stream)).is_err() {
                return;
            }
        }
    });
    println!("{}", json!({"event": "ready", "session": session.facts}));
    if let Some(script_file) = start.script.clone() {
        let lines = script(&script_file)?;
        thread::spawn(move || {
            let ran = TcpStream::connect(("127.0.0.1", port))
                .map_err(|error| error.to_string())
                .and_then(|stream| lines_over(stream, lines));
            if let Err(error) = ran {
                eprintln!("{error}");
            }
        });
    }
    let ended = session.serve();
    let _ = fs::remove_file(&file);
    println!("{ended}");
    let code = ended["exit_code"].as_i64().and_then(|code| u8::try_from(code).ok());
    let closed = ended["exit"]["error"]["code"] == "session_closed";
    Ok(match (start.host.as_str(), code) {
        ("native", Some(code)) => code,
        _ => u8::from(!closed),
    })
}

fn native_projects(root: &Path) -> Result<Vec<(Project, Option<String>)>, String> {
    native_folders()?
        .into_iter()
        .map(|folder| {
            let game = clients()
                .iter()
                .find(|client| client.folder == folder)
                .map(|client| client.game.clone());
            Ok((Project::read(root, &folder)?, game))
        })
        .collect()
}

fn native_session(
    root: &Path,
    project: &Project,
    start: &Start,
    name: &str,
    run: Run,
    settings: Settings,
) -> Result<Session, String> {
    let program = timed("build", || {
        if start.profiling {
            pipeline::build_profiling(root, project)
        } else {
            pipeline::build_native(root, project)
        }
    })?;
    let mut arguments = start.arguments.clone();
    arguments.extend(["--dev".to_owned(), "--run".to_owned(), name.to_owned()]);
    if !start.window {
        arguments.push("--offscreen".to_owned());
    }
    if let Some((width, height)) = start.size {
        arguments.extend(["--size".to_owned(), format!("{width}x{height}")]);
    }
    let channel = channel();
    let started = Instant::now();
    let (child, stdin) = native::spawn(root, &program, &arguments, &channel.0)?;
    let mut session = Session::new(channel, child, Host::Native(Some(stdin)), run, settings);
    let ready = session.wait_ready();
    println!("{}", json!({"event": "step", "step": "game", "seconds": started.elapsed().as_secs_f64()}));
    ready?;
    session.facts = json!({"program": program, "arguments": arguments, "identity": session.identity});
    Ok(session)
}

fn browser_session(root: &Path, client: &Client, start: &Start, run: Run, settings: Settings) -> Result<Session, String> {
    let mut query = String::new();
    let mut given = start.arguments.iter();
    while let Some(option) = given.next() {
        match (option.strip_prefix("--"), given.next()) {
            (Some(name), Some(value)) if !name.is_empty() => {
                query.push_str(&format!("&{}={}", encoded(name), encoded(value)));
            }
            _ => return Err(format!("browser game arguments are --name value pairs, each becoming ?name=value; got {option:?}")),
        }
    }
    browser::start(
        browser::Page {
            root,
            game: &client.game,
            page: &client.page,
            query,
            profiling: start.profiling,
            window: start.window,
            size: start.size,
            chrome: start.chrome.clone(),
        },
        run,
        settings,
    )
}

pub fn timed<T>(step: &str, work: impl FnOnce() -> T) -> T {
    let started = Instant::now();
    let done = work();
    println!("{}", json!({"event": "step", "step": step, "seconds": started.elapsed().as_secs_f64()}));
    done
}

fn encoded(text: &str) -> String {
    text.bytes()
        .map(|byte| {
            if byte.is_ascii_alphanumeric() || b"-._~".contains(&byte) {
                char::from(byte).to_string()
            } else {
                format!("%{byte:02X}")
            }
        })
        .collect()
}

fn run_option(arguments: &[String]) -> Result<(Option<String>, Vec<String>), String> {
    let mut run = None;
    let mut rest = Vec::new();
    let mut given = arguments.iter();
    while let Some(argument) = given.next() {
        if argument == "--run" {
            run = Some(given.next().ok_or("--run takes a run name")?.clone());
        } else {
            rest.push(argument.clone());
        }
    }
    Ok((run, rest))
}

fn script(file: &Path) -> Result<Vec<(usize, String)>, String> {
    let text = fs::read_to_string(file).map_err(|error| format!("cannot read {}: {error}", file.display()))?;
    Ok(text
        .lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(index, line)| (index + 1, line.to_owned()))
        .collect())
}

fn send(root: &Path, run: Option<&str>, lines: impl IntoIterator<Item = (usize, String)>) -> Result<u8, String> {
    let file = session_file(root, run)?;
    let session: Value = fs::read_to_string(&file)
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .ok_or_else(|| format!("cannot read {}", file.display()))?;
    let port = session["port"].as_u64().and_then(|port| u16::try_from(port).ok()).unwrap_or_default();
    let stream = TcpStream::connect(("127.0.0.1", port)).map_err(|error| {
        format!("{} names port {port}, where nothing answers ({error}); the driver (pid {}) has stopped", file.display(), session["pid"])
    })?;
    lines_over(stream, lines.into_iter().collect())?;
    Ok(0)
}

fn lines_over(stream: TcpStream, lines: Vec<(usize, String)>) -> Result<(), String> {
    let mut writer = stream.try_clone().map_err(|error| error.to_string())?;
    let mut replies = BufReader::new(stream).lines();
    for (line, request) in lines {
        writeln!(writer, "{}", request.trim()).map_err(|error| format!("cannot send line {line}: {error}"))?;
        let reply = replies
            .next()
            .ok_or_else(|| format!("the session closed before answering line {line}"))?
            .map_err(|error| error.to_string())?;
        println!("{reply}");
        let refused = serde_json::from_str::<Value>(&reply).map_or(true, |reply| reply["ok"] != true);
        if refused {
            return Err(json!({"event": "refused", "line": line, "request": request}).to_string());
        }
    }
    Ok(())
}

fn run_folder(root: &Path, name: &str) -> Result<PathBuf, String> {
    if !session::plain(name) {
        return Err(format!("{name:?} is not a plain run name"));
    }
    Ok(root.join(RUNS).join(name))
}

fn session_file(root: &Path, run: Option<&str>) -> Result<PathBuf, String> {
    if let Some(run) = run {
        return Ok(run_folder(root, run)?.join("session.json"));
    }
    let folder = root.join(RUNS);
    let open: Vec<PathBuf> = fs::read_dir(&folder)
        .map(|entries| {
            entries
                .flatten()
                .map(|entry| entry.path().join("session.json"))
                .filter(|file| file.is_file())
                .collect()
        })
        .unwrap_or_default();
    match open.as_slice() {
        [file] => Ok(file.clone()),
        [] => Err(format!("no session is open under {}; start one with `cargo xtask agent start`", folder.display())),
        several => Err(format!(
            "{} sessions are open; pick one with --run: {}",
            several.len(),
            several
                .iter()
                .filter_map(|file| file.parent()?.file_name())
                .map(|name| name.to_string_lossy())
                .collect::<Vec<_>>()
                .join(", ")
        )),
    }
}

fn calendar(time: SystemTime) -> [u64; 6] {
    let seconds = time.duration_since(UNIX_EPOCH).map_or(0, |since| since.as_secs());
    let days = seconds / 86_400 + 719_468;
    let era = days / 146_097;
    let day_of_era = days % 146_097;
    let year_of_era = (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let shifted = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * shifted + 2) / 5 + 1;
    let month = if shifted < 10 { shifted + 3 } else { shifted - 9 };
    let year = year_of_era + era * 400 + u64::from(month <= 2);
    let clock = seconds % 86_400;
    [year, month, day, clock / 3600, clock / 60 % 60, clock % 60]
}

pub fn utc(time: SystemTime) -> String {
    let [year, month, day, hour, minute, second] = calendar(time);
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}
