use std::collections::{BTreeMap, VecDeque};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::net::{Shutdown, TcpStream};
use std::path::PathBuf;
use std::process::{Child, ChildStdin};
use std::sync::mpsc::{Receiver, Sender};
use std::thread;
use std::time::{Duration, Instant, SystemTime};

use serde_json::{Map, Value, json};

use super::browser::Browser;
use super::utc;

pub enum Event {
    Connected(TcpStream),
    Line(u64, String),
    Gone(u64),
    Stdout(String),
    Stderr(String),
    Devtools(Value),
    Closed,
    Built(Result<(), String>),
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Source {
    Console,
    Errors,
    FailedFetches,
    Stderr,
}

const SOURCES: [Source; 4] = [
    Source::Console,
    Source::Errors,
    Source::FailedFetches,
    Source::Stderr,
];

impl Source {
    fn name(self) -> &'static str {
        match self {
            Self::Console => "console",
            Self::Errors => "errors",
            Self::FailedFetches => "failed_fetches",
            Self::Stderr => "stderr",
        }
    }
}

pub enum Host {
    Native(Option<ChildStdin>),
    Browser(Box<Browser>),
}

pub struct Settings {
    pub deadline: Duration,
    pub reply_limit: usize,
}

pub struct Run {
    pub folder: PathBuf,
    script: File,
    host: File,
    record: Option<File>,
    replies: u32,
}

pub fn plain(name: &str) -> bool {
    !name.is_empty()
        && !name.starts_with('.')
        && name
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || "._-".contains(character))
}

impl Run {
    pub fn create(folder: PathBuf) -> Result<Self, String> {
        if let Some(kept) = ["record.jsonl", "session.json"]
            .into_iter()
            .find(|file| folder.join(file).exists())
        {
            return Err(format!(
                "run {} holds {kept}; runs are kept until deleted, so pick another --run",
                folder.display()
            ));
        }
        fs::create_dir_all(&folder)
            .map_err(|error| format!("cannot create {}: {error}", folder.display()))?;
        let file = |name: &str| {
            File::create(folder.join(name))
                .map_err(|error| format!("cannot create {name} in {}: {error}", folder.display()))
        };
        Ok(Self {
            script: file("script.jsonl")?,
            host: file("host.jsonl")?,
            record: None,
            replies: 0,
            folder,
        })
    }

    fn folder_of(&self, run: Option<&str>) -> Result<PathBuf, String> {
        match run {
            None => Ok(self.folder.clone()),
            Some(run) if plain(run) => {
                let folder = self.folder.with_file_name(run);
                fs::create_dir_all(&folder)
                    .map_err(|error| format!("cannot create {}: {error}", folder.display()))?;
                Ok(folder)
            }
            Some(run) => Err(format!("{run:?} is not a plain run name")),
        }
    }

    pub fn record(&mut self, line: &str) -> Result<(), String> {
        let header: Value = serde_json::from_str(line)
            .map_err(|error| format!("a record line is not JSON: {error}"))?;
        if header.get("machine").is_some() {
            let path = self.folder_of(header["name"].as_str())?.join("record.jsonl");
            self.record = Some(
                File::create(&path)
                    .map_err(|error| format!("cannot create {}: {error}", path.display()))?,
            );
        }
        let record = self
            .record
            .as_mut()
            .ok_or("a record entry came before its header")?;
        writeln!(record, "{}", line.trim_end())
            .map_err(|error| format!("cannot write record.jsonl: {error}"))
    }

    pub fn file(&self, run: Option<&str>, name: &str, bytes: &[u8]) -> Result<(), String> {
        if !plain(name) {
            return Err(format!("{name:?} is not a plain file name"));
        }
        let path = self.folder_of(run)?.join(name);
        fs::write(&path, bytes).map_err(|error| format!("cannot write {}: {error}", path.display()))
    }

    fn spill(&mut self, reply: &Value, bytes: usize) -> Result<Value, String> {
        self.replies += 1;
        let folder = self.folder.join("replies");
        fs::create_dir_all(&folder).map_err(|error| error.to_string())?;
        let path = folder.join(format!("{}.json", self.replies));
        fs::write(&path, reply.to_string())
            .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
        let mut short = reply.as_object().cloned().unwrap_or_default();
        short.remove("value");
        short.insert("file".into(), json!(path));
        short.insert("bytes".into(), json!(bytes));
        Ok(Value::Object(short))
    }
}

struct Holder {
    stream: TcpStream,
    generation: u64,
    peer: String,
    since: String,
}

pub struct Request {
    client: u64,
    id: Option<Value>,
    line: u64,
    body: Value,
    kept: Value,
    deadline: Instant,
}

impl Request {
    fn field(&self, key: &str) -> &str {
        self.body[key].as_str().unwrap_or_default()
    }

    fn named(&self) -> Value {
        json!({"verb": self.field("verb"), "name": self.field("name")})
    }
}

pub struct Running {
    request: Request,
    started: Instant,
    pub wire: u64,
    answered: bool,
}

pub struct Session {
    pub events: Receiver<Event>,
    pub sender: Sender<Event>,
    pub host: Host,
    pub run: Run,
    pub facts: Value,
    pub started: Instant,
    pub running: Option<Running>,
    pub wire: u64,
    pub answers: BTreeMap<u64, Value>,
    pub panic: Option<Value>,
    pub exit: Option<Value>,
    pub identity: Option<Value>,
    pub built: Option<Result<(), String>>,
    pub listening: Option<String>,
    settings: Settings,
    holder: Option<Holder>,
    generation: u64,
    lines: u64,
    queue: VecDeque<Request>,
    log: [Vec<Value>; 4],
    since: [usize; 4],
    refused: Vec<Value>,
    ended: Option<Option<i32>>,
    closing: bool,
    closed: Option<Instant>,
    child: Child,
}

impl Drop for Session {
    fn drop(&mut self) {
        self.reap(true);
    }
}

pub const NATIVE_REBUILD: &str = "native_program_cannot_be_swapped";

fn unanswered_by_game() -> Value {
    json!({"unavailable": "answered_by_driver"})
}

pub fn refusal(code: &str, got: Value) -> Value {
    json!({
        "ok": false,
        "frame": unanswered_by_game(),
        "tick": unanswered_by_game(),
        "error": {"code": code, "path": "$", "got": got},
    })
}

pub fn accepted(value: Value) -> Value {
    json!({
        "ok": true,
        "frame": unanswered_by_game(),
        "tick": unanswered_by_game(),
        "ready": unanswered_by_game(),
        "value": value,
    })
}

pub fn panic_in(text: &str) -> Option<Value> {
    let (_, report) = text.split_once(engine_panic::PANIC_PREFIX)?;
    let report = report.lines().next().unwrap_or_default();
    Some(serde_json::from_str(report).unwrap_or_else(|_| json!(report)))
}

fn is_host(name: &str) -> bool {
    name == "host" || name.starts_with("host.")
}

impl Session {
    pub fn new(
        (sender, events): (Sender<Event>, Receiver<Event>),
        child: Child,
        host: Host,
        run: Run,
        settings: Settings,
    ) -> Self {
        Self {
            events,
            sender,
            host,
            run,
            facts: Value::Null,
            started: Instant::now(),
            running: None,
            wire: 0,
            answers: BTreeMap::new(),
            panic: None,
            exit: None,
            identity: None,
            built: None,
            listening: None,
            settings,
            holder: None,
            generation: 0,
            lines: 0,
            queue: VecDeque::new(),
            log: Default::default(),
            since: [0; 4],
            refused: Vec::new(),
            ended: None,
            closing: false,
            closed: None,
            child,
        }
    }

    pub fn native(&self) -> bool {
        matches!(self.host, Host::Native(_))
    }

    pub fn browser(&mut self) -> Result<&mut Browser, String> {
        match &mut self.host {
            Host::Browser(browser) => Ok(browser),
            Host::Native(_) => Err("the native host has no page".into()),
        }
    }

    pub fn step(&mut self) {
        self.step_until(None);
    }

    pub fn step_until(&mut self, limit: Option<Instant>) {
        let deadline = self
            .running
            .iter()
            .filter(|running| !running.answered)
            .map(|running| running.request.deadline)
            .chain(self.queue.iter().map(|request| request.deadline))
            .chain(limit)
            .min();
        let event = match deadline {
            Some(at) => self
                .events
                .recv_timeout(at.saturating_duration_since(Instant::now()))
                .ok(),
            None => self.events.recv().ok(),
        };
        if let Some(event) = event {
            self.handle(event);
        }
        self.expire();
    }

    fn handle(&mut self, event: Event) {
        match event {
            Event::Connected(stream) => self.connect(stream),
            Event::Line(generation, text) => {
                if self.holder.as_ref().is_some_and(|holder| holder.generation == generation) {
                    self.intake(generation, &text);
                }
            }
            Event::Gone(generation) => {
                if self.holder.as_ref().is_some_and(|holder| holder.generation == generation) {
                    self.holder = None;
                    self.queue.retain(|request| request.client != generation);
                }
            }
            Event::Stdout(line) => self.stdout(line),
            Event::Stderr(line) => self.stderr(line),
            Event::Devtools(message) => self.devtools(message),
            Event::Closed => self.reap(false),
            Event::Built(result) => self.built = Some(result),
        }
    }

    fn connect(&mut self, mut stream: TcpStream) {
        let peer = stream
            .peer_addr()
            .map_or_else(|error| error.to_string(), |address| address.to_string());
        if let Some(holder) = &self.holder {
            let busy = refusal(
                "session_busy",
                json!({"holder": holder.peer, "since": holder.since}),
            );
            thread::spawn(move || {
                let _ = writeln!(stream, "{busy}");
                let _ = stream.shutdown(Shutdown::Write);
                let _ = std::io::copy(&mut stream, &mut std::io::sink());
            });
            return;
        }
        let Ok(reader) = stream.try_clone() else {
            return;
        };
        self.generation += 1;
        let generation = self.generation;
        let sender = self.sender.clone();
        thread::spawn(move || {
            for line in BufReader::new(reader).lines() {
                let Ok(line) = line else { break };
                if sender.send(Event::Line(generation, line)).is_err() {
                    return;
                }
            }
            let _ = sender.send(Event::Gone(generation));
        });
        self.holder = Some(Holder {
            stream,
            generation,
            peer,
            since: utc(SystemTime::now()),
        });
    }

    fn intake(&mut self, client: u64, text: &str) {
        if text.trim().is_empty() {
            return;
        }
        self.lines += 1;
        let line = self.lines;
        let Ok(Value::Object(mut body)) = serde_json::from_str::<Value>(text) else {
            let reply = refusal("invalid_json", json!(text));
            return self.reply_to(client, None, reply);
        };
        let id = body.remove("id");
        let kept = Value::Object(body.clone());
        let deadline = match body.remove("deadline_ms") {
            None => self.settings.deadline,
            Some(Value::Number(milliseconds)) if milliseconds.as_u64().is_some() => {
                Duration::from_millis(milliseconds.as_u64().unwrap_or_default())
            }
            Some(other) => {
                let reply = refusal("invalid_deadline", json!({"deadline_ms": other}));
                return self.reply_to(client, id, reply);
            }
        };
        let request = Request {
            client,
            id,
            line,
            body: Value::Object(body),
            kept,
            deadline: Instant::now() + deadline,
        };
        match (request.field("verb"), request.field("name")) {
            ("call", "host.close") => {
                self.cancel("host.close");
                self.closing = true;
                self.finish(request, accepted(json!({"closing": true})), false);
                self.close_host();
            }
            ("call", "host.rebuild") if !self.native() => {
                self.cancel("host.rebuild");
                self.queue.push_front(request);
            }
            (verb, name) if is_host(name) && verb != "call" => {
                let reply = self.host_answer(&request);
                self.finish(request, reply, false);
            }
            _ => self.queue.push_back(request),
        }
    }

    fn cancel(&mut self, by: &str) {
        let cancelled = refusal("cancelled", json!({"by": by}));
        if let Some(running) = self.running.take() {
            self.finish(running.request, cancelled.clone(), running.answered);
        }
        for request in self.queue.drain(..).collect::<Vec<_>>() {
            self.finish(request, cancelled.clone(), false);
        }
    }

    pub fn serve(&mut self) -> Value {
        while self.ended.is_none() {
            if self.exit.is_some() {
                self.close_host();
                self.fail_all();
                let limit = self.closed.map(|at| at + self.settings.deadline);
                if limit.is_some_and(|limit| limit <= Instant::now()) {
                    self.reap(true);
                } else {
                    self.step_until(limit);
                }
                continue;
            }
            match (&self.running, self.queue.pop_front()) {
                (None, Some(request)) => self.dispatch(request),
                (_, request) => {
                    if let Some(request) = request {
                        self.queue.push_front(request);
                    }
                    self.step();
                }
            }
        }
        json!({
            "event": "ended",
            "exit_code": self.ended.flatten(),
            "exit": self.exit,
            "refused": self.refused,
            "host": SOURCES.map(|source| json!({source.name(): self.log[source as usize].len()})),
        })
    }

    fn dispatch(&mut self, request: Request) {
        match request.field("name") {
            "host.rebuild" => {
                self.start_running(request, 0);
                let reply = self.rebuild();
                if let Some(running) = self.running.take() {
                    self.finish(running.request, reply, running.answered);
                }
            }
            name if is_host(name) => {
                let reply = self.host_answer(&request);
                self.finish(request, reply, false);
            }
            _ => {
                self.since = SOURCES.map(|source| self.log[source as usize].len());
                self.wire += 1;
                let wire = self.wire;
                let sent = match &mut self.host {
                    Host::Native(stdin) => super::native::send(stdin.as_mut(), wire, &request.body),
                    Host::Browser(browser) => browser.request(wire, &request.body),
                };
                match sent {
                    Ok(()) => self.start_running(request, wire),
                    Err(error) => self.finish(request, refusal("game_unreachable", json!(error)), false),
                }
            }
        }
    }

    fn start_running(&mut self, request: Request, wire: u64) {
        self.running = Some(Running {
            request,
            started: Instant::now(),
            wire,
            answered: false,
        });
    }

    pub fn complete(&mut self, mut reply: Value) {
        if let Some(running) = self.running.take() {
            if let Some(reply) = reply.as_object_mut() {
                reply.remove("id");
            }
            if running.request.field("verb") == "describe"
                && running.request.field("name").is_empty()
                && let Some(names) = reply.pointer_mut("/value/names").and_then(Value::as_array_mut)
            {
                names.extend(self.host_names());
            }
            self.finish(running.request, reply, running.answered);
        }
    }

    fn finish(&mut self, request: Request, reply: Value, answered: bool) {
        if reply["ok"] == true {
            if !is_host(request.field("name")) {
                let _ = writeln!(self.run.script, "{}", request.kept);
            }
        } else if !answered {
            self.refused.push(json!({
                "line": request.line,
                "request": request.kept,
                "error": reply["error"]["code"],
            }));
        }
        if answered {
            println!("{}", json!({"event": "late", "line": request.line, "reply": reply}));
        } else {
            self.reply_to(request.client, request.id, reply);
        }
    }

    fn reply_to(&mut self, client: u64, id: Option<Value>, mut reply: Value) {
        let Some(holder) = self.holder.as_mut().filter(|holder| holder.generation == client) else {
            println!("{}", json!({"event": "unanswered", "reply": reply}));
            return;
        };
        if let (Some(id), Some(fields)) = (id, reply.as_object_mut()) {
            fields.insert("id".into(), id);
        }
        let mut text = reply.to_string();
        if text.len() > self.settings.reply_limit {
            text = match self.run.spill(&reply, text.len()) {
                Ok(short) => short.to_string(),
                Err(error) => refusal("reply_not_written", json!(error)).to_string(),
            };
        }
        if writeln!(holder.stream, "{text}").is_err() {
            self.holder = None;
        }
    }

    fn expire(&mut self) {
        let now = Instant::now();
        let mut late = Vec::new();
        let behind = self.running.as_mut().map(|running| {
            let running_ms = running.started.elapsed().as_millis();
            let named = running.request.named();
            if !running.answered && running.request.deadline <= now {
                running.answered = true;
                late.push((
                    running.request.client,
                    running.request.id.clone(),
                    refusal("still_running", json!({"request": named, "running_ms": running_ms})),
                    running.request.line,
                    running.request.kept.clone(),
                ));
            }
            json!({"request": named, "running_ms": running_ms})
        });
        let (expired, waiting): (Vec<_>, Vec<_>) =
            self.queue.drain(..).partition(|request| request.deadline <= now);
        self.queue = waiting.into();
        for request in expired {
            let reply = refusal(
                "waiting",
                json!({"request": request.named(), "behind": behind}),
            );
            self.finish(request, reply, false);
        }
        for (client, id, reply, line, kept) in late {
            self.refused.push(json!({"line": line, "request": kept, "error": "still_running"}));
            self.reply_to(client, id, reply);
        }
    }

    fn stdout(&mut self, line: String) {
        let parsed = serde_json::from_str::<Value>(&line).ok();
        match parsed {
            Some(identity) if self.identity.is_none() && identity.get("ready").is_some() => {
                self.identity = Some(identity);
            }
            Some(reply)
                if reply["id"].as_u64().is_some_and(|id| {
                    self.running.as_ref().is_some_and(|running| running.wire == id)
                }) =>
            {
                self.complete(reply);
            }
            _ => self.note(Source::Console, json!({"text": line})),
        }
    }

    fn stderr(&mut self, line: String) {
        if self.listening.is_none()
            && let Some(url) = line.strip_prefix("DevTools listening on ")
        {
            self.listening = Some(url.trim().to_owned());
        }
        if let Some(report) = panic_in(&line) {
            self.panic = Some(report.clone());
            self.note(Source::Errors, json!({"panic": report}));
        }
        self.note(Source::Stderr, json!({"text": line}));
    }

    pub fn note(&mut self, source: Source, mut entry: Value) {
        entry["t"] = json!(self.started.elapsed().as_secs_f64());
        let _ = writeln!(
            self.run.host,
            "{}",
            json!({"source": source.name(), "entry": entry})
        );
        self.log[source as usize].push(entry);
    }

    pub fn since(&self, source: Source) -> Vec<Value> {
        self.log[source as usize]
            .get(self.since[source as usize]..)
            .unwrap_or_default()
            .to_vec()
    }

    fn reap(&mut self, kill: bool) {
        if self.ended.is_some() {
            return;
        }
        if kill && matches!(self.child.try_wait(), Ok(None)) {
            let pid = self.child.id();
            let killed = self.child.kill().map_err(|error| error.to_string());
            let text = format!("the driver killed the process it started (pid {pid}) by its handle: {killed:?}");
            self.note(Source::Stderr, json!({"driver": "killed", "pid": pid, "text": text}));
        }
        let code = self.child.wait().ok().and_then(|status| status.code());
        self.exited(code);
    }

    fn exited(&mut self, code: Option<i32>) {
        self.ended = Some(code);
        if self.exit.is_none() {
            self.exit = Some(if self.native() {
                refusal(
                    "game_exited",
                    json!({"exit_code": code, "panic": self.panic, "stderr": self.since(Source::Stderr)}),
                )
            } else {
                refusal(
                    "browser_exited",
                    json!({"exit_code": code, "stderr": self.since(Source::Stderr)}),
                )
            });
        }
        self.fail_all();
    }

    pub fn fail_all(&mut self) {
        let Some(exit) = self.exit.clone() else {
            return;
        };
        if let Some(running) = self.running.take() {
            self.finish(running.request, exit.clone(), running.answered);
        }
        for request in self.queue.drain(..).collect::<Vec<_>>() {
            self.finish(request, exit.clone(), false);
        }
    }

    fn close_host(&mut self) {
        if self.closed.is_some() {
            return;
        }
        self.closed = Some(Instant::now());
        match &mut self.host {
            Host::Native(stdin) => drop(stdin.take()),
            Host::Browser(browser) => browser.close(),
        }
        if self.exit.is_none() && self.closing {
            self.exit = Some(refusal("session_closed", json!({"by": "host.close"})));
        }
    }

    pub fn wait_ready(&mut self) -> Result<(), String> {
        while self.identity.is_none() {
            if let Some(exit) = &self.exit {
                return Err(exit["error"].to_string());
            }
            self.step();
        }
        Ok(())
    }

    fn host_names(&self) -> Vec<Value> {
        let row = |name: &str, kind: &str, verb: &str, args: Value, example: Value| {
            json!({
                "name": name,
                "kind": kind,
                "owner": "driver",
                "verbs": [verb],
                "args": args,
                "examples": [{"verb": verb, "name": name, "args": example}],
            })
        };
        let integer = json!({"type": "integer", "minimum": 0});
        let pointer = json!({"type": "string"});
        let mut names = vec![row(
            "host",
            "state",
            "read",
            json!({"type": "object", "properties": {"pointer": pointer}, "additionalProperties": false}),
            json!({}),
        )];
        for source in SOURCES {
            let mut entry = row(
                &format!("host.{}", source.name()),
                "state",
                "read",
                json!({"type": "object", "properties": {"offset": integer, "limit": integer, "pointer": pointer}, "additionalProperties": false}),
                json!({"offset": 0, "limit": 20}),
            );
            if let Some(reason) = self.unavailable(source) {
                entry["unavailable"] = json!(reason);
            }
            names.push(entry);
        }
        let none = json!({"type": "object", "additionalProperties": false});
        let mut rebuild = row("host.rebuild", "call", "call", none.clone(), json!({}));
        if self.native() {
            rebuild["unavailable"] = json!(NATIVE_REBUILD);
        }
        names.push(rebuild);
        names.push(row("host.close", "call", "call", none, json!({})));
        names
    }

    fn unavailable(&self, source: Source) -> Option<&'static str> {
        (self.native() && source == Source::FailedFetches)
            .then_some("native_host_fetches_nothing")
    }

    fn host_answer(&self, request: &Request) -> Value {
        let name = request.field("name");
        let args = &request.body["args"];
        if request.field("verb") == "describe" {
            let names: Vec<Value> = self
                .host_names()
                .into_iter()
                .filter(|entry| entry["name"].as_str().is_some_and(|entry| entry.starts_with(name)))
                .collect();
            return accepted(json!({"names": names}));
        }
        if request.field("verb") != "read" {
            return refusal(
                "invalid_verb",
                json!({"name": name, "verb": request.field("verb"), "expected": ["describe", "read"]}),
            );
        }
        let value = if name == "host" {
            let counts: Map<String, Value> = SOURCES
                .iter()
                .map(|source| (source.name().to_owned(), json!(self.log[*source as usize].len())))
                .collect();
            json!({"session": self.facts, "counts": counts})
        } else if let Some(source) = SOURCES
            .into_iter()
            .find(|source| name.strip_prefix("host.") == Some(source.name()))
        {
            if let Some(reason) = self.unavailable(source) {
                json!({"unavailable": reason})
            } else {
                let entries = &self.log[source as usize];
                let offset = args["offset"].as_u64().map_or(0, |offset| offset as usize);
                let limit = args["limit"].as_u64().map_or(usize::MAX, |limit| limit as usize);
                let page: Vec<Value> = entries.iter().skip(offset).take(limit).cloned().collect();
                json!({"total": entries.len(), "entries": page})
            }
        } else {
            let nearest: Vec<Value> = self.host_names().into_iter().map(|entry| entry["name"].clone()).collect();
            let mut reply = refusal("unknown_name", json!(name));
            reply["error"]["path"] = json!("name");
            reply["error"]["nearest"] = json!(nearest);
            return reply;
        };
        match args["pointer"].as_str() {
            None => accepted(value),
            Some(pointer) => match value.pointer(pointer) {
                Some(found) => accepted(found.clone()),
                None => {
                    let mut reply = refusal("invalid_pointer", json!(pointer));
                    reply["error"]["path"] = json!("args.pointer");
                    reply
                }
            },
        }
    }
}
