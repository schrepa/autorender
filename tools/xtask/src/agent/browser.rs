use std::collections::BTreeMap;
use std::env;
use std::fs::{self, File, OpenOptions};
use std::path::{Path, PathBuf};
use std::process::{ChildStdout, Command, Stdio};
use std::sync::mpsc::channel;
use std::thread;
use std::time::{Duration, Instant};

use base64::Engine as _;
use serde_json::{Value, json};

use super::native::watch;
use super::timed;
use super::session::{
    Event, Host, Run, Session, Settings, Source, accepted, panic_in, refusal,
};
use super::socket::{self, Socket};
use crate::pipeline;
use crate::serve::{Server, free_port};

const SHADER_CACHES: [&str; 6] = [
    "DawnWebGPUCache",
    "DawnGraphiteCache",
    "GraphiteDawnCache",
    "GPUCache",
    "GrShaderCache",
    "ShaderCache",
];
const BINDINGS: [&str; 2] = ["__engineRecord", "__engineFile"];
const READY: &str = r##"new Promise(done => { const look = () => { if (typeof window.__engineDev?.request === "function") done(true); else if (document.querySelector("#status")?.getAttribute("data-state") === "error") done(document.querySelector("#message")?.textContent ?? document.querySelector("#status").textContent); else setTimeout(look, 50); }; look(); })"##;
const ADAPTER: &str = r#"(async () => { if (!navigator.gpu) return "navigator.gpu is missing"; const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" }); if (!adapter) return "navigator.gpu.requestAdapter() returned null"; const { vendor, architecture, device, description } = adapter.info; return { vendor, architecture, device, description }; })()"#;

pub struct Page<'a> {
    pub root: &'a Path,
    pub game: &'a str,
    pub page: &'a str,
    pub query: String,
    pub profiling: bool,
    pub window: bool,
    pub size: Option<(u32, u32)>,
    pub chrome: Option<PathBuf>,
}

const PORT_FILE_POLL: Duration = Duration::from_millis(50);

pub struct Browser {
    socket: Option<Socket>,
    profile: PathBuf,
    page: String,
    root: PathBuf,
    game: String,
    profiling: bool,
    pub chrome: PathBuf,
    urls: BTreeMap<String, String>,
    navigations: u64,
    loaded: bool,
    _profile: File,
    _server: Server,
}

impl Browser {
    fn send(&self, id: u64, method: &str, params: &Value, page: bool) -> Result<(), String> {
        let socket = self.socket.as_ref().ok_or("the DevTools socket is not open")?;
        let mut message = json!({"id": id, "method": method, "params": params});
        if page {
            message["sessionId"] = json!(self.page);
        }
        socket.send(&message.to_string())
    }

    pub fn request(&self, wire: u64, body: &Value) -> Result<(), String> {
        let line = Value::String(body.to_string());
        self.send(
            wire,
            "Runtime.evaluate",
            &json!({"expression": format!("window.__engineDev.request({line})"), "awaitPromise": true, "returnByValue": true}),
            true,
        )
    }

    pub fn close(&mut self) {
        let _ = self.send(0, "Browser.close", &json!({}), false);
    }
}

pub fn start(page: Page<'_>, run: Run, settings: Settings) -> Result<Session, String> {
    let chrome = find(page.chrome.clone())?;
    timed("build", || build(page.root, page.game, page.profiling))?;
    let mode: &[&str] = if page.profiling { &["--mode", "profiling"] } else { &[] };
    let server = timed("vite", || {
        Server::vite(page.root, free_port()?, &run.folder.join("vite.log"), mode)
    })?;
    let address = format!("{}/{}?engine_dev=1{}", server.url, page.page, page.query);
    let (profile, lock) = timed("profile", claim_profile)?;
    let mut flags = vec![
        format!("--user-data-dir={}", profile.display()),
        "--remote-debugging-port=0".to_owned(),
        "--enable-unsafe-webgpu".to_owned(),
        "--no-first-run".to_owned(),
        "--no-default-browser-check".to_owned(),
        "--disable-background-timer-throttling".to_owned(),
        "--disable-backgrounding-occluded-windows".to_owned(),
        "--disable-renderer-backgrounding".to_owned(),
    ];
    if !page.window {
        flags.push("--headless=new".to_owned());
    }
    if let Some((width, height)) = page.size {
        flags.push(format!("--window-size={width},{height}"));
    }
    flags.push("about:blank".to_owned());
    let channel = channel();
    let started = Instant::now();
    let mut child = Command::new(&chrome)
        .args(&flags)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("cannot start {}: {error}", chrome.display()))?;
    watch(&mut child, None::<ChildStdout>, &channel.0);
    let browser = Browser {
        socket: None,
        profile,
        page: String::new(),
        root: page.root.to_path_buf(),
        game: page.game.to_owned(),
        profiling: page.profiling,
        chrome,
        urls: BTreeMap::new(),
        navigations: 0,
        loaded: false,
        _profile: lock,
        _server: server,
    };
    let mut session = Session::new(channel, child, Host::Browser(Box::new(browser)), run, settings);
    session.facts = json!({"chrome": session.browser()?.chrome, "flags": flags, "page": address});
    session.attach()?;
    println!("{}", json!({"event": "step", "step": "chrome", "seconds": started.elapsed().as_secs_f64()}));
    session.open(&address)?;
    Ok(session)
}

impl Session {
    fn attach(&mut self) -> Result<(), String> {
        let profile = self.browser()?.profile.clone();
        let url = loop {
            if let Some(url) = self.listening.clone().or_else(|| active_port(&profile)) {
                break url;
            }
            if let Some(exit) = &self.exit {
                return Err(json!({"code": "browser_unavailable", "path": "$", "got": {"exit": exit["error"], "chrome": self.facts}}).to_string());
            }
            self.step_until(Some(Instant::now() + PORT_FILE_POLL));
        };
        let (socket, messages) = socket::connect(&url)?;
        let sender = self.sender.clone();
        thread::spawn(move || {
            for message in messages {
                if let Ok(message) = serde_json::from_str(&message)
                    && sender.send(Event::Devtools(message)).is_err()
                {
                    return;
                }
            }
        });
        self.browser()?.socket = Some(socket);
        Ok(())
    }

    fn call(&mut self, method: &str, params: Value, page: bool) -> Result<Value, String> {
        self.wire += 1;
        let id = self.wire;
        self.browser()?.send(id, method, &params, page)?;
        loop {
            if let Some(answer) = self.answers.remove(&id) {
                return match answer.get("error") {
                    Some(error) => Err(format!("{method}: {}", error["message"])),
                    None => Ok(answer["result"].clone()),
                };
            }
            if let Some(exit) = &self.exit {
                return Err(exit["error"].to_string());
            }
            self.step();
        }
    }

    fn open(&mut self, address: &str) -> Result<(), String> {
        let started = Instant::now();
        let targets = self.call("Target.getTargets", json!({}), false)?;
        let target = targets["targetInfos"]
            .as_array()
            .and_then(|targets| targets.iter().find(|target| target["type"] == "page"))
            .map(|target| target["targetId"].clone())
            .ok_or("Chrome opened no page")?;
        let attached = self.call(
            "Target.attachToTarget",
            json!({"targetId": target, "flatten": true}),
            false,
        )?;
        attached["sessionId"]
            .as_str()
            .ok_or("Chrome attached to the page without a session id")?
            .clone_into(&mut self.browser()?.page);
        for method in ["Runtime.enable", "Page.enable", "Network.enable", "Inspector.enable"] {
            self.call(method, json!({}), true)?;
        }
        for name in BINDINGS {
            self.call("Runtime.addBinding", json!({"name": name}), true)?;
        }
        let version = self.call("Browser.getVersion", json!({}), false)?;
        self.facts["version"] = version["product"].clone();
        println!("{}", json!({"event": "step", "step": "devtools", "seconds": started.elapsed().as_secs_f64()}));
        let started = Instant::now();
        let navigations = self.browser()?.navigations;
        let navigated = self.call("Page.navigate", json!({"url": address}), true)?;
        if let Some(error) = navigated["errorText"].as_str() {
            return Err(format!("cannot load {address}: {error}"));
        }
        let waited = self.wait_page(navigations);
        println!("{}", json!({"event": "step", "step": "page", "seconds": started.elapsed().as_secs_f64()}));
        waited?;
        self.browser()?.loaded = true;
        Ok(())
    }

    fn wait_page(&mut self, navigations: u64) -> Result<(), String> {
        while self.browser()?.navigations == navigations {
            if let Some(exit) = &self.exit {
                return Err(exit["error"].to_string());
            }
            self.step();
        }
        let evaluate = json!({"expression": READY, "awaitPromise": true, "returnByValue": true});
        let ready = loop {
            match self.call("Runtime.evaluate", evaluate.clone(), true) {
                Err(error) if self.exit.is_none() && error.contains("context") => {}
                other => break other?,
            }
        };
        let adapter = self.call(
            "Runtime.evaluate",
            json!({"expression": ADAPTER, "awaitPromise": true, "returnByValue": true}),
            true,
        )?;
        let adapter = adapter["result"]["value"].clone();
        self.facts["adapter"] = adapter.clone();
        let failed = |code: &str, reason: Value, session: &Self| {
            json!({"code": code, "path": "$", "got": {
                "reason": reason,
                "chrome": session.facts,
                "errors": session.since(Source::Errors),
                "console": session.since(Source::Console),
                "failed_fetches": session.since(Source::FailedFetches),
            }})
            .to_string()
        };
        if !adapter.is_object() {
            return Err(failed("webgpu_unavailable", adapter, self));
        }
        match &ready["result"]["value"] {
            Value::Bool(true) => Ok(()),
            _ if ready.get("exceptionDetails").is_some() => {
                Err(failed("page_unavailable", ready["exceptionDetails"].clone(), self))
            }
            status => Err(failed("page_unavailable", status.clone(), self)),
        }
    }

    pub fn rebuild(&mut self) -> Value {
        let (root, game, profiling) = match self.browser() {
            Ok(browser) => (browser.root.clone(), browser.game.clone(), browser.profiling),
            Err(_) => {
                return refusal(
                    "unavailable",
                    json!("the native program cannot be swapped under a running game; close the session and start again"),
                );
            }
        };
        let started = Instant::now();
        let sender = self.sender.clone();
        self.built = None;
        thread::spawn(move || {
            let _ = sender.send(Event::Built(build(&root, &game, profiling)));
        });
        while self.built.is_none() {
            if let Some(exit) = &self.exit {
                return exit.clone();
            }
            self.step();
        }
        if let Some(Err(error)) = self.built.take() {
            return refusal("build_failed", json!(error));
        }
        let reloaded = self.browser().map(|browser| browser.navigations).and_then(|navigations| {
            self.call("Page.reload", json!({"ignoreCache": true}), true)?;
            self.wait_page(navigations)
        });
        match reloaded {
            Ok(()) => accepted(json!({"rebuilt": true, "seconds": started.elapsed().as_secs_f64()})),
            Err(error) => refusal("page_unavailable", json!(error)),
        }
    }

    pub fn devtools(&mut self, message: Value) {
        if let Some(id) = message["id"].as_u64() {
            if self.running.as_ref().is_some_and(|running| running.wire == id) {
                self.complete(evaluated(&message));
            } else if id != 0 {
                self.answers.insert(id, message);
            }
            return;
        }
        let params = &message["params"];
        let Ok(browser) = self.browser() else {
            return;
        };
        match message["method"].as_str().unwrap_or_default() {
            "Network.requestWillBeSent" => {
                if let (Some(id), Some(url)) = (params["requestId"].as_str(), params["request"]["url"].as_str()) {
                    browser.urls.insert(id.to_owned(), url.to_owned());
                }
            }
            "Network.loadingFinished" => {
                browser.urls.remove(params["requestId"].as_str().unwrap_or_default());
            }
            "Network.loadingFailed" => {
                let url = browser.urls.remove(params["requestId"].as_str().unwrap_or_default());
                let entry = json!({"url": url, "error": params["errorText"], "canceled": params["canceled"]});
                self.note(Source::FailedFetches, entry);
            }
            "Network.responseReceived" if params["response"]["status"].as_u64() >= Some(400) => {
                let entry = json!({"url": params["response"]["url"], "status": params["response"]["status"]});
                self.note(Source::FailedFetches, entry);
            }
            "Page.frameNavigated" if params["frame"]["parentId"].is_null() => {
                browser.navigations += 1;
                if browser.loaded {
                    let url = params["frame"]["url"].clone();
                    println!("{}", json!({"event": "page_reloaded", "url": url}));
                    self.note(Source::Console, json!({"level": "driver", "text": "page reloaded", "url": url}));
                }
            }
            "Runtime.consoleAPICalled" => {
                let text: Vec<String> = params["args"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .map(|argument| match &argument["value"] {
                        Value::String(text) => text.clone(),
                        Value::Null => argument["description"].as_str().unwrap_or_default().to_owned(),
                        value => value.to_string(),
                    })
                    .collect();
                self.note(Source::Console, json!({"level": params["type"], "text": text.join(" ")}));
            }
            "Runtime.exceptionThrown" => {
                let details = &params["exceptionDetails"];
                let text = details["exception"]["description"]
                    .as_str()
                    .or(details["text"].as_str())
                    .unwrap_or_default()
                    .to_owned();
                if let Some(report) = panic_in(&text) {
                    self.panic = Some(report);
                }
                let entry = json!({"text": text, "url": details["url"], "line": details["lineNumber"]});
                self.note(Source::Errors, entry);
            }
            "Runtime.bindingCalled" => self.bound(params["name"].as_str(), params["payload"].as_str()),
            method @ ("Inspector.targetCrashed" | "Inspector.detached" | "Target.detachedFromTarget")
                if self.exit.is_none() =>
            {
                self.exit = Some(refusal(
                    "page_crashed",
                    json!({
                        "event": method,
                        "reason": params["reason"],
                        "panic": self.panic,
                        "console": self.since(Source::Console),
                        "errors": self.since(Source::Errors),
                    }),
                ));
                self.fail_all();
            }
            _ => {}
        }
    }

    fn bound(&mut self, name: Option<&str>, payload: Option<&str>) {
        let payload = payload.unwrap_or_default();
        let written = match name {
            Some("__engineRecord") => self.run.record(payload),
            Some("__engineFile") => serde_json::from_str::<Value>(payload)
                .map_err(|error| error.to_string())
                .and_then(|file| {
                    let bytes = base64::engine::general_purpose::STANDARD
                        .decode(file["base64"].as_str().unwrap_or_default())
                        .map_err(|error| error.to_string())?;
                    let name = file["name"].as_str().unwrap_or_default();
                    self.run.file(file["run"].as_str(), name, &bytes)
                }),
            _ => return,
        };
        if let Err(error) = written {
            self.note(Source::Errors, json!({"driver": name, "text": error}));
        }
    }
}

fn evaluated(message: &Value) -> Value {
    if let Some(error) = message.get("error") {
        return refusal("page_unreachable", error["message"].clone());
    }
    let result = &message["result"];
    if let Some(details) = result.get("exceptionDetails") {
        let text = details["exception"]["description"]
            .as_str()
            .or(details["text"].as_str())
            .unwrap_or_default();
        let thrown = text.strip_prefix("Error: ").unwrap_or(text);
        let code = ["runtime_panicked", "runtime_failed"]
            .into_iter()
            .find(|code| thrown.starts_with(code))
            .unwrap_or("page_threw");
        return refusal(code, json!(thrown));
    }
    match result["result"]["value"].as_str().map(serde_json::from_str::<Value>) {
        Some(Ok(reply)) if reply["ok"].is_boolean() => reply,
        _ => refusal("page_replied", result["result"].clone()),
    }
}

fn active_port(profile: &Path) -> Option<String> {
    let text = fs::read_to_string(profile.join("DevToolsActivePort")).ok()?;
    let mut lines = text.lines();
    let port: u16 = lines.next()?.trim().parse().ok()?;
    let path = lines.next()?.trim();
    path.starts_with('/').then(|| format!("ws://127.0.0.1:{port}{path}"))
}

fn build(root: &Path, game: &str, profiling: bool) -> Result<(), String> {
    if profiling {
        pipeline::build_profiling_wasm(root)?;
    } else {
        pipeline::build_dev(root, Some(game))?;
    }
    pipeline::install_web_dependencies(root)
}

fn find(given: Option<PathBuf>) -> Result<PathBuf, String> {
    if let Some(path) = given {
        return if path.is_file() {
            Ok(path)
        } else {
            Err(format!("--chrome {} is not a file", path.display()))
        };
    }
    let mut candidates = Vec::new();
    if cfg!(windows) {
        for variable in ["PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA"] {
            if let Some(base) = env::var_os(variable) {
                candidates.push(PathBuf::from(base).join("Google/Chrome/Application/chrome.exe"));
            }
        }
    } else if cfg!(target_os = "macos") {
        candidates.push("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome".into());
    } else {
        for folder in env::split_paths(&env::var_os("PATH").unwrap_or_default()) {
            for name in ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser"] {
                candidates.push(folder.join(name));
            }
        }
    }
    candidates.iter().find(|path| path.is_file()).cloned().ok_or_else(|| {
        let looked: Vec<_> = candidates.iter().map(|path| path.display().to_string()).collect();
        format!("no Chrome found at {}; pass --chrome PATH", looked.join(", "))
    })
}

fn claim_profile() -> Result<(PathBuf, File), String> {
    let root = env::temp_dir().join("engine-chrome");
    fs::create_dir_all(&root).map_err(|error| format!("cannot create {}: {error}", root.display()))?;
    let mut index = 0_u32;
    loop {
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(root.join(format!("{index}.lock")))
            .map_err(|error| format!("cannot open a Chrome profile lock in {}: {error}", root.display()))?;
        if lock.try_lock().is_ok() {
            let folder = root.join(index.to_string());
            fs::create_dir_all(&folder).map_err(|error| error.to_string())?;
            clear_profile(&folder);
            return Ok((folder, lock));
        }
        index += 1;
    }
}

fn clear_profile(folder: &Path) {
    let Ok(entries) = fs::read_dir(folder) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let path = entry.path();
        if name == "Default" && path.is_dir() {
            clear_profile(&path);
        } else if !SHADER_CACHES.iter().any(|cache| name == *cache) {
            let _ = if path.is_dir() {
                fs::remove_dir_all(&path)
            } else {
                fs::remove_file(&path)
            };
        }
    }
}
