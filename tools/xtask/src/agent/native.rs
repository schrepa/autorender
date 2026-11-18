use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::Sender;
use std::thread;

use serde_json::Value;

use super::session::Event;

pub fn spawn(
    root: &Path,
    program: &str,
    arguments: &[String],
    sender: &Sender<Event>,
) -> Result<(Child, ChildStdin), String> {
    let mut child = Command::new(program)
        .args(arguments)
        .current_dir(root)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("cannot start {program}: {error}"))?;
    let stdin = child.stdin.take().ok_or("the game has no stdin")?;
    let stdout = child.stdout.take().ok_or("the game has no stdout")?;
    watch(&mut child, Some(stdout), sender);
    Ok((child, stdin))
}

pub fn watch(child: &mut Child, stdout: Option<impl Read + Send + 'static>, sender: &Sender<Event>) {
    let errors = child.stderr.take().map(|stderr| {
        let sender = sender.clone();
        thread::spawn(move || lines(stderr, |line| sender.send(Event::Stderr(line)).is_ok()))
    });
    let sender = sender.clone();
    thread::spawn(move || {
        if let Some(stdout) = stdout {
            lines(stdout, |line| sender.send(Event::Stdout(line)).is_ok());
        }
        if let Some(errors) = errors {
            let _ = errors.join();
        }
        let _ = sender.send(Event::Closed);
    });
}

pub fn lines(reader: impl Read, mut each: impl FnMut(String) -> bool) {
    let mut reader = BufReader::new(reader);
    let mut line = Vec::new();
    loop {
        line.clear();
        match reader.read_until(b'\n', &mut line) {
            Ok(0) | Err(_) => return,
            Ok(_) => {
                let text = String::from_utf8_lossy(&line);
                if !each(text.trim_end_matches(['\r', '\n']).to_owned()) {
                    return;
                }
            }
        }
    }
}

pub fn send(stdin: Option<&mut ChildStdin>, wire: u64, body: &Value) -> Result<(), String> {
    let stdin = stdin.ok_or("the game's input is closed")?;
    let mut line = body.clone();
    line["id"] = wire.into();
    writeln!(stdin, "{line}").map_err(|error| format!("cannot write to the game: {error}"))
}
