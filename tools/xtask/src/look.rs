use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

use serde_json::{Value, json};

use crate::Failure;
use crate::measure::read_json;
use crate::pipeline::QUICK;
use crate::process::Project;
use crate::serve::Server;

const USAGE: &str = "usage: cargo xtask look PROJECT --limit-ms MS [--browser] [--out DIR]";
const PLAN: &str = "look.json";
const SHEET: &str = "sheet.png";
const SHEET_WIDTH: usize = 2048;
const GAP: usize = 4;
const BACKGROUND: u8 = 24;

struct Options {
    project: String,
    limit_ms: u64,
    browser: bool,
    out: Option<PathBuf>,
}

fn options(arguments: &[String]) -> Result<Options, String> {
    let (mut project, mut limit_ms, mut browser, mut out) = (None, None, false, None);
    let mut given = arguments.iter();
    while let Some(argument) = given.next() {
        match argument.as_str() {
            "--browser" => browser = true,
            "--limit-ms" => {
                limit_ms = Some(
                    given
                        .next()
                        .and_then(|limit| limit.parse::<u64>().ok())
                        .filter(|limit| *limit > 0)
                        .ok_or(USAGE)?,
                );
            }
            "--out" => out = Some(PathBuf::from(given.next().ok_or(USAGE)?)),
            flag if flag.starts_with("--") => return Err(format!("{flag}\n{USAGE}")),
            _ if project.is_none() => project = Some(argument.clone()),
            _ => return Err(format!("{argument}\n{USAGE}")),
        }
    }
    Ok(Options {
        project: project.ok_or(USAGE)?,
        limit_ms: limit_ms.ok_or(USAGE)?,
        browser,
        out,
    })
}

pub fn look(root: &Path, arguments: &[String]) -> Result<(), Failure> {
    let options = options(arguments)?;
    let project = Project::read(root, &options.project)?;
    let host = if options.browser { "browser" } else { "native" };
    let out = options.out.clone().unwrap_or_else(|| {
        root.join("target")
            .join("look")
            .join(format!("{}-{host}", project.package))
    });
    clear(&out)?;
    let plan = root.join(&options.project).join(PLAN);
    let source = if plan.is_file() {
        read_json(&plan)?["source"].as_str().map(str::to_owned)
    } else {
        None
    };
    let mut command = Command::new("node");
    command
        .args(["scripts/agent.mjs", "look", "--limit-ms"])
        .arg(options.limit_ms.to_string())
        .arg("--out")
        .arg(&out)
        .current_dir(root.join("web"));
    if plan.is_file() {
        command.arg("--plan").arg(&plan);
    }
    let server = if options.browser {
        let page = project
            .client()
            .and_then(|client| client.page.clone())
            .ok_or_else(|| format!("{}: page", project.package))?;
        let query = match &source {
            Some(source) => {
                let served = source.strip_prefix("web/").ok_or_else(|| {
                    format!("{}: source {source} is outside web/", plan.display())
                })?;
                format!("?source=/{served}")
            }
            None => String::new(),
        };
        let server = Server::page(
            root,
            &out.join("server.log"),
            Duration::from_millis(options.limit_ms),
        )?;
        command.args(["--url", &format!("{}/{page}{query}", server.url)]);
        Some(server)
    } else {
        let program = crate::smoke::build(root, &project, QUICK)?;
        command.args(["--native", &program]);
        if let Some(source) = &source {
            command.args(["--source", source]);
        }
        None
    };
    let mut child = command
        .stdout(Stdio::piped())
        .spawn()
        .map_err(|error| format!("cannot run node: {error}"))?;
    let (mut views, mut problems) = (Vec::new(), 0);
    if let Some(output) = child.stdout.take() {
        for line in BufReader::new(output).lines() {
            let line = line.map_err(|error| format!("scripts/agent.mjs look: {error}"))?;
            println!("{line}");
            let record = serde_json::from_str::<Value>(&line).unwrap_or(Value::Null);
            match record["event"].as_str() {
                Some("view") => {
                    problems += usize::from(!record["unsettled_after_s"].is_null());
                    views.push(record);
                }
                Some(_) | None => problems += 1,
            }
        }
    }
    let status = child
        .wait()
        .map_err(|error| format!("scripts/agent.mjs look: {error}"))?;
    drop(server);
    if !status.success() && problems == 0 {
        problems += 1;
    }
    if !views.is_empty() {
        println!("{}", sheet(&views, &out.join(SHEET))?);
    }
    if problems == 0 {
        return Ok(());
    }
    Err(Failure::found(format!(
        "look: {problems} errors or unsettled views; see the lines above"
    )))
}

fn clear(out: &Path) -> Result<(), String> {
    fs::create_dir_all(out).map_err(|error| format!("cannot create {}: {error}", out.display()))?;
    let entries =
        fs::read_dir(out).map_err(|error| format!("cannot read {}: {error}", out.display()))?;
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let shot = name.ends_with(".png") && name.starts_with(|first: char| first.is_ascii_digit());
        if shot || name == SHEET {
            fs::remove_file(entry.path())
                .map_err(|error| format!("cannot remove {}: {error}", entry.path().display()))?;
        }
    }
    Ok(())
}

struct Image {
    width: usize,
    height: usize,
    channels: usize,
    pixels: Vec<u8>,
}

fn decode(path: &Path) -> Result<Image, String> {
    let file =
        fs::File::open(path).map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    let mut decoder = png::Decoder::new(BufReader::new(file));
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder
        .read_info()
        .map_err(|error| format!("{}: {error}", path.display()))?;
    let mut pixels = vec![0; reader.output_buffer_size()];
    let info = reader
        .next_frame(&mut pixels)
        .map_err(|error| format!("{}: {error}", path.display()))?;
    pixels.truncate(info.buffer_size());
    let channels = match info.color_type {
        png::ColorType::Rgba => 4,
        png::ColorType::Rgb => 3,
        other => return Err(format!("{} is {other:?}, not RGB or RGBA", path.display())),
    };
    Ok(Image {
        width: info.width as usize,
        height: info.height as usize,
        channels,
        pixels,
    })
}

fn sheet(views: &[Value], path: &Path) -> Result<Value, String> {
    let shots: Vec<(&Value, Image)> = views
        .iter()
        .filter_map(|view| Some((view, view["png"].as_str()?)))
        .map(|(view, png)| decode(Path::new(png)).map(|image| (view, image)))
        .collect::<Result<_, _>>()?;
    let columns = (1..=shots.len())
        .find(|columns| columns * columns >= shots.len())
        .unwrap_or(1);
    let rows = shots.len().div_ceil(columns);
    let widest = shots
        .iter()
        .map(|(_, image)| image.width)
        .max()
        .unwrap_or(1);
    let tallest = shots
        .iter()
        .map(|(_, image)| image.height)
        .max()
        .unwrap_or(1);
    let factor = (columns * widest).div_ceil(SHEET_WIDTH).max(1);
    let (cell_width, cell_height) = (widest.div_ceil(factor), tallest.div_ceil(factor));
    let width = columns * (cell_width + GAP) + GAP;
    let height = rows * (cell_height + GAP) + GAP;
    let mut pixels = vec![BACKGROUND; width * height * 3];
    let mut cells = Vec::new();
    for (index, (view, image)) in shots.iter().enumerate() {
        let (row, column) = (index / columns, index % columns);
        let (left, top) = (
            GAP + column * (cell_width + GAP),
            GAP + row * (cell_height + GAP),
        );
        for y in 0..image.height.div_ceil(factor) {
            for x in 0..image.width.div_ceil(factor) {
                let mut sum = [0usize; 3];
                let mut count = 0;
                for from_y in y * factor..((y + 1) * factor).min(image.height) {
                    for from_x in x * factor..((x + 1) * factor).min(image.width) {
                        let at = (from_y * image.width + from_x) * image.channels;
                        for (total, value) in sum.iter_mut().zip(&image.pixels[at..at + 3]) {
                            *total += usize::from(*value);
                        }
                        count += 1;
                    }
                }
                let at = ((top + y) * width + left + x) * 3;
                for (pixel, total) in pixels[at..at + 3].iter_mut().zip(sum) {
                    *pixel = (total / count) as u8;
                }
            }
        }
        cells.push(json!({"row": row, "column": column, "view": view["view"]}));
    }
    let file = fs::File::create(path)
        .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), width as u32, height as u32);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    encoder
        .write_header()
        .and_then(|mut writer| writer.write_image_data(&pixels))
        .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
    Ok(json!({
        "event": "legend",
        "sheet": path.display().to_string(),
        "columns": columns,
        "rows": rows,
        "cells": cells,
    }))
}
