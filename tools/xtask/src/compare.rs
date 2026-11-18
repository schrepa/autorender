use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::{BufReader, BufWriter},
    path::{Path, PathBuf},
};

use engine_command_line::{Command, CommandOption};
use serde_json::{Value, json};

use crate::{
    Failure,
    runs::{self, Run},
};

enum Span {
    Whole,
    To(String),
    Between(String, String),
}

struct Side {
    text: String,
    runs: Vec<Run>,
    span: Option<Span>,
}

struct Options {
    only: Vec<String>,
    tolerance: u8,
    output: PathBuf,
}

enum Leaf {
    Number(f64),
    Unavailable(String),
}

#[derive(Default)]
struct Column {
    kind: String,
    unit: String,
    numbers: Vec<f64>,
    reasons: BTreeSet<String>,
}

#[derive(Default)]
struct Notes {
    only_a: BTreeSet<String>,
    only_b: BTreeSet<String>,
    refused: BTreeSet<String>,
    undeclared: BTreeSet<String>,
    skipped: BTreeSet<String>,
}

type Segment<'a> = (&'a Run, Option<usize>, usize);

pub fn command() -> Command {
    let side = "RUN[,RUN...][@MARK|@MARK..MARK]";
    Command::new(
        "compare",
        [
            CommandOption::taking("A", side).required(),
            CommandOption::taking("B", side).required(),
            CommandOption::taking("--only", "NAME..."),
            CommandOption::taking("--tolerance", "0..255").defaults_to(0),
        ],
    )
}

pub fn compare(root: &Path, arguments: &[String]) -> Result<(), Failure> {
    let (a, b, options) = parse(root, arguments)?;
    println!("a: {}", a.text);
    println!("b: {}", b.text);
    for key in ["build", "pack", "picture"] {
        flag(key, &a, &b);
    }
    let mut notes = Notes::default();
    match (&a.span, &b.span) {
        (None, None) => by_mark(&a, &b, &options, &mut notes),
        (span_a, span_b) => {
            let segments_a = segments(&a, span_a.as_ref().unwrap_or(&Span::Whole))?;
            let segments_b = segments(&b, span_b.as_ref().unwrap_or(&Span::Whole))?;
            section("span", &segments_a, &segments_b, &options, &mut notes);
        }
    }
    print_notes(&notes);
    Ok(())
}

fn parse(root: &Path, arguments: &[String]) -> Result<(Side, Side, Options), Failure> {
    let mut sides = Vec::new();
    let mut only = Vec::new();
    let mut tolerance = 0;
    let mut rest = arguments.iter();
    while let Some(argument) = rest.next() {
        match argument.as_str() {
            "--only" => only.push(value(rest.next(), "--only", "a dotted name")?.to_owned()),
            "--tolerance" => {
                let text = value(rest.next(), "--tolerance", "0..255")?;
                tolerance = text
                    .parse()
                    .map_err(|_| invalid("--tolerance", text, "0..255"))?;
            }
            _ => sides.push(side(root, argument)?),
        }
    }
    let mut sides = sides.into_iter();
    let (Some(a), Some(b), None) = (sides.next(), sides.next(), sides.next()) else {
        return Err(invalid(
            "compare",
            &arguments.join(" "),
            "<a> <b> [--only <name>]... [--tolerance <0..255>], a side = run[,run...][@mark|@mark..mark]",
        ));
    };
    let output = root.join("artifacts").join("compare").join(format!(
        "{}_{}",
        plain(&a.text),
        plain(&b.text)
    ));
    Ok((
        a,
        b,
        Options {
            only,
            tolerance,
            output,
        },
    ))
}

fn value<'a>(next: Option<&'a String>, path: &str, expected: &str) -> Result<&'a str, Failure> {
    next.map(String::as_str)
        .ok_or_else(|| invalid(path, "nothing", expected))
}

fn invalid(path: &str, got: &str, expected: &str) -> Failure {
    json!({
        "code": "invalid_arguments",
        "path": path,
        "got": got,
        "expected": expected,
    })
    .to_string()
    .into()
}

fn side(root: &Path, text: &str) -> Result<Side, Failure> {
    let (names, span) = match text.split_once('@') {
        None => (text, None),
        Some((names, marks)) => match marks.split_once("..") {
            None => (names, Some(Span::To(marks.to_owned()))),
            Some((from, to)) => (names, Some(Span::Between(from.to_owned(), to.to_owned()))),
        },
    };
    let runs = names
        .split(',')
        .map(|name| runs::read(root, name))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Side {
        text: text.to_owned(),
        runs,
        span,
    })
}

fn plain(text: &str) -> String {
    text.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn flag(key: &str, a: &Side, b: &Side) {
    let headers: BTreeMap<&str, &Value> = a
        .runs
        .iter()
        .chain(&b.runs)
        .map(|run| (run.name.as_str(), &run.header[key]))
        .collect();
    let mut known = BTreeMap::new();
    for (run, value) in headers {
        match value["unavailable"].as_str() {
            Some(reason) => println!("header {key}: unavailable in {run}: {reason}"),
            None => {
                known.insert(run, shown(value));
            }
        }
    }
    if known.values().collect::<BTreeSet<_>>().len() > 1 {
        let listed: Vec<String> = known
            .iter()
            .map(|(run, value)| format!("{run} {value}"))
            .collect();
        println!("flag: {key}s differ: {}", listed.join(", "));
    }
}

fn shown(value: &Value) -> String {
    match value {
        Value::Null => "unknown".to_owned(),
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

fn marks(run: &Run) -> Vec<(String, usize)> {
    let mut seen: BTreeMap<&str, usize> = BTreeMap::new();
    let mut labels = Vec::new();
    for (at, entry) in run.entries.iter().enumerate() {
        let Some(name) = entry["mark"].as_str() else {
            continue;
        };
        let count = seen.entry(name).or_default();
        *count += 1;
        let label = if *count == 1 {
            name.to_owned()
        } else {
            format!("{name}#{count}")
        };
        labels.push((label, at));
    }
    labels
}

fn segments<'a>(side: &'a Side, span: &Span) -> Result<Vec<Segment<'a>>, Failure> {
    side.runs
        .iter()
        .map(|run| {
            let marks = marks(run);
            let find = |label: &str| {
                marks
                    .iter()
                    .find(|(name, _)| name == label)
                    .map(|(_, at)| *at)
                    .ok_or_else(|| Failure::found(format!("run {} has no mark {label}", run.name)))
            };
            match span {
                Span::Whole => run
                    .entries
                    .iter()
                    .enumerate()
                    .max_by_key(|(_, entry)| entry["frame"].as_u64())
                    .map(|(end, _)| (run, None, end))
                    .ok_or_else(|| Failure::found(format!("run {} has no entries", run.name))),
                Span::To(to) => Ok((run, None, find(to)?)),
                Span::Between(from, to) => Ok((run, Some(find(from)?), find(to)?)),
            }
        })
        .collect()
}

fn by_mark(a: &Side, b: &Side, options: &Options, notes: &mut Notes) {
    let labels = |side: &Side| -> Vec<String> {
        side.runs
            .first()
            .map(|run| marks(run).into_iter().map(|(label, _)| label).collect())
            .unwrap_or_default()
    };
    let (labels_a, labels_b) = (labels(a), labels(b));
    for label in &labels_a {
        if !labels_b.contains(label) {
            notes.skipped.insert(format!("mark {label} only in a"));
            continue;
        }
        let segments_a = mark_segments(a, label, notes);
        let segments_b = mark_segments(b, label, notes);
        section(
            &format!("mark {label}"),
            &segments_a,
            &segments_b,
            options,
            notes,
        );
    }
    for label in labels_b.iter().filter(|label| !labels_a.contains(label)) {
        notes.skipped.insert(format!("mark {label} only in b"));
    }
}

fn mark_segments<'a>(side: &'a Side, label: &str, notes: &mut Notes) -> Vec<Segment<'a>> {
    let mut segments = Vec::new();
    for run in &side.runs {
        let marks = marks(run);
        match marks.iter().position(|(name, _)| name == label) {
            Some(at) => {
                let start = at.checked_sub(1).map(|before| marks[before].1);
                segments.push((run, start, marks[at].1));
            }
            None => {
                notes
                    .skipped
                    .insert(format!("run {} has no mark {label}", run.name));
            }
        }
    }
    segments
}

fn section(title: &str, a: &[Segment], b: &[Segment], options: &Options, notes: &mut Notes) {
    let (span_a, frames_a) = covered(a);
    let (span_b, frames_b) = covered(b);
    println!();
    println!("== {title}");
    println!("a: {span_a}");
    println!("b: {span_b}");
    if frames_a != frames_b {
        let launch = a.iter().chain(b).any(|(_, start, _)| start.is_none());
        println!(
            "flag: the sides ran different numbers of frames{}; a count that grows each frame differs for that reason, not for the engine",
            if launch {
                " (a span from launch holds every frame before the mark, loading and free-running included)"
            } else {
                ""
            }
        );
    }
    let columns_a = gather(a, "a", options, notes);
    let columns_b = gather(b, "b", options, notes);
    let mut counts = Vec::new();
    let mut times = Vec::new();
    let mut zeros = 0;
    for (name, column_a) in &columns_a {
        let Some(column_b) = columns_b.get(name) else {
            notes.only_a.insert(name.clone());
            continue;
        };
        if column_a.unit != column_b.unit {
            notes.refused.insert(format!(
                "{name}: unit {} in a, {} in b",
                column_a.unit, column_b.unit
            ));
            continue;
        }
        if options.only.is_empty() && zero(column_a) && zero(column_b) {
            zeros += 1;
            continue;
        }
        let row = [
            name.clone(),
            column_a.unit.clone(),
            cell(column_a),
            cell(column_b),
            change(column_a, column_b),
        ];
        match column_a.kind.as_str() {
            "time" => times.push(row),
            _ => counts.push(row),
        }
    }
    for name in columns_b
        .keys()
        .filter(|name| !columns_a.contains_key(*name))
    {
        notes.only_b.insert(name.clone());
    }
    table("counts", &counts);
    table("times", &times);
    if zeros > 0 {
        println!("{zeros} facts are 0 on both sides and left out; --only <name> shows them");
    }
    pictures(title, a, b, options);
}

fn zero(column: &Column) -> bool {
    column.reasons.is_empty()
        && !column.numbers.is_empty()
        && column.numbers.iter().all(|number| *number == 0.0)
}

fn reach(run: &Run, at: Option<usize>, key: &str) -> u64 {
    at.and_then(|at| run.entries[at][key].as_u64()).unwrap_or(0)
}

fn covered(segments: &[Segment]) -> (String, BTreeSet<u64>) {
    let lengths: BTreeSet<(u64, u64)> = segments
        .iter()
        .map(|(run, start, end)| {
            let length =
                |key: &str| reach(run, Some(*end), key).saturating_sub(reach(run, *start, key));
            (length("frame"), length("tick"))
        })
        .collect();
    let text = segments
        .first()
        .map_or_else(String::new, |(run, start, end)| {
            let point = |at: Option<usize>| {
                at.map_or("launch".to_owned(), |at| {
                    format!("frame {}", reach(run, Some(at), "frame"))
                })
            };
            let lengths: Vec<String> = lengths
                .iter()
                .map(|(frames, ticks)| format!("{frames} frames, {ticks} ticks"))
                .collect();
            format!(
                "{} to {} ({})",
                point(*start),
                point(Some(*end)),
                lengths.join("; ")
            )
        });
    (text, lengths.iter().map(|(frames, _)| *frames).collect())
}

fn gather(
    segments: &[Segment],
    side: &str,
    options: &Options,
    notes: &mut Notes,
) -> BTreeMap<String, Column> {
    let mut columns: BTreeMap<String, Column> = BTreeMap::new();
    let mut refused = BTreeSet::new();
    for (run, start, end) in segments {
        for (name, (kind, unit, leaf)) in span_values(run, *start, *end, options, notes) {
            let column = columns.entry(name.clone()).or_insert_with(|| Column {
                kind: kind.to_owned(),
                unit: unit.to_owned(),
                ..Column::default()
            });
            if column.unit != unit {
                notes.refused.insert(format!(
                    "{name}: unit {} and {unit} within {side} (run {})",
                    column.unit, run.name
                ));
                refused.insert(name);
                continue;
            }
            match leaf {
                Leaf::Number(number) => column.numbers.push(number),
                Leaf::Unavailable(reason) => {
                    column.reasons.insert(reason);
                }
            }
        }
    }
    columns.retain(|name, _| !refused.contains(name));
    for column in columns.values_mut() {
        column.numbers.sort_by(f64::total_cmp);
    }
    columns
}

fn span_values<'a>(
    run: &'a Run,
    start: Option<usize>,
    end: usize,
    options: &Options,
    notes: &mut Notes,
) -> BTreeMap<String, (&'a str, &'a str, Leaf)> {
    let declared: BTreeMap<&str, (&str, &str)> = run.header["facts"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|fact| {
            let name = fact["name"].as_str()?;
            let kind = fact["kind"].as_str().unwrap_or_default();
            Some((name, (kind, fact["unit"].as_str().unwrap_or_default())))
        })
        .collect();
    let mut ends = BTreeMap::new();
    leaves("", &run.entries[end]["values"], &mut ends);
    let mut starts = BTreeMap::new();
    if let Some(start) = start {
        leaves("", &run.entries[start]["values"], &mut starts);
    }
    let mut facts = BTreeMap::new();
    for (name, leaf) in ends {
        if !options.only.is_empty()
            && !options
                .only
                .iter()
                .any(|only| name == *only || name.starts_with(&format!("{only}.")))
        {
            continue;
        }
        let Some((kind, unit)) = declaration(&declared, &name) else {
            notes.undeclared.insert(name);
            continue;
        };
        if !matches!(kind, "count" | "bytes" | "time") {
            continue;
        }
        let leaf = match (&leaf, start.map(|_| starts.get(&name))) {
            (Leaf::Unavailable(reason), _) | (_, Some(Some(Leaf::Unavailable(reason)))) => {
                Leaf::Unavailable(reason.clone())
            }
            (Leaf::Number(_), Some(None)) => {
                Leaf::Unavailable("not_recorded_at_the_start".to_owned())
            }
            (Leaf::Number(to), Some(Some(Leaf::Number(from)))) => Leaf::Number(to - from),
            (Leaf::Number(to), None) => Leaf::Number(*to),
        };
        facts.insert(name, (kind, unit, leaf));
    }
    facts
}

fn leaves(prefix: &str, value: &Value, out: &mut BTreeMap<String, Leaf>) {
    let joined = |key: &str| {
        if prefix.is_empty() {
            key.to_owned()
        } else {
            format!("{prefix}.{key}")
        }
    };
    match value {
        Value::Number(number) => {
            if let Some(number) = number.as_f64() {
                out.insert(prefix.to_owned(), Leaf::Number(number));
            }
        }
        Value::Object(fields) => {
            if let (1, Some(reason)) = (
                fields.len(),
                fields.get("unavailable").and_then(Value::as_str),
            ) {
                out.insert(prefix.to_owned(), Leaf::Unavailable(reason.to_owned()));
                return;
            }
            for (key, field) in fields {
                leaves(&joined(key), field, out);
            }
        }
        Value::Array(items) => {
            for (at, item) in items.iter().enumerate() {
                let key = match item["name"].as_str() {
                    Some(name) => name.to_owned(),
                    None => item
                        .as_object()
                        .map(|fields| {
                            fields
                                .values()
                                .filter_map(Value::as_str)
                                .collect::<Vec<_>>()
                        })
                        .filter(|names| !names.is_empty())
                        .map_or_else(|| at.to_string(), |names| names.join("/")),
                };
                leaves(&joined(&key), item, out);
            }
        }
        _ => {}
    }
}

fn declaration<'a>(
    declared: &BTreeMap<&str, (&'a str, &'a str)>,
    name: &str,
) -> Option<(&'a str, &'a str)> {
    let mut path = name;
    loop {
        if let Some(found) = declared.get(path) {
            return Some(*found);
        }
        path = &path[..path.rfind('.')?];
    }
}

fn cell(column: &Column) -> String {
    if !column.reasons.is_empty() {
        let reasons: Vec<&str> = column.reasons.iter().map(String::as_str).collect();
        return format!("unavailable: {}", reasons.join("; "));
    }
    let numbers = &column.numbers;
    match (numbers.as_slice(), median(numbers)) {
        ([first, .., last], Some(median)) if numbers.len() > 1 => format!(
            "{} ({}..{}, n={})",
            number(median),
            number(*first),
            number(*last),
            numbers.len()
        ),
        (_, median) => median.map_or("none".to_owned(), number),
    }
}

fn change(a: &Column, b: &Column) -> String {
    let (Some(from), Some(to)) = (median(&a.numbers), median(&b.numbers)) else {
        return String::new();
    };
    if !a.reasons.is_empty() || !b.reasons.is_empty() {
        return String::new();
    }
    let change = to - from;
    if change > 0.0 {
        format!("+{}", number(change))
    } else {
        number(change)
    }
}

fn median(sorted: &[f64]) -> Option<f64> {
    let middle = sorted.len() / 2;
    match sorted.len() % 2 {
        _ if sorted.is_empty() => None,
        1 => Some(sorted[middle]),
        _ => Some((sorted[middle - 1] + sorted[middle]) / 2.0),
    }
}

fn number(value: f64) -> String {
    let text = format!("{value:.3}");
    text.trim_end_matches('0').trim_end_matches('.').to_owned()
}

fn table(title: &str, rows: &[[String; 5]]) {
    if rows.is_empty() {
        return;
    }
    let header = ["fact", "unit", "a", "b", "b-a"].map(str::to_owned);
    let mut widths = [0; 5];
    for row in std::iter::once(&header).chain(rows) {
        for (width, text) in widths.iter_mut().zip(row) {
            *width = (*width).max(text.len());
        }
    }
    println!("{title}");
    for row in std::iter::once(&header).chain(rows) {
        let cells: Vec<String> = row
            .iter()
            .zip(widths)
            .map(|(text, width)| format!("{text:<width$}"))
            .collect();
        println!("  {}", cells.join("  ").trim_end());
    }
}

fn pictures(title: &str, a: &[Segment], b: &[Segment], options: &Options) {
    let (pictures_a, pictures_b) = (taken("a", a), taken("b", b));
    for (name, pictures) in [("a", &pictures_a), ("b", &pictures_b)] {
        if let Some(((_, first), rest)) = pictures.split_first() {
            let differ: Vec<&str> = rest
                .iter()
                .filter(|(_, other)| other["sha256"] != first["sha256"])
                .map(|(run, _)| run.name.as_str())
                .collect();
            if !differ.is_empty() {
                println!(
                    "picture: within {name}, runs whose picture differs from the first: {}",
                    differ.join(", ")
                );
            }
        }
    }
    let (Some((run_a, first_a)), Some((run_b, first_b))) = (pictures_a.first(), pictures_b.first())
    else {
        println!(
            "picture: a has {}, b has {}",
            pictures_a.len(),
            pictures_b.len()
        );
        return;
    };
    if first_a["sha256"].is_string() && first_a["sha256"] == first_b["sha256"] {
        println!("picture: same (sha256 {})", shown(&first_a["sha256"]));
        return;
    }
    let file = |run: &Run, picture: &Value| {
        run.folder
            .join(picture["path"].as_str().unwrap_or_default())
    };
    let target = options.output.join(format!("{}.png", plain(title)));
    match difference(
        &file(run_a, first_a),
        &file(run_b, first_b),
        options.tolerance,
        &target,
    ) {
        Ok(line) => println!("{line}"),
        Err((side, reason)) => println!("picture: unavailable in {side}: {reason}"),
    }
}

fn taken<'a>(side: &str, segments: &[Segment<'a>]) -> Vec<(&'a Run, &'a Value)> {
    let mut pictures = Vec::new();
    for (run, _, end) in segments {
        let picture = &run.entries[*end]["picture"];
        match picture["unavailable"].as_str() {
            Some(reason) => println!("picture: unavailable in {side} ({}): {reason}", run.name),
            None if !picture.is_null() => pictures.push((*run, picture)),
            None => {}
        }
    }
    pictures
}

fn difference(
    a: &Path,
    b: &Path,
    tolerance: u8,
    target: &Path,
) -> Result<String, (&'static str, &'static str)> {
    let (width, height, pixels_a) = decode(a).map_err(|reason| ("a", reason))?;
    let (width_b, height_b, pixels_b) = decode(b).map_err(|reason| ("b", reason))?;
    if (width, height) != (width_b, height_b) {
        return Ok(format!(
            "flag: picture sizes differ: {width}x{height}, {width_b}x{height_b}"
        ));
    }
    let mut image = Vec::with_capacity(pixels_a.len());
    let mut changed = 0_u64;
    let mut low = (u32::MAX, u32::MAX);
    let mut high = (0, 0);
    for (at, (pixel_a, pixel_b)) in pixels_a
        .as_chunks::<4>()
        .0
        .iter()
        .zip(pixels_b.as_chunks::<4>().0)
        .enumerate()
    {
        let most = pixel_a
            .iter()
            .zip(pixel_b)
            .map(|(x, y)| x.abs_diff(*y))
            .max()
            .unwrap_or(0);
        if most > tolerance {
            changed += 1;
            let (x, y) = ((at % width as usize) as u32, (at / width as usize) as u32);
            low = (low.0.min(x), low.1.min(y));
            high = (high.0.max(x), high.1.max(y));
            image.extend([255, 0, 0, 255]);
        } else {
            let gray =
                ((u32::from(pixel_a[0]) + u32::from(pixel_a[1]) + u32::from(pixel_a[2])) / 9) as u8;
            image.extend([gray, gray, gray, 255]);
        }
    }
    let total = u64::from(width) * u64::from(height);
    if changed == 0 {
        return Ok(format!(
            "picture: hashes differ, no pixel differs by more than {tolerance}"
        ));
    }
    encode(target, width, height, &image).map_err(|reason| ("the difference", reason))?;
    Ok(format!(
        "picture: {:.4}% of pixels changed ({changed} of {total}) beyond {tolerance}, box x {}..{} y {}..{}, difference {}",
        changed as f64 * 100.0 / total as f64,
        low.0,
        high.0,
        low.1,
        high.1,
        target.display()
    ))
}

fn decode(path: &Path) -> Result<(u32, u32, Vec<u8>), &'static str> {
    let file = File::open(path).map_err(|_| "picture_file_missing")?;
    let mut decoder = png::Decoder::new(BufReader::new(file));
    decoder.set_transformations(png::Transformations::normalize_to_color8());
    let mut reader = decoder.read_info().map_err(|_| "picture_not_png")?;
    let mut buffer = vec![0; reader.output_buffer_size()];
    let info = reader
        .next_frame(&mut buffer)
        .map_err(|_| "picture_not_png")?;
    buffer.truncate(info.buffer_size());
    let pixels = match reader.output_color_type().0 {
        png::ColorType::Rgba => buffer,
        png::ColorType::Rgb => buffer
            .as_chunks::<3>()
            .0
            .iter()
            .flat_map(|[red, green, blue]| [*red, *green, *blue, 255])
            .collect(),
        _ => return Err("picture_not_rgb_or_rgba"),
    };
    Ok((info.width, info.height, pixels))
}

fn encode(path: &Path, width: u32, height: u32, pixels: &[u8]) -> Result<(), &'static str> {
    let unwritten = |_| "difference_not_written";
    fs::create_dir_all(path.parent().unwrap_or(path)).map_err(|_| "difference_not_written")?;
    let file = File::create(path).map_err(|_| "difference_not_written")?;
    let mut encoder = png::Encoder::new(BufWriter::new(file), width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().map_err(unwritten)?;
    writer.write_image_data(pixels).map_err(unwritten)?;
    writer.finish().map_err(unwritten)
}

fn print_notes(notes: &Notes) {
    let lists = [
        ("only in a", &notes.only_a),
        ("only in b", &notes.only_b),
        ("refused", &notes.refused),
        ("not declared in the header", &notes.undeclared),
        ("left out", &notes.skipped),
    ];
    for (title, list) in lists {
        if !list.is_empty() {
            println!();
            println!("{title}:");
            for item in list {
                println!("  {item}");
            }
        }
    }
}
