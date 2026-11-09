use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use engine_checks::bench;
use serde_json::{Map, Value};

use super::matrix::Cell;
use super::session::{Run, Shot};

struct Difference {
    pixels: usize,
    total: usize,
    largest: u8,
}

impl Difference {
    fn describe(&self, tolerance: u8, diff: &Path) -> String {
        format!(
            "{} of {} pixels differ by more than {tolerance} (largest {}); {}",
            self.pixels,
            self.total,
            self.largest,
            diff.display()
        )
    }
}

pub(crate) fn decode(path: &Path) -> Result<(u32, u32, Vec<u8>), String> {
    let file =
        fs::File::open(path).map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    let mut decoder = png::Decoder::new(std::io::BufReader::new(file));
    decoder.set_transformations(png::Transformations::EXPAND);
    let mut reader = decoder
        .read_info()
        .map_err(|error| format!("{}: {error}", path.display()))?;
    let mut pixels = vec![0; reader.output_buffer_size()];
    let info = reader
        .next_frame(&mut pixels)
        .map_err(|error| format!("{}: {error}", path.display()))?;
    pixels.truncate(info.buffer_size());
    if info.bit_depth != png::BitDepth::Eight {
        return Err(format!("{} is not 8 bits a channel", path.display()));
    }
    match info.color_type {
        png::ColorType::Rgba => Ok((info.width, info.height, pixels)),
        png::ColorType::Rgb => {
            let opaque = pixels
                .as_chunks::<3>()
                .0
                .iter()
                .flat_map(|&[r, g, b]| [r, g, b, 255])
                .collect();
            Ok((info.width, info.height, opaque))
        }
        other => Err(format!("{} is {other:?}, not RGB or RGBA", path.display())),
    }
}

fn difference(
    base: &Path,
    candidate: &Path,
    tolerance: u8,
    diff: Option<&Path>,
) -> Result<Difference, String> {
    let (width, height, a) = decode(base)?;
    let (other_width, other_height, b) = decode(candidate)?;
    if (width, height) != (other_width, other_height) {
        return Err(format!(
            "the captures differ in size: {width}x{height} and {other_width}x{other_height}"
        ));
    }
    let (mut pixels, mut largest) = (0, 0);
    let mut image = Vec::with_capacity(a.len());
    for (x, y) in a.as_chunks::<4>().0.iter().zip(b.as_chunks::<4>().0) {
        let channel = x
            .iter()
            .zip(y)
            .map(|(x, y)| x.abs_diff(*y))
            .max()
            .unwrap_or_default();
        largest = largest.max(channel);
        if channel > tolerance {
            pixels += 1;
            image.extend([255, 0, 0, 255]);
        } else {
            let gray = ((u16::from(x[0]) + u16::from(x[1]) + u16::from(x[2])) / 12) as u8;
            image.extend([gray, gray, gray, 255]);
        }
    }
    if let Some(diff) = diff.filter(|_| pixels > 0) {
        let file = fs::File::create(diff).map_err(|error| error.to_string())?;
        let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), width, height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        encoder
            .write_header()
            .and_then(|mut writer| writer.write_image_data(&image))
            .map_err(|error| format!("cannot write {}: {error}", diff.display()))?;
    }
    Ok(Difference {
        pixels,
        total: (width as usize) * (height as usize),
        largest,
    })
}

pub(super) fn pixel_difference(a: &Path, b: &Path) -> Result<(usize, usize, u8), String> {
    difference(a, b, 0, None).map(|found| (found.pixels, found.total, found.largest))
}

pub(super) enum Verdict {
    Equal,
    WithinVariation {
        pixels: usize,
        largest: u8,
        spread: (usize, u8),
    },
    EqualUnstable {
        distinct: usize,
    },
    Different(String),
    Unavailable(String),
    Unchecked(String),
}

pub(super) struct Changed {
    pub(super) key: String,
    pub(super) name: String,
    pub(super) base: Value,
    pub(super) candidate: Value,
    pub(super) rule: String,
    pub(super) fails: bool,
}

fn rule<'a>(catalog: &'a [Value], name: &str) -> Option<&'a str> {
    catalog
        .iter()
        .find(|entry| {
            entry["kind"] == "work"
                && bench::covers(entry["name"].as_str().unwrap_or_default(), name)
        })
        .and_then(|entry| entry["rule"].as_str())
}

fn compare_work(
    key: &str,
    base: &Map<String, Value>,
    candidate: &Map<String, Value>,
    catalog: &[Value],
) -> Vec<Changed> {
    let names: BTreeSet<&String> = base.keys().chain(candidate.keys()).collect();
    let mut changed = Vec::new();
    for name in names {
        let Some(rule) = rule(catalog, name) else {
            continue;
        };
        let (before, after) = (
            base.get(name).cloned().unwrap_or(Value::Null),
            candidate.get(name).cloned().unwrap_or(Value::Null),
        );
        let fails = match rule {
            "false" => {
                after != Value::Bool(false)
                    || (base.contains_key(name) && before != Value::Bool(false))
            }
            _ if !base.contains_key(name) => false,
            _ if before == after => false,
            "equal" => true,
            "not-above" => match (before.as_f64(), after.as_f64()) {
                (Some(before), Some(after)) => after > before,
                _ => true,
            },
            _ => false,
        };
        if fails || before != after {
            changed.push(Changed {
                key: key.to_owned(),
                name: name.clone(),
                base: before,
                candidate: after,
                rule: rule.to_owned(),
                fails,
            });
        }
    }
    changed
}

pub(super) fn hashes<'a>(runs: &'a [Run], key: &str) -> Vec<&'a str> {
    runs.iter()
        .filter_map(|run| match run.shots.get(key) {
            Some(Shot::Taken { sha256, .. }) => Some(sha256.as_str()),
            _ => None,
        })
        .collect()
}

fn first<'a>(runs: &'a [Run], key: &str) -> Option<(&'a Path, &'a str, &'a Map<String, Value>)> {
    runs.iter().find_map(|run| match run.shots.get(key) {
        Some(Shot::Taken { path, sha256, work }) => Some((path.as_path(), sha256.as_str(), work)),
        _ => None,
    })
}

pub(super) fn run_problems(role: &str, runs: &[Run]) -> Vec<String> {
    runs.iter()
        .enumerate()
        .flat_map(|(index, run)| {
            let at = format!("{role} run {}", index + 1);
            run.error
                .iter()
                .map(move |error| format!("{at} stopped: {error}"))
                .chain(
                    run.unsettled.iter().map(move |view| {
                        format!("{role} run {} did not settle: {view}", index + 1)
                    }),
                )
                .collect::<Vec<_>>()
        })
        .collect()
}

pub(super) fn judge(
    cell: &Cell,
    base: &[Run],
    candidate: &[Run],
    catalog: &[Value],
    folder: &Path,
) -> (Vec<(String, Verdict)>, Vec<Changed>) {
    let (mut verdicts, mut changed) = (Vec::new(), Vec::new());
    let problems: Vec<String> = run_problems("base", base)
        .into_iter()
        .chain(run_problems("candidate", candidate))
        .collect();
    for key in cell.keys() {
        let unavailable = |runs: &[Run]| {
            runs.iter().find_map(|run| match run.shots.get(&key) {
                Some(Shot::Unavailable(why)) => Some(why.clone()),
                _ => None,
            })
        };
        match (unavailable(base), unavailable(candidate)) {
            (Some(why), Some(_)) => {
                verdicts.push((key, Verdict::Unavailable(why)));
                continue;
            }
            (Some(why), None) | (None, Some(why)) => {
                verdicts.push((
                    key,
                    Verdict::Unchecked(format!("one side only could not take it: {why}")),
                ));
                continue;
            }
            (None, None) => {}
        }
        let (base_hashes, candidate_hashes) = (hashes(base, &key), hashes(candidate, &key));
        let (Some((base_path, _, base_work)), Some((candidate_path, _, candidate_work))) =
            (first(base, &key), first(candidate, &key))
        else {
            verdicts.push((key, Verdict::Unchecked(problems.join("; "))));
            continue;
        };
        let mut work = compare_work(&key, base_work, candidate_work, catalog);
        if cell.unstable.contains_key(&key) {
            let against: Vec<Vec<Changed>> = base
                .iter()
                .filter_map(|run| match run.shots.get(&key) {
                    Some(Shot::Taken { work, .. }) => Some(work),
                    _ => None,
                })
                .map(|other| compare_work(&key, other, candidate_work, catalog))
                .collect();
            for change in &mut work {
                change.fails = against.iter().all(|changes| {
                    changes
                        .iter()
                        .any(|other| other.name == change.name && other.fails)
                });
            }
        }
        changed.extend(work);
        let distinct: BTreeSet<&str> = base_hashes.iter().copied().collect();
        let verdict = if cell.unstable.contains_key(&key) {
            if candidate_hashes.iter().any(|hash| distinct.contains(hash)) {
                Verdict::EqualUnstable {
                    distinct: distinct.len(),
                }
            } else {
                nearest_difference(cell, &key, base, candidate_path, folder)
            }
        } else if base_hashes.first() == candidate_hashes.first() {
            Verdict::Equal
        } else {
            describe_difference(cell, &key, base_path, candidate_path, folder)
        };
        let verdict = match verdict {
            Verdict::Different(what) if !problems.is_empty() => {
                Verdict::Unchecked(format!("{what}, but {}", problems.join("; ")))
            }
            verdict => verdict,
        };
        verdicts.push((key, verdict));
    }
    (verdicts, changed)
}

fn nearest_difference(
    cell: &Cell,
    key: &str,
    base: &[Run],
    candidate: &Path,
    folder: &Path,
) -> Verdict {
    let mut images: Vec<(&str, &Path)> = base
        .iter()
        .filter_map(|run| match run.shots.get(key) {
            Some(Shot::Taken { path, sha256, .. }) => Some((sha256.as_str(), path.as_path())),
            _ => None,
        })
        .collect();
    images.sort_unstable();
    images.dedup_by_key(|(hash, _)| *hash);
    let mut nearest: Option<(Difference, &Path)> = None;
    for (_, image) in &images {
        match difference(image, candidate, cell.tolerance, None) {
            Ok(found) if found.pixels == 0 => return Verdict::Equal,
            Ok(found) => {
                if nearest
                    .as_ref()
                    .is_none_or(|(least, _)| found.pixels < least.pixels)
                {
                    nearest = Some((found, image));
                }
            }
            Err(error) => return Verdict::Different(error),
        }
    }
    let Some((least, image)) = nearest else {
        return Verdict::Unchecked("the base took no image".into());
    };
    let mut spread = (0, 0);
    for (index, (_, a)) in images.iter().enumerate() {
        for (_, b) in &images[index + 1..] {
            match difference(a, b, cell.tolerance, None) {
                Ok(found) => spread = (spread.0.max(found.pixels), spread.1.max(found.largest)),
                Err(error) => return Verdict::Different(error),
            }
        }
    }
    if least.pixels <= spread.0 && least.largest <= spread.1 {
        return Verdict::WithinVariation {
            pixels: least.pixels,
            largest: least.largest,
            spread,
        };
    }
    match describe_difference(cell, key, image, candidate, folder) {
        Verdict::Different(what) => Verdict::Different(format!(
            "matches none of the base's {} distinct images, and differs from the nearest by more than they differ from each other ({} pixels, up to {} levels): {what}",
            images.len(),
            spread.0,
            spread.1
        )),
        verdict => verdict,
    }
}

fn diff_path(cell: &Cell, key: &str, folder: &Path) -> std::path::PathBuf {
    folder.join(format!(
        "{}-{}-diff.png",
        cell.name,
        key.replace([' ', ':'], "-")
    ))
}

fn describe_difference(
    cell: &Cell,
    key: &str,
    base: &Path,
    candidate: &Path,
    folder: &Path,
) -> Verdict {
    let diff = diff_path(cell, key, folder);
    match difference(base, candidate, cell.tolerance, Some(&diff)) {
        Ok(found) if found.pixels == 0 => Verdict::Equal,
        Ok(found) => Verdict::Different(found.describe(cell.tolerance, &diff)),
        Err(error) => Verdict::Different(error),
    }
}

pub(super) fn compare_digests(
    cell: &Cell,
    base: &[Run],
    candidate: &[Run],
) -> Vec<(String, String)> {
    let Some(theirs) = candidate.first() else {
        return Vec::new();
    };
    let views: BTreeSet<&String> = base.iter().flat_map(|run| run.digests.keys()).collect();
    views
        .into_iter()
        .map(|view| {
            let digest = theirs.digests.get(view).filter(|digest| !digest.is_null());
            let ours: Vec<&Value> = base
                .iter()
                .filter_map(|run| run.digests.get(view))
                .filter(|digest| !digest.is_null())
                .collect();
            let verdict = match digest {
                Some(digest) if !ours.is_empty() => {
                    let judged = |part: &String| {
                        !cell.unstable.contains_key(&format!("{view} digest:{part}"))
                    };
                    let nearest = ours
                        .iter()
                        .map(|ours| {
                            digest_parts(ours, digest)
                                .into_iter()
                                .filter(judged)
                                .collect::<Vec<_>>()
                        })
                        .min_by_key(Vec::len)
                        .unwrap_or_default();
                    if nearest.is_empty() {
                        "equal".to_owned()
                    } else {
                        format!("different: {}", nearest.join(", "))
                    }
                }
                _ => "unavailable".to_owned(),
            };
            (view.clone(), verdict)
        })
        .collect()
}

pub(super) fn digest_parts(a: &Value, b: &Value) -> Vec<String> {
    let mut parts = Vec::new();
    if a["entities"] != b["entities"] {
        parts.push("entities".to_owned());
    }
    for section in ["components", "assets"] {
        let names: BTreeSet<&String> = [a, b]
            .iter()
            .flat_map(|digest| digest[section].as_object().into_iter().flatten())
            .map(|(name, _)| name)
            .collect();
        for name in names {
            if a[section][name.as_str()] != b[section][name.as_str()] {
                parts.push(format!("{section}.{name}"));
            }
        }
    }
    parts
}

pub(super) struct Judged<'a> {
    pub(super) cell: &'a Cell,
    pub(super) verdicts: Vec<(String, Verdict)>,
    pub(super) changed: Vec<Changed>,
    pub(super) digests: Vec<(String, String)>,
    pub(super) base: Vec<Run>,
    pub(super) candidate: Vec<Run>,
}
