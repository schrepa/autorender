//! `cargo xtask ladder`: the distance ladder, a check of our own. A ladder
//! file names places, looks and distances. For each place, look and
//! distance a standing eye (the Stress Lab's `stand:` view) stands that far
//! back from the place's target along its heading and looks down at the
//! target's ground, in one offscreen run of the native Stress Lab, one at a
//! time under the machine's heavy-run lock ([`crate::heavy`]), its last
//! frame captured. Each capture's luminance, in stops, splits into a
//! Laplacian pyramid, and each band's RMS in a window around the image's
//! center, where the target lies, is the contrast the view holds at that
//! octave of size, in pixels and in meters across the view at the target.
//! Natural images' power spectra fall about as 1/f² (Torralba and Oliva
//! 2003, "Statistics of natural image categories", Network: Computation in
//! Neural Systems), which puts about the same contrast in every octave: a
//! band well below its capture's median band is a size the view lacks. It
//! writes a JSON report, a contact sheet of each place and look's distances
//! and a printed table of distance by world size, and judges nothing: it
//! exits 0 whenever the runs complete, whatever the numbers, and no build
//! or check runs it.

use std::collections::BTreeSet;
use std::fs;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Instant, SystemTime};

use serde_json::{Value, json};

const USAGE: &str = "usage: cargo xtask ladder FILE --out DIR [--program PATH] [-- ARGUMENTS...]";

/// Frames each run draws once prepared, 5 s at 60 Hz, the last captured.
const FRAMES: u32 = 300;

/// Each capture's width and height, in pixels.
const SIZE: (u32, u32) = (1600, 900);

/// The standing eye's height above the ground, in meters, as the Stress
/// Lab's `stand:` view stands it (`EYE_HEIGHT` in
/// apps/stress-lab/src/scene.rs): the eye pitches down to the target's
/// ground this far below it.
const EYE_HEIGHT: f64 = 1.7;

/// The `stand:` view's vertical field of view, in degrees (its camera in
/// apps/stress-lab/src/scene.rs): the angle a capture's rows share.
const VERTICAL_FOV_DEGREES: f64 = 60.0;

/// Luminance's weights of linear red, green and blue (ITU-R BT.709).
const REC_709: [f64; 3] = [0.2126, 0.7152, 0.0722];

/// The darkest luminance a capture resolves: the gray of 8-bit sRGB code
/// 1, linear 1 / (255 × 12.92) (IEC 61966-2-1). Darker pixels read as it,
/// so black lies a finite number of stops down.
const DARKEST: f64 = 1.0 / (255.0 * 12.92);

/// The pyramid's blur: the 5-tap binomial kernel, (1 4 6 4 1) / 16 (Burt
/// and Adelson 1983, "The Laplacian pyramid as a compact image code").
const BINOMIAL: [f64; 5] = [0.0625, 0.25, 0.375, 0.25, 0.0625];

/// The measuring window's side: the image's height over this, a quarter,
/// around the image's center, where the target lies.
const WINDOW_PARTS: f64 = 4.0;

/// The world sizes the tables hold, in meters: octaves from an eighth of a
/// meter doubling to 64 m.
const OCTAVES: [f64; 10] = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0];

/// How many times smaller each capture is on its contact sheet, along each
/// side: a ladder's five 1600-pixel captures side by side in 2000 pixels.
const SHEET_SCALE: usize = 4;

/// Pixels between a contact sheet's captures.
const SHEET_GAP: usize = 8;

/// The 8-bit sRGB gray between a contact sheet's captures and in place of
/// a failed run's.
const SHEET_BACKGROUND: u8 = 24;

/// The fields a ladder file holds.
const FIELDS: [&str; 5] = ["note", "preset", "looks", "distances", "places"];

/// The fields each of a ladder's places holds.
const PLACE_FIELDS: [&str; 3] = ["name", "target", "yaw"];

/// What the command line asks for.
struct Options {
    ladder: PathBuf,
    out: PathBuf,
    program: Option<PathBuf>,
    /// The arguments after `--`, which every run takes after its own.
    passed: Vec<String>,
}

fn options(arguments: &[String]) -> Result<Options, String> {
    let (ours, passed) = match arguments.iter().position(|argument| argument == "--") {
        Some(at) => (&arguments[..at], arguments[at + 1..].to_vec()),
        None => (arguments, Vec::new()),
    };
    let (mut ladder, mut out, mut program) = (None, None, None);
    let mut ours = ours.iter();
    while let Some(argument) = ours.next() {
        match argument.as_str() {
            "--out" | "--program" => {
                let value = ours
                    .next()
                    .ok_or_else(|| format!("{argument} takes a path\n{USAGE}"))?;
                let slot = if argument == "--out" {
                    &mut out
                } else {
                    &mut program
                };
                *slot = Some(PathBuf::from(value));
            }
            option if option.starts_with("--") => {
                return Err(format!("unknown option {option}\n{USAGE}"));
            }
            file if ladder.is_none() => ladder = Some(PathBuf::from(file)),
            other => return Err(format!("unexpected argument {other}\n{USAGE}")),
        }
    }
    Ok(Options {
        ladder: ladder.ok_or_else(|| format!("a ladder file is required\n{USAGE}"))?,
        out: out.ok_or_else(|| format!("--out is required\n{USAGE}"))?,
        program,
        passed,
    })
}

/// A ladder file (`bench/ladders/NAME.json`): plain JSON, its fields read
/// as follows, a `note` beside them saying what the ladder is for.
struct Ladder {
    /// `preset`: the Stress Lab preset every run starts (`--preset`).
    preset: String,
    /// `looks`: the lighting presets each place is seen under (`--look`),
    /// `scene` the scene's own.
    looks: Vec<String>,
    /// `distances`: the meters the eye stands back from each target.
    distances: Vec<f64>,
    /// `places`: what the eye looks at.
    places: Vec<Place>,
}

/// A place of a ladder: its `name`, its `target` (`[x, z]`, world
/// meters), the ground the eye looks at, and its `yaw`, the heading the eye
/// looks along (degrees from -z toward +x, as the `stand:` view takes it).
struct Place {
    name: String,
    target: [f64; 2],
    yaw: f64,
}

/// `value` as a name runs and files take: letters, digits, `-` and `_`.
fn name(value: &Value) -> Option<String> {
    value
        .as_str()
        .filter(|name| {
            !name.is_empty()
                && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        })
        .map(str::to_owned)
}

/// `value` as a place of a ladder.
fn place(value: &Value) -> Option<Place> {
    let fields = value.as_object()?;
    if fields.keys().any(|field| !PLACE_FIELDS.contains(&field.as_str())) {
        return None;
    }
    let number = |value: &Value| value.as_f64().filter(|number| number.is_finite());
    let target = value["target"].as_array().filter(|target| target.len() == 2)?;
    Some(Place {
        name: name(&value["name"])?,
        target: [number(&target[0])?, number(&target[1])?],
        yaw: number(&value["yaw"])?,
    })
}

/// The ladder in the file at `path`, and the file as read.
fn read_ladder(path: &Path) -> Result<(Ladder, Value), String> {
    let file = crate::measure::read_json(path)?;
    let wrong = |what: &str| format!("{}: {what}", path.display());
    let fields = file.as_object().ok_or_else(|| wrong("a ladder is a JSON object"))?;
    if let Some(field) = fields.keys().find(|field| !FIELDS.contains(&field.as_str())) {
        return Err(wrong(&format!("{field} is not a field ({})", FIELDS.join(", "))));
    }
    let list = |key: &str| file[key].as_array().filter(|items| !items.is_empty());
    let preset = name(&file["preset"])
        .ok_or_else(|| wrong("preset names the Stress Lab preset every run starts"))?;
    let looks = list("looks")
        .and_then(|looks| looks.iter().map(name).collect::<Option<Vec<_>>>())
        .ok_or_else(|| wrong("looks lists the lighting presets each place is seen under"))?;
    let distances = list("distances")
        .and_then(|distances| {
            distances
                .iter()
                .map(|distance| {
                    distance
                        .as_f64()
                        .filter(|meters| meters.is_finite() && *meters > 0.0)
                })
                .collect::<Option<Vec<_>>>()
        })
        .ok_or_else(|| wrong("distances lists the meters the eye stands back, each above 0"))?;
    let places = list("places")
        .and_then(|places| places.iter().map(place).collect::<Option<Vec<_>>>())
        .ok_or_else(|| {
            wrong("places lists each place's name, target ([x, z], meters) and yaw (degrees)")
        })?;
    // Runs' files are named after them.
    let once = |names: Vec<String>| {
        let count = names.len();
        names.into_iter().collect::<BTreeSet<_>>().len() == count
    };
    let listed_once = once(places.iter().map(|place| place.name.clone()).collect())
        && once(looks.clone())
        && once(distances.iter().map(f64::to_string).collect());
    if !listed_once {
        return Err(wrong("each place's name, each look and each distance is listed once"));
    }
    let ladder = Ladder {
        preset,
        looks,
        distances,
        places,
    };
    Ok((ladder, file))
}

/// The program the runs start, and its build's identity where this task
/// built it: `--program`'s, or the working tree's built into
/// `target/ladder/` as measure builds an arm
/// ([`crate::measure::build_arm`]), with DXC's library beside it where
/// there is one, as measure's arms have it. Its own default `--assets` is
/// the checkout's `web/public`, which holds the cook's packs.
fn program(root: &Path, given: Option<&Path>) -> Result<(PathBuf, Value), String> {
    if let Some(given) = given {
        let program = std::path::absolute(given)
            .map_err(|error| format!("cannot resolve {}: {error}", given.display()))?;
        if !program.is_file() {
            return Err(format!("--program {} is not a file", program.display()));
        }
        return Ok((program, Value::Null));
    }
    let sha = crate::measure::git(root, &["rev-parse", "HEAD"])?;
    let changed = crate::measure::git(root, &["status", "--porcelain", "--untracked-files=no"])?;
    let backend = std::env::var("WGPU_BACKEND").unwrap_or_default();
    let folder = root.join("target").join("ladder");
    let mut build = crate::measure::build_arm(root, root, &folder, &sha, &backend)?;
    build["uncommitted_changes"] = (!changed.is_empty()).into();
    if cfg!(windows) && let Some(library) = crate::measure::dxc() {
        let beside = folder.join("dxcompiler.dll");
        if crate::measure::file_hash(&beside).ok() != crate::measure::file_hash(&library).ok() {
            fs::copy(&library, &beside)
                .map_err(|error| format!("cannot copy {}: {error}", library.display()))?;
        }
    }
    Ok((folder.join(crate::measure::program_name()), build))
}

/// Where a run's eye stands: world x and z (meters), and the yaw and pitch
/// it looks along (degrees, as the `stand:` view takes them).
#[derive(Clone, Copy)]
struct Eye {
    x: f64,
    z: f64,
    yaw: f64,
    pitch: f64,
}

impl Eye {
    /// The eye `distance` meters back from `place`'s target along its
    /// heading, pitched down to the target's ground [`EYE_HEIGHT`] below
    /// it.
    fn back_from(place: &Place, distance: f64) -> Self {
        let (sin, cos) = place.yaw.to_radians().sin_cos();
        Self {
            x: place.target[0] - distance * sin,
            z: place.target[1] + distance * cos,
            yaw: place.yaw,
            pitch: -(EYE_HEIGHT / distance).atan().to_degrees(),
        }
    }

    /// The Stress Lab's view of it: `stand:x,z,yaw,pitch`.
    fn view(&self) -> String {
        format!("stand:{:.3},{:.3},{:.3},{:.3}", self.x, self.z, self.yaw, self.pitch)
    }
}

/// One band of a capture's pyramid: its level, the sizes of detail it
/// holds (from and to, in pixels and in meters across the view at the
/// target's distance), its contrast (the RMS in the window, in stops) and
/// that over its capture's median band's.
struct Band {
    level: u32,
    pixels: [f64; 2],
    meters: [f64; 2],
    contrast: f64,
    ratio: Option<f64>,
}

/// A capture measured: its size, its window's side (pixels) and its bands.
struct Measured {
    width: usize,
    height: usize,
    window: f64,
    bands: Vec<Band>,
}

/// One run: its place, look and distance (indices into the ladder), where
/// its eye stood, what it started and wrote, how long it took, what it
/// printed of its frames, and its capture measured or why it is not.
struct Run {
    place: usize,
    look: usize,
    distance: usize,
    eye: Eye,
    command: Vec<String>,
    capture: PathBuf,
    log: PathBuf,
    seconds: f64,
    printed: Value,
    measured: Result<Measured, String>,
}

/// What every run shares: the repository, the program, the ladder, the
/// folder captures and logs go into, and the arguments passed on.
struct Runner<'a> {
    root: &'a Path,
    program: &'a Path,
    ladder: &'a Ladder,
    folder: &'a Path,
    passed: &'a [String],
}

impl Runner<'_> {
    /// Runs the ladder's place `place` under its look `look` from its
    /// distance `distance` (indices), and measures the capture.
    fn run(&self, place: usize, look: usize, distance: usize) -> Run {
        let ladder = self.ladder;
        let (spot, name) = (&ladder.places[place], &ladder.looks[look]);
        let meters = ladder.distances[distance];
        let eye = Eye::back_from(spot, meters);
        let stem = format!("{}-{name}-{meters}m", spot.name);
        let capture = self.folder.join(format!("{stem}.png"));
        let log = self.folder.join(format!("{stem}.log"));
        let mut arguments: Vec<String> = [
            ("--preset", ladder.preset.clone()),
            ("--look", name.clone()),
            ("--view", eye.view()),
            ("--dynamic-resolution", "off".to_owned()),
            ("--offscreen", FRAMES.to_string()),
            ("--size", format!("{}x{}", SIZE.0, SIZE.1)),
            ("--capture", capture.display().to_string()),
        ]
        .into_iter()
        .flat_map(|(option, value)| [option.to_owned(), value])
        .collect();
        arguments.extend_from_slice(self.passed);
        eprintln!("{} ({name}) from {meters} m: {}.", spot.name, eye.view());
        // A run that fails leaves no earlier capture to measure.
        let _ = fs::remove_file(&capture);
        let started = Instant::now();
        let output = Command::new(self.program)
            .args(&arguments)
            .current_dir(self.root)
            .output();
        let seconds = started.elapsed().as_secs_f64();
        let (printed, measured) = match output {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);
                let kept = fs::write(&log, format!("{stdout}\n{stderr}"))
                    .map_err(|error| format!("cannot write {}: {error}", log.display()));
                let printed: Value = stdout
                    .find('{')
                    .and_then(|start| serde_json::from_str(&stdout[start..]).ok())
                    .unwrap_or_default();
                let measured = if output.status.success() {
                    kept.and_then(|()| measure_capture(&capture, meters))
                } else {
                    let last: Vec<&str> = stderr.lines().rev().take(3).collect();
                    Err(format!(
                        "the run failed ({}; {}): {}",
                        output.status,
                        log.display(),
                        last.join(" | ")
                    ))
                };
                let printed = json!({
                    "frames": printed["frames"],
                    "prepare_ms": printed["prepare_ms"],
                    "loading": printed["stats"]["loading"],
                });
                (printed, measured)
            }
            Err(error) => (
                Value::Null,
                Err(format!("cannot start {}: {error}", self.program.display())),
            ),
        };
        let mut command = vec![self.program.display().to_string()];
        command.extend(arguments);
        Run {
            place,
            look,
            distance,
            eye,
            command,
            capture,
            log,
            seconds,
            printed,
            measured,
        }
    }
}

/// One level of a pyramid: its width, its height and its values, row by
/// row.
struct Level {
    width: usize,
    height: usize,
    values: Vec<f64>,
}

/// The capture at `path`, seen from `distance` meters, measured. Its
/// pyramid filters, subtracts and decimates (Burt and Adelson's): each
/// band is what [`BINOMIAL`] blurs out of its level, the blurred level
/// halved is the next, and band `k` holds detail 2^k to 2^(k+1) pixels
/// across. Bands go on while the window spans a sample of their level.
fn measure_capture(path: &Path, distance: f64) -> Result<Measured, String> {
    let (width, height, pixels) = crate::equivalence::decode(path)?;
    let (width, height) = (width as usize, height as usize);
    let window = height as f64 / WINDOW_PARTS;
    let center = [width as f64 / 2.0, height as f64 / 2.0];
    // Meters a pixel spans across the view at the target's distance.
    let meters = distance * VERTICAL_FOV_DEGREES.to_radians() / height as f64;
    let mut level = Level {
        width,
        height,
        values: stops(&pixels),
    };
    let (mut scale, mut found) = (1.0, Vec::new());
    while window / scale >= 1.0 {
        let blurred = blur(&level);
        let detail = Level {
            width: level.width,
            height: level.height,
            values: level
                .values
                .iter()
                .zip(&blurred.values)
                .map(|(value, smooth)| value - smooth)
                .collect(),
        };
        found.push((scale, window_rms(&detail, center, window, scale)));
        level = halve(&blurred);
        scale *= 2.0;
    }
    let contrasts: Vec<f64> = found.iter().map(|&(_, contrast)| contrast).collect();
    let median = engine::host::median(&contrasts).unwrap_or_default();
    let bands = found
        .into_iter()
        .zip(0_u32..)
        .map(|((scale, contrast), level)| Band {
            level,
            pixels: [scale, 2.0 * scale],
            meters: [scale * meters, 2.0 * scale * meters],
            contrast,
            ratio: (median > 0.0).then_some(contrast / median),
        })
        .collect();
    Ok(Measured {
        width,
        height,
        window,
        bands,
    })
}

/// The linear light of each 8-bit sRGB code (IEC 61966-2-1).
fn linear_codes() -> [f64; 256] {
    std::array::from_fn(|code| {
        let encoded = code as f64 / 255.0;
        if encoded <= 0.040_45 {
            encoded / 12.92
        } else {
            ((encoded + 0.055) / 1.055).powf(2.4)
        }
    })
}

/// Linear light `linear` as its nearest 8-bit sRGB code (IEC 61966-2-1).
fn srgb_code(linear: f64) -> u8 {
    let encoded = if linear <= 0.003_130_8 {
        linear * 12.92
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    };
    (encoded * 255.0).round().clamp(0.0, 255.0) as u8
}

/// Each pixel's luminance in stops: its 8-bit sRGB channels (of RGBA
/// `pixels`) decoded to linear light and weighted as [`REC_709`] weighs
/// them, no darker than [`DARKEST`], in log2.
fn stops(pixels: &[u8]) -> Vec<f64> {
    let linear = linear_codes();
    pixels
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&[red, green, blue, _]| {
            let luminance = REC_709[0] * linear[usize::from(red)]
                + REC_709[1] * linear[usize::from(green)]
                + REC_709[2] * linear[usize::from(blue)];
            luminance.max(DARKEST).log2()
        })
        .collect()
}

/// `level` blurred by [`BINOMIAL`] along its rows, then its columns, its
/// edge samples standing in for those past its edges.
fn blur(level: &Level) -> Level {
    let (width, height) = (level.width, level.height);
    // The sample `offset` taps along from two before `at`, of `count`.
    let tap =
        |at: usize, offset: usize, count: usize| (at + offset).saturating_sub(2).min(count - 1);
    let mut across = vec![0.0; level.values.len()];
    for row in 0..height {
        for column in 0..width {
            across[row * width + column] = BINOMIAL
                .iter()
                .enumerate()
                .map(|(offset, weight)| {
                    weight * level.values[row * width + tap(column, offset, width)]
                })
                .sum();
        }
    }
    let mut values = vec![0.0; across.len()];
    for row in 0..height {
        for column in 0..width {
            values[row * width + column] = BINOMIAL
                .iter()
                .enumerate()
                .map(|(offset, weight)| weight * across[tap(row, offset, height) * width + column])
                .sum();
        }
    }
    Level {
        width,
        height,
        values,
    }
}

/// Every other sample of `level` along each side, from its first: the
/// next level, half as wide and high.
fn halve(level: &Level) -> Level {
    let (width, height) = (level.width.div_ceil(2), level.height.div_ceil(2));
    let values = (0..height)
        .flat_map(|row| {
            (0..width).map(move |column| level.values[2 * (row * level.width + column)])
        })
        .collect();
    Level {
        width,
        height,
        values,
    }
}

/// The RMS of `band`'s samples, `scale` image pixels apart, within the
/// square `side` image pixels across around image position `center`.
fn window_rms(band: &Level, center: [f64; 2], side: f64, scale: f64) -> f64 {
    let columns = within(center[0], side, scale, band.width);
    let rows = within(center[1], side, scale, band.height);
    let count = (columns.len() * rows.len()) as f64;
    let sum: f64 = rows
        .flat_map(|row| {
            columns
                .clone()
                .map(move |column| band.values[row * band.width + column])
        })
        .map(|value| value * value)
        .sum();
    (sum / count).sqrt()
}

/// The samples of a line of `count`, `scale` image pixels apart from the
/// image's first pixel, whose centers lie within `side` image pixels
/// around image position `center`: the one nearest it where none do.
fn within(center: f64, side: f64, scale: f64, count: usize) -> Range<usize> {
    // Sample `i` stands where image pixel `i × scale` does, its center at
    // `i × scale + 0.5`.
    let first_from =
        |position: f64| ((position - 0.5) / scale).ceil().clamp(0.0, count as f64) as usize;
    let samples = first_from(center - side / 2.0)..first_from(center + side / 2.0);
    if samples.is_empty() {
        let nearest = ((center - 0.5) / scale)
            .round()
            .clamp(0.0, (count - 1) as f64) as usize;
        nearest..nearest + 1
    } else {
        samples
    }
}

/// The band of `bands` holding detail `size` meters across, by the middle
/// of the octave from it (`size` × √2), where one does.
fn covering(bands: &[Band], size: f64) -> Option<&Band> {
    let middle = size * std::f64::consts::SQRT_2;
    bands.iter().find(|band| (band.meters[0]..band.meters[1]).contains(&middle))
}

/// The runs of place `place` under look `look`, in the ladder's order of
/// distances: each with the band covering each of [`OCTAVES`], where one
/// does.
fn table(runs: &[Run], place: usize, look: usize) -> Vec<(&Run, Vec<Option<&Band>>)> {
    runs.iter()
        .filter(|run| run.place == place && run.look == look)
        .map(|run| {
            let bands = run
                .measured
                .as_ref()
                .map(|measured| measured.bands.as_slice())
                .unwrap_or_default();
            let covered = OCTAVES.iter().map(|&size| covering(bands, size)).collect();
            (run, covered)
        })
        .collect()
}

/// A capture's thumbnail: its width, its height and its 8-bit sRGB pixels
/// (RGB), row by row.
struct Thumbnail {
    width: usize,
    height: usize,
    pixels: Vec<u8>,
}

/// The capture at `path`, [`SHEET_SCALE`] times smaller along each side:
/// each block of its pixels averaged in linear light.
fn thumbnail(path: &Path) -> Result<Thumbnail, String> {
    let (width, height, pixels) = crate::equivalence::decode(path)?;
    let width = width as usize;
    let (small_width, small_height) = (width / SHEET_SCALE, height as usize / SHEET_SCALE);
    let linear = linear_codes();
    let block = (SHEET_SCALE * SHEET_SCALE) as f64;
    let mut small = Vec::with_capacity(small_width * small_height * 3);
    for row in 0..small_height {
        for column in 0..small_width {
            let mut sum = [0.0; 3];
            for y in row * SHEET_SCALE..(row + 1) * SHEET_SCALE {
                for x in column * SHEET_SCALE..(column + 1) * SHEET_SCALE {
                    let at = (y * width + x) * 4;
                    for (total, code) in sum.iter_mut().zip(&pixels[at..at + 3]) {
                        *total += linear[usize::from(*code)];
                    }
                }
            }
            small.extend(sum.map(|total| srgb_code(total / block)));
        }
    }
    Ok(Thumbnail {
        width: small_width,
        height: small_height,
        pixels: small,
    })
}

/// Writes `captures` (a failed run's `None`) side by side into the PNG
/// `path`: each one's [`thumbnail`], [`SHEET_GAP`] pixels apart.
fn contact_sheet(captures: &[Option<&Path>], path: &Path) -> Result<(), String> {
    let thumbnails = captures
        .iter()
        .map(|capture| capture.map(thumbnail).transpose())
        .collect::<Result<Vec<_>, _>>()?;
    let blank = (SIZE.0 as usize / SHEET_SCALE, SIZE.1 as usize / SHEET_SCALE);
    let slots: Vec<(usize, usize)> = thumbnails
        .iter()
        .map(|small| small.as_ref().map_or(blank, |small| (small.width, small.height)))
        .collect();
    let width = slots.iter().map(|&(width, _)| width).sum::<usize>()
        + SHEET_GAP * slots.len().saturating_sub(1);
    let height = slots.iter().map(|&(_, height)| height).max().unwrap_or(blank.1);
    let mut sheet = vec![SHEET_BACKGROUND; width * height * 3];
    let mut left = 0;
    for (small, &(slot_width, _)) in thumbnails.iter().zip(&slots) {
        if let Some(small) = small {
            let line = small.width * 3;
            for row in 0..small.height {
                let to = (row * width + left) * 3;
                sheet[to..to + line].copy_from_slice(&small.pixels[row * line..(row + 1) * line]);
            }
        }
        left += slot_width + SHEET_GAP;
    }
    write_png(path, width, height, &sheet)
}

/// Writes 8-bit RGB `pixels`, `width` by `height`, as the PNG `path`.
fn write_png(path: &Path, width: usize, height: usize, pixels: &[u8]) -> Result<(), String> {
    let failed =
        |error: &dyn std::fmt::Display| format!("cannot write {}: {error}", path.display());
    let file = fs::File::create(path).map_err(|error| failed(&error))?;
    let mut encoder =
        png::Encoder::new(std::io::BufWriter::new(file), width as u32, height as u32);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().map_err(|error| failed(&error))?;
    writer
        .write_image_data(pixels)
        .map_err(|error| failed(&error))?;
    writer.finish().map_err(|error| failed(&error))
}

/// Writes each place and look's contact sheet into `out`, its distances'
/// captures side by side in the ladder's order: the sheets' paths.
fn contact_sheets(out: &Path, ladder: &Ladder, runs: &[Run]) -> Result<Vec<PathBuf>, String> {
    let mut sheets = Vec::new();
    for (place, spot) in ladder.places.iter().enumerate() {
        for (look, name) in ladder.looks.iter().enumerate() {
            let captures: Vec<Option<&Path>> = runs
                .iter()
                .filter(|run| run.place == place && run.look == look)
                .map(|run| run.measured.is_ok().then_some(run.capture.as_path()))
                .collect();
            let sheet = out.join(format!("{}-{name}.png", spot.name));
            contact_sheet(&captures, &sheet)?;
            sheets.push(sheet);
        }
    }
    Ok(sheets)
}

/// `value` to a ten-thousandth.
fn rounded(value: f64) -> f64 {
    (value * 1e4).round() / 1e4
}

/// A run as the report gives it.
fn run_json(run: &Run, ladder: &Ladder) -> Value {
    let eye = run.eye;
    let mut record = json!({
        "place": ladder.places[run.place].name,
        "look": ladder.looks[run.look],
        "distance_m": ladder.distances[run.distance],
        "eye": {
            "x": eye.x,
            "z": eye.z,
            "yaw": eye.yaw,
            "pitch": eye.pitch,
            "view": eye.view(),
        },
        "command": run.command,
        "capture": run.capture.display().to_string(),
        "log": run.log.display().to_string(),
        "seconds": rounded(run.seconds),
        "printed": run.printed,
    });
    match &run.measured {
        Ok(measured) => {
            record["size"] = json!([measured.width, measured.height]);
            record["window_px"] = json!(measured.window);
            record["bands"] = measured
                .bands
                .iter()
                .map(|band| {
                    json!({
                        "level": band.level,
                        "pixels": band.pixels,
                        "meters": band.meters.map(rounded),
                        "contrast_stops": rounded(band.contrast),
                        "ratio": band.ratio.map(rounded),
                    })
                })
                .collect();
        }
        Err(error) => record["error"] = error.clone().into(),
    }
    record
}

/// The report: what the ladder ran with, every run, each place and look's
/// contact sheet, and each place and look's table of contrast and of its
/// share of the capture's median band, by distance and world size.
fn report(inputs: Value, ladder: &Ladder, runs: &[Run], sheets: &[PathBuf]) -> Value {
    let mut tables = Vec::new();
    for (place, spot) in ladder.places.iter().enumerate() {
        for (look, name) in ladder.looks.iter().enumerate() {
            let rows: Vec<Value> = table(runs, place, look)
                .into_iter()
                .map(|(run, bands)| {
                    let contrast: Vec<Option<f64>> = bands
                        .iter()
                        .map(|band| band.map(|band| rounded(band.contrast)))
                        .collect();
                    let ratio: Vec<Option<f64>> = bands
                        .iter()
                        .map(|band| band.and_then(|band| band.ratio).map(rounded))
                        .collect();
                    json!({
                        "distance_m": ladder.distances[run.distance],
                        "contrast_stops": contrast,
                        "ratio": ratio,
                    })
                })
                .collect();
            tables.push(json!({
                "place": spot.name,
                "look": name,
                "octaves_m": OCTAVES,
                "rows": rows,
            }));
        }
    }
    let runs: Vec<Value> = runs.iter().map(|run| run_json(run, ladder)).collect();
    let sheets: Vec<String> = sheets.iter().map(|sheet| sheet.display().to_string()).collect();
    json!({
        "inputs": inputs,
        "runs": runs,
        "contact_sheets": sheets,
        "tables": tables,
    })
}

/// One printed row of a table: the distance, then each value to three
/// places, or a dash where no band covers the size.
fn row(distance: f64, values: impl IntoIterator<Item = Option<f64>>) -> String {
    let mut line = format!("  {distance:>8} m");
    for value in values {
        match value {
            Some(value) => line.push_str(&format!("{value:>8.3}")),
            None => line.push_str(&format!("{:>8}", "-")),
        }
    }
    line
}

/// Prints each place and look's tables by distance and world size: the
/// contrast, then its share of each capture's median band.
fn print_tables(ladder: &Ladder, runs: &[Run]) {
    let header: String = OCTAVES.iter().map(|size| format!("{size:>8}")).collect();
    for (place, spot) in ladder.places.iter().enumerate() {
        for (look, name) in ladder.looks.iter().enumerate() {
            let rows = table(runs, place, look);
            println!("{} ({name}): contrast (RMS, stops) by world size (m)", spot.name);
            println!("  {:>10}{header}", "distance");
            for (run, bands) in &rows {
                let contrast = bands.iter().map(|band| band.map(|band| band.contrast));
                println!("{}", row(ladder.distances[run.distance], contrast));
            }
            println!("  as a share of its capture's median band");
            for (run, bands) in &rows {
                let ratio = bands.iter().map(|band| band.and_then(|band| band.ratio));
                println!("{}", row(ladder.distances[run.distance], ratio));
            }
        }
    }
}

pub fn ladder(root: &Path, arguments: &[String]) -> Result<(), String> {
    let options = options(arguments)?;
    let (ladder, file) = read_ladder(&options.ladder)?;
    let out = std::path::absolute(&options.out)
        .map_err(|error| format!("cannot resolve {}: {error}", options.out.display()))?;
    let folder = out.join("captures");
    fs::create_dir_all(&folder)
        .map_err(|error| format!("cannot create {}: {error}", folder.display()))?;
    let (program, build) = program(root, options.program.as_deref())?;
    let inputs = json!({
        "ladder": options.ladder.display().to_string(),
        "file": file,
        "program": program.display().to_string(),
        "build": build,
        "arguments": options.passed,
        "frames": FRAMES,
        "size": [SIZE.0, SIZE.1],
        "eye_height_m": EYE_HEIGHT,
        "vertical_fov_degrees": VERTICAL_FOV_DEGREES,
        "started_at": crate::measure::iso(SystemTime::now()),
    });
    let runner = Runner {
        root,
        program: &program,
        ladder: &ladder,
        folder: &folder,
        passed: &options.passed,
    };
    // The runs hold the machine's heavy-run lock: nothing else draws on its
    // GPU meanwhile.
    let heavy = crate::heavy::HeavyRun::hold("cargo xtask ladder")?;
    let mut runs = Vec::new();
    for place in 0..ladder.places.len() {
        for look in 0..ladder.looks.len() {
            for distance in 0..ladder.distances.len() {
                runs.push(runner.run(place, look, distance));
            }
        }
    }
    drop(heavy);
    let sheets = contact_sheets(&out, &ladder, &runs)?;
    let path = out.join("report.json");
    let report = report(inputs, &ladder, &runs, &sheets);
    fs::write(
        &path,
        serde_json::to_string_pretty(&report).expect("reports serialize") + "\n",
    )
    .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
    print_tables(&ladder, &runs);
    println!("Wrote {} and {} contact sheets beside it.", path.display(), sheets.len());
    let failed: Vec<String> = runs
        .iter()
        .filter_map(|run| {
            let error = run.measured.as_ref().err()?;
            Some(format!("{}: {error}", run.capture.display()))
        })
        .collect();
    if failed.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "{} of {} runs did not complete:\n  {}",
            failed.len(),
            runs.len(),
            failed.join("\n  ")
        ))
    }
}
