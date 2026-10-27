//! Handovers by the camera, which `cargo xtask gpu` refuses: no shader
//! decides detail (a `smoothstep` or `step`, a `select` or `mix` weight, an
//! `if`, a loop's bound or a comparison turned into a number) by where the
//! camera stands or how it looks at a surface, but through the measures of
//! the detail eye or in optics. Reads of the camera (its eye,
//! `camera.position`; the detail eye, `camera.detail`, and a pixel's angle,
//! `camera.options.y`, outside the measures; a camera matrix's eye or axes)
//! and positions relative to the eye (named `relative`) are followed
//! through expressions, locals, arguments, results and calls. What the
//! measures give (`scene.wgsl`'s `detail_distance`, `detail_across` and
//! `detail_span`) and an eye-relative position turned back into a world
//! position (`eye_world`) depend on the camera no more; neither do the
//! optics [`OPTICS`] lists, nor a line marked `// optics: <why>`. What a
//! decision does with the sizes the measures give is the size check's
//! (`fades`), which follows sizes on screen through the same flow
//! ([`sweep`]).

use std::collections::{BTreeMap, BTreeSet};

use naga::Span;

use super::Checked;

mod flow;

pub(super) use flow::Watch;
use flow::{Call, Decision, Flow, Read, Source};

/// Where the shader files lie, from the repository's root.
const SHADERS: &str = "crates/engine_render/src/";

/// The one way from an eye-relative position back to a world position:
/// what it gives depends on where the camera stands no more, as what the
/// detail eye's measures (`flow::SIZES`) give does.
const EYE_WORLD: &str = "eye_world";

/// Optics: functions depending on where the camera stands or how it looks
/// as light and sight do, each with what it is. They are not followed:
/// what they give, and what they pass the functions they call, depends on
/// nothing.
const OPTICS: &[(&str, &str)] = &[
    ("air_light", "the air's light toward the eye"),
    ("air_sky", "the sky's air along a view"),
    ("air_transmittance", "the sky's air along a view"),
    ("atmospheric", "fog and air between a point and the eye"),
    (
        "atmospheric_cover",
        "fog and air between a point and the eye",
    ),
    ("behind_clouds", "the sky behind the clouds along a view"),
    ("box_projected", "reflection probes' parallax"),
    ("cap_share", "a highlight's energy"),
    ("cloud_dome_direction", "the clouds' dome along a view"),
    ("cloud_passing", "the clouds' shadow lookup"),
    ("cover_seen", "what a view through blades meets"),
    ("crown_softness", "a shadow map lookup"),
    ("crown_spread", "a shadow map lookup"),
    ("debug_sun", "the debug view of the shadow lookups"),
    ("decal_point", "the side of a decal's surface the eye sees"),
    ("distribution", "the BRDF's microfacets"),
    ("dominant_direction", "a highlight's reflection"),
    ("environment_brdf", "the BRDF's split sum"),
    (
        "environment_light",
        "reflections of environment maps and probes",
    ),
    ("fall_cover", "what the eye sees through falling water"),
    (
        "finish",
        "lighting a surface: its BRDF, highlights and shadows",
    ),
    ("fog", "fog between a point and the eye"),
    ("fresnel", "Fresnel's reflectance"),
    ("fs_particle", "soft particles against what lies behind"),
    ("fs_particle_mist", "the view's path through mist"),
    ("grade", "grading the light that reaches the eye"),
    ("haze", "the sky's haze along a view"),
    ("height_depth", "fog and air between a point and the eye"),
    ("impostor_basis", "an impostor's frames, seen from the eye"),
    ("impostor_card", "an impostor's frames, seen from the eye"),
    (
        "impostor_direction",
        "an impostor's frames, seen from the eye",
    ),
    (
        "impostor_octahedral",
        "an impostor's frames, seen from the eye",
    ),
    ("level_visibility", "a shadow map lookup"),
    ("local_light", "lighting by local lights"),
    ("map_visibility", "a shadow map lookup"),
    ("mist_noise", "the view's path through mist"),
    ("octahedral_coordinates", "encoding a normal facing the eye"),
    ("open_sky", "the sky along a view"),
    ("plane_visibility", "a shadow map lookup"),
    ("shade_ocean_below", "the sea seen from under it"),
    ("shaped_light", "lighting by local lights"),
    ("sky", "the sky along a view"),
    ("sky_above", "the sky a surface facing the eye sees"),
    ("sky_average", "the sky a surface facing the eye sees"),
    ("sky_behind", "the sky behind fog along a view"),
    ("sky_ray", "the sky's view ray"),
    ("sky_view_uv", "the sky's air along a view"),
    ("sphere_hit", "the sky's air along a view"),
    ("sun_highlight", "the sun's highlight"),
    ("sun_visibility", "a shadow map lookup"),
    ("sunlight_over", "the sky's air along a view"),
    ("terrain_canopy", "the ground a canopy hides along the view"),
    ("terrain_canopy_look", "a canopy seen along the view"),
    ("terrain_canopy_through", "a canopy seen along the view"),
    ("terrain_parallax", "the relief's ray march"),
    ("terrain_resolved", "the depth edges of what hides what"),
    ("terrain_visibility", "the terrain's horizon lookup"),
    ("texel_slope", "a shadow map lookup"),
    ("toward_viewer", "an impostor's frames, seen from the eye"),
    ("transmitted", "light through leaves toward the eye"),
    ("transparent_targets", "refraction through glass"),
    (
        "underwater_cover",
        "the sea's water between a point and the eye",
    ),
    ("underwater_light", "the sea's water along a view"),
    ("visibility", "the BRDF's shadowing and masking"),
    ("vs_particle", "particles facing the eye"),
    ("water_behind", "refraction through water"),
    ("water_edge", "water seen against what lies under it"),
    ("water_facing", "the side of water's surface the eye sees"),
    ("water_scattered", "light the water scatters toward the eye"),
    (
        "with_night_sky",
        "the night sky along a view, its stars a pixel wide",
    ),
];

/// A line marked optics: its dependence on the camera is legitimate.
const MARK: &str = "// optics:";

/// The sources a report shows for each decision, and the calls for each.
const SHOWN: usize = 4;
const CALLS_SHOWN: usize = 3;

/// A shader module's source as the files it concatenates (each file's
/// start in the module, its path and its lines' starts within it) and the
/// lines carrying a check's mark (their starts and ends in the module).
struct Layout {
    module: &'static str,
    files: Vec<File>,
    marked: Vec<(usize, usize)>,
}

struct File {
    start: usize,
    path: &'static str,
    lines: Vec<usize>,
}

impl Layout {
    /// The file and line (from 1) holding the start of `span`.
    fn at(&self, span: Span) -> (String, usize) {
        let Some(range) = span.to_range() else {
            return (self.module.to_owned(), 0);
        };
        let index = self.files.partition_point(|file| file.start <= range.start);
        match index.checked_sub(1).and_then(|index| self.files.get(index)) {
            Some(file) => {
                let line = file
                    .lines
                    .partition_point(|&start| start <= range.start - file.start);
                (format!("{SHADERS}{}", file.path), line)
            }
            None => (self.module.to_owned(), 0),
        }
    }

    /// Whether `span` touches a marked line.
    fn marked(&self, span: Span) -> bool {
        let Some(range) = span.to_range() else {
            return false;
        };
        let end = range.end.max(range.start + 1);
        let first = self
            .marked
            .partition_point(|&(_, line_end)| line_end <= range.start);
        self.marked
            .get(first)
            .is_some_and(|&(line_start, _)| line_start < end)
    }
}

/// Every shader module's layout with the lines carrying a mark, the marked
/// lines by file and line, and those saying nothing after the mark.
struct Layouts {
    layouts: Vec<Layout>,
    lines: BTreeSet<(&'static str, usize)>,
    unexplained: Vec<String>,
}

/// Every shader module's layout, from the files `engine_render::shaders`
/// lists in the modules' order, with the lines carrying `mark`, of which
/// those saying nothing after it, which should say `says`, are problems.
fn layouts(mark: &str, says: &str) -> Layouts {
    let mut lines = BTreeSet::new();
    let mut unexplained = Vec::new();
    let mut files = engine_render::shaders::FILES.iter().copied().peekable();
    let mut layouts = Vec::new();
    for &(module, source) in engine_render::shaders::MODULES {
        let mut layout = Layout {
            module,
            files: Vec::new(),
            marked: Vec::new(),
        };
        let mut at = 0;
        while let Some(&(path, text)) = files.peek() {
            if !source.get(at..).is_some_and(|rest| rest.starts_with(text)) {
                break;
            }
            files.next();
            let mut starts = Vec::new();
            let mut offset = 0;
            for (index, line) in text.split_inclusive('\n').enumerate() {
                starts.push(offset);
                if let Some(found) = line.find(mark) {
                    layout.marked.push((at + offset, at + offset + line.len()));
                    lines.insert((path, index + 1));
                    if line[found + mark.len()..].trim().is_empty() {
                        unexplained.push(format!(
                            "{SHADERS}{path}:{}: marks a line `{mark}` without saying {says}",
                            index + 1
                        ));
                    }
                }
                offset += line.len();
            }
            layout.files.push(File {
                start: at,
                path,
                lines: starts,
            });
            at += text.len();
        }
        layouts.push(layout);
    }
    unexplained.sort();
    unexplained.dedup();
    Layouts {
        layouts,
        lines,
        unexplained,
    }
}

/// Whether the function `name` is trusted to read the camera besides the
/// measures the flow trusts (`flow::SIZES`): the way back to the world, or
/// an optics.
fn trusted(name: &str) -> bool {
    name == EYE_WORLD || OPTICS.iter().any(|&(optics, _)| optics == name)
}

/// A decision reading what a flow watches: its file, line, function and
/// kind, and what feeds it, in every module holding it.
type Found = BTreeMap<(String, usize, String, Decision), BTreeSet<String>>;

/// What following a watch through every module found: each decision it
/// reaches outside the trusted functions and the marked lines, as reports
/// show it (its file, line, function and kind, and what feeds it); the
/// problems besides (marks saying nothing, listed functions no shader
/// declares, flows that never settled); how many decisions the followed
/// functions make; and how many lines carry the mark.
pub(super) struct Swept {
    pub found: Vec<String>,
    pub problems: Vec<String>,
    pub decisions: usize,
    pub marks: usize,
}

/// Follows `watch` through every module of `modules`: the functions
/// `trusted` names are not followed, each function `listed` (in `list`) is
/// one some shader declares, and the lines carrying `mark`, which should
/// say `says`, decide nothing.
pub(super) fn sweep(
    modules: &[Checked],
    watch: Watch,
    trusted: &dyn Fn(&str) -> bool,
    listed: &[(&str, &str)],
    list: &str,
    mark: &str,
    says: &str,
) -> Swept {
    let marks = layouts(mark, says);
    let mut problems = marks.unexplained;
    let mut found = Found::new();
    let mut declared = BTreeSet::new();
    let mut decisions = 0;
    for checked in modules {
        let Some(layout) = marks
            .layouts
            .iter()
            .find(|layout| layout.module == checked.name)
        else {
            continue;
        };
        let module = &checked.module;
        declared.extend(
            module
                .functions
                .iter()
                .filter_map(|(_, function)| function.name.clone()),
        );
        declared.extend(module.entry_points.iter().map(|entry| entry.name.clone()));
        let marked = |span: Span| layout.marked(span);
        let followed = flow::follow(module, &checked.info, watch, trusted, &marked);
        decisions += followed.decisions();
        report(layout, &followed, &mut found);
        for name in &followed.unsettled {
            problems.push(format!(
                "{}: the flow of {} through `{name}` never settled",
                checked.name,
                watched(watch)
            ));
        }
    }
    for &(name, _) in listed {
        if !declared.contains(name) {
            problems.push(format!(
                "{list} lists `{name}`, which no shader declares: remove it"
            ));
        }
    }
    let found = found
        .into_iter()
        .map(|((path, line, function, kind), feeds)| {
            let shown: Vec<&str> = feeds.iter().take(SHOWN).map(String::as_str).collect();
            let mut text = format!(
                "{path}:{line}: in `{function}`, {} reads {}",
                decision(kind),
                shown.join("; ")
            );
            if feeds.len() > SHOWN {
                text.push_str(&format!("; and {} more", feeds.len() - SHOWN));
            }
            text
        })
        .collect();
    Swept {
        found,
        problems,
        decisions,
        marks: marks.lines.len(),
    }
}

/// Every module's handovers by the camera; returns how many decisions the
/// shaders make.
pub(super) fn check(modules: &[Checked]) -> Result<usize, String> {
    let swept = sweep(
        modules,
        Watch::Camera,
        &trusted,
        OPTICS,
        "tools/xtask/src/gpu/handovers.rs (OPTICS)",
        MARK,
        "why",
    );
    let mut problems = swept.problems;
    if !swept.found.is_empty() {
        problems.push(format!(
            "{} decisions read the camera outside the detail eye's measures and the optics; \
             measure detail from the detail eye (scene.wgsl's `detail_distance`, \
             `detail_across`, `detail_span`), turn eye-relative positions back with \
             `eye_world`, or, where it is optics, list the function in \
             tools/xtask/src/gpu/handovers.rs (OPTICS) or mark the line `{MARK} <why>`:",
            swept.found.len()
        ));
        problems.extend(swept.found);
    }
    if problems.is_empty() {
        Ok(swept.decisions)
    } else {
        Err(problems.join("\n"))
    }
}

/// Adds the module's decisions by what its flow watches to `found`: each
/// decision, where it lies, and what reads feed it.
fn report(layout: &Layout, flow: &Flow<'_>, found: &mut Found) {
    for (&(sink, source), via) in &flow.findings {
        let site = &flow.sinks[sink];
        let (path, line) = layout.at(site.span);
        found
            .entry((path, line, site.function.clone(), site.decision))
            .or_default()
            .insert(fed(layout, &flow.sources[source], via));
    }
}

/// A read feeding a decision through the calls `via`.
fn fed(layout: &Layout, source: &Source, via: &[Call]) -> String {
    let (path, line) = layout.at(source.span);
    let mut text = format!(
        "{} ({path}:{line}, in `{}`)",
        read(source.read),
        source.function
    );
    if !via.is_empty() {
        let calls: Vec<String> = via
            .iter()
            .take(CALLS_SHOWN)
            .map(|call| {
                let (path, line) = layout.at(call.span);
                format!("`{}` ({path}:{line})", call.function)
            })
            .collect();
        text.push_str(" through the calls in ");
        text.push_str(&calls.join(", "));
        if via.len() > CALLS_SHOWN {
            text.push_str(" and further");
        }
    }
    text
}

/// What a flow watching `watch` follows, as reports name it.
fn watched(watch: Watch) -> &'static str {
    match watch {
        Watch::Camera => "the camera",
        Watch::Sizes => "sizes on screen",
    }
}

fn decision(decision: Decision) -> &'static str {
    match decision {
        Decision::Smoothstep => "`smoothstep`",
        Decision::Step => "`step`",
        Decision::MixWeight => "`mix`'s weight",
        Decision::Select => "a `select` condition",
        Decision::If => "an `if` condition",
        Decision::Loop => "a loop's condition",
        Decision::BreakIf => "a `break if` condition",
        Decision::Switch => "a `switch` selector",
        Decision::Number => "a comparison turned into a number",
        Decision::Saturate => "`saturate`",
    }
}

fn read(read: Read) -> &'static str {
    match read {
        Read::Eye => "the camera's eye (`camera.position`)",
        Read::Detail => "the detail eye outside the measures (`camera.detail`)",
        Read::PixelAngle => "a pixel's angle outside the measures (`camera.options.y`)",
        Read::ViewAxis => "a camera matrix's eye or axis",
        Read::Relative => "an eye-relative position (`relative`)",
        Read::Measure => {
            "a size the detail eye measures (`detail_distance`, `detail_across` or `detail_span`)"
        }
        Read::Derivative => "a pixel's derivatives (`dpdx`, `dpdy` or `fwidth`)",
    }
}
