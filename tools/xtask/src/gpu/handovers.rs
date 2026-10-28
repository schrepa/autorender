use std::collections::{BTreeMap, BTreeSet};

use naga::Span;

use super::Checked;

mod flow;

pub(super) use flow::Watch;
use flow::{Call, Decision, Flow, Read, Source};

const SHADERS: &str = "crates/engine_render/src/";

const EYE_WORLD: &str = "eye_world";

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
    (
        "finish",
        "lighting a surface: its BRDF, highlights and shadows",
    ),
    (
        "finish_reflecting",
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
    ("plane_visibility", "a shadow map lookup"),
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

const MARK: &str = "// optics:";

const SHOWN: usize = 4;
const CALLS_SHOWN: usize = 3;

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

struct Layouts {
    layouts: Vec<Layout>,
    lines: BTreeSet<(&'static str, usize)>,
    unexplained: Vec<String>,
}

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

fn trusted(name: &str) -> bool {
    name == EYE_WORLD || OPTICS.iter().any(|&(optics, _)| optics == name)
}

type Found = BTreeMap<(String, usize, String, Decision), BTreeSet<String>>;

pub(super) struct Swept {
    pub found: Vec<String>,
    pub problems: Vec<String>,
    pub decisions: usize,
    pub marks: usize,
}

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
