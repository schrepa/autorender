pub const PREPARE_ENTRY: &str = "prepare";
pub const DERIVE_ENTRY: &str = "derive";

impl DrawState {
    pub const PLAIN: Self = Self {
        alpha_test: false,
        fixed: false,
        alpha: Alpha::Opaque,
        no_depth_test: false,
        no_depth_write: false,
    };
}

type LaidOpen = (PointInput, String, &'static str);

    pub(crate) open: Vec<(OpenSlot, String, &'static str)>,

    pub(crate) sets: Vec<(PointSet, &'static str)>,
    pub(crate) terms: Vec<(SurfaceTerms, String)>,

    pub(crate) prepares: Vec<String>,
    pub(crate) kinds: Vec<SurfaceKind>,

fn set_by(brick: &str, set: PointSet) -> Value {
    json!({ "brick": brick, "type": set.ty })
}

        layout.lay_open(&declared.open, problems);

        layout.lay_terms(&declared.terms, problems);
        layout.lay_sets(declared, problems);

        for kind in &declared.kinds {
            if layout.kinds.iter().any(|known| known.name == kind.name) {
                problems.push(wrong_name(
                    format!("$.surface_kinds.{}", kind.name),
                    kind.name,
                ));
                continue;
            }
            layout.kinds.push(*kind);
        }

    fn lay_open(&mut self, open: &[(OpenSlot, String, &'static str)], problems: &mut Vec<Error>) {
        for (declared, path, brick) in open {
            let name = declared.input.name;
            let at = format!("$.open_slots.{name}");
            if let Some((_, _, first)) = self.open.iter().find(|(input, ..)| input.name == name) {
                problems.push(Error::new(ErrorCode::NameTwice, at).got([*first, *brick]));
                continue;
            }
            if SurfaceSlot::named(name).is_some() || self.point_field(name) {
                problems.push(wrong_name(at, name));
                continue;
            }
            self.open.push((declared.input, path.clone(), *brick));
            for value in declared.values {
                self.value(value.name, value.standard, Some(value.rule));
            }
        }
    }

    fn lay_terms(&mut self, terms: &[(SurfaceTerms, String)], problems: &mut Vec<Error>) {
        for (declared, path) in terms {
            for &(name, ty) in declared.fields {
                if self.slot_index(name).is_some() || self.point_field(name) {
                    problems.push(wrong_name(format!("$.surface_terms.{name}"), name));
                    continue;
                }
                self.terms.push((name, ty));
            }
            self.derives.push(path.clone());
        }
    }

    fn lay_sets(&mut self, declared: &SurfaceDecls, problems: &mut Vec<Error>) {
        let mut sets: Vec<(PointSet, Vec<&'static str>)> = Vec::new();
        for &(set, brick) in &declared.sets {
            match sets.iter().position(|(known, _)| known.name == set.name) {
                Some(at) if sets[at].0.ty != set.ty => {
                    let first = set_by(sets[at].1[0], sets[at].0);
                    let at = format!("$.point_sets.{}", set.name);
                    problems.push(disagree(at, first, set_by(brick, set)));
                }
                Some(at) => {
                    if !sets[at].1.contains(&brick) {
                        sets[at].1.push(brick);
                    }
                }
                None => sets.push((set, vec![brick])),
            }
        }
        for (set, by) in sets {
            let at = format!("$.point_sets.{}", set.name);
            match self.input(set.name) {
                Some((read, reader)) if read.ty != set.ty => {
                    problems.push(disagree(at, read_by(reader, read), set_by(by[0], set)));
                }
                Some(_) => self.sets.push((set, true)),
                None if SurfaceSlot::named(set.name).is_some() || self.point_field(set.name) => {
                    problems.push(wrong_name(at, set.name));
                }
                None => {
                    let notice = Error::new(ErrorCode::InputUnread, at).got(&by);
                    self.notices.push(notice);
                    self.sets.push((set, false));
                }
            }
        }
    }

        for (index, path) in self.derives.iter().enumerate() {
            let id = index + 1;
            let _ = writeln!(
                imports,
                "import package::{path}::{DERIVE_ENTRY} as derive_{id};"
            );
            let _ = writeln!(body, "    result = derive_{id}(result);");
        }

        for &(PointSet { name, ty }, read) in &self.sets {
            let _ = writeln!(
                body,
                "\nfn set_{name}(point: SurfacePoint, value: {ty}) -> SurfacePoint {{"
            );
            if read {
                let _ = writeln!(
                    body,
                    "    var result = point;\n    result.{name} = value;\n    return result;"
                );
            } else {
                body.push_str("    return point;\n");
            }
            body.push_str("}\n");
        }

                    SurfaceSlot::Position => ("vec3<f32>", "input: PositionInput", "input"),

        signed.extend(self.open.iter().map(|(input, path, _)| {
            (input.name, input.ty, surface, "input, point", path.as_str())
        }));

                    if let Some(at) = self.prepares.iter().position(|known| known == path) {
                        let term = format!("material_module(SLOT_{}) == {id}u", upper(name));
                        prepared.push((at, term));
                    }

        for (at, path) in self.prepares.iter().enumerate() {
            let picked: Vec<&str> = prepared
                .iter()
                .filter(|(known, _)| *known == at)
                .map(|(_, term)| term.as_str())
                .collect();
            if picked.is_empty() {
                continue;
            }
            let id = at + 1;
            let _ = writeln!(
                imports,
                "import package::{path}::{PREPARE_ENTRY} as prepare_{id};"
            );
            let _ = writeln!(
                body,
                "    if {} {{\n        prepare_{id}(input, point);\n    }}",
                picked.join("\n        || ")
            );
        }

        for (input, ..) in &self.open {
            let name = input.name;
            let _ = writeln!(body, "    filled.{name} = slot_{name}(input, filled);");
        }

fn map_gradient(index: u32, d0: vec2<f32>, d1: vec2<f32>) -> vec2<f32> {{
    let second = {RECORD}.maps[index * 2u + 1u].w != 0.0;
    let change = vec3<f32>(select(d0, d1, second), 0.0);
    return vec2<f32>(dot({RECORD}.maps[index * 2u].xyz, change),
        dot({RECORD}.maps[index * 2u + 1u].xyz, change));
}}

fn material_alpha() -> u32 {{
    if fixed_material() {{
        return draw_state_alpha(DRAW_STATE);
    }}
    return {RECORD}.choice.y;
}}

#[derive(Clone, Copy, Debug)]
pub struct SlotPrepare {
    pub module: Module,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaterialValue {
    pub name: &'static str,
    pub standard: f32,
    pub rule: Rule,
}

#[derive(Clone, Copy, Debug)]
pub struct OpenSlot {
    pub input: PointInput,
    pub standard: Module,
    pub values: &'static [MaterialValue],
}

#[derive(Clone, Copy, Debug)]
pub struct SurfaceTerms {
    pub fields: &'static [(&'static str, &'static str)],
    pub derive: Module,
}

pub fn material_state(material: Option<&Material>) -> u32 {
    material
        .map_or(DrawState::PLAIN, |material| {
            draw_of(material.render_state())
        })
        .pack()
}
