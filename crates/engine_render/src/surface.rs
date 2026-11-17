use std::fmt::Write as _;

use engine_core::{
    AlphaMode, BitValue, Error, ErrorCode, GpuBits, LightingModel, MAPPED, MAPPED_SLOTS, Material,
    MaterialHandle, Module, Rule, Setting, SurfaceSlot, UvTransform, gpu_bits,
};
use engine_place::{PointInput, PointSet};
use serde_json::{Value, json};

use crate::linker::{CORE_PACKAGE, Declaration, Ty};

pub const SHADING_MODULE: &str = "engine::shading";
pub const MATERIALS_MODULE: &str = "engine::materials";
pub const SLOTS_MODULE: &str = "engine::slots";
pub const SURFACES_MODULE: &str = "engine::surfaces";
pub const MAPPED_MODULE: &str = "engine::mapped";
pub const STANDARD_MODULE: &str = "engine::standard";
pub const RECORD: &str = "material_record";
pub const LIGHTING_ENTRY: &str = "lit";
pub const DERIVE_ENTRY: &str = "derive";
pub const MASK_MIP_SCALE: f32 = 0.25;

#[gpu_bits]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Alpha {
    Opaque = 0,
    Mask = 1,
    Blend = 2,
    Additive = 3,
}

#[gpu_bits]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Lighting {
    Lit = 0,
    Unlit = 1,
}

#[gpu_bits]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DrawState {
    pub alpha_test: bool,
    pub fixed: bool,
    #[bits(2)]
    pub alpha: Alpha,
    pub no_depth_test: bool,
    pub no_depth_write: bool,
}

impl DrawState {
    pub const PLAIN: Self = Self {
        alpha_test: false,
        fixed: false,
        alpha: Alpha::Opaque,
        no_depth_test: false,
        no_depth_write: false,
    };
}

#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum FaceCull {
    #[default]
    Back,
    Front,
    None,
}

impl FaceCull {
    pub(crate) fn of(cull: engine_core::Cull) -> Self {
        match cull {
            engine_core::Cull::Back => Self::Back,
            engine_core::Cull::Front => Self::Front,
            engine_core::Cull::None => Self::None,
        }
    }

    pub(crate) fn face(self) -> Option<wgpu::Face> {
        match self {
            Self::Back => Some(wgpu::Face::Back),
            Self::Front => Some(wgpu::Face::Front),
            Self::None => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct MaterialVariant {
    pub state: u32,
    pub bias: [i32; 2],
    pub maps: u32,
    pub model: u32,
    pub cull: FaceCull,
}

impl MaterialVariant {
    pub const PLAIN: Self = Self {
        state: 0,
        bias: [0, 0],
        maps: 0,
        model: 0,
        cull: FaceCull::Back,
    };

    pub fn draw(self) -> DrawState {
        <DrawState as GpuBits>::unpack(self.state).unwrap_or(DrawState::PLAIN)
    }

    #[must_use]
    pub fn with_draw(self, draw: DrawState) -> Self {
        Self {
            state: draw.pack(),
            ..self
        }
    }

    pub fn blended(self) -> bool {
        matches!(self.draw().alpha, Alpha::Blend | Alpha::Additive)
    }

    pub(crate) fn constants(self) -> [(&'static str, f64); 3] {
        [
            ("DRAW_STATE", f64::from(self.state)),
            ("MATERIAL_MAPS", f64::from(self.maps)),
            ("MATERIAL_MODEL", f64::from(self.model)),
        ]
    }

    pub(crate) fn dynamic(self) -> Option<Self> {
        let draw = self.draw();
        draw.fixed.then(|| Self {
            state: DrawState {
                fixed: false,
                ..draw
            }
            .pack(),
            maps: 0,
            model: 0,
            ..self
        })
    }

    pub(crate) fn keyed(self, used: [bool; 3], styled: bool, faced: bool) -> Option<Self> {
        let [stated, mapped, modeled] = used;
        if !used.contains(&true) && !styled && !faced {
            return None;
        }
        let draw = self.draw();
        let state = DrawState {
            alpha_test: (stated || faced) && draw.alpha_test,
            fixed: stated && draw.fixed,
            alpha: if stated || styled {
                draw.alpha
            } else {
                Alpha::Opaque
            },
            no_depth_test: styled && draw.no_depth_test,
            no_depth_write: styled && draw.no_depth_write,
        };
        Some(Self {
            state: state.pack(),
            bias: if styled { self.bias } else { [0, 0] },
            maps: if mapped { self.maps } else { 0 },
            model: if modeled { self.model } else { 0 },
            cull: if faced { self.cull } else { FaceCull::Back },
        })
    }

    pub(crate) fn restyle(self, state: &mut wgpu::DepthStencilState) {
        let draw = self.draw();
        if draw.no_depth_write {
            state.depth_write_enabled = Some(false);
        }
        if draw.no_depth_test {
            state.depth_compare = Some(wgpu::CompareFunction::Always);
        }
        if self.bias != [0, 0] {
            state.bias = wgpu::DepthBiasState {
                constant: self.bias[0],
                slope_scale: f32::from_bits(self.bias[1] as u32),
                clamp: 0.0,
            };
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SurfaceKind {
    pub name: &'static str,
}

impl SurfaceKind {
    pub const SOLID: Self = Self { name: "solid" };
}

#[derive(Clone, Copy, Debug)]
pub struct SlotModule {
    pub slot: &'static str,
    pub name: &'static str,
    pub module: Module,
    pub maps: &'static [&'static str],
    pub values: &'static [&'static str],
    pub reach: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct LightingModule {
    pub name: &'static str,
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

const POINT_FIELDS: [(&str, &str); 15] = [
    ("base_color", "vec3<f32>"),
    ("opacity", "f32"),
    ("emissive", "vec3<f32>"),
    ("normal", "vec3<f32>"),
    ("normal_dx", "vec3<f32>"),
    ("normal_dy", "vec3<f32>"),
    ("geometric", "vec3<f32>"),
    ("view", "vec3<f32>"),
    ("n_v", "f32"),
    ("world", "vec3<f32>"),
    ("world_dx", "vec3<f32>"),
    ("world_dy", "vec3<f32>"),
    ("distance", "f32"),
    ("kind", "u32"),
    ("lighting", "u32"),
];

type LaidSlot = (&'static str, &'static str, String, &'static [&'static str]);

type LaidOpen = (PointInput, String, &'static str);

#[derive(Clone, Copy, Debug)]
pub(crate) struct LeftInput {
    pub(crate) brick: &'static str,
    pub(crate) name: &'static str,
    pub(crate) opens: bool,
}

#[derive(Default)]
pub(crate) struct SurfaceDecls {
    pub(crate) slots: Vec<(SlotModule, String, &'static str)>,
    pub(crate) open: Vec<(OpenSlot, String, &'static str)>,
    pub(crate) inputs: Vec<(PointInput, &'static str)>,
    pub(crate) sets: Vec<(PointSet, &'static str)>,
    pub(crate) terms: Vec<(SurfaceTerms, String)>,
    pub(crate) models: Vec<(LightingModule, String)>,
    pub(crate) kinds: Vec<SurfaceKind>,
    pub(crate) left: Vec<LeftInput>,
}

impl SurfaceDecls {
    fn left_of(&self, name: &str, opens: bool) -> Vec<&'static str> {
        let mut bricks = Vec::new();
        for left in &self.left {
            if left.name == name && (left.opens || !opens) && !bricks.contains(&left.brick) {
                bricks.push(left.brick);
            }
        }
        bricks
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct LaidValue {
    name: &'static str,
    standard: f32,
    rule: Option<Rule>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct MaterialLayout {
    maps: Vec<&'static str>,
    values: Vec<LaidValue>,
    open: Vec<LaidOpen>,
    inputs: Vec<(PointInput, &'static str)>,
    sets: Vec<(PointSet, bool)>,
    terms: Vec<(&'static str, &'static str)>,
    derives: Vec<String>,
    slots: Vec<LaidSlot>,
    skipped: Vec<LaidSlot>,
    models: Vec<(&'static str, String)>,
    kinds: Vec<SurfaceKind>,
    notices: Vec<Error>,
}

pub(crate) fn core_slots() -> Vec<(SlotModule, String, &'static str)> {
    let module = crate::shaders::core_module(MAPPED_MODULE);
    MAPPED_SLOTS
        .iter()
        .filter(|(slot, _)| SurfaceSlot::named(slot).is_some())
        .map(|&(slot, maps)| {
            (
                SlotModule {
                    slot,
                    name: MAPPED,
                    module,
                    maps,
                    values: &[],
                    reach: 0.0,
                },
                MAPPED_MODULE.to_owned(),
                CORE_PACKAGE,
            )
        })
        .collect()
}

fn upper(name: &str) -> String {
    name.to_ascii_uppercase()
}

fn wrong_name(path: String, name: &str) -> Error {
    Error::new(ErrorCode::NameTwice, path).got(name)
}

fn read_by(brick: &str, input: PointInput) -> Value {
    json!({ "brick": brick, "type": input.ty, "standard": input.standard })
}

fn set_by(brick: &str, set: PointSet) -> Value {
    json!({ "brick": brick, "type": set.ty })
}

fn disagree(path: String, first: Value, second: Value) -> Error {
    Error::new(ErrorCode::UsageConflict, path).got([first, second])
}

impl MaterialLayout {
    pub(crate) fn standard() -> Self {
        let declared = SurfaceDecls {
            slots: core_slots(),
            ..SurfaceDecls::default()
        };
        Self::of(&declared, &mut Vec::new())
    }

    pub(crate) fn notices(&self) -> &[Error] {
        &self.notices
    }

    fn point_field(&self, name: &str) -> bool {
        POINT_FIELDS.iter().any(|(known, _)| *known == name)
            || self.input(name).is_some()
            || self.terms.iter().any(|(known, _)| *known == name)
    }

    fn input(&self, name: &str) -> Option<(PointInput, &'static str)> {
        self.open
            .iter()
            .map(|(input, _, brick)| (*input, *brick))
            .chain(self.inputs.iter().copied())
            .find(|(input, _)| input.name == name)
    }

    fn fields(&self) -> impl Iterator<Item = PointInput> + '_ {
        self.open
            .iter()
            .map(|(input, ..)| *input)
            .chain(self.inputs.iter().map(|(input, _)| *input))
    }

    fn value(&mut self, name: &'static str, standard: f32, rule: Option<Rule>) {
        if !self.values.iter().any(|known| known.name == name) {
            self.values.push(LaidValue {
                name,
                standard,
                rule,
            });
        }
    }

    pub(crate) fn of(declared: &SurfaceDecls, problems: &mut Vec<Error>) -> Self {
        let mut layout = Self {
            maps: Vec::new(),
            values: Vec::new(),
            open: Vec::new(),
            inputs: Vec::new(),
            sets: Vec::new(),
            terms: Vec::new(),
            derives: Vec::new(),
            slots: Vec::new(),
            skipped: Vec::new(),
            models: Vec::new(),
            kinds: vec![SurfaceKind::SOLID],
            notices: Vec::new(),
        };
        layout.lay_open(&declared.open, problems);
        layout.lay_inputs(&declared.inputs, problems);
        layout.lay_terms(&declared.terms, problems);
        layout.lay_sets(declared, problems);
        layout.lay_slots(declared, problems);
        layout.lay_models(&declared.models, problems);
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
        layout
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

    fn lay_inputs(&mut self, inputs: &[(PointInput, &'static str)], problems: &mut Vec<Error>) {
        for &(input, brick) in inputs {
            let at = format!("$.point_inputs.{}", input.name);
            if let Some((known, first)) = self.input(input.name) {
                if known != input {
                    problems.push(disagree(at, read_by(first, known), read_by(brick, input)));
                }
                continue;
            }
            if SurfaceSlot::named(input.name).is_some() || self.point_field(input.name) {
                problems.push(wrong_name(at, input.name));
                continue;
            }
            self.inputs.push((input, brick));
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
                    let notice = Error::new(ErrorCode::InputUnread, at)
                        .got(&by)
                        .nearest(declared.left_of(set.name, false));
                    self.notices.push(notice);
                    self.sets.push((set, false));
                }
            }
        }
    }

    fn lay_slots(&mut self, declared: &SurfaceDecls, problems: &mut Vec<Error>) {
        let mut unopened: Vec<(&'static str, Vec<Value>)> = Vec::new();
        for (slot, path, brick) in &declared.slots {
            if self.slot_index(slot.slot).is_none() {
                let filler = json!({ "brick": brick, "module": slot.name });
                match unopened.iter().position(|(name, _)| *name == slot.slot) {
                    Some(at) => unopened[at].1.push(filler),
                    None => unopened.push((slot.slot, vec![filler])),
                }
                self.skipped
                    .push((slot.slot, slot.name, path.clone(), slot.maps));
                continue;
            }
            if self
                .slots
                .iter()
                .any(|(known, name, _, _)| *known == slot.slot && *name == slot.name)
            {
                problems.push(wrong_name(
                    format!("$.material_slots.{}.{}", slot.slot, slot.name),
                    slot.name,
                ));
                continue;
            }
            for map in slot.maps {
                if !self.maps.contains(map) {
                    self.maps.push(map);
                }
            }
            for &value in slot.values {
                self.value(value, 0.0, None);
            }
            self.slots
                .push((slot.slot, slot.name, path.clone(), slot.maps));
        }
        for (slot, fillers) in unopened {
            let at = format!("$.material_slots.{slot}");
            let notice = Error::new(ErrorCode::SlotUnopened, at).got(fillers);
            let openers = declared.left_of(slot, true);
            let notice = if openers.is_empty() {
                notice.variants(self.slot_names())
            } else {
                notice.nearest(openers)
            };
            self.notices.push(notice);
        }
    }

    fn lay_models(&mut self, models: &[(LightingModule, String)], problems: &mut Vec<Error>) {
        for (declared, path) in models {
            let core = [Lighting::Lit, Lighting::Unlit]
                .iter()
                .any(|model| snake(model) == declared.name);
            if core || self.models.iter().any(|(name, _)| *name == declared.name) {
                problems.push(wrong_name(
                    format!("$.lighting_models.{}", declared.name),
                    declared.name,
                ));
                continue;
            }
            self.models.push((declared.name, path.clone()));
        }
    }

    fn value_vec4s(&self) -> usize {
        self.values.len().div_ceil(4).max(1)
    }

    fn slot_names(&self) -> impl Iterator<Item = &'static str> + '_ {
        SurfaceSlot::ALL
            .into_iter()
            .map(SurfaceSlot::name)
            .chain(self.open.iter().map(|(input, ..)| input.name))
    }

    fn slot_index(&self, slot: &str) -> Option<usize> {
        SurfaceSlot::named(slot)
            .map(SurfaceSlot::index)
            .or_else(|| {
                self.open
                    .iter()
                    .position(|(input, ..)| input.name == slot)
                    .map(|at| SurfaceSlot::ALL.len() + at)
            })
    }

    fn slot_count(&self) -> usize {
        SurfaceSlot::ALL.len() + self.open.len()
    }

    fn module_rows(&self) -> usize {
        self.slot_count().div_ceil(4)
    }

    fn maps_at(&self) -> usize {
        4 * (3 + self.module_rows())
    }

    pub(crate) fn record_bytes(&self) -> usize {
        (3 + self.module_rows() + 2 * self.maps.len() + self.value_vec4s()) * 16
    }

    pub(crate) fn maps(&self) -> &[&'static str] {
        &self.maps
    }

    pub(crate) fn slot_module(&self, slot: &str, name: &str) -> Option<u32> {
        self.slots
            .iter()
            .filter(|(known, _, _, _)| *known == slot)
            .position(|(_, known, _, _)| *known == name)
            .map(|at| at as u32 + 1)
    }

    fn read_maps<'a>(&'a self, material: &'a Material) -> impl Iterator<Item = &'static str> + 'a {
        material.slots().iter().flat_map(move |(slot, module)| {
            self.slots
                .iter()
                .filter(move |(known, name, _, _)| {
                    *known == slot.as_str() && *name == module.as_str()
                })
                .flat_map(|(_, _, _, maps)| maps.iter().copied())
        })
    }

    pub(crate) fn reads(&self, material: &Material, map: &str) -> bool {
        self.read_maps(material).any(|read| read == map)
    }

    fn skipped_slot(&self, slot: &str) -> bool {
        self.skipped.iter().any(|(known, ..)| *known == slot)
    }

    fn skipped_reads(&self, material: &Material, map: &str) -> bool {
        material.slots().iter().any(|(slot, module)| {
            self.skipped.iter().any(|(known, name, _, maps)| {
                *known == slot.as_str() && *name == module.as_str() && maps.contains(&map)
            })
        })
    }

    pub(crate) fn model(&self, lighting: &LightingModel) -> Option<u32> {
        match lighting {
            LightingModel::Lit => Some(Lighting::Lit.bits()),
            LightingModel::Unlit => Some(Lighting::Unlit.bits()),
            LightingModel::Named(name) => self
                .models
                .iter()
                .position(|(known, _)| known == name)
                .map(|at| at as u32 + 2),
        }
    }

    pub(crate) fn materials_source(&self) -> String {
        let mut source = String::new();
        let _ = writeln!(
            source,
            "@group(1) @binding(0) var<uniform> {RECORD}: MaterialRecord;"
        );
        for (index, name) in self.maps.iter().enumerate() {
            let binding = 1 + 2 * index;
            let _ = writeln!(
                source,
                "@group(1) @binding({binding}) var {name}_map: texture_2d<f32>;"
            );
            let _ = writeln!(
                source,
                "@group(1) @binding({}) var {name}_sampler: sampler;",
                binding + 1
            );
        }
        let _ = writeln!(
            source,
            "\nstruct MaterialRecord {{\n    emissive: vec4<f32>,\n    params: vec4<f32>,\n    choice: vec4<u32>,\n    modules: array<vec4<u32>, {}>,\n    maps: array<vec4<f32>, {}>,\n    values: array<vec4<f32>, {}>,\n}}\n",
            self.module_rows(),
            2 * self.maps.len(),
            self.value_vec4s()
        );
        for name in ["DRAW_STATE", "MATERIAL_MAPS", "MATERIAL_MODEL"] {
            let _ = writeln!(source, "override {name}: u32 = 0u;");
        }
        source.push_str(<DrawState as GpuBits>::WGSL);
        source.push_str(<Alpha as BitValue>::WGSL);
        source.push_str(<Lighting as BitValue>::WGSL);
        let _ = writeln!(
            source,
            "const MASK_MIP_SCALE: f32 = {:?};",
            f64::from(MASK_MIP_SCALE)
        );
        let _ = writeln!(
            source,
            "const SURFACE_EMISSION: f32 = {:?};",
            f64::from(Material::MAX_EMISSION)
        );
        for (index, name) in self.slot_names().enumerate() {
            let _ = writeln!(source, "const SLOT_{}: u32 = {index}u;", upper(name));
        }
        for (index, name) in self.maps.iter().enumerate() {
            let _ = writeln!(source, "const MAP_{}: u32 = {index}u;", upper(name));
        }
        for (index, value) in self.values.iter().enumerate() {
            let _ = writeln!(source, "const VALUE_{}: u32 = {index}u;", upper(value.name));
        }
        material_functions(&mut source);
        source
    }

    fn point_functions(&self, imports: &mut String, body: &mut String) {
        body.push_str(
            "\nfn derived(point: SurfacePoint) -> SurfacePoint {\n    var result = point;\n    result.n_v = max(dot(point.normal, point.view), 1.0e-4);\n",
        );
        for (index, path) in self.derives.iter().enumerate() {
            let id = index + 1;
            let _ = writeln!(
                imports,
                "import package::{path}::{DERIVE_ENTRY} as derive_{id};"
            );
            let _ = writeln!(body, "    result = derive_{id}(result);");
        }
        body.push_str("    return result;\n}\n");
        body.push_str("\nfn standard_point() -> SurfacePoint {\n    var point: SurfacePoint;\n");
        for input in self.fields() {
            let _ = writeln!(body, "    point.{} = {};", input.name, input.standard);
        }
        body.push_str("    return point;\n}\n");
        let _ = writeln!(imports, "import package::engine::surfaces::SURFACE_SOLID;");
        body.push_str(
            "\nfn flat_point(color: vec4<f32>) -> SurfacePoint {\n    var point = standard_point();\n    point.base_color = color.rgb;\n    point.opacity = color.a;\n    point.normal = vec3<f32>(0.0, 0.0, 1.0);\n    point.geometric = point.normal;\n    point.view = point.normal;\n    point.kind = SURFACE_SOLID;\n    return derived(point);\n}\n",
        );
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
    }

    pub(crate) fn slots_source(&self, full: bool) -> String {
        let mut imports = String::from(
            "import package::engine::shading::{PositionInput, Shaded, SurfaceInput, SurfacePoint, unlit};\n",
        );
        let mut body = String::new();
        let surface = "input: SurfaceInput, point: SurfacePoint";
        let mut signed: Vec<(&'static str, &'static str, &str, &str, &str)> = SurfaceSlot::ALL
            .into_iter()
            .map(|slot| {
                let (returns, params, args) = match slot {
                    SurfaceSlot::Position => ("vec3<f32>", "input: PositionInput", "input"),
                    SurfaceSlot::BaseColor | SurfaceSlot::Normal | SurfaceSlot::Emissive => {
                        ("vec3<f32>", surface, "input, point")
                    }
                    SurfaceSlot::Opacity | SurfaceSlot::Cutoff => ("f32", surface, "input, point"),
                    SurfaceSlot::Kind => ("u32", surface, "input, point"),
                    SurfaceSlot::Output => (
                        "Shaded",
                        "input: SurfaceInput, point: SurfacePoint, shaded: Shaded",
                        "input, point, shaded",
                    ),
                };
                (slot.name(), returns, params, args, STANDARD_MODULE)
            })
            .collect();
        signed.extend(self.open.iter().map(|(input, path, _)| {
            (input.name, input.ty, surface, "input, point", path.as_str())
        }));
        for (name, returns, params, args, standard) in signed {
            let _ = writeln!(
                imports,
                "import package::{standard}::{name} as standard_{name};"
            );
            let _ = writeln!(body, "\nfn slot_{name}({params}) -> {returns} {{");
            let modules: Vec<&LaidSlot> = self
                .slots
                .iter()
                .filter(|(known, _, _, _)| *known == name)
                .collect();
            if full && !modules.is_empty() {
                let _ = writeln!(
                    body,
                    "    let picked = material_module(SLOT_{});",
                    upper(name)
                );
                for (index, (_, _, path, _)) in modules.iter().enumerate() {
                    let id = index + 1;
                    let _ = writeln!(imports, "import package::{path}::{name} as {name}_{id};");
                    let _ = writeln!(
                        body,
                        "    if picked == {id}u {{\n        return {name}_{id}({args});\n    }}"
                    );
                }
            }
            let _ = writeln!(body, "    return standard_{name}({args});\n}}");
        }
        body.push_str(
            "\nfn open_slots(input: SurfaceInput, point: SurfacePoint) -> SurfacePoint {\n    var filled = point;\n",
        );
        for (input, ..) in &self.open {
            let name = input.name;
            let _ = writeln!(body, "    filled.{name} = slot_{name}(input, filled);");
        }
        body.push_str("    return filled;\n}\n");
        self.point_functions(&mut imports, &mut body);
        if full {
            let _ = writeln!(imports, "import package::face::lit::{LIGHTING_ENTRY};");
            body.push_str(
                "\nfn shade(point: SurfacePoint, pixel: vec2<f32>) -> Shaded {\n    var shaded = modeled(point, pixel);\n    shaded.opacity = point.opacity;\n    return shaded;\n}\n",
            );
            body.push_str(
                "\nfn modeled(point: SurfacePoint, pixel: vec2<f32>) -> Shaded {\n    if point.lighting == LIGHTING_UNLIT {\n        return unlit(point);\n    }\n",
            );
            for (index, (_, path)) in self.models.iter().enumerate() {
                let id = index + 2;
                let _ = writeln!(
                    imports,
                    "import package::{path}::{LIGHTING_ENTRY} as model_{id};"
                );
                let _ = writeln!(
                    body,
                    "    if point.lighting == {id}u {{\n        return model_{id}(point, pixel);\n    }}"
                );
            }
            let _ = writeln!(body, "    return {LIGHTING_ENTRY}(point, pixel);\n}}");
        }
        let mut used = vec!["material_module".to_owned(), "LIGHTING_UNLIT".to_owned()];
        used.extend(
            self.slot_names()
                .map(|name| format!("SLOT_{}", upper(name))),
        );
        let named: Vec<String> = used
            .into_iter()
            .filter(|name| body.contains(name.as_str()))
            .collect();
        if !named.is_empty() {
            let _ = writeln!(
                imports,
                "import package::engine::materials::{{{}}};",
                named.join(", ")
            );
        }
        format!("{imports}{body}")
    }

    pub(crate) fn shading_source(&self) -> String {
        let mut source = String::from(include_str!("shading.wgsl"));
        source.push_str("\nstruct SurfacePoint {\n");
        let inputs = self.fields().map(|input| (input.name, input.ty));
        for (name, ty) in POINT_FIELDS
            .iter()
            .copied()
            .chain(inputs)
            .chain(self.terms.iter().copied())
        {
            let _ = writeln!(source, "    {name}: {ty},");
        }
        source.push_str("}\n");
        source
    }

    pub(crate) fn surfaces_source(&self) -> String {
        let mut source = String::new();
        for (index, kind) in self.kinds.iter().enumerate() {
            let _ = writeln!(
                source,
                "const SURFACE_{}: u32 = {index}u;",
                upper(kind.name)
            );
        }
        source
    }

    pub(crate) fn record(
        &self,
        handle: MaterialHandle,
        material: &Material,
    ) -> (Vec<u32>, Vec<Error>) {
        let path = format!("$.materials.{}", handle.slot());
        let mut problems = Vec::new();
        let mut words = vec![0_u32; self.record_bytes() / 4];
        let [r, g, b] = material.emissive();
        words[..4].copy_from_slice(&[r, g, b, 0.0].map(f32::to_bits));
        let (alpha, cutoff) = alpha_of(material.alpha_mode());
        words[4..8].copy_from_slice(&[material.normal_scale(), 0.0, cutoff, 0.0].map(f32::to_bits));
        let model = self.model(material.lighting()).unwrap_or_else(|| {
            problems.push(
                Error::new(ErrorCode::UnknownVariable, format!("{path}.lighting"))
                    .got(format!("{:?}", material.lighting())),
            );
            Lighting::Lit.bits()
        });
        words[8] = model;
        words[9] = alpha.bits();
        for (slot, module) in material.slots() {
            let Some(index) = self.slot_index(slot) else {
                if self.skipped_slot(slot) {
                    continue;
                }
                problems.push(
                    Error::new(ErrorCode::UnknownVariable, format!("{path}.slots.{slot}"))
                        .got(slot.as_str())
                        .variants(self.slot_names()),
                );
                continue;
            };
            match self.slot_module(slot, module) {
                Some(id) => words[12 + index] = id,
                None => problems.push(
                    Error::new(ErrorCode::UnknownVariable, format!("{path}.slots.{slot}"))
                        .got(module.as_str()),
                ),
            }
        }
        for (name, map) in material.maps() {
            let index = self
                .maps
                .iter()
                .position(|known| known == name)
                .filter(|_| self.reads(material, name));
            let Some(index) = index else {
                if self.skipped_reads(material, name) {
                    continue;
                }
                problems.push(
                    Error::new(ErrorCode::UnknownVariable, format!("{path}.maps.{name}"))
                        .got(name.as_str())
                        .variants(self.read_maps(material)),
                );
                continue;
            };
            let [first, second] = transform_rows(&map.transform);
            let at = self.maps_at() + index * 8;
            words[at..at + 4]
                .copy_from_slice(&[first[0], first[1], first[2], 1.0].map(f32::to_bits));
            words[at + 4..at + 8].copy_from_slice(
                &[second[0], second[1], second[2], map.tex_coord as f32].map(f32::to_bits),
            );
        }
        let values_at = self.maps_at() + self.maps.len() * 8;
        for (index, declared) in self.values.iter().enumerate() {
            let value = material.value(declared.name).unwrap_or(declared.standard);
            if let Some(rule) = declared.rule
                && let Err(invalid) = value.check(declared.name, rule)
            {
                problems.push(invalid.error(ErrorCode::InvalidMaterial, &format!("{path}.values")));
            }
            words[values_at + index] = value.to_bits();
        }
        (words, problems)
    }

    pub(crate) fn plain_record(&self) -> Vec<u32> {
        let mut words = vec![0_u32; self.record_bytes() / 4];
        words[4..8].copy_from_slice(&[1.0_f32, 0.0, 0.5, 0.0].map(f32::to_bits));
        let values_at = self.maps_at() + self.maps.len() * 8;
        for (index, declared) in self.values.iter().enumerate() {
            words[values_at + index] = declared.standard.to_bits();
        }
        words
    }

    pub(crate) fn variant(&self, material: Option<&Material>, words: &[u32]) -> MaterialVariant {
        let fixed = self.maps.len() <= u32::BITS as usize;
        let maps_at = self.maps_at();
        let mut maps = 0_u32;
        for index in 0..self.maps.len().min(u32::BITS as usize) {
            if f32::from_bits(words[maps_at + index * 8 + 3]) != 0.0 {
                maps |= 1 << index;
            }
        }
        let state = material.map_or_else(engine_core::RenderState::default, Material::render_state);
        let draw = DrawState {
            fixed,
            ..draw_of(state)
        };
        MaterialVariant {
            state: draw.pack(),
            bias: [state.bias.constant, state.bias.slope.to_bits() as i32],
            maps: if fixed { maps } else { 0 },
            model: if fixed { words[8] } else { 0 },
            cull: FaceCull::of(state.cull),
        }
    }
}

fn alpha_of(mode: AlphaMode) -> (Alpha, f32) {
    match mode {
        AlphaMode::Opaque => (Alpha::Opaque, 0.5),
        AlphaMode::Mask { cutoff } => (Alpha::Mask, cutoff),
        AlphaMode::Blend => (Alpha::Blend, 0.5),
        AlphaMode::Additive => (Alpha::Additive, 0.5),
    }
}

fn draw_of(state: engine_core::RenderState) -> DrawState {
    DrawState {
        alpha_test: state.alpha.masked(),
        fixed: false,
        alpha: alpha_of(state.alpha).0,
        no_depth_test: !state.depth_test,
        no_depth_write: !state.depth_write,
    }
}

fn snake(model: &Lighting) -> &'static str {
    match model {
        Lighting::Lit => "lit",
        Lighting::Unlit => "unlit",
    }
}

fn transform_rows(transform: &UvTransform) -> [[f32; 3]; 2] {
    let (sin, cos) = engine_core::math::sin_cos(transform.rotation);
    let [sx, sy] = transform.scale;
    let [ox, oy] = transform.offset;
    [[cos * sx, sin * sy, ox], [-sin * sx, cos * sy, oy]]
}

pub fn material_state(material: Option<&Material>) -> u32 {
    material
        .map_or(DrawState::PLAIN, |material| {
            draw_of(material.render_state())
        })
        .pack()
}

pub(crate) fn material_entry(declared: &Declaration) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding: declared.binding,
        visibility: if declared.ty == Ty::Uniform {
            wgpu::ShaderStages::VERTEX_FRAGMENT
        } else {
            wgpu::ShaderStages::FRAGMENT
        },
        ty: declared.ty.binding(true, false),
        count: None,
    }
}

fn material_functions(source: &mut String) {
    let _ = write!(
        source,
        "
fn alpha_test() -> bool {{
    return draw_state_alpha_test(DRAW_STATE);
}}

fn fixed_material() -> bool {{
    return draw_state_fixed(DRAW_STATE);
}}

fn map_present(index: u32) -> bool {{
    if fixed_material() {{
        return ((MATERIAL_MAPS >> index) & 1u) != 0u;
    }}
    return {RECORD}.maps[index * 2u].w != 0.0;
}}

fn map_uv(index: u32, uv0: vec2<f32>, uv1: vec2<f32>) -> vec2<f32> {{
    let second = {RECORD}.maps[index * 2u + 1u].w != 0.0;
    let point = vec3<f32>(select(uv0, uv1, second), 1.0);
    return vec2<f32>(dot({RECORD}.maps[index * 2u].xyz, point),
        dot({RECORD}.maps[index * 2u + 1u].xyz, point));
}}

fn map_gradient(index: u32, d0: vec2<f32>, d1: vec2<f32>) -> vec2<f32> {{
    let second = {RECORD}.maps[index * 2u + 1u].w != 0.0;
    let change = vec3<f32>(select(d0, d1, second), 0.0);
    return vec2<f32>(dot({RECORD}.maps[index * 2u].xyz, change),
        dot({RECORD}.maps[index * 2u + 1u].xyz, change));
}}

fn material_value(index: u32) -> f32 {{
    return {RECORD}.values[index / 4u][index % 4u];
}}

fn material_module(slot: u32) -> u32 {{
    return {RECORD}.modules[slot / 4u][slot % 4u];
}}

fn material_model() -> u32 {{
    if fixed_material() {{
        return MATERIAL_MODEL;
    }}
    return {RECORD}.choice.x;
}}

fn material_alpha() -> u32 {{
    if fixed_material() {{
        return draw_state_alpha(DRAW_STATE);
    }}
    return {RECORD}.choice.y;
}}

fn alpha_cutoff() -> f32 {{
    return {RECORD}.params.z;
}}

fn normal_scale() -> f32 {{
    return {RECORD}.params.x;
}}

fn emissive_factor() -> vec3<f32> {{
    return {RECORD}.emissive.rgb;
}}
"
    );
}
