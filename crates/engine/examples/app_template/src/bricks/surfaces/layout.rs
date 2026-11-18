use std::fmt::Write as _;
use std::sync::{Mutex, PoisonError};

use engine_core::{
    BitValue, Error, ErrorCode, GpuBits, LightingModel, MAPPED, MAPPED_SLOTS, Material, Module,
    Rule, ShaderText, Source, SurfaceSlot, Var, gpu_bits,
};
use engine_render::{Entries, Made};
use serde_json::{Value, json};

use super::PointInput;

pub(super) const SHADING_MODULE: &str = "surfaces::shading";
pub(super) const MATERIALS_MODULE: &str = "surfaces::materials";
pub(super) const SLOTS_MODULE: &str = "surfaces::slots";
pub(super) const SURFACES_MODULE: &str = "surfaces::surfaces";
pub(super) const WRITES_MODULE: &str = "surfaces::writes";
const MAPPED_MODULE: &str = "surfaces::mapped";
const STANDARD_MODULE: &str = "surfaces::standard";
pub(super) const RECORD: &str = "material_record";
const LIGHTING_ENTRY: &str = "lit";
pub(super) const MASK_MIP_SCALE: f32 = 0.25;

static GENERATED: Source = Source::generated(
    super::SOURCE.package(),
    super::SOURCE.game(),
    super::SOURCE.hash(),
    &[ShaderText::generated(MATERIALS_MODULE, "", "")],
);

pub(super) const MATERIALS: Module = Module::generated(&GENERATED, 0);
pub(super) const RECORD_VAR: Var = Var::generated(MATERIALS, RECORD);

const WRITES: &str = "import package::surfaces::shading::{Shaded, SurfaceInput, SurfacePoint};\nimport package::surfaces::slots::{shade, slot_output};\n\nstruct Outputs {\n    @location(0) color: vec4<f32>,\n}\n\nfn outputs(point: SurfacePoint, shaded: Shaded) -> Outputs {\n    var written: Outputs;\n    written.color = vec4<f32>(shaded.linear, 1.0);\n    return written;\n}\n\nfn blended_outputs(point: SurfacePoint, shaded: Shaded) -> Outputs {\n    var written = outputs(point, shaded);\n    written.color = vec4<f32>(shaded.linear * shaded.opacity, shaded.opacity);\n    return written;\n}\n\nfn slot_shaded(input: SurfaceInput, point: SurfacePoint, pixel: vec2<f32>) -> Shaded {\n    return slot_output(input, point, shade(point, pixel));\n}\n";

#[gpu_bits]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum Alpha {
    Opaque = 0,
    Mask = 1,
    Blend = 2,
    Additive = 3,
}

#[gpu_bits]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum Lighting {
    Unlit = 0,
}

#[gpu_bits]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct DrawState {
    pub(super) alpha_test: bool,
    pub(super) fixed: bool,
    #[bits(2)]
    pub(super) alpha: Alpha,
    pub(super) no_depth_test: bool,
    pub(super) no_depth_write: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SurfaceKind {
    name: &'static str,
}

impl SurfaceKind {
    const SOLID: Self = Self { name: "solid" };
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
    pub form: ModelForm,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelForm {
    Shades,
}

impl ModelForm {
    const fn entry(self) -> &'static str {
        match self {
            Self::Shades => LIGHTING_ENTRY,
        }
    }
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

pub(super) type LaidSlot = (&'static str, &'static str, String, &'static [&'static str]);

#[derive(Default)]
struct SurfaceDecls {
    slots: Vec<(SlotModule, String, &'static str)>,
    inputs: Vec<(PointInput, &'static str)>,
    models: Vec<(LightingModule, String)>,
}

impl SurfaceDecls {
    fn of(entries: &Entries) -> Self {
        let mut slots = mapped_slots();
        slots.extend(
            entries
                .of::<SlotModule>()
                .map(|(package, slot)| (*slot, path_of(slot.module).to_owned(), package)),
        );
        Self {
            slots,
            inputs: entries
                .of::<PointInput>()
                .map(|(package, input)| (*input, package))
                .collect(),
            models: entries
                .of::<LightingModule>()
                .map(|(_, model)| (*model, path_of(model.module).to_owned()))
                .collect(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct LaidValue {
    pub(super) name: &'static str,
    pub(super) standard: f32,
    pub(super) rule: Option<Rule>,
}

pub(super) type MapVars = (&'static str, Var, Var);

#[derive(Clone, Debug)]
pub(super) struct SurfaceLayout {
    pub(super) maps: Vec<&'static str>,
    pub(super) values: Vec<LaidValue>,
    pub(super) inputs: Vec<(PointInput, &'static str)>,
    pub(super) slots: Vec<LaidSlot>,
    pub(super) skipped: Vec<LaidSlot>,
    pub(super) models: Vec<(&'static str, String, ModelForm)>,
    kinds: Vec<SurfaceKind>,
    map_vars: Vec<MapVars>,
    problems: Vec<Error>,
    notices: Vec<Error>,
}

pub(super) fn made(entries: &Entries) -> Made {
    let declared = SurfaceDecls::of(entries);
    let layout = SurfaceLayout::laid(&declared);
    let mut sources = vec![MATERIALS];
    let mut keep = Vec::new();
    for (slot, path, _) in &declared.slots {
        sources.push(slot.module);
        keep.push((path.clone(), slot.slot.to_owned()));
    }
    for (model, path) in &declared.models {
        sources.push(model.module);
        keep.push((path.clone(), model.form.entry().to_owned()));
    }
    let modules = vec![
        (SHADING_MODULE.to_owned(), layout.shading_source()),
        (MATERIALS_MODULE.to_owned(), layout.materials_source()),
        (SLOTS_MODULE.to_owned(), layout.slots_source()),
        (SURFACES_MODULE.to_owned(), layout.surfaces_source()),
        (WRITES_MODULE.to_owned(), WRITES.to_owned()),
    ];
    Made {
        sources,
        modules,
        keep,
        problems: layout.problems,
        notices: layout.notices,
    }
}

fn path_of(module: Module) -> &'static str {
    module
        .source()
        .modules()
        .get(module.index() as usize)
        .map_or("", ShaderText::path)
}

fn mapped_slots() -> Vec<(SlotModule, String, &'static str)> {
    let module = super::shaders::surfaces::mapped::MODULE;
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
                super::SOURCE.package(),
            )
        })
        .collect()
}

fn interned(name: String) -> &'static str {
    static NAMES: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());
    let mut names = NAMES.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(known) = names.iter().copied().find(|known| *known == name) {
        return known;
    }
    let name: &'static str = Box::leak(name.into_boxed_str());
    names.push(name);
    name
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

fn disagree(path: String, first: Value, second: Value) -> Error {
    Error::new(ErrorCode::UsageConflict, path).got([first, second])
}

impl SurfaceLayout {
    pub(super) fn of(entries: &Entries) -> Self {
        Self::laid(&SurfaceDecls::of(entries))
    }

    pub(super) fn map_vars(&self) -> &[MapVars] {
        &self.map_vars
    }

    fn point_field(&self, name: &str) -> bool {
        POINT_FIELDS.iter().any(|(known, _)| *known == name) || self.input(name).is_some()
    }

    fn input(&self, name: &str) -> Option<(PointInput, &'static str)> {
        self.inputs
            .iter()
            .copied()
            .find(|(input, _)| input.name == name)
    }

    fn fields(&self) -> impl Iterator<Item = PointInput> + '_ {
        self.inputs.iter().map(|(input, _)| *input)
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

    fn laid(declared: &SurfaceDecls) -> Self {
        let mut layout = Self {
            maps: Vec::new(),
            values: Vec::new(),
            inputs: Vec::new(),
            slots: Vec::new(),
            skipped: Vec::new(),
            models: Vec::new(),
            kinds: vec![SurfaceKind::SOLID],
            map_vars: Vec::new(),
            problems: Vec::new(),
            notices: Vec::new(),
        };
        let mut problems = Vec::new();
        layout.lay_inputs(&declared.inputs, &mut problems);
        layout.lay_slots(declared, &mut problems);
        layout.lay_models(&declared.models, &mut problems);
        layout.problems = problems;
        layout.map_vars = layout
            .maps
            .iter()
            .map(|&map| {
                (
                    map,
                    Var::generated(MATERIALS, interned(format!("{map}_map"))),
                    Var::generated(MATERIALS, interned(format!("{map}_sampler"))),
                )
            })
            .collect();
        layout
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
            let notice = Error::new(ErrorCode::SlotUnopened, at)
                .got(fillers)
                .variants(self.slot_names());
            self.notices.push(notice);
        }
    }

    fn lay_models(&mut self, models: &[(LightingModule, String)], problems: &mut Vec<Error>) {
        for (declared, path) in models {
            let core = snake(&Lighting::Unlit) == declared.name;
            if core || self.models.iter().any(|(name, ..)| *name == declared.name) {
                problems.push(wrong_name(
                    format!("$.lighting_models.{}", declared.name),
                    declared.name,
                ));
                continue;
            }
            self.models
                .push((declared.name, path.clone(), declared.form));
        }
    }

    fn value_vec4s(&self) -> usize {
        self.values.len().div_ceil(4).max(1)
    }

    pub(super) fn slot_names(&self) -> impl Iterator<Item = &'static str> + '_ {
        SurfaceSlot::ALL.into_iter().map(SurfaceSlot::name)
    }

    pub(super) fn slot_index(&self, slot: &str) -> Option<usize> {
        SurfaceSlot::named(slot).map(SurfaceSlot::index)
    }

    fn slot_count(&self) -> usize {
        SurfaceSlot::ALL.len()
    }

    fn module_rows(&self) -> usize {
        self.slot_count().div_ceil(4)
    }

    pub(super) fn maps_at(&self) -> usize {
        4 * (3 + self.module_rows())
    }

    pub(super) fn record_bytes(&self) -> usize {
        (3 + self.module_rows() + 2 * self.maps.len() + self.value_vec4s()) * 16
    }

    pub(super) fn maps(&self) -> &[&'static str] {
        &self.maps
    }

    pub(super) fn slot_module(&self, slot: &str, name: &str) -> Option<u32> {
        self.slots
            .iter()
            .filter(|(known, _, _, _)| *known == slot)
            .position(|(_, known, _, _)| *known == name)
            .map(|at| at as u32 + 1)
    }

    pub(super) fn read_maps<'a>(
        &'a self,
        material: &'a Material,
    ) -> impl Iterator<Item = &'static str> + 'a {
        material.slots().iter().flat_map(move |(slot, module)| {
            self.slots
                .iter()
                .filter(move |(known, name, _, _)| {
                    *known == slot.as_str() && *name == module.as_str()
                })
                .flat_map(|(_, _, _, maps)| maps.iter().copied())
        })
    }

    pub(super) fn reads(&self, material: &Material, map: &str) -> bool {
        self.read_maps(material).any(|read| read == map)
    }

    pub(super) fn skipped_slot(&self, slot: &str) -> bool {
        self.skipped.iter().any(|(known, ..)| *known == slot)
    }

    pub(super) fn skipped_reads(&self, material: &Material, map: &str) -> bool {
        material.slots().iter().any(|(slot, module)| {
            self.skipped.iter().any(|(known, name, _, maps)| {
                *known == slot.as_str() && *name == module.as_str() && maps.contains(&map)
            })
        })
    }

    pub(super) fn model(&self, lighting: &LightingModel) -> Option<u32> {
        match lighting {
            LightingModel::Unlit => Some(Lighting::Unlit.bits()),
            LightingModel::Named(name) => self
                .models
                .iter()
                .position(|(known, ..)| known == name)
                .map(|at| at as u32 + 1),
        }
    }

    fn materials_source(&self) -> String {
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
        body.push_str("    return result;\n}\n");
        body.push_str("\nfn standard_point() -> SurfacePoint {\n    var point: SurfacePoint;\n");
        for input in self.fields() {
            let _ = writeln!(body, "    point.{} = {};", input.name, input.standard);
        }
        body.push_str("    return point;\n}\n");
        let _ = writeln!(imports, "import package::{SURFACES_MODULE}::SURFACE_SOLID;");
        body.push_str(
            "\nfn flat_point(color: vec4<f32>) -> SurfacePoint {\n    var point = standard_point();\n    point.base_color = color.rgb;\n    point.opacity = color.a;\n    point.normal = vec3<f32>(0.0, 0.0, 1.0);\n    point.geometric = point.normal;\n    point.view = point.normal;\n    point.kind = SURFACE_SOLID;\n    return derived(point);\n}\n",
        );
    }

    fn model_functions(&self, imports: &mut String, body: &mut String) {
        body.push_str(
            "\nfn shade(point: SurfacePoint, pixel: vec2<f32>) -> Shaded {\n    var shaded = modeled(point, pixel);\n    shaded.opacity = point.opacity;\n    return shaded;\n}\n",
        );
        body.push_str("\nfn modeled(point: SurfacePoint, pixel: vec2<f32>) -> Shaded {\n");
        for (index, (_, path, form)) in self.models.iter().enumerate() {
            let id = index + 1;
            let entry = form.entry();
            let _ = writeln!(imports, "import package::{path}::{entry} as model_{id};");
            let _ = writeln!(
                body,
                "    if point.lighting == {id}u {{\n        return model_{id}(point, pixel);\n    }}"
            );
        }
        body.push_str("    return unlit(point);\n}\n");
    }

    fn slots_source(&self) -> String {
        let mut imports = format!(
            "import package::{SHADING_MODULE}::{{Shaded, SurfaceInput, SurfacePoint, unlit}};\n"
        );
        let mut body = String::new();
        let surface = "input: SurfaceInput, point: SurfacePoint";
        let signed = SurfaceSlot::ALL.into_iter().filter_map(|slot| {
            let (returns, params, args) = match slot {
                SurfaceSlot::Position => return None,
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
            Some((slot.name(), returns, params, args))
        });
        for (name, returns, params, args) in signed {
            let _ = writeln!(
                imports,
                "import package::{STANDARD_MODULE}::{name} as standard_{name};"
            );
            let _ = writeln!(body, "\nfn slot_{name}({params}) -> {returns} {{");
            let modules: Vec<&LaidSlot> = self
                .slots
                .iter()
                .filter(|(known, _, _, _)| *known == name)
                .collect();
            if !modules.is_empty() {
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
        body.push_str("\nfn prepare_slots(input: SurfaceInput, point: SurfacePoint) {\n}\n");
        body.push_str(
            "\nfn open_slots(input: SurfaceInput, point: SurfacePoint) -> SurfacePoint {\n    var filled = point;\n    return filled;\n}\n",
        );
        self.point_functions(&mut imports, &mut body);
        self.model_functions(&mut imports, &mut body);
        let mut used = vec!["material_module".to_owned()];
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
                "import package::{MATERIALS_MODULE}::{{{}}};",
                named.join(", ")
            );
        }
        format!("{imports}{body}")
    }

    fn shading_source(&self) -> String {
        let mut source = String::from(include_str!("shading.wgsl"));
        source.push_str("\nstruct SurfacePoint {\n");
        let inputs = self.fields().map(|input| (input.name, input.ty));
        for (name, ty) in POINT_FIELDS.iter().copied().chain(inputs) {
            let _ = writeln!(source, "    {name}: {ty},");
        }
        source.push_str("}\n");
        source
    }

    fn surfaces_source(&self) -> String {
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
}

fn snake(model: &Lighting) -> &'static str {
    match model {
        Lighting::Unlit => "unlit",
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
