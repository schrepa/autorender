use std::fmt::Write as _;
use std::sync::Mutex;

use engine_core::{
    AlphaMode, BitValue, Error, ErrorCode, GpuBits, LightingModel, Material, MaterialHandle,
    Module, STANDARD_MAPS, SurfaceSlot, UvTransform, gpu_bits,
};

use crate::bindings::{Sample, Sampling, Stages, Table, View, sampler, texture, uniform_of};
use crate::pipeline_layouts::{PipelineLayout, REQUIRED_LIMITS};

pub const MATERIALS_MODULE: &str = "engine::materials";
pub const SLOTS_MODULE: &str = "engine::slots";
pub const SURFACES_MODULE: &str = "engine::surfaces";
pub const RECORD: &str = "material_record";
pub const MASK_MIP_SCALE: f32 = 0.25;
const MODULE_ROWS: usize = SurfaceSlot::ALL.len().div_ceil(4);
const MAPS_AT: usize = 4 * (3 + MODULE_ROWS);

#[gpu_bits]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TraceDepth {
    Solid = 0,
    Thin = 1,
    Surface = 2,
    Opaque = 3,
}

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
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DrawState {
    pub alpha_test: bool,
    pub fading: bool,
}

impl DrawState {
    pub(crate) fn constants(self) -> [(&'static str, f64); 1] {
        [("DRAW_STATE", f64::from(self.pack()))]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SurfaceKind {
    pub name: &'static str,
    pub depth: TraceDepth,
}

impl SurfaceKind {
    pub const SOLID: Self = Self {
        name: "solid",
        depth: TraceDepth::Solid,
    };
    pub const THIN: Self = Self {
        name: "thin",
        depth: TraceDepth::Thin,
    };
}

#[derive(Clone, Copy, Debug)]
pub struct SlotModule {
    pub slot: SurfaceSlot,
    pub name: &'static str,
    pub module: Module,
    pub maps: &'static [&'static str],
    pub values: &'static [&'static str],
}

#[derive(Clone, Copy, Debug)]
pub struct LightingModule {
    pub name: &'static str,
    pub module: Module,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct MaterialLayout {
    maps: Vec<&'static str>,
    values: Vec<&'static str>,
    slots: Vec<(SurfaceSlot, &'static str, String)>,
    models: Vec<(&'static str, String)>,
    kinds: Vec<SurfaceKind>,
}

fn upper(name: &str) -> String {
    name.to_ascii_uppercase()
}

fn wrong_name(path: String, name: &str) -> Error {
    Error::new(ErrorCode::NameTwice, path).got(name)
}

impl MaterialLayout {
    pub(crate) fn standard() -> Self {
        Self {
            maps: STANDARD_MAPS.to_vec(),
            values: Vec::new(),
            slots: Vec::new(),
            models: Vec::new(),
            kinds: vec![SurfaceKind::SOLID, SurfaceKind::THIN],
        }
    }

    pub(crate) fn of(
        slots: &[(SlotModule, String)],
        models: &[(LightingModule, String)],
        kinds: &[SurfaceKind],
        problems: &mut Vec<Error>,
    ) -> Self {
        let mut layout = Self::standard();
        for (declared, path) in slots {
            if layout
                .slots
                .iter()
                .any(|(slot, name, _)| *slot == declared.slot && *name == declared.name)
            {
                problems.push(wrong_name(
                    format!("$.material_slots.{}.{}", declared.slot.name(), declared.name),
                    declared.name,
                ));
                continue;
            }
            for map in declared.maps {
                if !layout.maps.contains(map) {
                    layout.maps.push(map);
                }
            }
            for value in declared.values {
                if !layout.values.contains(value) {
                    layout.values.push(value);
                }
            }
            layout
                .slots
                .push((declared.slot, declared.name, path.clone()));
        }
        for (declared, path) in models {
            let core = [Lighting::Lit, Lighting::Unlit]
                .iter()
                .any(|model| snake(model) == declared.name);
            if core || layout.models.iter().any(|(name, _)| *name == declared.name) {
                problems.push(wrong_name(
                    format!("$.lighting_models.{}", declared.name),
                    declared.name,
                ));
                continue;
            }
            layout.models.push((declared.name, path.clone()));
        }
        for kind in kinds {
            if layout.kinds.iter().any(|known| known.name == kind.name) {
                problems.push(wrong_name(format!("$.surface_kinds.{}", kind.name), kind.name));
                continue;
            }
            layout.kinds.push(*kind);
        }
        layout
    }

    fn value_vec4s(&self) -> usize {
        self.values.len().div_ceil(4).max(1)
    }

    pub(crate) fn record_bytes(&self) -> usize {
        (3 + MODULE_ROWS + 2 * self.maps.len() + self.value_vec4s()) * 16
    }

    pub(crate) fn maps(&self) -> &[&'static str] {
        &self.maps
    }

    pub(crate) fn slot_module(&self, slot: SurfaceSlot, name: &str) -> Option<u32> {
        self.slots
            .iter()
            .filter(|(known, _, _)| *known == slot)
            .position(|(_, known, _)| *known == name)
            .map(|at| at as u32 + 1)
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
        let mut binding = 0;
        let declare_map = |source: &mut String, name: &str, binding: &mut u32| {
            let _ = writeln!(
                source,
                "@group(1) @binding({}) var {name}_map: texture_2d<f32>;",
                *binding
            );
            let _ = writeln!(
                source,
                "@group(1) @binding({}) var {name}_sampler: sampler;",
                *binding + 1
            );
            *binding += 2;
        };
        for name in &self.maps[..STANDARD_MAPS.len()] {
            declare_map(&mut source, name, &mut binding);
        }
        let _ = writeln!(
            source,
            "@group(1) @binding({binding}) var<uniform> {RECORD}: MaterialRecord;"
        );
        binding += 1;
        for name in &self.maps[STANDARD_MAPS.len()..] {
            declare_map(&mut source, name, &mut binding);
        }
        let _ = writeln!(
            source,
            "\nstruct MaterialRecord {{\n    emissive: vec4<f32>,\n    params: vec4<f32>,\n    choice: vec4<u32>,\n    modules: array<vec4<u32>, {MODULE_ROWS}>,\n    maps: array<vec4<f32>, {}>,\n    values: array<vec4<f32>, {}>,\n}}\n",
            2 * self.maps.len(),
            self.value_vec4s()
        );
        source.push_str("override DRAW_STATE: u32 = 0u;\n");
        source.push_str(<DrawState as GpuBits>::WGSL);
        source.push_str(<Alpha as BitValue>::WGSL);
        source.push_str(<Lighting as BitValue>::WGSL);
        let _ = writeln!(source, "const MASK_MIP_SCALE: f32 = {:?};", f64::from(MASK_MIP_SCALE));
        let _ = writeln!(
            source,
            "const SURFACE_EMISSION: f32 = {:?};",
            f64::from(engine_core::Surface::MAX_EMISSION)
        );
        for slot in SurfaceSlot::ALL {
            let _ = writeln!(
                source,
                "const SLOT_{}: u32 = {}u;",
                upper(slot.name()),
                slot.index()
            );
        }
        for (index, name) in self.maps.iter().enumerate() {
            let _ = writeln!(source, "const MAP_{}: u32 = {index}u;", upper(name));
        }
        for (index, name) in self.values.iter().enumerate() {
            let _ = writeln!(source, "const VALUE_{}: u32 = {index}u;", upper(name));
        }
        let _ = write!(
            source,
            "
fn alpha_test() -> bool {{
    return draw_state_alpha_test(DRAW_STATE);
}}

fn fade_list() -> bool {{
    return draw_state_fading(DRAW_STATE);
}}

fn map_present(index: u32) -> bool {{
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
    return {RECORD}.choice.x;
}}

fn material_alpha() -> u32 {{
    return {RECORD}.choice.y;
}}

fn alpha_cutoff() -> f32 {{
    return {RECORD}.params.z;
}}

fn normal_scale() -> f32 {{
    return {RECORD}.params.x;
}}

fn occlusion_strength() -> f32 {{
    return {RECORD}.params.y;
}}

fn emissive_factor() -> vec3<f32> {{
    return {RECORD}.emissive.rgb;
}}

fn transmission_factor() -> f32 {{
    return {RECORD}.emissive.w;
}}
"
        );
        source
    }

    pub(crate) fn slots_source(&self, full: bool) -> String {
        let mut imports = String::from(
            "import package::engine::shading::{Shaded, SurfaceInput, SurfacePoint, unlit};\n",
        );
        let mut body = String::new();
        for slot in SurfaceSlot::ALL {
            let name = slot.name();
            let (returns, params, args) = match slot {
                SurfaceSlot::BaseColor | SurfaceSlot::Normal | SurfaceSlot::Emissive => {
                    ("vec3<f32>", "input: SurfaceInput, point: SurfacePoint", "input, point")
                }
                SurfaceSlot::Opacity
                | SurfaceSlot::Roughness
                | SurfaceSlot::Metalness
                | SurfaceSlot::Occlusion
                | SurfaceSlot::Transmission => {
                    ("f32", "input: SurfaceInput, point: SurfacePoint", "input, point")
                }
                SurfaceSlot::Output => (
                    "Shaded",
                    "input: SurfaceInput, point: SurfacePoint, shaded: Shaded",
                    "input, point, shaded",
                ),
            };
            let _ = writeln!(
                imports,
                "import package::engine::standard::{name} as standard_{name};"
            );
            let _ = writeln!(
                body,
                "\nfn slot_{name}({params}) -> {returns} {{"
            );
            let modules: Vec<&(SurfaceSlot, &'static str, String)> = self
                .slots
                .iter()
                .filter(|(known, _, _)| *known == slot)
                .collect();
            if full && !modules.is_empty() {
                let _ = writeln!(
                    body,
                    "    let module = material_module(SLOT_{});",
                    upper(name)
                );
                for (index, (_, _, path)) in modules.iter().enumerate() {
                    let id = index + 1;
                    let _ = writeln!(
                        imports,
                        "import package::{path}::{name} as {name}_{id};"
                    );
                    let _ = writeln!(
                        body,
                        "    if module == {id}u {{\n        return {name}_{id}({args});\n    }}"
                    );
                }
            }
            let _ = writeln!(body, "    return standard_{name}({args});\n}}");
        }
        if full {
            imports.push_str("import package::face::lit::{finish, local_light};\n");
            body.push_str(
                "\nfn shade_model(point: SurfacePoint, pixel: vec2<f32>) -> Shaded {\n    let model = material_model();\n    if model == LIGHTING_UNLIT {\n        return unlit(point);\n    }\n",
            );
            for (index, (_, path)) in self.models.iter().enumerate() {
                let id = index + 2;
                let _ = writeln!(imports, "import package::{path}::finish as model_{id};");
                let _ = writeln!(
                    body,
                    "    if model == {id}u {{\n        return model_{id}(point, local_light(point, pixel));\n    }}"
                );
            }
            body.push_str("    return finish(point, local_light(point, pixel));\n}\n");
        }
        let mut used = vec!["material_module", "material_model", "LIGHTING_UNLIT"];
        used.extend(SurfaceSlot::ALL.map(|slot| match slot {
            SurfaceSlot::BaseColor => "SLOT_BASE_COLOR",
            SurfaceSlot::Opacity => "SLOT_OPACITY",
            SurfaceSlot::Normal => "SLOT_NORMAL",
            SurfaceSlot::Roughness => "SLOT_ROUGHNESS",
            SurfaceSlot::Metalness => "SLOT_METALNESS",
            SurfaceSlot::Emissive => "SLOT_EMISSIVE",
            SurfaceSlot::Occlusion => "SLOT_OCCLUSION",
            SurfaceSlot::Transmission => "SLOT_TRANSMISSION",
            SurfaceSlot::Output => "SLOT_OUTPUT",
        }));
        let named: Vec<&str> = used
            .into_iter()
            .filter(|name| body.contains(name))
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

    pub(crate) fn surfaces_source(&self) -> String {
        let mut source = String::from(<TraceDepth as BitValue>::WGSL);
        for (index, kind) in self.kinds.iter().enumerate() {
            let _ = writeln!(source, "const SURFACE_{}: u32 = {index}u;", upper(kind.name));
        }
        source.push_str("\nfn surface_depth(kind: u32) -> u32 {\n");
        for (index, kind) in self.kinds.iter().enumerate().skip(1) {
            let _ = writeln!(
                source,
                "    if kind == {index}u {{\n        return {}u;\n    }}",
                kind.depth.bits()
            );
        }
        let _ = writeln!(source, "    return {}u;\n}}", self.kinds[0].depth.bits());
        source
    }

    pub(crate) fn table(&self) -> (&'static Table, PipelineLayout) {
        static TABLES: Mutex<Vec<(Vec<&'static str>, usize, &'static Table, PipelineLayout)>> =
            Mutex::new(Vec::new());
        let size = self.record_bytes();
        let mut tables = TABLES
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some((_, _, table, layout)) = tables
            .iter()
            .find(|(maps, bytes, _, _)| *maps == self.maps && *bytes == size)
        {
            return (table, *layout);
        }
        let number = tables.len();
        let leak = |text: String| -> &'static str { Box::leak(text.into_boxed_str()) };
        let mut bindings = Vec::with_capacity(2 * self.maps.len() + 1);
        let add_map = |bindings: &mut Vec<_>, name: &str| {
            bindings.push(texture(
                leak(format!("{name}_map")),
                Sample::Float,
                View::D2,
                Stages::FRAGMENT,
            ));
            bindings.push(sampler(
                leak(format!("{name}_sampler")),
                Sampling::Filtering,
                Stages::FRAGMENT,
            ));
        };
        for name in &self.maps[..STANDARD_MAPS.len()] {
            add_map(&mut bindings, name);
        }
        bindings.push(uniform_of(RECORD, size, Stages::FRAGMENT));
        for name in &self.maps[STANDARD_MAPS.len()..] {
            add_map(&mut bindings, name);
        }
        let table: &'static Table = Box::leak(Box::new(Table {
            label: leak(format!("engine/material-layout-{number}")),
            bindings: Box::leak(bindings.into_boxed_slice()),
        }));
        let groups: &'static [&'static Table] = Box::leak(Box::new([
            &crate::bindings::EMPTY,
            table,
            &crate::bindings::IMPOSTOR_BAKE,
        ]));
        let layout = PipelineLayout {
            label: leak(format!("engine/impostor-bake-pipelines-{number}")),
            groups,
            limits: REQUIRED_LIMITS,
        };
        tables.push((self.maps.clone(), size, table, layout));
        (table, layout)
    }

    pub(crate) fn record(&self, handle: MaterialHandle, material: &Material) -> (Vec<u32>, Vec<Error>) {
        let path = format!("$.materials.{}", handle.slot());
        let mut problems = Vec::new();
        let mut words = vec![0_u32; self.record_bytes() / 4];
        let [r, g, b] = material.emissive();
        words[..4].copy_from_slice(&[r, g, b, material.transmission()].map(f32::to_bits));
        let state = material.render_state();
        let (alpha, cutoff) = match state.alpha {
            AlphaMode::Opaque => (Alpha::Opaque, 0.5),
            AlphaMode::Mask { cutoff } => (Alpha::Mask, cutoff),
            AlphaMode::Blend => (Alpha::Blend, 0.5),
            AlphaMode::Additive => (Alpha::Additive, 0.5),
        };
        words[4..8].copy_from_slice(
            &[
                material.normal_scale(),
                material.occlusion_strength(),
                cutoff,
                0.0,
            ]
            .map(f32::to_bits),
        );
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
            match self.slot_module(*slot, module) {
                Some(id) => words[12 + slot.index()] = id,
                None => problems.push(
                    Error::new(
                        ErrorCode::UnknownVariable,
                        format!("{path}.slots.{}", slot.name()),
                    )
                    .got(module.as_str()),
                ),
            }
        }
        for (name, map) in material.maps() {
            let Some(index) = self.maps.iter().position(|known| known == name) else {
                problems.push(
                    Error::new(ErrorCode::UnknownVariable, format!("{path}.maps.{name}"))
                        .got(name.as_str()),
                );
                continue;
            };
            let [first, second] = transform_rows(&map.transform);
            let at = MAPS_AT + index * 8;
            words[at..at + 4].copy_from_slice(&[first[0], first[1], first[2], 1.0].map(f32::to_bits));
            words[at + 4..at + 8].copy_from_slice(
                &[second[0], second[1], second[2], map.tex_coord as f32].map(f32::to_bits),
            );
        }
        let values_at = MAPS_AT + self.maps.len() * 8;
        for (name, value) in material.values() {
            if let Some(index) = self.values.iter().position(|known| known == name) {
                words[values_at + index] = value.to_bits();
            }
        }
        (words, problems)
    }

    pub(crate) fn plain_record(&self) -> Vec<u32> {
        let mut words = vec![0_u32; self.record_bytes() / 4];
        words[4..8].copy_from_slice(&[1.0_f32, 1.0, 0.5, 0.0].map(f32::to_bits));
        words
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
    DrawState {
        alpha_test: material.is_some_and(|material| material.alpha_mode().masked()),
        fading: false,
    }
    .pack()
}
