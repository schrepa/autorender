use std::fmt::Write as _;

use engine_core::{
    AlphaMode, BitValue, Error, ErrorCode, GpuBits, LightingModel, MAPPED, MAPPED_SLOTS, Material,
    MaterialHandle, Module, SurfaceSlot, UvTransform, gpu_bits,
};

use crate::linker::{Declaration, Ty};

pub const MATERIALS_MODULE: &str = "engine::materials";
pub const SLOTS_MODULE: &str = "engine::slots";
pub const SURFACES_MODULE: &str = "engine::surfaces";
pub const MAPPED_MODULE: &str = "engine::mapped";
pub const RECORD: &str = "material_record";
pub const MASK_MIP_SCALE: f32 = 0.25;
const MODULE_ROWS: usize = SurfaceSlot::ALL.len().div_ceil(4);
const MAPS_AT: usize = 4 * (3 + MODULE_ROWS);

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

const SLOT_BITS: u32 = 4;
const SLOTS_PER_WORD: usize = (u32::BITS / SLOT_BITS) as usize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct MaterialVariant {
    pub state: u32,
    pub bias: [i32; 2],
    pub maps: u32,
    pub model: u32,
    pub slots: [u32; 2],
    pub cull: FaceCull,
}

impl MaterialVariant {
    pub const PLAIN: Self = Self {
        state: 0,
        bias: [0, 0],
        maps: 0,
        model: 0,
        slots: [0, 0],
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

    pub(crate) fn constants(self) -> [(&'static str, f64); 5] {
        [
            ("DRAW_STATE", f64::from(self.state)),
            ("MATERIAL_MAPS", f64::from(self.maps)),
            ("MATERIAL_MODEL", f64::from(self.model)),
            ("MATERIAL_SLOTS_LOW", f64::from(self.slots[0])),
            ("MATERIAL_SLOTS_HIGH", f64::from(self.slots[1])),
        ]
    }

    pub(crate) fn keyed(self, used: [bool; 5], styled: bool, faced: bool) -> Option<Self> {
        let [stated, mapped, modeled, low, high] = used;
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
            slots: [
                if low { self.slots[0] } else { 0 },
                if high { self.slots[1] } else { 0 },
            ],
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
    pub slot: SurfaceSlot,
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

type LaidSlot = (SurfaceSlot, &'static str, String, &'static [&'static str]);

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct MaterialLayout {
    maps: Vec<&'static str>,
    values: Vec<&'static str>,
    slots: Vec<LaidSlot>,
    models: Vec<(&'static str, String)>,
    kinds: Vec<SurfaceKind>,
}

pub(crate) fn core_slots() -> Vec<(SlotModule, String)> {
    let module = crate::shaders::core_module(MAPPED_MODULE);
    MAPPED_SLOTS
        .iter()
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

impl MaterialLayout {
    pub(crate) fn standard() -> Self {
        Self::of(&core_slots(), &[], &[], &mut Vec::new())
    }

    pub(crate) fn of(
        slots: &[(SlotModule, String)],
        models: &[(LightingModule, String)],
        kinds: &[SurfaceKind],
        problems: &mut Vec<Error>,
    ) -> Self {
        let mut layout = Self {
            maps: Vec::new(),
            values: Vec::new(),
            slots: Vec::new(),
            models: Vec::new(),
            kinds: vec![SurfaceKind::SOLID],
        };
        for (declared, path) in slots {
            if layout
                .slots
                .iter()
                .any(|(slot, name, _, _)| *slot == declared.slot && *name == declared.name)
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
                .push((declared.slot, declared.name, path.clone(), declared.maps));
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
            .filter(|(known, _, _, _)| *known == slot)
            .position(|(_, known, _, _)| *known == name)
            .map(|at| at as u32 + 1)
    }

    fn read_maps<'a>(&'a self, material: &'a Material) -> impl Iterator<Item = &'static str> + 'a {
        material.slots().iter().flat_map(move |(slot, module)| {
            self.slots
                .iter()
                .filter(move |(known, name, _, _)| known == slot && name == module)
                .flat_map(|(_, _, _, maps)| maps.iter().copied())
        })
    }

    pub(crate) fn reads(&self, material: &Material, map: &str) -> bool {
        self.read_maps(material).any(|read| read == map)
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
            "\nstruct MaterialRecord {{\n    emissive: vec4<f32>,\n    params: vec4<f32>,\n    choice: vec4<u32>,\n    modules: array<vec4<u32>, {MODULE_ROWS}>,\n    maps: array<vec4<f32>, {}>,\n    values: array<vec4<f32>, {}>,\n}}\n",
            2 * self.maps.len(),
            self.value_vec4s()
        );
        for name in [
            "DRAW_STATE",
            "MATERIAL_MAPS",
            "MATERIAL_MODEL",
            "MATERIAL_SLOTS_LOW",
            "MATERIAL_SLOTS_HIGH",
        ] {
            let _ = writeln!(source, "override {name}: u32 = 0u;");
        }
        let _ = writeln!(source, "const SLOT_BITS: u32 = {SLOT_BITS}u;");
        let _ = writeln!(source, "const SLOTS_PER_WORD: u32 = {SLOTS_PER_WORD}u;");
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
        material_functions(&mut source);
        source
    }

    pub(crate) fn slots_source(&self, full: bool) -> String {
        let mut imports = String::from(
            "import package::engine::shading::{PositionInput, Shaded, SurfaceInput, SurfacePoint, unlit};\n",
        );
        let mut body = String::new();
        let surface = "input: SurfaceInput, point: SurfacePoint";
        for slot in SurfaceSlot::ALL {
            let name = slot.name();
            let (returns, params, args) = match slot {
                SurfaceSlot::Position => ("vec3<f32>", "input: PositionInput", "input"),
                SurfaceSlot::BaseColor | SurfaceSlot::Normal | SurfaceSlot::Emissive => {
                    ("vec3<f32>", surface, "input, point")
                }
                SurfaceSlot::Opacity
                | SurfaceSlot::Cutoff
                | SurfaceSlot::Roughness
                | SurfaceSlot::Metalness
                | SurfaceSlot::Occlusion
                | SurfaceSlot::Transmission => ("f32", surface, "input, point"),
                SurfaceSlot::Kind => ("u32", surface, "input, point"),
                SurfaceSlot::LightInputs => ("LightInputs", surface, "input, point"),
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
            let modules: Vec<&LaidSlot> = self
                .slots
                .iter()
                .filter(|(known, _, _, _)| *known == slot)
                .collect();
            if full && !modules.is_empty() {
                let _ = writeln!(
                    body,
                    "    let picked = material_module(SLOT_{});",
                    upper(name)
                );
                for (index, (_, _, path, _)) in modules.iter().enumerate() {
                    let id = index + 1;
                    let _ = writeln!(
                        imports,
                        "import package::{path}::{name} as {name}_{id};"
                    );
                    let _ = writeln!(
                        body,
                        "    if picked == {id}u {{\n        return {name}_{id}({args});\n    }}"
                    );
                }
            }
            let _ = writeln!(body, "    return standard_{name}({args});\n}}");
        }
        if full {
            imports.push_str("import package::face::lit::{LightInputs, finish};\n");
            body.push_str(
                "\nfn shade_model(point: SurfacePoint, pixel: vec2<f32>) -> Shaded {\n    var shaded = modeled(point, pixel);\n    shaded.opacity = point.opacity;\n    return shaded;\n}\n",
            );
            body.push_str(
                "\nfn modeled(point: SurfacePoint, pixel: vec2<f32>) -> Shaded {\n    let model = material_model();\n    if model == LIGHTING_UNLIT {\n        return unlit(point);\n    }\n",
            );
            for (index, (_, path)) in self.models.iter().enumerate() {
                let id = index + 2;
                let _ = writeln!(imports, "import package::{path}::finish as model_{id};");
                let _ = writeln!(
                    body,
                    "    if model == {id}u {{\n        return model_{id}(point, pixel, vec3<f32>(0.0));\n    }}"
                );
            }
            body.push_str("    return finish(point, pixel, vec3<f32>(0.0));\n}\n");
        } else {
            body.push_str("\nalias LightInputs = u32;\n");
        }
        let mut used = vec![
            "material_module".to_owned(),
            "material_model".to_owned(),
            "LIGHTING_UNLIT".to_owned(),
        ];
        used.extend(
            SurfaceSlot::ALL
                .iter()
                .map(|slot| format!("SLOT_{}", upper(slot.name()))),
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

    pub(crate) fn surfaces_source(&self) -> String {
        let mut source = String::new();
        for (index, kind) in self.kinds.iter().enumerate() {
            let _ = writeln!(source, "const SURFACE_{}: u32 = {index}u;", upper(kind.name));
        }
        source
    }

    pub(crate) fn record(&self, handle: MaterialHandle, material: &Material) -> (Vec<u32>, Vec<Error>) {
        let path = format!("$.materials.{}", handle.slot());
        let mut problems = Vec::new();
        let mut words = vec![0_u32; self.record_bytes() / 4];
        let [r, g, b] = material.emissive();
        words[..4].copy_from_slice(&[r, g, b, material.transmission()].map(f32::to_bits));
        let (alpha, cutoff) = alpha_of(material.alpha_mode());
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
            let index = self
                .maps
                .iter()
                .position(|known| known == name)
                .filter(|_| self.reads(material, name));
            let Some(index) = index else {
                problems.push(
                    Error::new(ErrorCode::UnknownVariable, format!("{path}.maps.{name}"))
                        .got(name.as_str())
                        .variants(self.read_maps(material)),
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

    pub(crate) fn variant(&self, material: Option<&Material>, words: &[u32]) -> MaterialVariant {
        let modules: Vec<u32> = SurfaceSlot::ALL
            .iter()
            .map(|slot| words[12 + slot.index()])
            .collect();
        let fixed = self.maps.len() <= u32::BITS as usize
            && modules.iter().all(|&id| id < 1 << SLOT_BITS);
        let mut maps = 0_u32;
        for index in 0..self.maps.len().min(u32::BITS as usize) {
            if f32::from_bits(words[MAPS_AT + index * 8 + 3]) != 0.0 {
                maps |= 1 << index;
            }
        }
        let mut slots = [0_u32; 2];
        if fixed {
            for (at, id) in modules.iter().enumerate() {
                slots[at / SLOTS_PER_WORD] |= id << ((at % SLOTS_PER_WORD) as u32 * SLOT_BITS);
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
            slots,
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
        .map_or(DrawState::PLAIN, |material| draw_of(material.render_state()))
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
    if fixed_material() {{
        let word = select(MATERIAL_SLOTS_LOW, MATERIAL_SLOTS_HIGH, slot >= SLOTS_PER_WORD);
        return (word >> ((slot % SLOTS_PER_WORD) * SLOT_BITS)) & ((1u << SLOT_BITS) - 1u);
    }}
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
}
