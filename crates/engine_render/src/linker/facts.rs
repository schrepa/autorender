use wesl::syntax::{
    AccessMode, AddressSpace, Attribute, Attributes, DeclarationKind, Expression, Function,
    GlobalDeclaration, LiteralExpression, TranslationUnit, TypeExpression,
};
use wgpu::ShaderStages;

use super::{Declaration, Fragment, Sample, Ty};

#[derive(Default)]
pub(super) struct Facts {
    pub(super) declared: Vec<Declaration>,
    pub(super) overrides: Vec<String>,
    pub(super) functions: Vec<(String, Option<ShaderStages>)>,
    pub(super) items: Vec<String>,
}

pub(super) fn facts(text: &str) -> Facts {
    let Ok(unit) = text.parse::<TranslationUnit>() else {
        return Facts::default();
    };
    let mut facts = Facts::default();
    for declaration in &unit.global_declarations {
        match declaration.node() {
            GlobalDeclaration::Declaration(item) => {
                let name = item.ident.to_string();
                if item.kind == DeclarationKind::Override {
                    facts.overrides.push(name.clone());
                }
                let group = number(&item.attributes, |attribute| {
                    if let Attribute::Group(group) = attribute { Some(group) } else { None }
                });
                let binding = number(&item.attributes, |attribute| {
                    if let Attribute::Binding(binding) = attribute { Some(binding) } else { None }
                });
                if let (Some(group), Some(binding), Some(ty)) =
                    (group, binding, binding_ty(item.kind, item.ty.as_ref()))
                {
                    facts.declared.push(Declaration {
                        group,
                        binding,
                        name: name.clone(),
                        ty,
                    });
                }
                facts.items.push(name);
            }
            GlobalDeclaration::Function(item) => {
                let name = item.ident.to_string();
                facts.functions.push((name.clone(), stage(item)));
                facts.items.push(name);
            }
            GlobalDeclaration::Struct(item) => facts.items.push(item.ident.to_string()),
            GlobalDeclaration::TypeAlias(item) => facts.items.push(item.ident.to_string()),
            GlobalDeclaration::Void
            | GlobalDeclaration::ConstAssert(_)
            | GlobalDeclaration::Compound(_) => {}
        }
    }
    facts
}

pub(super) fn fragment(text: &str, entry: &str) -> Fragment {
    let Ok(unit) = text.parse::<TranslationUnit>() else {
        return Fragment::default();
    };
    let members = |name: &str| {
        unit.global_declarations.iter().find_map(|declaration| {
            let GlobalDeclaration::Struct(item) = declaration.node() else {
                return None;
            };
            (item.ident.to_string() == name).then(|| {
                (item.members.iter())
                    .map(|member| (member.attributes.clone(), member.ty.clone()))
                    .collect::<Vec<_>>()
            })
        })
    };
    for declaration in &unit.global_declarations {
        let GlobalDeclaration::Function(item) = declaration.node() else {
            continue;
        };
        if item.ident.to_string() != entry {
            continue;
        }
        let mut outputs = Vec::new();
        if let Some(returned) = &item.return_type {
            match location(&item.return_attributes) {
                Some(at) => outputs.push((at, sample_of(returned))),
                None => {
                    for (attributes, ty) in members(&returned.ident.to_string()).unwrap_or_default() {
                        if let Some(at) = location(&attributes) {
                            outputs.push((at, sample_of(&ty)));
                        }
                    }
                }
            }
        }
        let mut inputs = Vec::new();
        for parameter in &item.parameters {
            inputs.extend(location(&parameter.attributes));
            for (attributes, _) in members(&parameter.ty.ident.to_string()).unwrap_or_default() {
                inputs.extend(location(&attributes));
            }
        }
        inputs.sort_unstable();
        return Fragment { outputs, inputs };
    }
    Fragment::default()
}

fn stage(function: &Function) -> Option<ShaderStages> {
    function
        .attributes
        .iter()
        .find_map(|attribute| {
            let attribute = attribute.node();
            [
                (Attribute::Vertex, ShaderStages::VERTEX),
                (Attribute::Fragment, ShaderStages::FRAGMENT),
                (Attribute::Compute, ShaderStages::COMPUTE),
            ]
            .into_iter()
            .find_map(|(known, stage)| (*attribute == known).then_some(stage))
        })
}

fn location(attributes: &Attributes) -> Option<u32> {
    number(attributes, |attribute| {
        if let Attribute::Location(at) = attribute { Some(at) } else { None }
    })
}

fn number<'a>(
    attributes: &'a Attributes,
    pick: impl Fn(&'a Attribute) -> Option<&'a wesl::syntax::ExpressionNode>,
) -> Option<u32> {
    let expression = attributes.iter().find_map(|attribute| pick(attribute.node()))?;
    let Expression::Literal(literal) = expression.node() else {
        return None;
    };
    if let LiteralExpression::AbstractInt(value) = literal {
        u32::try_from(*value).ok()
    } else if let LiteralExpression::I32(value) = literal {
        u32::try_from(*value).ok()
    } else if let LiteralExpression::U32(value) = literal {
        Some(*value)
    } else {
        None
    }
}

fn names(ty: &TypeExpression) -> Vec<String> {
    let mut found = vec![ty.ident.to_string()];
    for argument in ty.template_args.iter().flatten() {
        if let Expression::TypeOrIdentifier(inner) = argument.expression.node() {
            found.extend(names(inner));
        }
    }
    found
}

fn sample_of(ty: &TypeExpression) -> wgpu::TextureSampleType {
    let names = names(ty);
    if names.iter().any(|name| name == "u32") {
        wgpu::TextureSampleType::Uint
    } else if names.iter().any(|name| name == "i32") {
        wgpu::TextureSampleType::Sint
    } else {
        wgpu::TextureSampleType::Float { filterable: true }
    }
}

fn binding_ty(kind: DeclarationKind, ty: Option<&TypeExpression>) -> Option<Ty> {
    match kind {
        DeclarationKind::Var(Some((AddressSpace::Uniform, _))) => Some(Ty::Uniform),
        DeclarationKind::Var(Some((AddressSpace::Storage, access))) => Some(Ty::Storage {
            write: access == Some(AccessMode::ReadWrite),
        }),
        DeclarationKind::Var(None | Some((AddressSpace::Handle, _))) => texture_ty(ty?),
        DeclarationKind::Var(Some(_))
        | DeclarationKind::Const
        | DeclarationKind::Override
        | DeclarationKind::Let => None,
    }
}

fn texture_ty(ty: &TypeExpression) -> Option<Ty> {
    use wgpu::TextureViewDimension as D;
    let head = ty.ident.to_string();
    let arguments: Vec<String> = (ty.template_args.iter().flatten())
        .filter_map(|argument| {
            let Expression::TypeOrIdentifier(inner) = argument.expression.node() else {
                return None;
            };
            Some(inner.ident.to_string())
        })
        .collect();
    match head.as_str() {
        "sampler" => return Some(Ty::Sampler),
        "sampler_comparison" => return Some(Ty::Comparison),
        _ => {}
    }
    let depth = [
        ("texture_depth_2d_array", D::D2Array),
        ("texture_depth_cube_array", D::CubeArray),
        ("texture_depth_cube", D::Cube),
        ("texture_depth_2d", D::D2),
    ];
    if let Some(&(_, view)) = depth.iter().find(|(name, _)| head == *name) {
        return Some(Ty::Texture {
            sample: Sample::Depth,
            view,
        });
    }
    let views = [
        ("2d_array", D::D2Array),
        ("cube_array", D::CubeArray),
        ("cube", D::Cube),
        ("3d", D::D3),
        ("2d", D::D2),
        ("1d", D::D1),
    ];
    if let Some(shape) = head.strip_prefix("texture_storage_") {
        let view = views.iter().find(|(name, _)| shape == *name)?.1;
        let format = storage_format(arguments.first()?)?;
        let access = match arguments.get(1).map(String::as_str) {
            Some("read") => wgpu::StorageTextureAccess::ReadOnly,
            Some("read_write") => wgpu::StorageTextureAccess::ReadWrite,
            _ => wgpu::StorageTextureAccess::WriteOnly,
        };
        return Some(Ty::StorageTexture {
            format,
            access,
            view,
        });
    }
    let shape = head.strip_prefix("texture_")?;
    let view = views.iter().find(|(name, _)| shape == *name)?.1;
    arguments.first()?;
    let sample = match arguments.first().map(String::as_str) {
        Some("u32") => Sample::Uint,
        Some("i32") => Sample::Sint,
        _ => Sample::Float,
    };
    Some(Ty::Texture { sample, view })
}

fn storage_format(format: &str) -> Option<wgpu::TextureFormat> {
    use wgpu::TextureFormat as F;
    Some(match format {
        "r32float" => F::R32Float,
        "r32uint" => F::R32Uint,
        "r32sint" => F::R32Sint,
        "rg32float" => F::Rg32Float,
        "rg32uint" => F::Rg32Uint,
        "rgba8unorm" => F::Rgba8Unorm,
        "rgba8snorm" => F::Rgba8Snorm,
        "rgba8uint" => F::Rgba8Uint,
        "rgba16float" => F::Rgba16Float,
        "rgba16uint" => F::Rgba16Uint,
        "rgba32float" => F::Rgba32Float,
        "rgba32uint" => F::Rgba32Uint,
        "r16float" => F::R16Float,
        "rg16float" => F::Rg16Float,
        "bgra8unorm" => F::Bgra8Unorm,
        _ => return None,
    })
}
