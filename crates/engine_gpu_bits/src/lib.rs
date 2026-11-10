#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

use std::fmt::Write as _;

use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as Tokens};
use quote::quote;
use syn::parse::{ParseStream, Parser};
use syn::punctuated::Punctuated;
use syn::{
    Attribute, Data, DataEnum, DeriveInput, Error, Expr, Field, Fields, Ident, Lit, LitInt, Path,
    Token, Type, Variant,
};

#[proc_macro_attribute]
pub fn gpu_bits(args: TokenStream, item: TokenStream) -> TokenStream {
    match expand(args.into(), item.into()) {
        Ok(tokens) => tokens.into(),
        Err(error) => error.to_compile_error().into(),
    }
}

fn expand(args: Tokens, item: Tokens) -> Result<Tokens, Error> {
    let root = root(args)?;
    let mut input: DeriveInput = syn::parse2(item)?;
    let name = input.ident.clone();
    let declared = match &mut input.data {
        Data::Struct(data) => match &mut data.fields {
            Fields::Named(named) => Declared::Word(fields(&mut named.named)?),
            Fields::Unnamed(_) | Fields::Unit => {
                return Err(Error::new_spanned(
                    &name,
                    "a gpu_bits word has named fields",
                ));
            }
        },
        Data::Enum(data) => Declared::Value(variants(data)?),
        Data::Union(data) => {
            return Err(Error::new(
                data.union_token.span,
                "gpu_bits takes a struct or an enum",
            ));
        }
    };
    let generated = match declared {
        Declared::Word(fields) => word(&root, &name, &fields),
        Declared::Value(variants) => value(&root, &name, &variants),
    };
    Ok(quote! {
        #input
        #generated
    })
}

enum Declared {
    Word(Vec<Bits>),
    Value(Vec<(Ident, u32)>),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Flag,
    Number(u32),
    Value,
}

struct Bits {
    ident: Ident,
    ty: Type,
    at: u32,
    width: u32,
    kind: Kind,
}

fn root(args: Tokens) -> Result<Path, Error> {
    let mut root: Path = syn::parse_quote!(::engine_core);
    if args.is_empty() {
        return Ok(root);
    }
    let parser = syn::meta::parser(|meta| {
        if meta.path.is_ident("crate") {
            root = meta.value()?.parse()?;
            Ok(())
        } else {
            Err(meta.error("gpu_bits takes crate = path"))
        }
    });
    parser.parse2(args)?;
    Ok(root)
}

fn kind_of(ty: &Type) -> Kind {
    let Type::Path(path) = ty else {
        return Kind::Value;
    };
    match path.path.get_ident().map(Ident::to_string).as_deref() {
        Some("bool") => Kind::Flag,
        Some("u8") => Kind::Number(8),
        Some("u16") => Kind::Number(16),
        Some("u32") => Kind::Number(32),
        Some(_) | None => Kind::Value,
    }
}

fn declared_bits(attr: &Attribute) -> Result<(u32, Option<u32>), Error> {
    attr.parse_args_with(|input: ParseStream<'_>| {
        let width: LitInt = input.parse()?;
        let width = width.base10_parse::<u32>()?;
        if input.is_empty() {
            return Ok((width, None));
        }
        input.parse::<Token![,]>()?;
        let key: Ident = input.parse()?;
        if key != "at" {
            return Err(Error::new(
                key.span(),
                "bits takes (width) or (width, at = bit)",
            ));
        }
        input.parse::<Token![=]>()?;
        let at: LitInt = input.parse()?;
        Ok((width, Some(at.base10_parse::<u32>()?)))
    })
}

fn fields(named: &mut Punctuated<Field, Token![,]>) -> Result<Vec<Bits>, Error> {
    let mut placed: Vec<Bits> = Vec::new();
    let mut next = 0;
    for field in named.iter_mut() {
        let Some(ident) = field.ident.clone() else {
            return Err(Error::new(
                Span::call_site(),
                "a gpu_bits word has named fields",
            ));
        };
        let mut declared = None;
        let mut kept = Vec::with_capacity(field.attrs.len());
        for attr in field.attrs.drain(..) {
            if attr.path().is_ident("bits") {
                declared = Some(declared_bits(&attr)?);
            } else {
                kept.push(attr);
            }
        }
        field.attrs = kept;
        let kind = kind_of(&field.ty);
        let (width, at) = match (kind, declared) {
            (Kind::Flag, None) => (1, None),
            (Kind::Flag, Some((1, at))) => (1, at),
            (Kind::Flag, Some(_)) => {
                return Err(Error::new(ident.span(), "a bool field is one bit"));
            }
            (Kind::Number(_) | Kind::Value, Some((width, at))) => (width, at),
            (Kind::Number(_) | Kind::Value, None) => {
                return Err(Error::new(ident.span(), "this field needs #[bits(width)]"));
            }
        };
        if width == 0 {
            return Err(Error::new(
                ident.span(),
                "a gpu_bits field has at least one bit",
            ));
        }
        if let Kind::Number(most) = kind
            && width > most
        {
            return Err(Error::new(
                ident.span(),
                "the field's type holds fewer bits",
            ));
        }
        let at = at.unwrap_or(next);
        if at + width > 32 {
            return Err(Error::new(ident.span(), "a gpu_bits word holds 32 bits"));
        }
        if let Some(other) = placed
            .iter()
            .find(|other| at < other.at + other.width && other.at < at + width)
        {
            return Err(Error::new(
                ident.span(),
                format!("these bits overlap the field {}", other.ident),
            ));
        }
        next = at + width;
        placed.push(Bits {
            ident,
            ty: field.ty.clone(),
            at,
            width,
            kind,
        });
    }
    if placed.is_empty() {
        return Err(Error::new(Span::call_site(), "a gpu_bits word has fields"));
    }
    Ok(placed)
}

fn variants(data: &DataEnum) -> Result<Vec<(Ident, u32)>, Error> {
    let mut listed: Vec<(Ident, u32)> = Vec::new();
    let mut next = 0_u32;
    for variant in &data.variants {
        let value = discriminant(variant)?.unwrap_or(next);
        if !matches!(variant.fields, Fields::Unit) {
            return Err(Error::new(
                variant.ident.span(),
                "gpu_bits values are unit variants",
            ));
        }
        if let Some((other, _)) = listed.iter().find(|(_, known)| *known == value) {
            return Err(Error::new(
                variant.ident.span(),
                format!("this value repeats {other}"),
            ));
        }
        next = value.saturating_add(1);
        listed.push((variant.ident.clone(), value));
    }
    if listed.is_empty() {
        return Err(Error::new(
            Span::call_site(),
            "a gpu_bits enum has variants",
        ));
    }
    Ok(listed)
}

fn discriminant(variant: &Variant) -> Result<Option<u32>, Error> {
    let Some((_, expr)) = &variant.discriminant else {
        return Ok(None);
    };
    let Expr::Lit(literal) = expr else {
        return Err(Error::new_spanned(
            expr,
            "a gpu_bits value is a whole number",
        ));
    };
    let Lit::Int(value) = &literal.lit else {
        return Err(Error::new_spanned(
            expr,
            "a gpu_bits value is a whole number",
        ));
    };
    value.base10_parse::<u32>().map(Some)
}

fn mask(width: u32) -> u32 {
    if width >= 32 {
        u32::MAX
    } else {
        (1 << width) - 1
    }
}

fn read_bits(field: &Bits) -> Tokens {
    let (at, mask) = (field.at, mask(field.width));
    match (at, field.width) {
        (0, 32) => quote! { word },
        (0, _) => quote! { word & #mask },
        (_, width) if at + width == 32 => quote! { word >> #at },
        (_, _) => quote! { (word >> #at) & #mask },
    }
}

fn written_bits(field: &Bits) -> Tokens {
    let (at, mask, width) = (field.at, mask(field.width), field.width);
    let ident = &field.ident;
    let held = if width < 32 {
        quote! {
            ::core::debug_assert!(
                value <= #mask,
                ::core::concat!("the gpu_bits field ", ::core::stringify!(#ident), " holds more bits than declared")
            );
            let value = value & #mask;
        }
    } else {
        quote! {}
    };
    let placed = if at == 0 {
        quote! { value }
    } else {
        quote! { value << #at }
    };
    quote! {
        #held
        word |= #placed;
    }
}

fn word(root: &Path, name: &Ident, fields: &[Bits]) -> Tokens {
    let wgsl = word_wgsl(&snake(&name.to_string()), fields);
    let entries = fields.iter().map(|field| {
        let (label, at, width, ty) = (field.ident.to_string(), field.at, field.width, &field.ty);
        quote! {
            #root::BitField {
                name: #label,
                at: #at,
                width: #width,
                values: <#ty as #root::BitValue>::VALUES,
            }
        }
    });
    let packs = fields.iter().map(|field| {
        let (ident, ty) = (&field.ident, &field.ty);
        let written = written_bits(field);
        quote! {
            let value = <#ty as #root::BitValue>::bits(self.#ident);
            #written
        }
    });
    let unpacks = fields.iter().map(|field| {
        let (ident, ty) = (&field.ident, &field.ty);
        let read = read_bits(field);
        quote! { #ident: <#ty as #root::BitValue>::of_bits(#read)?, }
    });
    let checks = fields
        .iter()
        .filter(|field| field.kind == Kind::Value)
        .map(|field| {
            let (ty, width) = (&field.ty, field.width);
            quote! {
                ::core::assert!(
                    <#ty as #root::BitValue>::WIDTH <= #width,
                    "a gpu_bits field is narrower than its values"
                );
            }
        });
    quote! {
        const _: () = {
            #(#checks)*
        };

        impl #root::GpuBits for #name {
            const WGSL: &'static str = #wgsl;
            const FIELDS: &'static [#root::BitField] = &[#(#entries),*];

            fn pack(self) -> u32 {
                let mut word = 0_u32;
                #(#packs)*
                word
            }

            fn unpack(word: u32) -> ::core::option::Option<Self> {
                ::core::option::Option::Some(Self {
                    #(#unpacks)*
                })
            }
        }
    }
}

fn word_wgsl(prefix: &str, fields: &[Bits]) -> String {
    let mut text = String::new();
    for field in fields {
        let name = field.ident.to_string();
        let (at, mask) = (field.at, mask(field.width));
        let clear = !(mask << at);
        if field.kind == Kind::Flag {
            let _ = writeln!(
                text,
                "fn {prefix}_{name}(word: u32) -> bool {{ return ((word >> {at}u) & 1u) != 0u; }}"
            );
            let _ = writeln!(
                text,
                "fn {prefix}_with_{name}(word: u32, value: bool) -> u32 {{ return (word & {clear}u) | (select(0u, 1u, value) << {at}u); }}"
            );
        } else {
            let _ = writeln!(
                text,
                "fn {prefix}_{name}(word: u32) -> u32 {{ return (word >> {at}u) & {mask}u; }}"
            );
            let _ = writeln!(
                text,
                "fn {prefix}_with_{name}(word: u32, value: u32) -> u32 {{ return (word & {clear}u) | ((value & {mask}u) << {at}u); }}"
            );
        }
    }
    let parameters: Vec<String> = fields
        .iter()
        .map(|field| {
            let ty = if field.kind == Kind::Flag {
                "bool"
            } else {
                "u32"
            };
            format!("{}: {ty}", field.ident)
        })
        .collect();
    let _ = writeln!(text, "fn {prefix}({}) -> u32 {{", parameters.join(", "));
    let _ = writeln!(text, "    var word = 0u;");
    for field in fields {
        let _ = writeln!(
            text,
            "    word = {prefix}_with_{0}(word, {0});",
            field.ident
        );
    }
    let _ = writeln!(text, "    return word;");
    let _ = writeln!(text, "}}");
    text
}

fn value(root: &Path, name: &Ident, variants: &[(Ident, u32)]) -> Tokens {
    let most = variants.iter().map(|(_, value)| *value).max().unwrap_or(0);
    let width = (32 - most.leading_zeros()).max(1);
    let prefix = snake(&name.to_string()).to_uppercase();
    let mut wgsl = String::new();
    for (ident, value) in variants {
        let _ = writeln!(
            wgsl,
            "const {prefix}_{}: u32 = {value}u;",
            snake(&ident.to_string()).to_uppercase()
        );
    }
    let entries = variants.iter().map(|(ident, value)| {
        let label = snake(&ident.to_string());
        quote! { (#value, #label) }
    });
    let bits = variants
        .iter()
        .map(|(ident, value)| quote! { Self::#ident => #value, });
    let found = variants.iter().map(|(ident, value)| {
        quote! { #value => ::core::option::Option::Some(Self::#ident), }
    });
    quote! {
        impl #root::BitValue for #name {
            const WIDTH: u32 = #width;
            const VALUES: &'static [(u32, &'static str)] = &[#(#entries),*];
            const WGSL: &'static str = #wgsl;

            fn bits(self) -> u32 {
                match self {
                    #(#bits)*
                }
            }

            fn of_bits(bits: u32) -> ::core::option::Option<Self> {
                match bits {
                    #(#found)*
                    _ => ::core::option::Option::None,
                }
            }
        }
    }
}

fn snake(name: &str) -> String {
    let characters: Vec<char> = name.chars().collect();
    let mut words: Vec<String> = Vec::new();
    let mut current = String::new();
    for (index, &character) in characters.iter().enumerate() {
        if character == '_' {
            if !current.is_empty() {
                words.push(std::mem::take(&mut current));
            }
            continue;
        }
        let starts = character.is_uppercase()
            && !current.is_empty()
            && index > 0
            && (characters[index - 1].is_lowercase()
                || characters[index - 1].is_ascii_digit()
                || characters
                    .get(index + 1)
                    .is_some_and(|next| next.is_lowercase()));
        if starts {
            words.push(std::mem::take(&mut current));
        }
        current.extend(character.to_lowercase());
    }
    if !current.is_empty() {
        words.push(current);
    }
    words.join("_")
}
