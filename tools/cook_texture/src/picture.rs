use std::collections::BTreeMap;

use cook::{Kind, Refusal, Step};
use serde::Deserialize;
use serde_json::{Map, json};

use crate::encode::{Encoding, encode};
use crate::image::decode;
use crate::mips::{Texels, Wrap, chain};

pub static PICTURE: Kind = Kind {
    name: "picture",
    version: 1,
    run: picture,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Picture {
    encodings: BTreeMap<String, Encoding>,
    mips: bool,
    wrap: [Wrap; 2],
}

fn identifier(name: &str) -> bool {
    let mut characters = name.chars();
    characters
        .next()
        .is_some_and(|first| first.is_ascii_lowercase())
        && characters.all(|character| {
            character.is_ascii_lowercase() || character.is_ascii_digit() || character == '_'
        })
}

fn picture(step: &mut Step<'_>) -> Result<(), Refusal> {
    let (name, params): (String, Picture) = (step.name().to_owned(), step.params()?);
    let usable = |(variant, encoding): (&String, &Encoding)| {
        identifier(variant) && *encoding != Encoding::Bc5
    };
    if params.encodings.is_empty() || !params.encodings.iter().all(usable) {
        let detail =
            "one or more lowercase variant names, each with a four-channel encoding (not bc5)";
        return Err(Refusal::new("PictureEncoding", &name, detail));
    }
    let [source] = step.inputs() else {
        return Err(Refusal::new("PictureInputs", &name, "one picture"));
    };
    let (file, bytes) = (source.name, source.bytes);
    let image = decode(bytes).map_err(|error| Refusal::new("PictureUndecodable", file, error))?;
    let (mut variants, mut notes) = (Map::new(), Map::new());
    for (variant, &encoding) in &params.encodings {
        let texels = if encoding.srgb() {
            Texels::Srgb
        } else {
            Texels::Linear
        };
        let levels = if params.mips {
            chain(texels, &image, params.wrap)
        } else {
            vec![image.clone()]
        };
        let made = format!("{name}.{variant}.ktx2");
        let encoded = encode(texels, encoding, &levels, false)
            .map_err(|error| Refusal::new("PictureEncode", &made, error))?;
        notes.insert(
            variant.clone(),
            json!({
                "encoding": encoding.name(),
                "levels": levels.len(),
                "file_bytes": encoded.file.len(),
                "gpu_bytes": encoded.gpu_bytes,
                "encode_seconds": encoded.seconds,
                "quality": encoded.quality,
            }),
        );
        variants.insert(variant.clone(), json!(made));
        step.write(&made, encoded.file);
    }
    step.item("textures", &name, json!({"id": name, "variants": variants}));
    step.note(
        json!({"picture": {"width": image.width, "height": image.height, "variants": notes}}),
    );
    Ok(())
}
