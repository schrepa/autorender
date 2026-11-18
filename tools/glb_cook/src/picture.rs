use std::path::Path;

use engine_textures::{Image, decode_image};
use gltf::texture::WrappingMode;
use serde_json::json;

use crate::config::PictureEntry;
use crate::encode::encode;
use crate::entry::Report;
use crate::fault::Fault;
use crate::maps::MapKind;
use crate::mips::chain;

fn read(path: &Path, file: &str) -> Result<Image, Fault> {
    let bytes = std::fs::read(path)
        .map_err(|error| Fault::new("PictureUnreadable", file, "file", error))?;
    let image = decode_image(file, &bytes)
        .map_err(|error| Fault::new("PictureUndecodable", file, "file", error))?;
    if image.pixels.len() != image.width as usize * image.height as usize * 4 {
        return Err(Fault::new(
            "PictureUndecodable",
            file,
            "file",
            "decoded pixels are not four channels",
        ));
    }
    Ok(image)
}

pub fn run(picture: &PictureEntry, path: &Path) -> Report {
    let mut report = Report::default();
    let file = path.display().to_string();
    let image = match read(path, &file) {
        Ok(image) => image,
        Err(fault) => {
            report.faults.push(fault);
            return report;
        }
    };
    let kind = if picture.encoding.srgb() {
        MapKind::BaseColor
    } else {
        MapKind::Orm
    };
    let levels = if picture.mips {
        chain(
            kind,
            &image,
            [WrappingMode::ClampToEdge, WrappingMode::ClampToEdge],
        )
    } else {
        vec![image]
    };
    match encode(kind, picture.encoding, &levels) {
        Ok(encoded) => {
            report.counts.insert(
                "picture",
                json!({
                    "width": levels[0].width,
                    "height": levels[0].height,
                    "encoding": picture.encoding.name(),
                    "levels": levels.len(),
                    "file_bytes": encoded.file.len(),
                    "gpu_bytes": encoded.gpu_bytes,
                    "encode_seconds": encoded.seconds,
                    "quality": encoded.quality,
                }),
            );
            report
                .files
                .push((format!("{}.ktx2", picture.name), encoded.file));
        }
        Err(error) => report
            .faults
            .push(Fault::new("PictureEncode", &file, &picture.name, error)),
    }
    report
}
