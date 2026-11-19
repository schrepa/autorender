use std::path::Path;

use serde_json::json;

use crate::config::{PictureEntry, Wrap};
use crate::encode::encode;
use crate::entry::Report;
use crate::fault::Fault;
use crate::image::read;
use crate::maps::MapKind;
use crate::mips::chain;

pub fn run(picture: &PictureEntry, path: &Path) -> Report {
    let mut report = Report::default();
    let file = path.display().to_string();
    if picture.optional && !path.is_file() {
        report.counts.insert(
            "left_out",
            json!(format!(
                "{file} is not there; scene parts that need `{}` are left out",
                picture.name
            )),
        );
        return report;
    }
    let image = match read(path, &file) {
        Ok(image) => image,
        Err(fault) => {
            report.faults.push(fault);
            return report;
        }
    };
    let mut variants = serde_json::Map::new();
    for (variant, &encoding) in &picture.encodings {
        let kind = if encoding.srgb() {
            MapKind::BaseColor
        } else {
            MapKind::Orm
        };
        let levels = if picture.mips {
            chain(kind, &image, picture.wrap.map(Wrap::mode))
        } else {
            vec![image.clone()]
        };
        match encode(kind, encoding, &levels, false) {
            Ok(encoded) => {
                variants.insert(
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
                report
                    .files
                    .push((format!("{}.{variant}.ktx2", picture.name), encoded.file));
            }
            Err(error) => report.faults.push(Fault::new(
                "PictureEncode",
                &file,
                format!("{}.{variant}", picture.name),
                error,
            )),
        }
    }
    report.counts.insert(
        "picture",
        json!({"width": image.width, "height": image.height, "variants": variants}),
    );
    report
}
