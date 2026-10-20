//! The highlands' surface materials: how each looks.

use engine::Cover;
use engine::terrain::SurfaceProjection;

/// How one surface material looks: its name; the material whose maps it
/// uses (whose files start with that name, its own unless it borrows
/// another's); its linear color and roughness without maps, and the color
/// its maps' colors are multiplied by; the meters its maps repeat over; how
/// sharply its heights win where it blends; how it lies on steep ground;
/// the cover growing where it lies; the meters its
/// heights sink between its stones and tufts; and the
/// meters tracks sink it at most.
pub(super) struct Look {
    pub(super) name: &'static str,
    pub(super) maps: &'static str,
    pub(super) color: [f32; 3],
    pub(super) roughness: f32,
    pub(super) tint: [f32; 3],
    pub(super) tiling: f32,
    pub(super) height_contrast: f32,
    pub(super) projection: SurfaceProjection,
    pub(super) cover: Option<Cover>,
    pub(super) relief: f32,
    pub(super) deform: f32,
}

pub(super) const LOOKS: [Look; 12] = [
    Look {
        name: "grass",
        maps: "grass",
        color: [0.105, 0.135, 0.045],
        roughness: 0.92,
        tint: [1.0, 1.0, 1.0],
        tiling: 3.0,
        height_contrast: 0.35,
        projection: SurfaceProjection::Top,
        cover: Some(Cover {
            density: 140.0,
            height: [0.1, 0.34],
            width: 0.035,
            color: [0.16, 0.23, 0.06],
            sway: 0.6,
        }),
        relief: 0.03,
        deform: 0.04,
    },
    Look {
        name: "dry_grass",
        maps: "dry_grass",
        color: [0.2, 0.16, 0.08],
        roughness: 0.92,
        tint: [0.9, 0.9, 0.9],
        tiling: 3.5,
        height_contrast: 0.35,
        projection: SurfaceProjection::Top,
        cover: Some(Cover {
            density: 70.0,
            height: [0.14, 0.46],
            width: 0.025,
            color: [0.34, 0.27, 0.12],
            sway: 0.7,
        }),
        relief: 0.03,
        deform: 0.03,
    },
    Look {
        name: "dirt",
        maps: "dirt",
        color: [0.21, 0.165, 0.11],
        roughness: 0.92,
        tint: [1.0, 1.0, 1.0],
        tiling: 3.0,
        height_contrast: 0.5,
        projection: SurfaceProjection::Top,
        cover: None,
        relief: 0.05,
        deform: 0.06,
    },
    Look {
        name: "scree",
        maps: "scree",
        color: [0.2, 0.19, 0.18],
        roughness: 0.82,
        tint: [0.78, 0.74, 0.68],
        tiling: 4.0,
        height_contrast: 0.8,
        projection: SurfaceProjection::Top,
        cover: None,
        relief: 0.12,
        deform: 0.02,
    },
    Look {
        name: "rock",
        maps: "rock",
        color: [0.25, 0.235, 0.22],
        roughness: 0.82,
        tint: [0.74, 0.7, 0.64],
        tiling: 8.0,
        height_contrast: 0.7,
        projection: SurfaceProjection::Biplanar,
        cover: None,
        relief: 0.0,
        deform: 0.0,
    },
    Look {
        name: "snow",
        maps: "snow",
        color: [0.82, 0.84, 0.88],
        roughness: 0.55,
        tint: [1.0, 1.0, 1.0],
        tiling: 6.0,
        height_contrast: 0.4,
        projection: SurfaceProjection::Top,
        cover: None,
        relief: 0.04,
        deform: 0.3,
    },
    // Needle litter under the spruces: the dirt's maps, darkened toward
    // olive, so forests read as dark stands from afar.
    Look {
        name: "forest_floor",
        maps: "dirt",
        color: [0.03, 0.036, 0.022],
        roughness: 0.95,
        tint: [0.18, 0.23, 0.15],
        tiling: 3.0,
        height_contrast: 0.5,
        projection: SurfaceProjection::Top,
        cover: None,
        relief: 0.06,
        deform: 0.05,
    },
    // Grass where water gathers: deeper green, taller and
    // denser, in hollows and along streams.
    Look {
        name: "wet_grass",
        maps: "grass",
        color: [0.07, 0.13, 0.04],
        roughness: 0.85,
        tint: [0.72, 0.95, 0.68],
        tiling: 3.0,
        height_contrast: 0.35,
        projection: SurfaceProjection::Top,
        cover: Some(Cover {
            density: 170.0,
            height: [0.16, 0.5],
            width: 0.035,
            color: [0.1, 0.2, 0.05],
            sway: 0.6,
        }),
        relief: 0.03,
        deform: 0.07,
    },
    // Stream and river beds: the scree's stones, washed darker and wetter.
    Look {
        name: "riverbed",
        maps: "scree",
        color: [0.13, 0.13, 0.12],
        roughness: 0.55,
        tint: [0.55, 0.56, 0.54],
        tiling: 3.0,
        height_contrast: 0.9,
        projection: SurfaceProjection::Top,
        cover: None,
        relief: 0.1,
        deform: 0.03,
    },
    // Roads: packed gravel, the scree's stones finer and
    // warmer, where no grass grows.
    Look {
        name: "road",
        maps: "scree",
        color: [0.27, 0.24, 0.2],
        roughness: 0.9,
        tint: [0.92, 0.85, 0.74],
        tiling: 1.6,
        height_contrast: 0.6,
        projection: SurfaceProjection::Top,
        cover: None,
        relief: 0.03,
        deform: 0.04,
    },
    // Rock the spray of falls and cascades keeps wet: as dark again as dry
    // rock, and glossy (Lekner and Dorf).
    Look {
        name: "wet_rock",
        maps: "rock",
        color: [0.12, 0.115, 0.11],
        roughness: 0.38,
        tint: [0.4, 0.39, 0.37],
        tiling: 8.0,
        height_contrast: 0.7,
        projection: SurfaceProjection::Biplanar,
        cover: None,
        relief: 0.0,
        deform: 0.0,
    },
    // Moss on that rock, as spray zones grow it: the grass's maps, dark
    // green and damp.
    Look {
        name: "moss",
        maps: "grass",
        color: [0.035, 0.07, 0.025],
        roughness: 0.72,
        tint: [0.32, 0.5, 0.22],
        tiling: 2.0,
        height_contrast: 0.5,
        projection: SurfaceProjection::Biplanar,
        cover: None,
        relief: 0.02,
        deform: 0.02,
    },
];
