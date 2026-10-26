//! The highlands' surface materials: how each looks.

use engine::Cover;
use engine::terrain::SurfaceProjection;

/// How one surface material looks: its name; the material whose maps it
/// uses (whose files start with that name, its own unless it borrows
/// another's); its linear color and roughness without maps, and the color
/// its maps' colors are multiplied by; the meters its maps repeat over; how
/// sharply its heights win where it blends; how it lies on steep ground;
/// the cover growing where it lies; the meters its
/// heights sink between its stones, tufts and ledges (relief, a few
/// centimeters at most); the meters they move the ground itself, deep
/// blocks and stones; and the meters tracks sink it at most.
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
    pub(super) displacement: f32,
    pub(super) deform: f32,
}

/// The looks, rock fifth. From 50 to 300 meters the grasses show as their
/// blades' tips (their canopy) over their maps' mean, the bare materials as
/// their maps' mean, so those set how neighboring patches differ: dry grass
/// half again as bright as meadow and straw-yellow, lush grass a third
/// darker and a deeper green, bare dirt half as bright and brown, silt as
/// bright and buff, scree a third darker and gray, each more than the light
/// varies within a patch.
pub(super) const LOOKS: [Look; 13] = [
    // Meadow, the grass the others differ from: fresh blades a light
    // yellow-green at their tips, about a fifth of the light reflected.
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
        displacement: 0.0,
        deform: 0.04,
    },
    // Dry grass on slopes the sun dries: cured blades lose their
    // chlorophyll and pale to straw, reflecting half again the meadow's
    // light and more red than green (tips 0.32 in luminance to its 0.2);
    // sparser and shorter, as drought stunts them, over their own maps'
    // thatch, which shows yellower between them.
    Look {
        name: "dry_grass",
        maps: "dry_grass",
        color: [0.22, 0.18, 0.085],
        roughness: 0.92,
        tint: [1.0, 0.95, 0.8],
        tiling: 3.5,
        height_contrast: 0.35,
        projection: SurfaceProjection::Top,
        cover: Some(Cover {
            density: 60.0,
            height: [0.1, 0.38],
            width: 0.025,
            color: [0.4, 0.32, 0.14],
            sway: 0.7,
        }),
        relief: 0.03,
        displacement: 0.0,
        deform: 0.03,
    },
    // Bare ground where erosion wore through the soil: its maps' brown as
    // they are, half as bright as the meadow's canopy; worn ground's macro
    // colors pale it toward its subsoil.
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
        displacement: 0.0,
        deform: 0.06,
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
        displacement: 0.0,
        deform: 0.05,
    },
    // The rock of every steep slope, laid across the slope's faces, its
    // blocks and ledges displacing the ground, so steep ground reads as rock
    // whatever stands on it.
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
        displacement: 0.25,
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
        displacement: 0.0,
        deform: 0.3,
    },
    // Lush grass where water gathers, in hollows and along streams, and on
    // slopes the sun leaves damp: dense blades rich in chlorophyll take
    // more of the red and blue, a deeper green a third darker than the
    // meadow's (tips 0.14 in luminance to its 0.2), taller and denser, over
    // the meadow's maps turned the same deeper green.
    Look {
        name: "wet_grass",
        maps: "grass",
        color: [0.05, 0.11, 0.03],
        roughness: 0.85,
        tint: [0.62, 0.9, 0.56],
        tiling: 3.0,
        height_contrast: 0.35,
        projection: SurfaceProjection::Top,
        cover: Some(Cover {
            density: 190.0,
            height: [0.18, 0.55],
            width: 0.035,
            color: [0.07, 0.17, 0.035],
            sway: 0.55,
        }),
        relief: 0.03,
        displacement: 0.0,
        deform: 0.07,
    },
    // Stream and river beds of the grains their floods leave, dry as their
    // maps make them, the ground's wetness darkening them where water runs.
    // Sand where the water runs slowest, rippled by it.
    Look {
        name: "sand",
        maps: "sand",
        color: [0.22, 0.2, 0.17],
        roughness: 0.84,
        tint: [1.0, 1.0, 1.0],
        tiling: 2.0,
        height_contrast: 0.3,
        projection: SurfaceProjection::Top,
        cover: None,
        relief: 0.012,
        displacement: 0.0,
        deform: 0.05,
    },
    // Gravel: small water-worn stones of every rock the water brings down.
    Look {
        name: "gravel",
        maps: "gravel",
        color: [0.14, 0.133, 0.122],
        roughness: 0.72,
        tint: [1.0, 1.0, 1.0],
        tiling: 1.5,
        height_contrast: 0.75,
        projection: SurfaceProjection::Top,
        cover: None,
        relief: 0.035,
        displacement: 0.0,
        deform: 0.02,
    },
    // Cobbles on steep beds: rounded stones packed with dark gaps between.
    Look {
        name: "cobble",
        maps: "cobble",
        color: [0.13, 0.125, 0.112],
        roughness: 0.66,
        tint: [1.0, 1.0, 1.0],
        tiling: 4.0,
        height_contrast: 0.9,
        projection: SurfaceProjection::Top,
        cover: None,
        relief: 0.0,
        displacement: 0.12,
        deform: 0.0,
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
        displacement: 0.0,
        deform: 0.04,
    },
    // Scree at the foot of the rock: talus of its stones, from 2 to 25
    // centimeters at four meters a repeat, broken fresher than the faces
    // above and so grayer and paler than their weathered tint, laid across
    // its steep slopes and displacing the ground with its larger stones.
    Look {
        name: "scree",
        maps: "scree",
        color: [0.2, 0.19, 0.18],
        roughness: 0.82,
        tint: [1.0, 0.98, 0.95],
        tiling: 4.0,
        height_contrast: 0.8,
        projection: SurfaceProjection::Biplanar,
        cover: None,
        relief: 0.0,
        displacement: 0.1,
        deform: 0.02,
    },
    // Fine silt erosion laid down at the feet of slopes and on fans, bare
    // of grass: the streams' sand's maps a shade darker and warmer, buff,
    // as bright as the meadow's canopy from afar (about 0.21 in luminance)
    // where that is green, its ripples larger and fainter, taking tracks
    // deeply.
    Look {
        name: "silt",
        maps: "sand",
        color: [0.24, 0.21, 0.17],
        roughness: 0.88,
        tint: [0.95, 0.93, 0.88],
        tiling: 3.0,
        height_contrast: 0.3,
        projection: SurfaceProjection::Top,
        cover: None,
        relief: 0.008,
        displacement: 0.0,
        deform: 0.06,
    },
];
