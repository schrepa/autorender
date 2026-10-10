//! What the page offers, from the lists the workloads use: scenes, looks,
//! effects, views and mesh batches, each with its name in menus, and the
//! names of the per-frame sample. `cargo xtask schema` writes it into the
//! page's generated module.

use serde::Serialize;

use crate::config::MESH_VARIETIES;
use crate::scene::{EFFECTS, LOOKS, SCENE_LOOK, VIEWS};
use crate::{StressSample, StressScene};

/// A menu choice: the value the workload takes, and its name in menus.
#[derive(Clone, Debug, Serialize)]
pub struct Choice {
    pub value: String,
    pub label: String,
}

fn choice(value: impl Into<String>, label: impl Into<String>) -> Choice {
    Choice {
        value: value.into(),
        label: label.into(),
    }
}

/// Everything the page offers.
#[derive(Clone, Debug, Serialize)]
pub struct Catalog {
    pub scenes: Vec<Choice>,
    pub looks: Vec<Choice>,
    pub effects: Vec<Choice>,
    pub views: Vec<Choice>,
    pub mesh_varieties: Vec<u32>,
    pub sample: Vec<&'static str>,
}

/// The page's catalog.
pub fn catalog() -> Catalog {
    Catalog {
        scenes: StressScene::ALL
            .iter()
            .map(|scene| choice(scene.name(), scene.label()))
            .collect(),
        looks: std::iter::once(choice(SCENE_LOOK.0, SCENE_LOOK.1))
            .chain(LOOKS.iter().map(|&(name, label, ..)| choice(name, label)))
            .collect(),
        effects: [choice("all", "All effects"), choice("none", "None")]
            .into_iter()
            .chain(
                EFFECTS
                    .iter()
                    .map(|&(name, what)| choice(format!("no-{name}"), format!("No {what}"))),
            )
            .collect(),
        views: VIEWS
            .iter()
            .map(|&(name, label)| choice(name, label))
            .collect(),
        mesh_varieties: MESH_VARIETIES.to_vec(),
        sample: StressSample::FIELDS.to_vec(),
    }
}
