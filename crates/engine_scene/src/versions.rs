//! What each scene version introduced: one table of rules, checked in
//! order. Validation refuses a document using a feature newer than its
//! version and names its first use; development tools describe the
//! versions from the same rules. Features found in referenced files
//! (compressed textures, buffer files' vertex extras) are checked where the
//! files are read, with [`version_required`].

use crate::{MeshSource, SCENE_VERSION, SceneDocument, SceneError, SceneObject};

/// A feature a scene version introduced: the version, what it is (as
/// messages and descriptions name it) and the authored path of a
/// document's first use of it.
struct Rule {
    since: u32,
    what: &'static str,
    first_use: fn(&SceneDocument) -> Option<String>,
}

/// The rules, newest version first: the order validation checks them in.
fn rules() -> [Rule; 27] {
    [
        Rule {
            since: 9,
            what: "streamed entities",
            first_use: |document| (!document.streamed.is_empty()).then(|| "$.streamed".to_owned()),
        },
        Rule {
            since: 9,
            what: "grounded objects",
            first_use: |document| {
                document.entity_use(|entity| entity.grounded.is_some().then_some("grounded"))
            },
        },
        Rule {
            since: 9,
            what: "terrains",
            first_use: |document| {
                document.entity_use(|entity| entity.terrain.is_some().then_some("terrain"))
            },
        },
        Rule {
            since: 9,
            what: "terrain settings, ground cover and aerial perspective",
            first_use: |document| {
                document
                    .environment
                    .as_ref()?
                    .version_nine_field()
                    .map(|field| format!("$.environment.{field}"))
            },
        },
        Rule {
            since: 8,
            what: "glTF material maps, texture settings, emissive colors, alpha modes, double-sided materials and transmission",
            first_use: |document| {
                document
                    .materials
                    .iter()
                    .enumerate()
                    .find_map(|(index, material)| {
                        material
                            .version_eight_field()
                            .map(|field| format!("$.materials[{index}].{field}"))
                    })
            },
        },
        Rule {
            since: 9,
            what: "world-mapped materials and top layers",
            first_use: |document| {
                document
                    .materials
                    .iter()
                    .enumerate()
                    .find_map(|(index, material)| {
                        material
                            .version_nine_field()
                            .map(|field| format!("$.materials[{index}].{field}"))
                    })
            },
        },
        Rule {
            since: 8,
            what: "decals",
            first_use: |document| {
                document.entity_use(|entity| entity.decal.is_some().then_some("decal"))
            },
        },
        Rule {
            since: 9,
            what: "floating objects",
            first_use: |document| {
                document.entity_use(|entity| entity.floating.is_some().then_some("floating"))
            },
        },
        Rule {
            since: 9,
            what: "deformers",
            first_use: |document| {
                document.entity_use(|entity| entity.deformer.is_some().then_some("deformer"))
            },
        },
        Rule {
            since: 9,
            what: "navigation agents",
            first_use: |document| {
                document.entity_use(|entity| entity.nav_agent.is_some().then_some("nav_agent"))
            },
        },
        Rule {
            since: 9,
            what: "spline meshes",
            first_use: |document| {
                document.entity_use(|entity| entity.spline.is_some().then_some("spline"))
            },
        },
        Rule {
            since: 9,
            what: "interiors",
            first_use: |document| {
                document.entity_use(|entity| {
                    entity
                        .reflection_probe
                        .is_some_and(|probe| probe.interior.is_some())
                        .then_some("reflection_probe.interior")
                })
            },
        },
        Rule {
            since: 8,
            what: "reflection probes",
            first_use: |document| {
                document.entity_use(|entity| {
                    entity
                        .reflection_probe
                        .is_some()
                        .then_some("reflection_probe")
                })
            },
        },
        Rule {
            since: 8,
            what: "object tints",
            first_use: |document| {
                document.entity_use(|entity| entity.tint.is_some().then_some("tint"))
            },
        },
        Rule {
            since: 8,
            what: "vertex colors, second texture coordinates, joints and weights",
            first_use: |document| {
                document
                    .meshes
                    .iter()
                    .position(|mesh| mesh.source.has_vertex_extras())
                    .map(|index| format!("$.meshes[{index}].source.vertices"))
            },
        },
        Rule {
            since: 8,
            what: "virtual meshes",
            first_use: |document| {
                document
                    .meshes
                    .iter()
                    .position(|mesh| matches!(mesh.source, MeshSource::Virtual { .. }))
                    .map(|index| format!("$.meshes[{index}].source"))
            },
        },
        Rule {
            since: 8,
            what: "refraction, decal, reflection probe, virtual geometry and level of detail settings",
            first_use: |document| {
                document
                    .environment
                    .as_ref()?
                    .section_since(8)
                    .map(|section| format!("$.environment.{section}"))
            },
        },
        Rule {
            since: 8,
            what: "animations and skins",
            first_use: |document| {
                if !document.animations.is_empty() {
                    return Some("$.animations".to_owned());
                }
                if !document.skins.is_empty() {
                    return Some("$.skins".to_owned());
                }
                document.entity_use(|entity| {
                    if entity.animation.is_some() {
                        Some("animation")
                    } else {
                        entity.skin.is_some().then_some("skin")
                    }
                })
            },
        },
        Rule {
            since: 7,
            what: "image and buffer files",
            first_use: |document| {
                document
                    .textures
                    .iter()
                    .position(|texture| texture.uri.is_some())
                    .map(|index| format!("$.textures[{index}].uri"))
                    .or_else(|| {
                        document
                            .meshes
                            .iter()
                            .position(|mesh| matches!(mesh.source, MeshSource::Buffer { .. }))
                            .map(|index| format!("$.meshes[{index}].source"))
                    })
            },
        },
        Rule {
            since: 6,
            what: "grading, shadow cascade, local light, particle, sun shaft and reflection settings",
            first_use: |document| {
                document
                    .environment
                    .as_ref()?
                    .version_six_field()
                    .map(|field| format!("$.environment.{field}"))
            },
        },
        Rule {
            since: 6,
            what: "material opacities",
            first_use: |document| {
                document
                    .materials
                    .iter()
                    .position(|material| material.opacity.is_some())
                    .map(|index| format!("$.materials[{index}].opacity"))
            },
        },
        Rule {
            since: 6,
            what: "entity lights and particles",
            first_use: |document| {
                document.entity_use(|entity| {
                    if entity.light.is_some() {
                        Some("light")
                    } else {
                        entity.particles.is_some().then_some("particles")
                    }
                })
            },
        },
        Rule {
            since: 5,
            what: "scene environments",
            first_use: |document| {
                document
                    .environment
                    .is_some()
                    .then(|| "$.environment".to_owned())
            },
        },
        Rule {
            since: 5,
            what: "material surfaces",
            first_use: |document| {
                document
                    .materials
                    .iter()
                    .position(|material| material.surface.is_some())
                    .map(|index| format!("$.materials[{index}].surface"))
            },
        },
        Rule {
            since: 4,
            what: "colliders",
            first_use: |document| {
                document.entity_use(|entity| entity.collider.is_some().then_some("collider"))
            },
        },
        Rule {
            since: 2,
            what: "textures and UV data",
            first_use: |document| {
                if !document.textures.is_empty() {
                    return Some("$.textures".to_owned());
                }
                document
                    .materials
                    .iter()
                    .position(|material| material.texture.is_some())
                    .map(|index| format!("$.materials[{index}].texture"))
                    .or_else(|| {
                        document
                            .meshes
                            .iter()
                            .position(|mesh| {
                                matches!(&mesh.source, MeshSource::Inline { vertices, .. }
                                    if vertices.iter().any(|vertex| vertex.uv != [0.0; 2]))
                            })
                            .map(|index| format!("$.meshes[{index}].source.vertices"))
                    })
            },
        },
        Rule {
            since: 3,
            what: "hierarchies, prefabs and transform-only nodes",
            first_use: |document| {
                if !document.prefabs.is_empty() {
                    return Some("$.prefabs".to_owned());
                }
                document.entity_use(|entity| {
                    if entity.parent.is_some() {
                        Some("parent")
                    } else if entity.prefab.is_some() {
                        Some("prefab")
                    } else if entity.mesh.is_none() {
                        Some("mesh")
                    } else {
                        entity.material.is_none().then_some("material")
                    }
                })
            },
        },
    ]
}

/// The error of a feature at `path`, `what`, that needs scene version
/// `since`.
pub(crate) fn version_required(path: impl Into<String>, what: &str, since: u32) -> SceneError {
    SceneError::new(
        "version_required",
        path,
        format!(
            "{what} require scene version {since}; set version to {since} and increment revision"
        ),
    )
}

/// What each scene version after the first introduced, oldest first:
/// `2 textures and UV data; 3 …`.
pub fn version_features() -> String {
    let rules = rules();
    (2..=SCENE_VERSION)
        .filter_map(|version| {
            let features: Vec<&str> = rules
                .iter()
                .filter(|rule| rule.since == version)
                .map(|rule| rule.what)
                .collect();
            (!features.is_empty()).then(|| format!("{version} {}", features.join(", ")))
        })
        .collect::<Vec<_>>()
        .join("; ")
}

impl SceneDocument {
    /// Refuses an unknown version, and a feature newer than the document's.
    pub(crate) fn check_version(&self) -> Result<(), SceneError> {
        if !(1..=SCENE_VERSION).contains(&self.version) {
            return Err(SceneError::new(
                "unsupported_version",
                "$.version",
                format!(
                    "expected version 1 through {SCENE_VERSION}; found {}",
                    self.version
                ),
            ));
        }
        for rule in rules() {
            if self.version < rule.since
                && let Some(path) = (rule.first_use)(self)
            {
                return Err(version_required(path, rule.what, rule.since));
            }
        }
        Ok(())
    }

    /// The authored path of the field `uses` names on the first entity,
    /// then streamed entity, then prefab entity, where it names one.
    fn entity_use(&self, uses: impl Fn(&SceneObject) -> Option<&'static str>) -> Option<String> {
        self.entities
            .iter()
            .enumerate()
            .find_map(|(index, entity)| {
                uses(entity).map(|field| format!("$.entities[{index}].{field}"))
            })
            .or_else(|| {
                self.streamed
                    .iter()
                    .enumerate()
                    .find_map(|(index, entity)| {
                        uses(entity).map(|field| format!("$.streamed[{index}].{field}"))
                    })
            })
            .or_else(|| {
                self.prefabs
                    .iter()
                    .enumerate()
                    .find_map(|(prefab, definition)| {
                        definition
                            .entities
                            .iter()
                            .enumerate()
                            .find_map(|(index, entity)| {
                                uses(entity).map(|field| {
                                    format!("$.prefabs[{prefab}].entities[{index}].{field}")
                                })
                            })
                    })
            })
    }
}
