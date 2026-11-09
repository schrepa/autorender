use engine_core::{AssetResidency, Drawn, MaterialHandle, MeshHandle};

use super::*;

impl DevTools {
    pub(super) fn register_components(&mut self, schemas: &Value) {
        self.insert(Inspector {
            brick: None,
            descriptor: InspectorDescriptor::generated(schemas, "DrawnMesh"),
            kind: InspectorKind::Component,
            has: Box::new(|world, entity| {
                entity.is_some_and(|entity| {
                    world
                        .get::<Drawn>(entity)
                        .is_some_and(|drawn| drawn.geometry.is_some())
                })
            }),
            read: Box::new(|world, entity, context| {
                let drawn = world.get::<Drawn>(entity?)?;
                let shown = (drawn.geometry?, drawn.material);
                let assets = world.get_resource::<AssetLibrary>()?;
                Some(json!(mesh_view(world, assets, shown, context)))
            }),
        })
        .expect("descriptor_bounded");
        self.register_resource::<JobExchange>(
            InspectorDescriptor::generated(schemas, "Jobs"),
            |jobs| {
                let stats = jobs.stats();
                json!(crate::views::JobsView {
                    workers: stats.workers,
                    running: stats.running,
                    submitted: stats.submitted,
                    completed: stats.completed,
                    failed: stats.failed,
                    last_failure: jobs.last_failure().map(str::to_owned),
                    cancelled: stats.cancelled,
                    shares: stats.shares,
                    shared_bytes: stats.shared_bytes,
                    cache_hits: stats.cache_hits,
                    cache_misses: stats.cache_misses,
                    cache_bytes: stats.cache_bytes,
                    cache_mismatches: stats.cache_mismatches,
                    held: stats.held,
                    in_place: stats.in_place,
                })
            },
        )
        .expect("descriptor_bounded");
    }
}

pub(super) fn mesh_view(
    world: &World,
    assets: &AssetLibrary,
    shown: (MeshHandle, Option<MaterialHandle>),
    context: &Context<'_>,
) -> DrawnMeshView {
    let Context { scene, renderer } = *context;
    let residency = world.get_resource::<AssetResidency>();
    let (handle, held) = shown;
    let mesh = assets.mesh(handle).map(|mesh| {
        let origin = scene.and_then(|scene| scene.mesh_origin(handle));
        let bounds = mesh.bounds();
        MeshView {
            id: origin.map(|(id, _)| id.to_owned()),
            source: origin.map(|(_, source)| source.clone()),
            vertices: mesh.vertices().len(),
            indices: mesh.indices().len(),
            local_bounds: MeshBoundsView {
                min: bounds.min().to_array(),
                max: bounds.max().to_array(),
            },
            colors: mesh.colors().is_some(),
            uv1: mesh.uv1().is_some(),
            rendered: renderer.and_then(|_| {
                let resident = residency?.mesh(handle)?;
                Some(MeshRenderView {
                    levels: resident
                        .lods()
                        .iter()
                        .map(|&(_, indices, error)| LodLevelView { indices, error })
                        .collect(),
                })
            }),
        }
    });
    let material = held.and_then(|material| assets.material(material));
    let textures = material
        .map(|material| {
            material
                .maps()
                .iter()
                .map(|(_, map)| {
                    let origin = scene.and_then(|scene| scene.texture_origin(map.texture));
                    let texture = assets.texture(map.texture);
                    let (uri, inline, provided) = match origin.map(|(_, source)| source) {
                        Some(TextureOrigin::File { uri }) => (Some(uri.clone()), false, None),
                        Some(TextureOrigin::Inline) => (None, true, None),
                        Some(TextureOrigin::Provided { name }) => (None, false, Some(name.clone())),
                        None => (None, false, None),
                    };
                    TextureView {
                        id: origin.map(|(id, _)| id.to_owned()),
                        uri,
                        inline,
                        provided,
                        alive: texture.is_some(),
                        width: texture.map(|texture| texture.width()),
                        height: texture.map(|texture| texture.height()),
                        gpu_resident: renderer
                            .map(|renderer| renderer.texture_resident(map.texture)),
                    }
                })
                .collect()
        })
        .unwrap_or_default();
    let material = material.map(|material| {
        let id = scene
            .and_then(|scene| {
                scene
                    .materials()
                    .find(|&(_, material)| Some(material) == held)
            })
            .map(|(id, _)| id.to_owned());
        let texture = |handle| {
            scene
                .and_then(|scene| scene.texture_origin(handle))
                .map(|(id, _)| id.to_owned())
                .unwrap_or_default()
        };
        SceneMaterial::from_runtime(id.unwrap_or_default(), material, texture)
    });
    DrawnMeshView {
        mesh,
        material,
        textures,
    }
}
