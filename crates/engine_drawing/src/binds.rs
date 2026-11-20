use std::collections::BTreeMap;

use engine_core::collections::{Cursor, Since};
use engine_core::{Asset, Assets, Material, Texture, TextureRef};

struct Held<K, T> {
    revision: u64,
    key: K,
    textures: Vec<(Asset<Texture>, u64)>,
    bound: Option<T>,
}

pub struct MaterialBinds<K, T> {
    held: BTreeMap<Asset<Material>, Held<K, T>>,
    cursor: Cursor,
}

impl<K, T> Default for MaterialBinds<K, T> {
    fn default() -> Self {
        Self {
            held: BTreeMap::new(),
            cursor: Cursor::default(),
        }
    }
}

impl<K: Copy + PartialEq, T> MaterialBinds<K, T> {
    pub fn changed(&mut self, assets: &Assets) -> bool {
        !matches!(assets.changed(&mut self.cursor), Since::Same)
    }

    pub fn dropped(
        &mut self,
        assets: &Assets,
        wanted: &BTreeMap<Asset<Material>, K>,
    ) -> Vec<(Asset<Material>, Option<T>)> {
        let gone: Vec<Asset<Material>> = self
            .held
            .iter()
            .filter(|(material, held)| {
                wanted.get(material) != Some(&held.key)
                    || !assets.ready(**material)
                    || held.textures.iter().any(|(texture, revision)| {
                        assets.revision_of(*texture) != Some(*revision)
                    })
            })
            .map(|(material, _)| *material)
            .collect();
        gone.into_iter()
            .filter_map(|material| {
                let held = self.held.remove(&material)?;
                Some((material, held.bound))
            })
            .collect()
    }

    pub fn fresh<'a>(
        &self,
        assets: &'a Assets,
        wanted: &BTreeMap<Asset<Material>, K>,
    ) -> Vec<(Asset<Material>, u64, K, &'a Material)> {
        wanted
            .iter()
            .filter_map(|(material, key)| {
                let revision = assets.revision_of(*material)?;
                if !assets.ready(*material)
                    || self
                        .held
                        .get(material)
                        .is_some_and(|held| held.revision == revision)
                {
                    return None;
                }
                let value = assets.get(*material).ok()?;
                Some((*material, revision, *key, value))
            })
            .collect()
    }

    pub fn keep(
        &mut self,
        assets: &Assets,
        (material, revision, key): (Asset<Material>, u64, K),
        textures: &[TextureRef],
        bound: Option<T>,
    ) -> Option<T> {
        let textures = textures
            .iter()
            .map(|map| {
                let known = assets.revision_of(map.texture).unwrap_or_default();
                (map.texture, known)
            })
            .collect();
        let held = Held {
            revision,
            key,
            textures,
            bound,
        };
        self.held.insert(material, held)?.bound
    }

    pub fn forget(&mut self, material: Asset<Material>) -> bool {
        self.held.remove(&material).is_some()
    }

    pub fn bound(&self, material: Asset<Material>) -> Option<&T> {
        self.held.get(&material)?.bound.as_ref()
    }

    pub fn each_bound(&self) -> impl Iterator<Item = (Asset<Material>, &T)> + '_ {
        self.held
            .iter()
            .filter_map(|(material, held)| Some((*material, held.bound.as_ref()?)))
    }

    pub fn clear(&mut self) -> Vec<Asset<Material>> {
        std::mem::take(&mut self.held).into_keys().collect()
    }
}
