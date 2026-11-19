use bevy_ecs::entity::Entity;
use engine_place::{Changes, NodePose, PoseNodes, PoseRows};

use super::GlobalTransforms;

impl PoseNodes for GlobalTransforms {
    fn count(&self) -> usize {
        self.links.len()
    }

    fn local(&self, node: usize) -> &NodePose {
        &self.locals[node]
    }

    fn rows(&self, node: usize) -> Option<&PoseRows> {
        self.links[node].valid.then(|| &self.globals[node])
    }

    fn parent(&self, node: usize) -> Option<usize> {
        let link = &self.links[node];
        link.parent().filter(|_| link.topology_valid)
    }

    fn children(&self, node: usize) -> impl Iterator<Item = usize> + '_ {
        std::iter::successors(self.links[node].first_child(), |&child| {
            self.links[child].next_sibling()
        })
    }

    fn index_of(&self, node: Entity) -> Option<usize> {
        self.node_index(node)
    }

    fn entity_at(&self, node: usize) -> Option<Entity> {
        self.live(node).then(|| self.entities[node])
    }

    fn moved(&self) -> &Changes {
        &self.moved
    }

    fn placed(&self) -> &Changes {
        &self.placed
    }
}
