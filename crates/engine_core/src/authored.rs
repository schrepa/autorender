use bevy_ecs::prelude::{Entity, Resource, World};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AuthoredIdentity {
    pub id: Option<String>,
    pub scene: Option<String>,
    pub name: Option<String>,
    pub in_active_scene: bool,
}

pub trait AuthoredScene: Send + Sync + 'static {
    fn entity_count(&self) -> Option<usize>;
    fn entity(&self, world: &World, id: &str) -> Option<Entity>;
    fn identity(&self, world: &World, entity: Entity) -> AuthoredIdentity;
}

#[derive(Resource)]
pub struct ActiveScene(pub Box<dyn AuthoredScene>);

impl ActiveScene {
    pub fn of(world: &World) -> Option<&dyn AuthoredScene> {
        world
            .get_resource::<Self>()
            .map(|active| active.0.as_ref())
    }
}
