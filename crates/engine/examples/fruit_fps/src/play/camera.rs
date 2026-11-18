use engine::prelude::Resource;

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
pub struct TestCamera {
    pub outside: bool,
    pub hud_hidden: bool,
}
