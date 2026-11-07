#[derive(Clone, Debug, Default, PartialEq)]
pub struct ScenePath {
    pub id: String,
    pub origin: [f32; 3],
    pub points: Vec<[f32; 2]>,
    pub at: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ScenePaths(pub Vec<ScenePath>);
