import package::engine::post::pixel::{PostPixel, SceneLight};

fn resolve(at: PostPixel, scene: SceneLight) -> SceneLight {
    return scene;
}

fn resolved_at(at: PostPixel) -> PostPixel {
    return at;
}
