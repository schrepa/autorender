use engine::browser::SceneFileSet;
use wasm_bindgen::prelude::*;

use crate::{ScenePlayer, bricks};

engine::export_game!(
    pub struct PlayerGame(ScenePlayer);
);

#[wasm_bindgen]
impl PlayerGame {
    pub async fn create(
        canvas: web_sys::HtmlCanvasElement,
        source: &str,
        files: SceneFileSet,
        without: &str,
    ) -> Result<PlayerGame, JsValue> {
        let without = bricks(without, "?without")?;
        Self::launch(canvas, move || {
            ScenePlayer::new(source, files.files(), &without)
        })
        .await
    }

    pub fn scene_progress(&self) -> String {
        let host = self.host();
        host.game().progress(host.engine()).to_string()
    }
}
