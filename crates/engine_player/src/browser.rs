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
        let bricks = bricks(without, "?without")?;
        Self::launch(canvas, move || ScenePlayer::new(source, files.files(), bricks)).await
    }

    pub fn scene_progress(&self) -> String {
        let host = self.host();
        host.game().progress(host.engine()).to_string()
    }
}

engine::__if_dev! {
    #[wasm_bindgen]
    impl PlayerGame {
        pub async fn smokes(
            canvas: web_sys::HtmlCanvasElement,
            names: String,
            report: JsValue,
        ) -> Result<bool, JsValue> {
            engine::smokes::page::run::<engine::smokes::Played<ScenePlayer>>(
                canvas,
                &names,
                &engine::smokes::standard(),
                &engine::bricks::standard(),
                &|source, bricks| engine::smokes::played(source, bricks, ScenePlayer::new),
                &report,
            )
            .await
        }
    }
}
