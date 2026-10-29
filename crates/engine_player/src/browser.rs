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
                &|source, without| engine::smokes::played(source, without, ScenePlayer::new),
                &report,
            )
            .await
        }
    }
}
