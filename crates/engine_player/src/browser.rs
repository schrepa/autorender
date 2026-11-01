use wasm_bindgen::prelude::*;

use crate::ScenePlayer;

engine::export_game!(
    pub struct PlayerGame(ScenePlayer);
);

#[wasm_bindgen]
impl PlayerGame {
    pub async fn create(
        canvas: web_sys::HtmlCanvasElement,
        source: &str,
        files: SceneFileSet,
    ) -> Result<PlayerGame, JsValue> {
        Self::launch(canvas, engine_player_bricks::bricks, move |engine| {
            ScenePlayer::new(source, files.files(), engine)
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
            let list = engine_player_bricks::bricks(engine::BrickList::default());
            let entries = engine::smokes::registered(&list).map_err(engine::settings::js_error)?;
            engine::smokes::page::run::<engine::smokes::Played<ScenePlayer>>(
                canvas,
                &names,
                &entries,
                &list,
                &|source, engine| engine::smokes::played(source, engine, ScenePlayer::new),
                &report,
            )
            .await
        }
    }
}
