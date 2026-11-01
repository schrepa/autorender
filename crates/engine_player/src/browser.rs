use wasm_bindgen::prelude::*;

use crate::{ScenePlayer, listed};

engine::export_game!(
    pub struct PlayerGame(ScenePlayer) with crate::bricks;
);

#[wasm_bindgen]
impl PlayerGame {
    pub async fn create(
        canvas: web_sys::HtmlCanvasElement,
        source: &str,
        files: SceneFileSet,
        without: &str,
    ) -> Result<PlayerGame, JsValue> {
        let bricks = listed(without, "?without").map_err(engine::settings::js_error)?;
        Self::launch(canvas, bricks, move |engine| {
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
            let list = crate::bricks(engine::BrickList::default());
            let entries = crate::smokes(&list).map_err(engine::settings::js_error)?;
            engine::smokes::page::run::<engine::smokes::Played<ScenePlayer>>(
                canvas,
                &names,
                &entries,
                &list,
                &|source, engine| {
                    let world = engine.app.world_mut();
                    if source.is_none() && engine::host::has_setting(world, crate::player::FLIES) {
                        engine::host::apply_setting(
                            world,
                            crate::player::FLIES,
                            &serde_json::json!(false),
                        )?;
                    }
                    engine::smokes::played(source, engine, ScenePlayer::new)
                },
                &report,
            )
            .await
        }
    }
}
