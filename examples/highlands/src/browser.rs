use engine::{Error, ErrorCode};
use wasm_bindgen::prelude::*;

use crate::{Highlands, Options, RockMaps};

#[wasm_bindgen]
#[derive(Default)]
pub struct HighlandsRocks {
    maps: RockMaps,
}

#[wasm_bindgen]
impl HighlandsRocks {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, name: &str, bytes: &[u8]) -> Result<(), JsValue> {
        let invalid =
            |message: String| Error::new(ErrorCode::InvalidTexture, format!("$.{name}"), message);
        if engine::host::image_format(bytes).map(|format| format.name) != Some("ktx2-rgba8") {
            return Err(invalid("ktx2-rgba8".to_owned()).into());
        }
        let image = engine::host::decode_image(name, bytes).map_err(invalid)?;
        self.maps.insert(name.to_owned(), image);
        Ok(())
    }
}

engine::export_game!(
    pub struct HighlandsGame(Highlands);
);

#[wasm_bindgen]
impl HighlandsGame {
    pub async fn create(
        canvas: web_sys::HtmlCanvasElement,
        rocks: HighlandsRocks,
        query: &str,
    ) -> Result<HighlandsGame, JsValue> {
        let options = page_options(query)?;
        let rocks = if options.rocks {
            rocks.maps
        } else {
            RockMaps::default()
        };
        Self::launch(canvas, move || Highlands::build(options, &rocks)).await
    }

    pub fn parameters() -> String {
        json(crate::parameters())
    }

    pub fn controls() -> String {
        json(crate::controls())
    }

    pub fn control(&mut self, name: &str, value: &str) -> Result<(), JsValue> {
        let (engine, highlands) = self.host_mut().parts_mut();
        highlands
            .control(engine, name, value)
            .map_err(JsValue::from)
    }
}

engine::__if_dev! {
    #[wasm_bindgen]
    impl HighlandsGame {
        pub async fn smokes(
            canvas: web_sys::HtmlCanvasElement,
            names: String,
            report: JsValue,
            query: String,
        ) -> Result<bool, JsValue> {
            let options = page_options(&query)?;
            let build = move |source: Option<&str>, without: &[&str]| match source {
                None => Ok((
                    engine::smokes::standard_engine(without)?,
                    Highlands::new(options.clone()),
                )),
                Some(_) => Err(
                    Error::new(ErrorCode::UnsupportedCommand, "$.source")
                        .expected(engine::Expected::Absent),
                ),
            };
            engine::smokes::page::run::<Highlands>(
                canvas,
                &names,
                &crate::smokes(),
                &build,
                &report,
            )
            .await
        }
    }
}

fn page_options(query: &str) -> Result<Options, JsValue> {
    let search = web_sys::UrlSearchParams::new_with_str(query)?;
    let mut options = Options::default();
    for (name, _) in crate::parameters() {
        if let Some(value) = search.get(name) {
            options.set(name, &value)?;
        }
    }
    Ok(options)
}

fn json(pairs: Vec<(&'static str, String)>) -> String {
    let object: serde_json::Map<String, serde_json::Value> = pairs
        .into_iter()
        .map(|(name, values)| (name.to_owned(), values.into()))
        .collect();
    serde_json::Value::Object(object).to_string()
}
