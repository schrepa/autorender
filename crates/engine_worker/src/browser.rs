use js_sys::wasm_bindgen::JsValue;
use js_sys::{Object, Reflect, Uint8Array};

use crate::{Image, Report};

fn set(object: &Object, field: &str, value: &JsValue) {
    let _ = Reflect::set(object, &JsValue::from_str(field), value);
}

pub fn report(report: Option<Report>) -> JsValue {
    let Some(report) = report else {
        return JsValue::UNDEFINED;
    };
    let object = Object::new();
    let (kind, id) = match &report {
        Report::Done(id, output) => {
            set(&object, "output", &Uint8Array::from(output.as_slice()));
            ("done", *id)
        }
        Report::Failed(id, error) => {
            set(&object, "error", &JsValue::from_str(&error.to_string()));
            ("failed", *id)
        }
    };
    set(&object, "type", &JsValue::from_str(kind));
    set(&object, "id", &JsValue::from_f64(id as f64));
    object.into()
}

pub fn image(image: &Image) -> JsValue {
    let object = Object::new();
    set(&object, "width", &JsValue::from(image.width));
    set(&object, "height", &JsValue::from(image.height));
    set(&object, "pixels", &Uint8Array::from(image.pixels.as_slice()));
    object.into()
}
