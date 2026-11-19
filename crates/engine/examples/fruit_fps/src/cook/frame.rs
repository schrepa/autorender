use cook::{Kind, Refusal, Step};
use serde_json::{Value, json};

pub static FRAME: Kind = Kind {
    name: "level_frame",
    version: 1,
    run: frame,
};

fn frame(step: &mut Step<'_>) -> Result<(), Refusal> {
    let [input] = step.inputs() else {
        return Err(Refusal::new("FrameInputs", step.name(), "one data file"));
    };
    let data: Value = serde_json::from_slice(input.bytes)
        .map_err(|error| Refusal::new("FrameData", input.name, error))?;
    let frame = json!({"frame": data["frame"], "bounds": data["bounds"], "raised": data["raised"], "collision": data["collision"]});
    step.pass("frame.json", frame.to_string().into_bytes());
    Ok(())
}
