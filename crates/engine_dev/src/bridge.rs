//! The development bridge between the engine and the host serving a
//! request: the engine checks each request and runs it, and the control it
//! names step by step ([`Step`]). Where a step needs what only the host
//! does, it hands the host an [`Effect`], which the host carries out and
//! answers with its [`Outcome`]. Hosts keep their transports (how requests
//! arrive and replies leave, how they wait for a frame, PNG files, abort
//! signals); what each control does, and in what order, is the engine's.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// What the engine answers a request, or the effect it waits for: the
/// request's reply, finished as every one is (its result with the
/// observation again, whether the host is paused, the authored scene's
/// source and the last frame's statistics), or what the host does next.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum Step {
    /// The request's reply.
    Reply(#[cfg_attr(not(target_arch = "wasm32"), schemars(with = "crate::Reply"))] Value),
    /// What the host does next, then resumes the request with its outcome.
    Effect(Effect),
}

/// What a host does for the request under way.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Effect {
    /// Pause the host's frames, or let them run again. Outcome: none.
    Pause { paused: bool },
    /// Wait for the host's next frame: the page's next animation frame,
    /// at most `within_ms`; natively until the GPU finished its work.
    /// Outcome: `now`, when the wait ended on the host's clock
    /// (`engine::host::now_ms`), or `interrupted` where the view stopped
    /// showing meanwhile.
    Wait { within_ms: f64 },
    /// Run an ordinary frame at `timestamp` without input, on the page
    /// after its next animation frame. Outcome: `failed` where it failed.
    Frame { timestamp: f64 },
    /// Show the last frame as drawn: the page waits two animation frames;
    /// a native host writes it as a PNG file. Outcome: the `artifact`
    /// written, or `failed`.
    Show,
    /// Run the game's own control `name` on `args`. Outcome: its `result`;
    /// `failed`, why the game refused; or `missing`, the controls the game
    /// has, where it has no `name`.
    Game {
        name: String,
        args: Map<String, Value>,
    },
    /// Read the authored scene and the files it references again, and hand
    /// them to the engine's reload. Outcome, where it cannot: `failed`,
    /// why they could not be read, or `missing` (empty) where the game has
    /// no authored scene.
    Source,
}

/// How a host's effect went: each [`Effect`] says what it sets.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct Outcome {
    /// When a wait ended, in milliseconds on the host's clock.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub now: Option<f64>,
    /// The view stopped showing during a wait, as a hidden page does.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub interrupted: bool,
    /// Why the effect failed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failed: Option<String>,
    /// The PNG file a host wrote of the frame it showed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact: Option<Artifact>,
    /// What the game's control returned.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    /// What the game has instead of what the effect named: its controls'
    /// names, or none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub missing: Option<Vec<String>>,
}

/// A file a host wrote for a request.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct Artifact {
    /// Where it is.
    pub path: String,
    /// Its MIME type, such as `image/png`.
    pub mime_type: String,
    /// Its size in bytes.
    pub bytes: u64,
    /// Its SHA-256, in lowercase hexadecimal.
    pub sha256: String,
}

/// The host's side of each call the bridge takes: its view, whether its
/// frames are paused, and what it is.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct HostContext {
    /// The view's width, in physical pixels.
    pub width: u32,
    /// The view's height, in physical pixels.
    pub height: u32,
    /// Whether the host's frames are paused.
    pub paused: bool,
    /// Device pixels per CSS pixel; 1 natively.
    pub device_pixel_ratio: f64,
    /// The host: the browser's user agent, or `engine_native`.
    pub user_agent: String,
    /// The authored scene the host reloads, where it has one: its URL or
    /// path.
    #[serde(default)]
    pub source: Option<String>,
    /// What a benchmark's frame intervals include on this host.
    pub timing: String,
}
