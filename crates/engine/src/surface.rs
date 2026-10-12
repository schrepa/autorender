//! The methods every exported game has: one table, which
//! [`crate::export_game!`] turns into the game's `#[wasm_bindgen]` methods and
//! `runtime_typescript` into the page's `Runtime` interface, so the two
//! cannot differ. A method returning JSON text names the type it carries
//! (`-> String as Json<FrameStats>`), which the page reads with the
//! generated `parseJson`. `PageHost` implements each, mostly as `GameHost`.

/// Calls `$callback` with the game surface after `{ $context }`, in groups:
/// methods taking `&mut self` or `&self`, in every build, development builds
/// or profiling builds.
#[doc(hidden)]
#[macro_export]
macro_rules! __game_surface {
    ($callback:ident { $($context:tt)* }) => {
        $crate::$callback! {
            { $($context)* }
            always_mut {
                /// Runs one frame at `timestamp` on a `width` × `height` canvas
                /// with the input gathered since the last one.
                fn frame(timestamp: f64, width: u32, height: u32) -> Result<(), $crate::__private::JsValue>;
                /// A source (a keyboard key, a finger on a touch button) pressed
                /// (`down`) or released key `code` (`KeyboardEvent.code`); whether
                /// the game uses the key, which the page then keeps from the
                /// browser.
                fn key(code: &str, down: bool) -> bool;
                /// The view turned by a drag of `dx`, `dy` CSS pixels.
                fn look(dx: f32, dy: f32);
                /// Nothing is held: the page took input away.
                fn clear_input();
                /// Loading work in place of a frame while the GPU compiles pipelines.
                fn work(width: u32, height: u32);
                /// Terrain loading for `milliseconds` between frames; whether tiles remain.
                fn work_between_frames(milliseconds: f64) -> bool;
                /// Workers the page runs jobs on.
                fn set_job_workers(workers: u32);
                /// Cancels, shares and jobs to hand workers since the last call.
                fn take_jobs() -> $crate::__private::js_sys::Array;
                /// Job `id`'s output from its worker.
                fn complete_job(id: f64, output: Vec<u8>);
                /// Job `id` failed on its worker.
                fn fail_job(id: f64, error: String);
                /// A part of job `id`'s output handed back before the rest.
                fn part_job(id: f64, part: Vec<u8>);
                /// The page's job output cache counters.
                fn report_job_cache(hits: f64, misses: f64, bytes: f64, mismatches: f64);
                /// Drops suspension time and held input before the next frame.
                fn suspend();
                /// Puts the camera where the game starts.
                fn reset_camera();
                /// Replaces the authored scene with `source` and the files it
                /// references; a scene that fails keeps the current one.
                fn reload_scene(source: &str, files: $crate::browser::SceneFileSet) -> Result<(), $crate::__private::JsValue>;
                /// GPU (default) or exact CPU presentation of ordinary frames.
                fn set_gpu_presentation(enabled: bool);
                /// Splits fixed ticks across frames around their due time (default).
                fn set_tick_pipelining(enabled: bool);
                /// Temporal antialiasing of GPU-presented frames, over the
                /// world's setting in this and later engines.
                fn set_temporal_antialiasing(enabled: bool);
                /// Occlusion culling of GPU-presented frames, over the world's
                /// setting in this and later engines.
                fn set_occlusion_culling(enabled: bool);
                /// Sub-pixel culling of GPU-presented frames, over the world's
                /// setting in this and later engines.
                fn set_subpixel_culling(enabled: bool);
                /// Applies the engine switches the page's URL search part
                /// `query` names (`engine::host::SETTINGS`), in this and later
                /// engines; a value a switch does not take fails.
                fn apply_host_settings(query: &str) -> Result<(), $crate::__private::JsValue>;
            }
            always_ref {
                /// The last frame's statistics, as the JSON object `FrameStats` types.
                fn stats() -> String as Json<FrameStats>;
                /// Whether ordinary frames are presented on the GPU.
                fn gpu_presentation() -> bool;
                /// Whether fixed ticks are split across frames.
                fn tick_pipelining() -> bool;
                /// Whether GPU-presented frames are temporally antialiased.
                fn temporal_antialiasing() -> bool;
                /// Whether GPU-presented frames are occlusion culled.
                fn occlusion_culling() -> bool;
                /// Whether GPU-presented frames are sub-pixel culled.
                fn subpixel_culling() -> bool;
                /// The spread of measured `values`, as the JSON object
                /// `Distribution` types; `null` without values, or with one
                /// that is not a finite number of at least 0.
                fn distribution(values: Vec<f64>) -> String as Json<Option<Distribution>>;
                /// The physical `[width, height]` a view `width` × `height`
                /// CSS pixels across draws at, at `pixel_ratio` device pixels
                /// per CSS pixel and the user's resolution `scale`, as every
                /// host sizes views (`engine::host::view_size`); empty for a
                /// view with no area.
                fn view_size(width: f64, height: f64, pixel_ratio: f64, scale: f64) -> Vec<u32>;
            }
            dev_mut {
                /// Runs development request `request` (JSON) on a `width` ×
                /// `height` canvas; `paused` says whether the page is paused.
                fn dev_command(request: &str, width: u32, height: u32, paused: bool) -> String as Json<Reply>;
                /// Reloads the authored scene's saved `source` when the active
                /// revision is `expected_revision` and the saved one `revision`.
                fn dev_reload(source: &str, expected_revision: &str, revision: &str, files: Option<$crate::browser::SceneFileSet>) -> String as Json<Reply>;
            }
            profiling_mut {
                /// Starts a capture of at most `max_events` CPU events.
                fn profiling_start(max_events: u32) -> Result<(), $crate::__private::JsValue>;
                /// Stops the capture: its report, as JSON.
                fn profiling_stop() -> Result<String, $crate::__private::JsValue>;
                /// Collects finished GPU timings.
                fn profiling_poll();
                /// The capture's GPU report, as JSON.
                fn profiling_report() -> String;
            }
            profiling_ref {
                /// Whether a capture runs, as JSON.
                fn profiling_status() -> String;
            }
        }
    };
}

/// Exports game `$game` to the page as `#[wasm_bindgen]` struct `$name`
/// with every method of the game surface; the game adds its own methods,
/// constructors included, in its own `#[wasm_bindgen] impl $name`, which
/// start it with `$name::launch`. Profiling builds also count allocations.
/// Expands to nothing outside the browser.
#[macro_export]
macro_rules! export_game {
    ($(#[$meta:meta])* $vis:vis struct $name:ident($game:ty);) => {
        $crate::__game_surface!(__export_game { $(#[$meta])* $vis $name $game });
    };
}

#[cfg(target_arch = "wasm32")]
#[doc(hidden)]
#[macro_export]
macro_rules! __export_game {
    (
        { $(#[$meta:meta])* $vis:vis $name:ident $game:ty }
        always_mut { $( $(#[doc = $am_doc:literal])* fn $am:ident($($am_arg:ident: $am_ty:ty),*) $(-> $am_ret:ty $(as $am_json:ty)?)?; )* }
        always_ref { $( $(#[doc = $ar_doc:literal])* fn $ar:ident($($ar_arg:ident: $ar_ty:ty),*) $(-> $ar_ret:ty $(as $ar_json:ty)?)?; )* }
        dev_mut { $( $(#[doc = $dm_doc:literal])* fn $dm:ident($($dm_arg:ident: $dm_ty:ty),*) $(-> $dm_ret:ty $(as $dm_json:ty)?)?; )* }
        profiling_mut { $( $(#[doc = $pm_doc:literal])* fn $pm:ident($($pm_arg:ident: $pm_ty:ty),*) $(-> $pm_ret:ty $(as $pm_json:ty)?)?; )* }
        profiling_ref { $( $(#[doc = $pr_doc:literal])* fn $pr:ident($($pr_arg:ident: $pr_ty:ty),*) $(-> $pr_ret:ty $(as $pr_json:ty)?)?; )* }
    ) => {
        $(#[$meta])*
        #[$crate::__private::bindgen(wasm_bindgen = $crate::__private::wasm_bindgen)]
        $vis struct $name {
            host: $crate::browser::PageHost<$game>,
        }

        #[$crate::__private::bindgen(wasm_bindgen = $crate::__private::wasm_bindgen)]
        impl $name {
            $( $(#[doc = $am_doc])* pub fn $am(&mut self, $($am_arg: $am_ty),*) $(-> $am_ret)? { self.host.$am($($am_arg),*) } )*
            $( $(#[doc = $ar_doc])* pub fn $ar(&self, $($ar_arg: $ar_ty),*) $(-> $ar_ret)? { self.host.$ar($($ar_arg),*) } )*
        }

        $crate::__if_dev! {
            #[$crate::__private::bindgen(wasm_bindgen = $crate::__private::wasm_bindgen)]
            impl $name {
                $( $(#[doc = $dm_doc])* pub fn $dm(&mut self, $($dm_arg: $dm_ty),*) $(-> $dm_ret)? { self.host.$dm($($dm_arg),*) } )*
            }
        }

        $crate::__if_profiling! {
            #[$crate::__private::bindgen(wasm_bindgen = $crate::__private::wasm_bindgen)]
            impl $name {
                $( $(#[doc = $pm_doc])* pub fn $pm(&mut self, $($pm_arg: $pm_ty),*) $(-> $pm_ret)? { self.host.$pm($($pm_arg),*) } )*
                $( $(#[doc = $pr_doc])* pub fn $pr(&self, $($pr_arg: $pr_ty),*) $(-> $pr_ret)? { self.host.$pr($($pr_arg),*) } )*
            }

            #[global_allocator]
            static ALLOCATOR: $crate::profiling::TrackingAllocator<::std::alloc::System> =
                $crate::profiling::TrackingAllocator::new(::std::alloc::System);
        }

        impl $name {
            /// Starts the renderer of `canvas` and, while the browser compiles
            /// its pipelines, the engine and game `build` makes.
            pub async fn launch<E: Into<$crate::Error>>(
                canvas: $crate::__private::web_sys::HtmlCanvasElement,
                build: impl FnOnce() -> Result<($crate::Engine, $game), E>,
            ) -> Result<Self, $crate::__private::JsValue> {
                Ok(Self {
                    host: $crate::browser::PageHost::launch(canvas, build).await?,
                })
            }

            /// The engine and game the page runs.
            pub fn host(&self) -> &$crate::GameHost<$game> {
                &self.host
            }

            /// The engine and game the page runs, to change them.
            pub fn host_mut(&mut self) -> &mut $crate::GameHost<$game> {
                &mut self.host
            }
        }
    };
}

#[cfg(not(target_arch = "wasm32"))]
#[doc(hidden)]
#[macro_export]
macro_rules! __export_game {
    ($($surface:tt)*) => {};
}

/// The page's `Runtime` interface: the game surface in TypeScript, which
/// `cargo xtask schema` writes to `web/src/generated/runtime.ts`.
#[cfg(not(target_arch = "wasm32"))]
pub fn runtime_typescript() -> String {
    crate::__game_surface!(__surface_typescript {})
}

#[cfg(not(target_arch = "wasm32"))]
#[doc(hidden)]
#[macro_export]
macro_rules! __surface_typescript {
    (
        {}
        $( $group:ident { $( $(#[doc = $doc:literal])* fn $method:ident($($arg:ident: $ty:ty),*) $(-> $ret:ty $(as $json:ty)?)?; )* } )*
    ) => {{
        let methods = [$( $(
            $crate::surface::SurfaceMethod {
                group: stringify!($group),
                name: stringify!($method),
                doc: &[$($doc),*],
                arguments: &[$((stringify!($arg), stringify!($ty))),*],
                returns: [$(stringify!($ret))?].first().copied(),
                json: [$($(stringify!($json))?)?].first().copied(),
            },
        )* )*];
        $crate::surface::typescript(&methods)
    }};
}

/// A method of the game surface, as its table declares it.
#[cfg(not(target_arch = "wasm32"))]
#[doc(hidden)]
pub struct SurfaceMethod {
    pub group: &'static str,
    pub name: &'static str,
    pub doc: &'static [&'static str],
    pub arguments: &'static [(&'static str, &'static str)],
    pub returns: Option<&'static str>,
    /// The JSON type the method's text carries, as `Json<FrameStats>`.
    pub json: Option<&'static str>,
}

/// `methods` as the TypeScript interface `Runtime`: methods of development
/// and profiling builds are optional.
#[cfg(not(target_arch = "wasm32"))]
#[doc(hidden)]
pub fn typescript(methods: &[SurfaceMethod]) -> String {
    use std::fmt::Write;
    let mut output = String::from(
        "// Generated by `cargo xtask schema` from the engine's game surface\n\
         // (`engine::export_game!`); do not edit.\n\
         import type { JobMessage } from \"../engine/jobs\";\n\
         import type { Distribution } from \"./host\";\n\
         import type { Reply } from \"./reply\";\n\
         import type { FrameStats } from \"./stats\";\n\n\
         /** JSON text the engine wrote of a `T`, which `parseJson` reads. */\n\
         export type Json<T> = string & { readonly json?: T };\n\n\
         /** The value the engine wrote as `json`. */\n\
         export function parseJson<T>(json: Json<T>): T {\n  \
         return JSON.parse(json) as T;\n\
         }\n\n\
         /** The methods every game the engine exports has; those of development\n \
         * and profiling builds are optional. */\n\
         export interface Runtime {\n",
    );
    for method in methods {
        if !method.doc.is_empty() {
            writeln!(output, "  /**{} */", method.doc.concat()).expect("strings write");
        }
        let optional = if method.group.starts_with("always") {
            ""
        } else {
            "?"
        };
        let arguments: Vec<String> = method
            .arguments
            .iter()
            .map(|(name, ty)| {
                let ty = ty.replace(' ', "");
                match ty
                    .strip_prefix("Option<")
                    .and_then(|ty| ty.strip_suffix('>'))
                {
                    Some(inner) => format!("{name}?: {}", typescript_type(inner)),
                    None => format!("{name}: {}", typescript_type(&ty)),
                }
            })
            .collect();
        let returns = match method.json {
            Some(json) => json_type(&json.replace(' ', "")),
            None => method.returns.map_or("void".to_owned(), |ty| {
                let ty = ty.replace(' ', "");
                match ty.strip_prefix("Result<").and_then(|ty| ty.split_once(',')) {
                    Some((ok, _)) => typescript_type(ok),
                    None => typescript_type(&ty),
                }
            }),
        };
        writeln!(
            output,
            "  {}{optional}({}): {returns};",
            method.name,
            arguments.join(", ")
        )
        .expect("strings write");
    }
    output.push_str("  free(): void;\n}\n");
    output
}

/// The TypeScript type of the JSON text `json` (`Json<T>`, spaces removed)
/// names: `Option<T>` is `T | null`; the payload types are the page's
/// generated ones, which the interface's file imports.
#[cfg(not(target_arch = "wasm32"))]
fn json_type(json: &str) -> String {
    let payload = json
        .strip_prefix("Json<")
        .and_then(|json| json.strip_suffix('>'))
        .unwrap_or_else(|| panic!("a method's JSON type is Json<T>, not {json}"));
    match payload
        .strip_prefix("Option<")
        .and_then(|payload| payload.strip_suffix('>'))
    {
        Some(inner) => format!("Json<{inner} | null>"),
        None => format!("Json<{payload}>"),
    }
}

/// The TypeScript type of Rust type `ty` (spaces removed) at the Wasm
/// boundary; a type the table adds without a mapping here is a programming
/// error.
#[cfg(not(target_arch = "wasm32"))]
fn typescript_type(ty: &str) -> String {
    let leaf = ty.rsplit("::").next().unwrap_or(ty);
    match leaf {
        "()" => "void",
        "f32" | "f64" | "u32" | "i32" => "number",
        "bool" => "boolean",
        "String" | "&str" => "string",
        "Vec<u8>" => "Uint8Array",
        "Vec<f64>" => "Float64Array",
        "Vec<u32>" => "Uint32Array",
        "Array" => "JobMessage[]",
        // Scene files are the page's bindings' own class.
        "SceneFileSet" => "unknown",
        _ => panic!("the game surface has no TypeScript type for {ty}"),
    }
    .to_owned()
}
