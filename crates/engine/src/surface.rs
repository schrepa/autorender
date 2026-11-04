#[doc(hidden)]
#[macro_export]
macro_rules! __game_surface {
    ($callback:ident { $($context:tt)* }) => {
        $crate::$callback! {
            { $($context)* }
            always_mut {
                fn frame(timestamp: f64, width: u32, height: u32) -> Result<(), $crate::__private::JsValue>;
                fn key(code: &str, down: bool) -> bool;
                fn look(dx: f32, dy: f32);
                fn motion(dx: f32, dy: f32);
                fn pointer(x: f32, y: f32);
                fn pointer_left();
                fn wheel(right_notches: f32, up_notches: f32);
                fn pad(buttons: u32, axes: Vec<f32>);
                fn clear_input();
                fn animation_frame(timestamp: f64, width: u32, height: u32) -> Result<u32, $crate::__private::JsValue>;
                fn work_while_compiling(width: u32, height: u32) -> bool;
                fn load_between_frames() -> bool;
                fn set_preparation(enabled: bool);
                fn set_job_workers(workers: u32);
                fn set_job_cache(caching: bool, verify: bool);
                fn take_jobs() -> $crate::__private::js_sys::Array;
                fn job_kept(id: f64, key: String, output: Option<Vec<u8>>);
                fn lose_job_worker(worker: u32, error: &str, replaced: bool);
                fn job_task() -> Option<u32>;
                fn job_task_done(worker: u32);
                fn complete_job(id: f64, output: Vec<u8>);
                fn fail_job(id: f64, error: String);
                fn report_job_cache(hits: f64, misses: f64, bytes: f64, mismatches: f64);
                fn suspend();
                fn reset_camera();
                fn reload_scene(source: &str, files: SceneFileSet) -> Result<(), $crate::__private::JsValue>;
                fn take_level_request() -> Option<String>;
                fn report_level(done: u32, total: u32);
                fn fail_level(message: &str);
                fn set_pixel_ratio(ratio: f64);
                fn set_setting(key: &str, value: &str) -> Result<(), $crate::__private::JsValue>;
                fn apply_host_settings(query: &str) -> Result<(), $crate::__private::JsValue>;
            }
            always_ref {
                fn stats() -> String as Json<FrameStats>;
                fn cursor_mode() -> u32;
                fn job_hash() -> Vec<u8>;
                fn distribution(values: Vec<f64>) -> String as Json<Option<Distribution>>;
            }
            dev_mut {
                fn dev_request(line: &str, context: &str) -> String as Json<Step>;
                fn dev_resume(outcome: &str, context: &str) -> String as Json<Step>;
                fn dev_reloaded(source: &str, context: &str, files: Option<SceneFileSet>) -> String as Json<Step>;
                fn dev_refusal(error: &str) -> String as Json<Reply>;
            }
            profiling_mut {
                fn profiling_start(max_events: u32) -> Result<(), $crate::__private::JsValue>;
                fn profiling_stop() -> Result<String, $crate::__private::JsValue>;
                fn profiling_poll();
                fn profiling_report() -> String;
            }
            profiling_ref {
                fn profiling_status() -> String;
            }
        }
    };
}

#[macro_export]
macro_rules! game {
    ($vis:vis struct $name:ident($game:ty), $title:literal, $bricks:path, $build:path) => {
        $crate::export_game!($vis struct $name($game););

        $crate::__worker::export_worker!($bricks);

        #[cfg(target_arch = "wasm32")]
        #[$crate::__private::bindgen(wasm_bindgen = $crate::__private::wasm_bindgen)]
        impl $name {
            pub async fn create(
                canvas: $crate::__private::web_sys::HtmlCanvasElement,
            ) -> Result<$name, $crate::__private::JsValue> {
                Self::launch(canvas, $bricks, $build).await
            }
        }

        #[cfg(not(target_arch = "wasm32"))]
        pub fn native(arguments: impl Iterator<Item = String>) -> ::std::process::ExitCode {
            ::engine_native::main($title, arguments, $bricks, $build)
        }
    };
}

#[macro_export]
macro_rules! game_main {
    ($run:path) => {
        #[cfg(not(target_arch = "wasm32"))]
        fn main() -> ::std::process::ExitCode {
            $run(::std::env::args().skip(1))
        }

        #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
        #[global_allocator]
        static ALLOCATOR: $crate::profiling::TrackingAllocator<::std::alloc::System> =
            $crate::profiling::TrackingAllocator::new(::std::alloc::System);

        #[cfg(target_arch = "wasm32")]
        fn main() {}
    };
}

#[macro_export]
macro_rules! export_game {
    ($(#[$meta:meta])* $vis:vis struct $name:ident($game:ty);) => {
        $crate::__game_surface!(__export_game { $(#[$meta])* $vis $name $game [] });
    };
    ($(#[$meta:meta])* $vis:vis struct $name:ident($game:ty) with $bricks:path;) => {
        $crate::__game_surface!(__export_game { $(#[$meta])* $vis $name $game [$bricks] });
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __listed {
    () => {
        ::core::option::Option::None
    };
    ($bricks:path) => {
        ::core::option::Option::Some($bricks as fn($crate::BrickList) -> $crate::BrickList)
    };
}

#[cfg(target_arch = "wasm32")]
#[doc(hidden)]
#[macro_export]
macro_rules! __export_game {
    (
        { $(#[$meta:meta])* $vis:vis $name:ident $game:ty [$($bricks:path)?] }
        always_mut { $( fn $am:ident($($am_arg:ident: $am_ty:ty),*) $(-> $am_ret:ty $(as $am_json:ty)?)?; )* }
        always_ref { $( fn $ar:ident($($ar_arg:ident: $ar_ty:ty),*) $(-> $ar_ret:ty $(as $ar_json:ty)?)?; )* }
        dev_mut { $( fn $dm:ident($($dm_arg:ident: $dm_ty:ty),*) $(-> $dm_ret:ty $(as $dm_json:ty)?)?; )* }
        profiling_mut { $( fn $pm:ident($($pm_arg:ident: $pm_ty:ty),*) $(-> $pm_ret:ty $(as $pm_json:ty)?)?; )* }
        profiling_ref { $( fn $pr:ident($($pr_arg:ident: $pr_ty:ty),*) $(-> $pr_ret:ty $(as $pr_json:ty)?)?; )* }
    ) => {
        #[$crate::__private::bindgen(wasm_bindgen = $crate::__private::wasm_bindgen)]
        #[derive(Default)]
        pub struct SceneFileSet {
            files: $crate::scene::SceneFiles,
        }

        #[$crate::__private::bindgen(wasm_bindgen = $crate::__private::wasm_bindgen)]
        impl SceneFileSet {
            #[wasm_bindgen(constructor)]
            pub fn new() -> Self {
                Self::default()
            }

            pub fn add_image(&mut self, uri: &str, width: u32, height: u32, pixels: Vec<u8>) {
                self.files.add_image(uri, width, height, pixels);
            }

            pub fn add_buffer(&mut self, uri: &str, bytes: Vec<u8>) {
                self.files.add_buffer(uri, bytes);
            }
        }

        impl SceneFileSet {
            pub fn files(&self) -> &$crate::scene::SceneFiles {
                &self.files
            }
        }

        impl ::core::convert::From<SceneFileSet> for $crate::scene::SceneFiles {
            fn from(set: SceneFileSet) -> Self {
                set.files
            }
        }

        #[$crate::__private::bindgen(wasm_bindgen = $crate::__private::wasm_bindgen, js_name = SceneLoad)]
        pub struct PageSceneLoad {
            load: $crate::browser::SceneLoading,
        }

        #[$crate::__private::bindgen(wasm_bindgen = $crate::__private::wasm_bindgen, js_class = SceneLoad)]
        impl PageSceneLoad {
            #[wasm_bindgen(constructor)]
            pub fn new(source: &str) -> Result<PageSceneLoad, $crate::__private::JsValue> {
                Ok(Self {
                    load: $crate::browser::SceneLoading::new(source, $crate::__listed!($($bricks)?))?,
                })
            }

            pub fn take_reads(
                &mut self,
            ) -> Result<$crate::__private::js_sys::Array, $crate::__private::JsValue> {
                self.load.take_reads()
            }

            pub fn read(&mut self, uri: &str, bytes: Vec<u8>) -> Result<bool, $crate::__private::JsValue> {
                self.load.read(uri, bytes)
            }

            pub fn image(
                &mut self,
                uri: &str,
                width: u32,
                height: u32,
                pixels: Vec<u8>,
            ) -> Result<(), $crate::__private::JsValue> {
                self.load.image(uri, width, height, pixels)
            }

            pub fn decode(&mut self, uri: &str, bytes: &[u8]) -> Result<(), $crate::__private::JsValue> {
                self.load.decode(uri, bytes)
            }

            pub fn finish(&mut self) -> Result<SceneFileSet, $crate::__private::JsValue> {
                Ok(SceneFileSet {
                    files: self.load.finish()?,
                })
            }
        }

        $(#[$meta])*
        #[$crate::__private::bindgen(wasm_bindgen = $crate::__private::wasm_bindgen)]
        $vis struct $name {
            host: $crate::browser::PageHost<$game>,
        }

        #[$crate::__private::bindgen(wasm_bindgen = $crate::__private::wasm_bindgen)]
        impl $name {
            $( pub fn $am(&mut self, $($am_arg: $am_ty),*) $(-> $am_ret)? { self.host.$am($($am_arg),*) } )*
            $( pub fn $ar(&self, $($ar_arg: $ar_ty),*) $(-> $ar_ret)? { self.host.$ar($($ar_arg),*) } )*
        }

        $crate::__if_dev! {
            #[$crate::__private::bindgen(wasm_bindgen = $crate::__private::wasm_bindgen)]
            impl $name {
                $( pub fn $dm(&mut self, $($dm_arg: $dm_ty),*) $(-> $dm_ret)? { self.host.$dm($($dm_arg),*) } )*
            }
        }

        $crate::__if_profiling! {
            #[$crate::__private::bindgen(wasm_bindgen = $crate::__private::wasm_bindgen)]
            impl $name {
                $( pub fn $pm(&mut self, $($pm_arg: $pm_ty),*) $(-> $pm_ret)? { self.host.$pm($($pm_arg),*) } )*
                $( pub fn $pr(&self, $($pr_arg: $pr_ty),*) $(-> $pr_ret)? { self.host.$pr($($pr_arg),*) } )*
            }

            #[global_allocator]
            static ALLOCATOR: $crate::profiling::TrackingAllocator<::std::alloc::System> =
                $crate::profiling::TrackingAllocator::new(::std::alloc::System);
        }

        impl $name {
            pub async fn launch<E: Into<$crate::Error>>(
                canvas: $crate::__private::web_sys::HtmlCanvasElement,
                bricks: impl FnOnce($crate::BrickList) -> $crate::BrickList,
                build: impl FnOnce(&mut $crate::Engine) -> Result<$game, E>,
            ) -> Result<Self, $crate::__private::JsValue> {
                Ok(Self {
                    host: $crate::browser::PageHost::launch(canvas, bricks, build).await?,
                })
            }

            pub fn host(&self) -> &$crate::GameHost<$game> {
                &self.host
            }

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
        $( $group:ident { $( fn $method:ident($($arg:ident: $ty:ty),*) $(-> $ret:ty $(as $json:ty)?)?; )* } )*
    ) => {{
        let methods = [$( $(
            $crate::surface::SurfaceMethod {
                group: stringify!($group),
                name: stringify!($method),
                arguments: &[$((stringify!($arg), stringify!($ty))),*],
                returns: [$(stringify!($ret))?].first().copied(),
                json: [$($(stringify!($json))?)?].first().copied(),
            },
        )* )*];
        $crate::surface::typescript(&methods)
    }};
}

#[cfg(not(target_arch = "wasm32"))]
#[doc(hidden)]
pub struct SurfaceMethod {
    pub group: &'static str,
    pub name: &'static str,
    pub arguments: &'static [(&'static str, &'static str)],
    pub returns: Option<&'static str>,
    pub json: Option<&'static str>,
}

#[cfg(not(target_arch = "wasm32"))]
#[doc(hidden)]
pub fn typescript(methods: &[SurfaceMethod]) -> String {
    use std::fmt::Write;
    let mut output = String::from(
        "import type { JobOrder } from \"../engine/jobs\";\n\
         import type { Distribution } from \"./host\";\n\
         import type { Reply, Step } from \"./reply\";\n\
         import type { FrameStats } from \"./stats\";\n\n\
         export type Json<T> = string & { readonly json?: T };\n\n\
         export function parseJson<T>(json: Json<T>): T {\n  \
         return JSON.parse(json) as T;\n\
         }\n\n\
         export interface Runtime {\n",
    );
    for method in methods {
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
                if let Some((ok, _)) = ty.strip_prefix("Result<").and_then(|ty| ty.split_once(','))
                {
                    typescript_type(ok)
                } else if let Some(inner) = ty
                    .strip_prefix("Option<")
                    .and_then(|ty| ty.strip_suffix('>'))
                {
                    format!("{} | undefined", typescript_type(inner))
                } else {
                    typescript_type(&ty)
                }
            }),
        };
        writeln!(
            output,
            "  {}{optional}({}): {returns};",
            method.name,
            arguments.join(", ")
        )
        .expect("string_writes");
    }
    output.push_str("  free(): void;\n}\n");
    output
}

#[cfg(not(target_arch = "wasm32"))]
fn json_type(json: &str) -> String {
    let payload = json
        .strip_prefix("Json<")
        .and_then(|json| json.strip_suffix('>'))
        .unwrap_or_else(|| panic!("json_method_type {json}"));
    match payload
        .strip_prefix("Option<")
        .and_then(|payload| payload.strip_suffix('>'))
    {
        Some(inner) => format!("Json<{inner} | null>"),
        None => format!("Json<{payload}>"),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn typescript_type(ty: &str) -> String {
    let leaf = ty.rsplit("::").next().unwrap_or(ty);
    match leaf {
        "()" => "void",
        "f32" | "f64" | "u32" | "i32" => "number",
        "bool" => "boolean",
        "String" | "&str" => "string",
        "Vec<u8>" => "Uint8Array",
        "Vec<f32>" => "Float32Array",
        "Vec<f64>" => "Float64Array",
        "Vec<u32>" => "Uint32Array",
        "Array" => "JobOrder[]",
        "SceneFileSet" => "unknown",
        _ => panic!("surface_typescript_type {ty}"),
    }
    .to_owned()
}
