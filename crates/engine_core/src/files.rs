use std::sync::{Arc, Mutex, PoisonError};

use bevy_ecs::resource::Resource;
use bevy_ecs::world::World;

use crate::{Background, Delivers, Error, ErrorCode, Interest, Requests, Ticket};

type Landing = Arc<Mutex<Vec<(String, Ticket, Result<Vec<u8>, String>)>>>;

struct Reading;

#[derive(Resource, Default)]
pub struct Files {
    requests: Requests<String, Arc<[u8]>, Reading>,
    root: String,
    landing: Landing,
}

impl Files {
    pub fn set_root(&mut self, root: impl Into<String>) {
        self.root = root.into();
    }

    pub fn root(&self) -> &str {
        &self.root
    }

    pub fn read<T>(
        &mut self,
        path: &str,
        decode: impl FnOnce(&[u8]) -> Result<T, Error> + Send + Sync + 'static,
    ) -> Background<T, String> {
        Background::asked(self.requests.ask(path.to_owned()), move |bytes| decode(&bytes[..]))
    }

    pub fn census(&self) -> crate::Census {
        self.requests.census()
    }
}

impl Delivers<String> for Files {
    fn landed(&self, held: &Interest<String>) -> Option<Result<Arc<[u8]>, Error>> {
        self.requests.landed(held)
    }
}

fn lock<T>(held: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    held.lock().unwrap_or_else(PoisonError::into_inner)
}

pub(crate) fn read_files(world: &mut World) {
    let now = world
        .get_resource::<crate::FrameTime>()
        .map_or(0.0, |time| time.real_elapsed);
    let Some(mut files) = world.get_resource_mut::<Files>() else {
        return;
    };
    let files = &mut *files;
    files.requests.settle(now);
    let landed = std::mem::take(&mut *lock(&files.landing));
    for (path, ticket, read) in landed {
        match read {
            Ok(bytes) => {
                files.requests.ready(&path, ticket, Arc::from(bytes));
            }
            Err(error) => {
                let error = Error::new(ErrorCode::MissingFile, "$.files").got(error);
                files.requests.fail(&path, ticket, error);
            }
        }
    }
    let fresh: Vec<(String, Ticket)> = files
        .requests
        .asked()
        .filter(|(path, _)| files.requests.work(path).is_none())
        .map(|(path, ticket)| (path.clone(), ticket))
        .collect();
    for (path, ticket) in fresh {
        files.requests.set_work(&path, Reading);
        start(joined(&files.root, &path), (path, ticket), Arc::clone(&files.landing));
    }
}

fn joined(root: &str, path: &str) -> String {
    if root.is_empty() || root.ends_with('/') {
        format!("{root}{path}")
    } else {
        format!("{root}/{path}")
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn start(at: String, (path, ticket): (String, Ticket), landing: Landing) {
    let spawned = std::thread::Builder::new()
        .name("engine_file_read".to_owned())
        .spawn({
            let landing = Arc::clone(&landing);
            let path = path.clone();
            move || {
                let read = std::fs::read(&at).map_err(|error| format!("{at}: {error}"));
                lock(&landing).push((path, ticket, read));
            }
        });
    if let Err(error) = spawned {
        lock(&landing).push((path, ticket, Err(error.to_string())));
    }
}

#[cfg(target_arch = "wasm32")]
fn start(at: String, (path, ticket): (String, Ticket), landing: Landing) {
    wasm_bindgen_futures::spawn_local(async move {
        let read = fetched(&at).await.map_err(|error| format!("{at}: {error}"));
        lock(&landing).push((path, ticket, read));
    });
}

#[cfg(target_arch = "wasm32")]
async fn fetched(at: &str) -> Result<Vec<u8>, String> {
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::JsFuture;
    let window = web_sys::window().ok_or("no window")?;
    let response = JsFuture::from(window.fetch_with_str(at))
        .await
        .map_err(|error| format!("{error:?}"))?;
    let response: web_sys::Response = response
        .dyn_into()
        .map_err(|error| format!("{error:?}"))?;
    if !response.ok() {
        return Err(format!("status {}", response.status()));
    }
    let buffer = response.array_buffer().map_err(|error| format!("{error:?}"))?;
    let buffer = JsFuture::from(buffer)
        .await
        .map_err(|error| format!("{error:?}"))?;
    Ok(js_sys::Uint8Array::new(&buffer).to_vec())
}
