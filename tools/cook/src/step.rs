use std::fs;
use std::path::{Component, Path, PathBuf};

use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::plan::Entry;
use crate::store::{Fact, Made, canonical, local, sha};
use crate::{Input, Refusal};

pub(crate) struct Resolved {
    pub(crate) name: String,
    pub(crate) sha256: String,
    pub(crate) path: PathBuf,
}

pub struct Step<'a> {
    name: &'a str,
    params: &'a Value,
    inputs: Vec<Input<'a>>,
    files: PathBuf,
    made: Made,
    notes: Vec<Value>,
    refused: Option<Refusal>,
}

impl<'a> Step<'a> {
    pub(crate) fn new(
        entry: &'a Entry,
        resolved: &'a [Resolved],
        bytes: &'a [Vec<u8>],
        dir: &Path,
    ) -> Self {
        let input = |(input, bytes): (&'a Resolved, &'a Vec<u8>)| Input {
            name: &input.name,
            bytes,
            path: &input.path,
        };
        Self {
            name: &entry.name,
            params: &entry.params,
            inputs: resolved.iter().zip(bytes).map(input).collect(),
            files: dir.join("files"),
            made: Made::default(),
            notes: Vec::new(),
            refused: None,
        }
    }

    pub(crate) fn finish(self) -> (Made, Vec<Value>, Option<Refusal>) {
        (self.made, self.notes, self.refused)
    }

    pub fn name(&self) -> &str {
        self.name
    }

    pub fn params<T: DeserializeOwned>(&self) -> Result<T, Refusal> {
        T::deserialize(self.params).map_err(|error| Refusal::new("Params", self.name, error))
    }

    pub fn inputs(&self) -> &[Input<'a>] {
        &self.inputs
    }

    pub fn input(&self, name: &str) -> Result<&Input<'a>, Refusal> {
        let found = self.inputs.iter().find(|input| input.name == name);
        found.ok_or_else(|| Refusal::new("UnknownInput", name, self.name))
    }

    pub fn write(&mut self, file: &str, bytes: Vec<u8>) {
        self.keep(file, &bytes, true);
    }

    pub fn pass(&mut self, file: &str, bytes: Vec<u8>) {
        self.keep(file, &bytes, false);
    }

    pub fn item(&mut self, section: &str, id: &str, item: Value) {
        let ids = self.made.items.entry(section.to_owned()).or_default();
        if ids.insert(id.to_owned(), canonical(&item)).is_some() {
            self.refuse(Refusal::new("DuplicateItem", id, section));
        }
    }

    pub fn note(&mut self, note: Value) {
        self.notes.push(note);
    }

    fn keep(&mut self, file: &str, bytes: &[u8], published: bool) {
        let normal = |part| matches!(part, Component::Normal(_));
        let plain =
            !file.is_empty() && !file.contains('\\') && Path::new(file).components().all(normal);
        let reserved =
            published && (file == "cooked.json" || file.split('/').next() == Some(".cook"));
        if !plain || reserved || self.made.files.contains_key(file) {
            return self.refuse(Refusal::new("FileName", file, "bad or twice"));
        }
        let path = local(&self.files, file);
        let folder = path.parent().map_or(Ok(()), fs::create_dir_all);
        match folder.and_then(|()| fs::write(&path, bytes)) {
            Err(error) => self.refuse(Refusal::new("Write", file, error)),
            Ok(()) => {
                let fact = Fact {
                    sha256: sha(bytes),
                    bytes: bytes.len() as u64,
                    published,
                };
                self.made.files.insert(file.to_owned(), fact);
            }
        }
    }

    fn refuse(&mut self, refusal: Refusal) {
        self.refused.get_or_insert(refusal);
    }
}
