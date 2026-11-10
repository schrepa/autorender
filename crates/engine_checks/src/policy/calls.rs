use std::collections::BTreeMap;

use engine_core::{Error, ErrorCode, Expected};

use super::{Problems, rust};
use crate::at;

const DECLARING: [&str; 2] = ["operation", "observation"];
const ANSWERING: [&str; 2] = ["Operation", "Callable"];

#[derive(Default)]
pub struct Calls {
    implemented: BTreeMap<String, Vec<String>>,
    declared: BTreeMap<String, Vec<String>>,
    called: Vec<(String, String)>,
}

impl Calls {
    pub fn read(&mut self, path: &str, split: &rust::Split) {
        for (index, code) in split.code.iter().enumerate() {
            let line = at(path, index + 1);
            if let Some(name) = implemented(code) {
                self.implemented.entry(name).or_default().push(line.clone());
            }
            for (named, ty) in turbofish(code) {
                if named == "call" {
                    self.called.push((ty, line.clone()));
                } else if DECLARING.contains(&named.as_str()) || named == "Call::of" {
                    self.declared.entry(ty).or_default().push(line.clone());
                }
            }
        }
    }

    pub fn check(&self, problems: &mut Problems) {
        for (name, places) in &self.implemented {
            let declared = self.declared.get(name).map_or(&[][..], Vec::as_slice);
            if declared.len() < places.len() {
                problems.push(
                    Error::new(ErrorCode::UnknownReference, places[declared.len()].as_str())
                        .expected(Expected::Present)
                        .got(name)
                        .nearest(format!("operation::<{name}>()")),
                );
            }
            if declared.len() > places.len() {
                problems.push(
                    Error::new(ErrorCode::NameTwice, declared[places.len()].as_str())
                        .got(name)
                        .nearest(&declared[0]),
                );
            }
        }
        for (name, line) in &self.called {
            if self.implemented.contains_key(name) && !self.declared.contains_key(name) {
                problems.push(
                    Error::new(ErrorCode::UnknownReference, line.as_str())
                        .expected(Expected::Present)
                        .got(name),
                );
            }
        }
    }
}

fn identifier(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

fn implemented(code: &str) -> Option<String> {
    ANSWERING.iter().find_map(|&answering| {
        rust::words(code, answering).find_map(|end| {
            let mut before = code[..end - answering.len()].trim_end();
            while let Some(path) = before.strip_suffix("::") {
                before = path.trim_end_matches(identifier);
            }
            if !rust::has_word(before, "impl") {
                return None;
            }
            let rest = code[end..].trim_start().strip_prefix("for")?;
            if !rest.starts_with(char::is_whitespace) {
                return None;
            }
            let name: String = rest
                .trim_start()
                .chars()
                .take_while(|&c| identifier(c))
                .collect();
            (!name.is_empty() && !generic(before, &name)).then_some(name)
        })
    })
}

fn generic(before: &str, name: &str) -> bool {
    before.split_once("impl<").is_some_and(|(_, parameters)| {
        parameters
            .split([',', '>'])
            .any(|parameter| parameter.split(':').next().is_some_and(|held| held.trim() == name))
    })
}

fn turbofish(code: &str) -> Vec<(String, String)> {
    code.match_indices("::<")
        .filter_map(|(at, _)| {
            let before = &code[..at];
            let named = &before[before.trim_end_matches(identifier).len()..];
            let named = if named == "of" && before.ends_with("Call::of") {
                "Call::of"
            } else {
                named
            };
            let path: String = code[at + 3..]
                .trim_start()
                .chars()
                .take_while(|&c| identifier(c) || c == ':')
                .collect();
            let ty = path.rsplit("::").next().unwrap_or_default().to_owned();
            ty.starts_with(|c: char| c.is_ascii_uppercase())
                .then(|| (named.to_owned(), ty))
        })
        .collect()
}
