use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;
use toml_edit::Document;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DependencyKind {
    Normal,
    Build,
    Development,
}

#[derive(Clone, Debug)]
pub struct Dependency {
    pub package: String,
    pub kind: DependencyKind,
}

#[derive(Clone, Debug)]
pub struct Package {
    pub name: String,
    pub folder: PathBuf,
    pub manifest: PathBuf,
    pub path: Option<String>,
    pub member: bool,
    pub dependencies: Vec<Dependency>,
    pub engine: Value,
}

impl Package {
    pub fn crate_name(&self) -> String {
        self.name.replace('-', "_")
    }

    pub fn game(&self) -> Option<&str> {
        self.engine["game"].as_str()
    }

    pub fn native(&self) -> Option<&str> {
        self.engine["native"].as_str()
    }

    pub fn project(&self) -> bool {
        self.game().is_some()
    }

    pub fn own(&self) -> bool {
        self.member && self.path.is_some()
    }

    pub fn file(&self, name: &str) -> Option<String> {
        self.path.as_deref().map(|folder| joined(folder, name))
    }
}

fn joined(folder: &str, name: &str) -> String {
    if folder.is_empty() {
        name.to_owned()
    } else {
        format!("{folder}/{name}")
    }
}

fn slashed(path: &Path) -> String {
    path.components()
        .map(|part| part.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/")
}

fn free_games(packages: &[Package]) -> Result<Vec<String>, String> {
    let mut free = Vec::new();
    for package in packages
        .iter()
        .filter(|package| package.own() && package.engine["kind"] == "game")
    {
        let shown = package.manifest.display();
        let text = fs::read_to_string(&package.manifest)
            .map_err(|error| format!("cannot read {shown}: {error}"))?;
        let manifest =
            Document::parse(text).map_err(|error| format!("{shown} is not TOML: {error}"))?;
        if manifest.as_item().get("lints").is_none() {
            free.extend(package.file(""));
        }
    }
    Ok(free)
}

pub struct Repository {
    pub root: PathBuf,
    pub workspace_root: PathBuf,
    pub packages: Vec<Package>,
    free: Vec<String>,
}

impl Repository {
    pub fn read(root: &Path) -> Result<Self, String> {
        let output = Command::new("cargo")
            .args(["metadata", "--format-version", "1", "--locked"])
            .current_dir(root)
            .output()
            .map_err(|error| format!("cannot run cargo metadata: {error}"))?;
        if !output.status.success() {
            return Err(String::from_utf8_lossy(&output.stderr).into_owned());
        }
        let metadata: Value = serde_json::from_slice(&output.stdout)
            .map_err(|error| format!("cargo metadata is not JSON: {error}"))?;
        Self::of(root, &metadata)
    }

    fn of(root: &Path, metadata: &Value) -> Result<Self, String> {
        let members: BTreeSet<&str> = metadata["workspace_members"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .collect();
        let mut packages = Vec::new();
        for package in metadata["packages"].as_array().into_iter().flatten() {
            let name = package["name"].as_str().unwrap_or_default().to_owned();
            let manifest = PathBuf::from(package["manifest_path"].as_str().unwrap_or_default());
            let folder = manifest
                .parent()
                .map(Path::to_path_buf)
                .ok_or_else(|| format!("cargo metadata gives {name} no manifest"))?;
            let path = folder.strip_prefix(root).ok().map(slashed);
            let member = package["id"]
                .as_str()
                .is_some_and(|id| members.contains(id));
            let dependencies = package["dependencies"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|dependency| {
                    let kind = match dependency["kind"].as_str() {
                        Some("build") => DependencyKind::Build,
                        Some("dev") => DependencyKind::Development,
                        Some(_) | None => DependencyKind::Normal,
                    };
                    dependency["name"].as_str().map(|name| Dependency {
                        package: name.to_owned(),
                        kind,
                    })
                })
                .collect();
            packages.push(Package {
                name,
                folder,
                manifest,
                path,
                member,
                dependencies,
                engine: package["metadata"]["engine"].clone(),
            });
        }
        packages.sort_by(|first, second| first.name.cmp(&second.name));
        let free = free_games(&packages)?;
        Ok(Self {
            root: root.to_path_buf(),
            workspace_root: PathBuf::from(metadata["workspace_root"].as_str().unwrap_or_default()),
            packages,
            free,
        })
    }

    pub fn ruled(&self, path: &str) -> bool {
        !self
            .free
            .iter()
            .any(|folder| path.starts_with(folder.as_str()))
    }

    pub fn package(&self, name: &str) -> Option<&Package> {
        self.packages.iter().find(|package| package.name == name)
    }

    pub fn of_crate(&self, name: &str) -> Option<&Package> {
        let named = |package: &&Package| package.crate_name() == name;
        self.packages
            .iter()
            .filter(|package| package.member)
            .find(named)
            .or_else(|| self.packages.iter().find(named))
    }

    pub fn own(&self) -> impl Iterator<Item = &Package> + '_ {
        self.packages.iter().filter(|package| package.own())
    }

    pub fn projects(&self) -> impl Iterator<Item = &Package> + '_ {
        self.own().filter(|package| package.game().is_some())
    }

    pub fn is_workspace_root(&self) -> bool {
        self.root
            .strip_prefix(&self.workspace_root)
            .is_ok_and(|rest| rest.as_os_str().is_empty())
    }

    pub fn runtime(&self) -> BTreeSet<&str> {
        let mut pending: Vec<&str> = self
            .own()
            .filter(|package| package.project())
            .map(|package| package.name.as_str())
            .collect();
        let mut reached = BTreeSet::new();
        while let Some(name) = pending.pop() {
            if !reached.insert(name) {
                continue;
            }
            if let Some(package) = self.package(name) {
                pending.extend(
                    package
                        .dependencies
                        .iter()
                        .filter(|dependency| dependency.kind == DependencyKind::Normal)
                        .map(|dependency| dependency.package.as_str()),
                );
            }
        }
        reached
    }
}
