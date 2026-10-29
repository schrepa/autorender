use serde::Serialize;
use serde_json::{Value, json};

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct CommandOption {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub required: bool,
}

impl CommandOption {
    pub fn flag(name: &str) -> Self {
        Self {
            name: name.to_owned(),
            ..Self::default()
        }
    }

    pub fn taking(name: &str, value: impl Into<String>) -> Self {
        Self {
            name: name.to_owned(),
            value: Some(value.into()),
            ..Self::default()
        }
    }

    #[must_use]
    pub fn defaults_to(mut self, default: impl ToString) -> Self {
        self.default = Some(default.to_string());
        self
    }

    #[must_use]
    pub fn in_unit(mut self, unit: &str) -> Self {
        self.unit = Some(unit.to_owned());
        self
    }

    #[must_use]
    pub fn required(mut self) -> Self {
        self.required = true;
        self
    }

    fn row(&self) -> Vec<String> {
        let cell = |value: &Option<String>| value.clone().unwrap_or_else(|| "-".to_owned());
        vec![
            self.name.clone(),
            cell(&self.value),
            cell(&self.default),
            cell(&self.unit),
            if self.required { "yes" } else { "-" }.to_owned(),
        ]
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Command {
    pub command: String,
    pub options: Vec<CommandOption>,
}

impl Command {
    pub fn new(command: &str, options: impl IntoIterator<Item = CommandOption>) -> Self {
        Self {
            command: command.to_owned(),
            options: options.into_iter().collect(),
        }
    }
}

const COLUMNS: [&str; 5] = ["option", "value", "default", "unit", "required"];

pub fn asks_help(arguments: &[String]) -> Option<bool> {
    let given = |name: &str| arguments.iter().any(|argument| argument == name);
    given("--help").then(|| given("--json"))
}

pub fn help(arguments: &[String], options: &[CommandOption]) -> Option<String> {
    asks_help(arguments).map(|json| {
        if json {
            options_json(options).to_string()
        } else {
            options_text(options)
        }
    })
}

pub fn options_json(options: &[CommandOption]) -> Value {
    json!({ "options": options })
}

pub fn options_text(options: &[CommandOption]) -> String {
    let mut rows = vec![COLUMNS.map(str::to_owned).to_vec()];
    rows.extend(options.iter().map(CommandOption::row));
    aligned(&rows)
}

pub fn commands_json(commands: &[Command]) -> Value {
    json!({ "commands": commands })
}

pub fn commands_text(commands: &[Command]) -> String {
    let mut header = vec!["command".to_owned()];
    header.extend(COLUMNS.map(str::to_owned));
    let mut rows = vec![header];
    for command in commands {
        if command.options.is_empty() {
            let mut row = vec![command.command.clone()];
            row.extend(COLUMNS.map(|_| "-".to_owned()));
            rows.push(row);
        }
        for option in &command.options {
            let mut row = vec![command.command.clone()];
            row.extend(option.row());
            rows.push(row);
        }
    }
    aligned(&rows)
}

fn aligned(rows: &[Vec<String>]) -> String {
    let columns = rows.iter().map(Vec::len).max().unwrap_or(0);
    let widths: Vec<usize> = (0..columns)
        .map(|column| {
            rows.iter()
                .filter_map(|row| row.get(column))
                .map(String::len)
                .max()
                .unwrap_or(0)
        })
        .collect();
    rows.iter()
        .map(|row| {
            let cells: Vec<String> = row
                .iter()
                .zip(&widths)
                .map(|(cell, &width)| format!("{cell:width$}"))
                .collect();
            cells.join("  ").trim_end().to_owned()
        })
        .collect::<Vec<_>>()
        .join("\n")
}
