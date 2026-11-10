use bevy_ecs::resource::Resource;
use engine_settings::Error;
use serde::Serialize;

use crate::{Census, Interest, Progress, Refusal, Request, Requests, Ticket};

#[derive(Resource, Default, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Level {
    requests: Requests<String, ()>,
    wanted: Option<Interest<String>>,
    shown: Option<Interest<String>>,
    handed: Option<Ticket>,
}

#[derive(Clone, Debug, PartialEq, Serialize, schemars::JsonSchema)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum LevelView {
    Idle,
    Asked {
        path: String,
        progress: Progress,
    },
    Ready {
        path: String,
    },
    Failed {
        path: String,
        error: Refusal,
    },
    Missing {
        path: String,
    },
}

impl Level {
    pub fn load(&mut self, path: impl Into<String>) {
        self.wanted = Some(self.requests.ask(path.into()));
    }

    pub fn settle(&mut self, now: f64) {
        let ready = self
            .wanted
            .as_ref()
            .and_then(|wanted| self.requests.of(wanted))
            .is_some_and(|request| request.ready().is_some());
        if ready {
            self.shown = self.wanted.take();
        }
        self.requests.settle(now);
    }

    pub fn take_asked(&mut self) -> Option<String> {
        let wanted = self.wanted.as_ref()?;
        let ticket = self.requests.ticket(wanted.key())?;
        if self.handed == Some(ticket) {
            return None;
        }
        self.handed = Some(ticket);
        Some(wanted.key().clone())
    }

    pub fn progress(&mut self, progress: Progress) {
        if let Some(wanted) = &self.wanted {
            self.requests.progress(wanted.key(), progress);
        }
    }

    pub fn built(&mut self) {
        self.end(|requests, path, ticket| {
            requests.ready(path, ticket, ());
        });
    }

    pub fn fail(&mut self, error: Error) {
        self.end(|requests, path, ticket| {
            requests.fail(path, ticket, error);
        });
    }

    fn end(&mut self, end: impl FnOnce(&mut Requests<String, ()>, &String, Ticket)) {
        let Some(wanted) = &self.wanted else {
            return;
        };
        if let Some(ticket) = self.requests.ticket(wanted.key()) {
            end(&mut self.requests, wanted.key(), ticket);
        }
    }

    pub fn census(&self) -> Census {
        self.requests.census()
    }

    pub fn failure(&self) -> Option<(&str, &Error)> {
        self.requests
            .failures()
            .next()
            .map(|(path, error)| (path.as_str(), error))
    }

    pub fn view(&self) -> LevelView {
        let Some(held) = self.wanted.as_ref().or(self.shown.as_ref()) else {
            return LevelView::Idle;
        };
        let path = held.key().clone();
        match self.requests.of(held) {
            Some(Request::Asked(_)) => LevelView::Asked {
                progress: self.requests.progress_of(&path).unwrap_or_default(),
                path,
            },
            Some(Request::Ready(())) => LevelView::Ready { path },
            Some(Request::Failed(error)) => LevelView::Failed {
                error: Refusal::of(error),
                path,
            },
            Some(Request::Missing) | None => LevelView::Missing { path },
        }
    }
}
