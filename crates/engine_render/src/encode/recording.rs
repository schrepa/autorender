use super::Open;

pub(crate) enum Recording {
    Idle,
    Encoding(wgpu::CommandEncoder),
    Rendering(Box<Open>, wgpu::CommandEncoder),
    Computing(u32, Box<wgpu::ComputePass<'static>>, wgpu::CommandEncoder),
    Collecting(Option<Opened>, Option<wgpu::CommandEncoder>),
}

#[derive(Clone, Copy)]
pub(crate) struct Opened {
    pub(crate) layout: u32,
}

impl Recording {
    pub(crate) fn of(encoder: Option<wgpu::CommandEncoder>) -> Self {
        encoder.map_or(Self::Idle, Self::Encoding)
    }

    pub(crate) fn encoder(&mut self) -> Option<&mut wgpu::CommandEncoder> {
        match self {
            Self::Encoding(encoder) => Some(encoder),
            Self::Idle | Self::Rendering(..) | Self::Computing(..) | Self::Collecting(..) => None,
        }
    }

    pub(crate) fn open(&self) -> Option<Opened> {
        match self {
            Self::Rendering(open, _) => Some(Opened {
                layout: open.layout,
            }),
            Self::Collecting(opened, _) => *opened,
            Self::Idle | Self::Encoding(_) | Self::Computing(..) => None,
        }
    }

    pub(crate) fn open_render(&self) -> Option<(bool, usize)> {
        match self {
            Self::Rendering(open, _) => Some((open.shared, open.planned_at)),
            Self::Idle | Self::Encoding(_) | Self::Computing(..) | Self::Collecting(..) => None,
        }
    }

    pub(crate) fn open_mut(&mut self) -> Option<&mut Open> {
        match self {
            Self::Rendering(open, _) => Some(&mut **open),
            Self::Idle | Self::Encoding(_) | Self::Computing(..) | Self::Collecting(..) => None,
        }
    }

    pub(crate) fn computing(&mut self, pass: u32) -> Option<&mut wgpu::ComputePass<'static>> {
        match self {
            Self::Computing(known, computing, _) if *known == pass => Some(&mut **computing),
            Self::Idle
            | Self::Encoding(_)
            | Self::Rendering(..)
            | Self::Computing(..)
            | Self::Collecting(..) => None,
        }
    }

    pub(crate) fn collecting(&self) -> bool {
        matches!(self, Self::Collecting(..))
    }

    pub(crate) fn collect(&mut self) {
        *self = match std::mem::replace(self, Self::Idle) {
            Self::Encoding(encoder) => Self::Collecting(None, Some(encoder)),
            Self::Idle => Self::Collecting(None, None),
            state @ (Self::Rendering(..) | Self::Computing(..) | Self::Collecting(..)) => state,
        };
    }

    pub(crate) fn record(&mut self) {
        *self = match std::mem::replace(self, Self::Idle) {
            Self::Collecting(_, encoder) => Self::of(encoder),
            state
            @ (Self::Idle | Self::Encoding(_) | Self::Rendering(..) | Self::Computing(..)) => state,
        };
    }

    pub(crate) fn collected(&mut self, opened: Opened) {
        if let Self::Collecting(open, _) = self {
            *open = Some(opened);
        }
    }

    pub(crate) fn transition(&mut self, step: Step) -> Ended {
        let (next, ended) = match (std::mem::replace(self, Self::Idle), step) {
            (Self::Encoding(encoder), Step::Render(open)) => {
                (Self::Rendering(open, encoder), Ended::Nothing)
            }
            (Self::Encoding(encoder), Step::Compute(pass, computing)) => {
                (Self::Computing(pass, computing, encoder), Ended::Nothing)
            }
            (Self::Rendering(open, encoder), Step::EndRender | Step::EndPass) => {
                (Self::Encoding(encoder), Ended::Render(open))
            }
            (Self::Computing(_, computing, encoder), Step::EndPass) => {
                drop(computing);
                (Self::Encoding(encoder), Ended::Nothing)
            }
            (Self::Encoding(encoder), Step::Take) => (Self::Idle, Ended::Encoder(encoder)),
            (Self::Collecting(_, encoder), Step::EndRender | Step::EndPass) => {
                (Self::Collecting(None, encoder), Ended::Nothing)
            }
            (Self::Collecting(_, Some(encoder)), Step::Take) => {
                (Self::Idle, Ended::Encoder(encoder))
            }
            (state @ (Self::Idle | Self::Encoding(_) | Self::Computing(..)), Step::EndRender) => {
                (state, Ended::Nothing)
            }
            (state @ (Self::Idle | Self::Encoding(_)), Step::EndPass) => (state, Ended::Nothing),
            (
                state @ (Self::Idle
                | Self::Rendering(..)
                | Self::Computing(..)
                | Self::Collecting(_, None)),
                Step::Take,
            ) => (state, Ended::Nothing),
            (
                state @ (Self::Idle
                | Self::Rendering(..)
                | Self::Computing(..)
                | Self::Collecting(..)),
                Step::Render(_) | Step::Compute(..),
            ) => (state, Ended::Nothing),
        };
        *self = next;
        ended
    }
}

pub(crate) enum Step {
    Render(Box<Open>),
    Compute(u32, Box<wgpu::ComputePass<'static>>),
    EndRender,
    EndPass,
    Take,
}

pub(crate) enum Ended {
    Nothing,
    Render(Box<Open>),
    Encoder(wgpu::CommandEncoder),
}
