use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll, Waker};

pub(crate) struct Lazy<F> {
    state: State<F>,
    #[cfg(feature = "dev")]
    fault: std::cell::Cell<Fault>,
}

enum State<F> {
    Absent,
    Created {
        part: F,
        validation: Option<Validation>,
    },
    Disabled,
}

type Validation = Pin<Box<dyn Future<Output = Option<wgpu::Error>>>>;

#[cfg(feature = "dev")]
#[derive(Default)]
enum Fault {
    #[default]
    None,
    Injected,
    Made(wgpu::Buffer),
}

impl<F> Default for Lazy<F> {
    fn default() -> Self {
        Self {
            state: State::Absent,
            #[cfg(feature = "dev")]
            fault: std::cell::Cell::default(),
        }
    }
}

impl<F> Lazy<F> {
    pub(crate) fn get(&self) -> Option<&F> {
        match &self.state {
            State::Created { part, .. } => Some(part),
            State::Absent | State::Disabled => None,
        }
    }

    pub(crate) fn get_mut(&mut self) -> Option<&mut F> {
        match &mut self.state {
            State::Created { part, .. } => Some(part),
            State::Absent | State::Disabled => None,
        }
    }

    pub(crate) fn create(
        &mut self,
        device: &wgpu::Device,
        label: &'static str,
        create: impl FnOnce() -> F,
    ) {
        if !matches!(self.state, State::Absent) {
            return;
        }
        let scope = crate::gpu_calls::validation_scope(device);
        let part = create();
        #[cfg(feature = "dev")]
        if let Fault::Injected = self.fault.take() {
            self.fault.set(Fault::Made(invalid_buffer(device, label)));
        }
        #[cfg(not(feature = "dev"))]
        let _ = label;
        let mut validation: Validation = Box::pin(scope.pop());
        self.state = match poll(&mut validation) {
            Poll::Ready(Some(_)) => {
                #[cfg(feature = "dev")]
                drop(self.fault.take());
                State::Disabled
            }
            Poll::Ready(None) => State::Created {
                part,
                validation: None,
            },
            Poll::Pending => State::Created {
                part,
                validation: Some(validation),
            },
        };
    }

    pub(crate) fn release(&mut self) {
        if matches!(self.state, State::Created { .. }) {
            self.state = State::Absent;
        }
    }

    pub(crate) fn disabled(&self) -> bool {
        matches!(self.state, State::Disabled)
    }

    pub(crate) fn poll(&mut self) -> bool {
        let State::Created {
            validation: Some(validation),
            ..
        } = &mut self.state
        else {
            return false;
        };
        match poll(validation) {
            Poll::Pending => false,
            Poll::Ready(None) => {
                if let State::Created { validation, .. } = &mut self.state {
                    *validation = None;
                }
                false
            }
            Poll::Ready(Some(_)) => self.disable(),
        }
    }

    pub(crate) fn disable(&mut self) -> bool {
        #[cfg(feature = "dev")]
        drop(self.fault.take());
        !matches!(
            std::mem::replace(&mut self.state, State::Disabled),
            State::Disabled
        )
    }

    #[cfg(feature = "dev")]
    pub(crate) fn inject(&mut self) -> bool {
        self.release();
        self.fault.set(Fault::Injected);
        true
    }

    #[cfg(feature = "dev")]
    pub(crate) fn encode_fault(&self, encoder: &mut wgpu::CommandEncoder) {
        match self.fault.take() {
            Fault::Made(buffer) => encoder.clear_buffer(&buffer, 0, None),
            fault @ (Fault::None | Fault::Injected) => self.fault.set(fault),
        }
    }
}

fn poll(validation: &mut Validation) -> Poll<Option<wgpu::Error>> {
    validation
        .as_mut()
        .poll(&mut Context::from_waker(Waker::noop()))
}

#[cfg(feature = "dev")]
fn invalid_buffer(device: &wgpu::Device, label: &'static str) -> wgpu::Buffer {
    let label = format!("{label}fault");
    crate::memory::buffer(
        device,
        &wgpu::BufferDescriptor {
            label: Some(&label),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::MAP_WRITE,
            mapped_at_creation: false,
        },
    )
}
