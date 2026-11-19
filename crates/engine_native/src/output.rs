use std::io::Cursor;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex, PoisonError};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SampleFormat, SizedSample};

use crate::mixer::{Command, Mixer, Pcm};

fn ogg_vorbis(bytes: &[u8]) -> bool {
    let packet = bytes
        .get(26)
        .and_then(|&segments| bytes.get(27 + usize::from(segments)..));
    bytes.starts_with(b"OggS") && packet.is_some_and(|packet| packet.starts_with(b"\x01vorbis"))
}

pub(crate) fn decode(bytes: &Arc<[u8]>) -> Result<Pcm, String> {
    use lewton::inside_ogg::OggStreamReader;
    use lewton::samples::InterleavedSamples;
    if !ogg_vorbis(bytes) {
        return Err("not_ogg_vorbis".to_owned());
    }
    let mut reader =
        OggStreamReader::new(Cursor::new(Arc::clone(bytes))).map_err(|error| error.to_string())?;
    let channels = usize::from(reader.ident_hdr.audio_channels);
    let rate = reader.ident_hdr.audio_sample_rate;
    let mut samples = Vec::new();
    while let Some(packet) = reader
        .read_dec_packet_generic::<InterleavedSamples<f32>>()
        .map_err(|error| error.to_string())?
    {
        samples.extend_from_slice(&packet.samples);
    }
    if channels == 0 || rate == 0 {
        return Err("empty_format".to_owned());
    }
    if let Some(end) = reader.get_last_absgp() {
        let end = usize::try_from(end)
            .unwrap_or(usize::MAX)
            .saturating_mul(channels);
        samples.truncate(end);
    }
    Ok(Pcm {
        samples,
        channels,
        rate,
    })
}

pub(crate) struct Device {
    pub(crate) _stream: cpal::Stream,
    pub(crate) commands: Sender<Command>,
    pub(crate) ended: Receiver<u32>,
    pub(crate) xruns: Arc<AtomicU64>,
    pub(crate) ahead_nanos: Arc<AtomicU64>,
    pub(crate) failure: Arc<Mutex<Option<String>>>,
    pub(crate) rate: u32,
    pub(crate) channels: u32,
    pub(crate) told: bool,
}

struct Shared {
    commands: Receiver<Command>,
    ended: Sender<u32>,
    xruns: Arc<AtomicU64>,
    ahead_nanos: Arc<AtomicU64>,
    failure: Arc<Mutex<Option<String>>>,
}

fn build<T: SizedSample + FromSample<f32>>(
    device: &cpal::Device,
    config: cpal::StreamConfig,
    mut mixer: Mixer,
    shared: Shared,
) -> Result<cpal::Stream, String> {
    let channels = usize::from(config.channels);
    let (mut left, mut right) = (Vec::<f32>::new(), Vec::<f32>::new());
    let Shared {
        commands,
        ended,
        xruns,
        ahead_nanos,
        failure,
    } = shared;
    device
        .build_output_stream::<T, _, _>(
            config,
            move |data: &mut [T], info: &cpal::OutputCallbackInfo| {
                let stamp = info.timestamp();
                let ahead = stamp
                    .playback
                    .checked_duration_since(stamp.callback)
                    .map_or(0, |ahead| {
                        u64::try_from(ahead.as_nanos()).unwrap_or(u64::MAX)
                    });
                ahead_nanos.store(ahead.saturating_add(1), Ordering::Relaxed);
                while let Ok(command) = commands.try_recv() {
                    mixer.apply(command);
                }
                let frames = data.len() / channels;
                left.clear();
                left.resize(frames, 0.0);
                right.clear();
                right.resize(frames, 0.0);
                mixer.render(&mut left, &mut right);
                for voice in mixer.take_ended() {
                    let _ = ended.send(voice);
                }
                for ((frame, l), r) in data.chunks_exact_mut(channels).zip(&left).zip(&right) {
                    let (l, r) = (
                        engine_core::math::clamp(*l, -1.0, 1.0),
                        engine_core::math::clamp(*r, -1.0, 1.0),
                    );
                    match frame {
                        [only] => *only = T::from_sample_((l + r) * 0.5),
                        [first, second, rest @ ..] => {
                            *first = T::from_sample_(l);
                            *second = T::from_sample_(r);
                            rest.fill(T::EQUILIBRIUM);
                        }
                        [] => {}
                    }
                }
            },
            move |error: cpal::Error| {
                if error.kind() == cpal::ErrorKind::Xrun {
                    xruns.fetch_add(1, Ordering::Relaxed);
                } else {
                    *failure.lock().unwrap_or_else(PoisonError::into_inner) =
                        Some(error.to_string());
                }
            },
            None,
        )
        .map_err(|error| error.to_string())
}

pub(crate) fn open(latency_seconds: Option<f32>) -> Result<Device, String> {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or_else(|| "no_output_device".to_owned())?;
    let supported = device
        .default_output_config()
        .map_err(|error| error.to_string())?;
    let rate = supported.sample_rate();
    let buffer_size = latency_seconds.map_or(cpal::BufferSize::Default, |seconds| {
        cpal::BufferSize::Fixed((f64::from(seconds) * f64::from(rate)).round() as u32)
    });
    let config = cpal::StreamConfig {
        channels: supported.channels(),
        sample_rate: rate,
        buffer_size,
    };
    let (commands, received) = channel::<Command>();
    let (ended_sender, ended) = channel::<u32>();
    let xruns = Arc::new(AtomicU64::new(0));
    let ahead_nanos = Arc::new(AtomicU64::new(0));
    let failure = Arc::new(Mutex::new(None));
    let shared = Shared {
        commands: received,
        ended: ended_sender,
        xruns: Arc::clone(&xruns),
        ahead_nanos: Arc::clone(&ahead_nanos),
        failure: Arc::clone(&failure),
    };
    let mixer = Mixer::new(rate);
    let format = supported.sample_format();
    let stream = if format == SampleFormat::F32 {
        build::<f32>(&device, config, mixer, shared)
    } else if format == SampleFormat::I16 {
        build::<i16>(&device, config, mixer, shared)
    } else if format == SampleFormat::U16 {
        build::<u16>(&device, config, mixer, shared)
    } else if format == SampleFormat::I32 {
        build::<i32>(&device, config, mixer, shared)
    } else {
        Err(format!("unsupported_sample_format {format}"))
    }?;
    stream.play().map_err(|error| error.to_string())?;
    Ok(Device {
        _stream: stream,
        commands,
        ended,
        xruns,
        ahead_nanos,
        failure,
        rate,
        channels: u32::from(supported.channels()),
        told: false,
    })
}
