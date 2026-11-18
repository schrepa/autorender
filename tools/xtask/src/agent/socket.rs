use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::sync::{Arc, Mutex};

const KEY: &str = "AAAAAAAAAAAAAAAAAAAAAA==";
const TEXT: u8 = 0x1;
const CLOSE: u8 = 0x8;
const PING: u8 = 0x9;
const PONG: u8 = 0xA;

#[derive(Clone)]
pub struct Socket(Arc<Mutex<TcpStream>>);

pub struct Messages {
    reader: BufReader<TcpStream>,
    socket: Socket,
}

pub fn connect(url: &str) -> Result<(Socket, Messages), String> {
    let failed = |error: std::io::Error| format!("cannot open the DevTools socket {url}: {error}");
    let rest = url
        .strip_prefix("ws://")
        .ok_or_else(|| format!("{url} is not a local unencrypted DevTools address"))?;
    let (address, path) = rest.split_once('/').unwrap_or((rest, ""));
    let mut stream = TcpStream::connect(address).map_err(failed)?;
    stream.set_nodelay(true).map_err(failed)?;
    write!(
        stream,
        "GET /{path} HTTP/1.1\r\nHost: {address}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {KEY}\r\nSec-WebSocket-Version: 13\r\n\r\n"
    )
    .map_err(failed)?;
    let mut reader = BufReader::new(stream.try_clone().map_err(failed)?);
    let mut status = String::new();
    reader.read_line(&mut status).map_err(failed)?;
    if !status.starts_with("HTTP/1.1 101") {
        return Err(format!("{url} refused the WebSocket upgrade: {}", status.trim()));
    }
    loop {
        let mut header = String::new();
        if reader.read_line(&mut header).map_err(failed)? == 0 || header.trim().is_empty() {
            break;
        }
    }
    let socket = Socket(Arc::new(Mutex::new(stream)));
    Ok((
        socket.clone(),
        Messages {
            reader,
            socket,
        },
    ))
}

impl Socket {
    pub fn send(&self, text: &str) -> Result<(), String> {
        self.frame(TEXT, text.as_bytes())
    }

    fn frame(&self, opcode: u8, payload: &[u8]) -> Result<(), String> {
        let mut frame = vec![0x80 | opcode];
        let length = payload.len();
        match (u8::try_from(length), u16::try_from(length)) {
            (Ok(short), _) if short < 126 => frame.push(0x80 | short),
            (_, Ok(medium)) => {
                frame.push(0x80 | 126);
                frame.extend(medium.to_be_bytes());
            }
            _ => {
                frame.push(0x80 | 127);
                frame.extend((length as u64).to_be_bytes());
            }
        }
        frame.extend([0; 4]);
        frame.extend(payload);
        let mut stream = self
            .0
            .lock()
            .map_err(|_| "the DevTools socket writer panicked".to_owned())?;
        stream
            .write_all(&frame)
            .map_err(|error| format!("cannot write to the DevTools socket: {error}"))
    }
}

impl Iterator for Messages {
    type Item = String;

    fn next(&mut self) -> Option<String> {
        let mut message = Vec::new();
        loop {
            let mut head = [0; 2];
            self.reader.read_exact(&mut head).ok()?;
            let length = match head[1] & 0x7f {
                126 => {
                    let mut bytes = [0; 2];
                    self.reader.read_exact(&mut bytes).ok()?;
                    u64::from(u16::from_be_bytes(bytes))
                }
                127 => {
                    let mut bytes = [0; 8];
                    self.reader.read_exact(&mut bytes).ok()?;
                    u64::from_be_bytes(bytes)
                }
                short => u64::from(short),
            };
            let mut mask = [0; 4];
            if head[1] & 0x80 != 0 {
                self.reader.read_exact(&mut mask).ok()?;
            }
            let mut payload = vec![0; usize::try_from(length).ok()?];
            self.reader.read_exact(&mut payload).ok()?;
            for (index, byte) in payload.iter_mut().enumerate() {
                *byte ^= mask[index % 4];
            }
            match head[0] & 0x0f {
                CLOSE => return None,
                PING => self.socket.frame(PONG, &payload).ok()?,
                PONG => {}
                _ if head[0] & 0x80 != 0 => {
                    message.extend(payload);
                    return Some(String::from_utf8_lossy(&message).into_owned());
                }
                _ => message.extend(payload),
            }
        }
    }
}
