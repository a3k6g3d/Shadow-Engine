//! The scheduler layer: one thread per drive, and a handle the async side can hold.
//! See docs/SHADOW_PLAN.md, Phase 1 item 1 - with one deviation, recorded there: `spawn_blocking` there, a
//! dedicated thread per drive here.
//!
//! The ring underneath (`lib.rs`'s `DirectFile`, or `ioring`'s `RingFile`) is single-threaded on purpose, and it
//! is `!Send` - both of those are load-bearing, so this layer does not fight them, it puts a thread in front of
//! them.  Each drive gets exactly one OS thread, which is the only thread that ever touches that drive's ring: it
//! opens the ring itself (a `!Send` type cannot be *moved* onto a thread, only built on one), issues the reads,
//! reaps the completions and answers them.
//!
//! What the tokio tasks hold is not the ring, it is a `Scheduler`: a sender and a waker, both of which can cross
//! a thread boundary.  A task hands a read over on a channel, wakes the drive thread, and awaits its completion -
//! so the only thing that has to be `Send` is the message, and the only thing that has to be `Sync` is the handle.
//! Which is what this file is: the point where the two halves of the plan - a `Send` future on one side, a
//! single-threaded ring on the other - are made to fit, with the ring's rules intact.
//!
//! The cost of that, said plainly: a request is two allocations (the channel's node and the oneshot's cell) on
//! top of the ring's zero-allocation path, and its completion costs a map lookup.  The ring's own hot path is
//! untouched; everything here is on the way *to* it.
//!
//! ```compile_fail,E0277
//! fn assert_send<T: Send>() {}
//! assert_send::<shadow_host::scheduler::Scheduler>();   // fine: the handle is Send, that is its job
//! assert_send::<shadow_host::DirectFile>();             // error[E0277]: `DirectFile` cannot be sent between
//! ```                                                 // threads safely - which is what the thread is for

use std::collections::HashMap;
use std::path::Path;

use tokio::sync::mpsc::{self, error::SendError, error::TryRecvError, UnboundedReceiver};
use tokio::sync::oneshot;

/// STRATA_HOST_IORING=1 asks for the I/O ring; it is taken only where this Windows has one, else the completion port.
fn want_ioring() -> bool {
    #[cfg(windows)]
    {
        std::env::var("STRATA_HOST_IORING").is_ok_and(|v| v == "1") && crate::ioring::RingFile::available()
    }
    #[cfg(not(windows))]
    false
}
use crate::{Buffer, Completion, DirectFile, Status, Waker, WAKE_TAG};
#[cfg(windows)]
use crate::ioring::{RingFile, RingWaker};

/// The answer to one request: the completion, or the refusal - with the buffer that never got queued, because a
/// read that was not queued produces no completion, so the error is the only place it can come home.
///
/// `None` on the error half happens once only: the drive thread died (panicked) with this request already in its
/// hands, and the buffer went with it.  Every refusal the ring itself makes hands the buffer back.
pub type Reply = Result<Completion, (Status, Option<Buffer>)>;

/// One read, as it crosses from a task to the thread that owns the ring.
///
/// The buffer moves in by value - the ring's rule, kept end to end: whoever owns the memory is the one who hands
/// it to the kernel, and the channel is the hand.
pub struct Request {
    pub tag: u64,
    pub offset: u64,
    pub buffer: Buffer,
    pub reply: oneshot::Sender<Reply>,
}

/// What a tokio task holds: a sender and a waker, both of which can cross a thread boundary (pinned at compile
/// time in `lib.rs`), and cheap to clone - every task gets its own copy of the same handle.
///
/// The channel is unbounded on purpose: the drive thread only ever *polls* it, because the place it normally
/// waits is the completion port, and a `send` that could block would have to be awaited by a thread that is not
/// running a runtime.  That does not make the ring's capacity fake - it is still the ring's own, and a read the
/// ring refuses comes back as a refusal, exactly as it does when the ring is called directly.
#[derive(Clone)]
pub struct Scheduler {
    tx: mpsc::UnboundedSender<Request>,
    waker: WakerKind,
    size: u64,
}

/// The drive thread's wake source, which is per-arm: the completion port for the IOCP ring, an event for the
/// I/O ring.  Both are `usize`-sized and both are `Send + Sync`, which is what lets one handle serve either.
#[derive(Clone)]
enum WakerKind {
    Port(Waker),
    #[cfg(windows)]
    Event(RingWaker),
}

impl WakerKind {
    fn wake(&self) {
        match self {
            WakerKind::Port(w) => w.wake(),
            #[cfg(windows)]
            WakerKind::Event(w) => w.wake(),
        }
    }
}

/// The ring this drive thread owns.  `Send` is what it is not, and that is the whole reason there is a thread.
enum Arm {
    Port(DirectFile),
    #[cfg(windows)]
    IoRing(RingFile),
}

/// How long the drive thread is willing to sleep *with reads in flight*, on an arm that cannot be woken from
/// another thread mid-wait.  The IOCP arm can be (its port is the wake source, and a wake posts to it from any
/// thread, so it lands at once); the I/O ring's `wait_with` only notices a wake at the next completion, or at
/// this timeout.  So one loop serves both, and the honest reading is: the IOCP arm pays a wake's latency, this
/// arm pays at most `POLL_MS`.
const POLL_MS: u32 = 2;

/// Completions taken per trip to the port - the same batch the ring itself asks for (`lib.rs`'s `REAP_BATCH`).
const REAP: usize = 64;

impl Arm {
    /// Open the ring on *this* thread and hand back the two things that are allowed to leave it: the waker and
    /// the size.  Which arm it is stays the plan's choice (`STRATA_HOST_IORING`, as `ffi` reads it).
    fn open(path: &Path, capacity: usize) -> Result<(Arm, WakerKind, u64), (Status, String)> {
        #[cfg(windows)]
        if want_ioring() {
            let f = RingFile::open(path, capacity)?;
            let (waker, size) = (f.waker(), f.size());
            return Ok((Arm::IoRing(f), WakerKind::Event(waker), size));
        }
        let f = DirectFile::open(path, capacity).map_err(|s| (s, format!("DirectFile::open: {s:?}")))?;
        let (waker, size) = (f.waker(), f.size());
        Ok((Arm::Port(f), WakerKind::Port(waker), size))
    }

    fn submit(&mut self, offset: u64, buffer: Buffer, tag: u64) -> Result<(), (Status, Buffer)> {
        match self {
            Arm::Port(f) => f.submit(offset, buffer, tag),
            #[cfg(windows)]
            Arm::IoRing(f) => f.submit(offset, buffer, tag),
        }
    }

    fn wait(&mut self, max: usize, timeout_ms: u32, f: impl FnMut(Completion)) -> usize {
        match self {
            Arm::Port(ring) => ring.wait_with(max, timeout_ms, f),
            #[cfg(windows)]
            Arm::IoRing(ring) => ring.wait_with(max, timeout_ms, f),
        }
    }
}

impl Scheduler {
    /// Open one drive and start the thread that will own it.  Returns the handle; the ring itself never comes
    /// back, because it cannot be carried here.
    ///
    /// The waker and the size are the ring's own report, sent back over a plain (blocking) channel before the
    /// loop starts - which is also what makes `open` fail here, rather than a second later on the first read.
    pub fn open(path: &Path, capacity: usize) -> Result<Scheduler, (Status, String)> {
        let (tx, rx) = mpsc::unbounded_channel();
        let (ready, received) = std::sync::mpsc::channel();
        let path = path.to_path_buf();
        std::thread::Builder::new()
            .name("shadow-drive".into())
            .spawn(move || drive(&path, capacity, rx, ready))
            .map_err(|e| (Status::DeviceUnavailable, format!("spawn the drive thread: {e}")))?;
        // The thread sends its answer and keeps running; `recv` returns as soon as the answer is there.
        match received.recv() {
            Ok(Ok((waker, size))) => Ok(Scheduler { tx, waker, size }),
            Ok(Err(e)) => Err(e),
            // The thread died between answering and running, so the channel it owned is gone with it.
            Err(_) => Err((Status::DeviceUnavailable, "the drive thread died before it answered".into())),
        }
    }

    pub fn size(&self) -> u64 {
        self.size
    }

    /// Hand a read to the drive thread, wake it, and await the completion.
    ///
    /// The three steps are the design: the channel is how the request crosses, the wake is what pulls the drive
    /// thread out of its wait so it can see the request *now* rather than at its next timeout, and the oneshot is
    /// how the completion - with the buffer inside it, still the caller's - comes back.
    ///
    /// A refusal (the ring was full, the offset or the buffer was not sector-aligned) comes back as an error with
    /// the buffer in it, the ring's own rule: the caller owns the retry.
    pub async fn submit(&self, offset: u64, buffer: Buffer, tag: u64) -> Reply {
        let (reply, awaited) = oneshot::channel();
        if let Err(req) = self.tx.send(Request { tag, offset, buffer, reply }) {
            // A closed channel hands the message back, so the buffer is still ours to give back.
            let SendError(Request { buffer, .. }) = req;
            return Err((Status::DeviceUnavailable, Some(buffer)));
        }
        self.waker.wake();
        awaited.await.map_err(|_| (Status::DeviceUnavailable, None))?
    }
}

/// What the drive thread keeps for itself: which oneshot answers which read it has in flight.
///
/// Keyed by the tag *this layer* gave the ring, not the caller's, so a caller may use the same tag twice in one
/// breath - the two completions are told apart here, and each one is handed back with the caller's tag written
/// into it.  `WAKE_TAG` is the ring's own and never reaches a caller, so no caller may use it.
type Waiting = HashMap<u64, (u64, oneshot::Sender<Reply>)>;

/// The drive thread's whole life: open the ring, answer the channel, sleep on the port until a wake or a
/// completion says otherwise.  Nothing here is shared with another thread, so nothing here is locked.
fn drive(
    path: &Path,
    capacity: usize,
    mut rx: UnboundedReceiver<Request>,
    ready: std::sync::mpsc::Sender<Result<(WakerKind, u64), (Status, String)>>,
) {
    let (mut arm, waker, size) = match Arm::open(path, capacity) {
        Ok(v) => v,
        Err(e) => {
            let _ = ready.send(Err(e));
            return;
        }
    };
    let _ = ready.send(Ok((waker.clone(), size))); // the handle keeps a copy; this thread keeps the original

    let mut waiting: Waiting = HashMap::new();
    let mut next_tag = 0u64; // the ring's tags, handed out in order; the last u64 stays the ring's (it is a wake)
    let mut closed = false;

    loop {
        // Everything the channel has, in the order it came.  Poll only: the port below is the only thing here
        // that sleeps, and the wake that follows every send is what makes it look up.
        loop {
            match rx.try_recv() {
                Ok(Request { tag, offset, buffer, reply }) => {
                    let mine = next_tag;
                    next_tag += 1;
                    match arm.submit(offset, buffer, mine) {
                        Ok(()) => {
                            waiting.insert(mine, (tag, reply));
                        }
                        Err((status, buffer)) => {
                            let _ = reply.send(Err((status, Some(buffer))));
                        }
                    }
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    closed = true;
                    break;
                }
            }
        }
        if closed && waiting.is_empty() {
            break; // nothing in flight, so the ring may drop here - every buffer it holds is still its own
        }
        // Nothing in flight: sleep as long as it takes, because the only thing that can end this sleep is the
        // wake that follows the next request.  Something in flight: see the note on POLL_MS.
        let timeout = if waiting.is_empty() && !closed { u32::MAX } else { POLL_MS };
        arm.wait(REAP, timeout, |mut c| {
            if c.tag == WAKE_TAG {
                return; // a wake answers nothing; it is the reason this loop is running again
            }
            if let Some((caller_tag, reply)) = waiting.remove(&c.tag) {
                c.tag = caller_tag;
                let _ = reply.send(Ok(c));
            }
        });
    }
    drop(arm);
}
