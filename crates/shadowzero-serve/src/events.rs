//! The push channel for the browser UI (docs/PLAN_UI_2026-11-17.md, feature 1).
//!
//! Until now the UI learned everything by polling `/v1/telemetry` once a second, and
//! the tool loop was driven entirely from the browser.  This hub lets the server
//! *push* instead: `/v1/events` is a Server-Sent-Events stream fed from here, and
//! every browser tab gets the same telemetry ticks, engine-status changes and tool
//! results the moment they happen, rather than up to a second after.
//!
//! The hub is one `tokio::sync::broadcast` channel of JSON values.  A broadcast
//! channel drops a message for a receiver that has fallen too far behind rather than
//! growing without bound, which is the right trade for a UI: a tab that stalls
//! simply misses a tick and the next one is still current (telemetry is a snapshot,
//! not a delta).  Subscribers that vanish are dropped by the channel itself.

use serde_json::Value;
use tokio::sync::broadcast;

/// Cap on the number of undelivered events a slow tab may fall behind by before its
/// missed ones are dropped.  Telemetry ticks are 1/s, so this is a few seconds of
/// grace before a lagging tab resynchronises on the next snapshot.
const CAPACITY: usize = 256;

#[derive(Clone)]
pub struct EventHub {
    tx: broadcast::Sender<Value>,
}

impl EventHub {
    pub fn new() -> Self {
        let (tx, _) = broadcast::channel(CAPACITY);
        Self { tx }
    }

    /// Send an event to every subscribed tab.  With no subscriber the message is
    /// simply dropped - publishing must never block or fail a request that produced
    /// it (a tool result is returned to the browser on its own HTTP response too).
    pub fn publish(&self, event: Value) {
        let _ = self.tx.send(event);
    }

    /// A fresh receiver starting at the next event published after this call.
    pub fn subscribe(&self) -> broadcast::Receiver<Value> {
        self.tx.subscribe()
    }
}
