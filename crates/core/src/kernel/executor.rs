//! Target-portable executor bridging Delta Kernel's synchronous handler traits to async IO.
//!
//! Kernel handler traits are synchronous, but the engine's IO is async. Natively the gap is
//! bridged by blocking on a Tokio runtime ([`TracedHandle`]); on `wasm32-unknown-unknown`
//! there is no runtime to block on and blocking the JS event loop would deadlock, so the
//! [`InlineExecutor`] polls futures that must already be ready — the `deltalake-wasm` facade
//! primes the `_delta_log` into memory before any sync kernel call, making every future the
//! engine drives immediately ready. A future that would block is reported as an error, never
//! a hang.

use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context, Poll, Wake, Waker};

use delta_kernel::{DeltaResult, Error};
use tracing::Instrument;

/// A Tokio runtime [`Handle`](tokio::runtime::Handle) that propagates the current `tracing`
/// span across the async/sync boundary.
///
/// Kernel handler traits are synchronous, but the engine's IO is async and driven on a
/// Tokio runtime. The runtime's worker threads carry no `tracing` context, so a naive
/// `handle.block_on(fut)` would run `fut` under a disconnected root span. [`TracedHandle`]
/// captures the span current at the call site and re-enters it inside the future, so
/// kernel→engine callbacks nest under the originating scan/operation span.
#[derive(Clone, Debug)]
pub struct TracedHandle(tokio::runtime::Handle);

impl From<tokio::runtime::Handle> for TracedHandle {
    fn from(handle: tokio::runtime::Handle) -> Self {
        TracedHandle(handle)
    }
}

impl TracedHandle {
    pub(crate) fn block_on<F: Future>(&self, future: F) -> F::Output {
        let current_span = tracing::Span::current();
        let task = async move { future.instrument(current_span).await };
        self.0.block_on(task)
    }
}

/// An executor that completes already-ready futures by polling them inline, without a runtime.
///
/// This is the wasm execution model: after the `_delta_log` tail has been primed into an
/// in-memory store, every future the sync engine drives resolves without genuine IO waits.
/// The poll loop tolerates cooperative yields (wake-then-`Pending`, e.g. `buffered` streams)
/// but reports a future that parks without waking — i.e. one waiting on real IO — as an
/// error rather than spinning or hanging.
///
/// It compiles (and works) natively too, so the wasm execution model is testable in native
/// test suites by constructing an engine with `ExecutorHandle::Inline`.
#[derive(Clone, Debug, Default)]
pub struct InlineExecutor;

/// Wake flag for [`InlineExecutor`]: records that the future signalled progress while polling.
struct FlagWaker(AtomicBool);

impl Wake for FlagWaker {
    fn wake(self: Arc<Self>) {
        self.0.store(true, Ordering::Release);
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.store(true, Ordering::Release);
    }
}

impl InlineExecutor {
    /// Upper bound on poll iterations, to surface pathological wake-loops as errors
    /// instead of busy-spinning forever.
    const MAX_POLLS: usize = 10_000;

    /// Drive `future` to completion by polling inline.
    ///
    /// Returns an error if the future returns `Pending` without having woken its waker
    /// (it is waiting on IO that will never be observed — on wasm this means the required
    /// data was not primed), or if it exceeds the poll budget.
    pub fn try_block_on<F: Future>(&self, future: F) -> DeltaResult<F::Output> {
        let flag = Arc::new(FlagWaker(AtomicBool::new(false)));
        let waker = Waker::from(flag.clone());
        let mut cx = Context::from_waker(&waker);
        let mut future = std::pin::pin!(future);

        for _ in 0..Self::MAX_POLLS {
            match future.as_mut().poll(&mut cx) {
                Poll::Ready(value) => return Ok(value),
                Poll::Pending => {
                    // A cooperative yield wakes the waker before returning `Pending`;
                    // clear the flag and poll again. No wake means a genuine IO wait.
                    if !flag.0.swap(false, Ordering::AcqRel) {
                        return Err(Error::generic(
                            "future would block on the inline executor — log data not primed?",
                        ));
                    }
                }
            }
        }
        Err(Error::generic(
            "inline executor exceeded its poll budget — busy wake-loop in a driven future?",
        ))
    }
}

/// The executor a `DataFusionEngine` uses to drive async IO from the synchronous kernel
/// handler traits.
///
/// Native code blocks on a Tokio runtime with span propagation ([`TracedHandle`]); wasm uses
/// the [`InlineExecutor`], whose contract is that all driven futures are already ready
/// (log data primed by the caller).
#[derive(Clone, Debug)]
pub enum ExecutorHandle {
    /// Block on a Tokio runtime handle (native default).
    Tokio(TracedHandle),
    /// Poll ready futures inline without a runtime (wasm default; forcible natively for tests).
    Inline(InlineExecutor),
}

impl ExecutorHandle {
    /// The ambient executor: the current Tokio runtime natively (panics outside a runtime,
    /// matching `Handle::current()`), the inline executor on wasm.
    pub fn current() -> Self {
        #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
        {
            ExecutorHandle::Tokio(tokio::runtime::Handle::current().into())
        }
        #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
        {
            ExecutorHandle::Inline(InlineExecutor)
        }
    }

    /// Drive `future` to completion, failing (instead of hanging) when the executor cannot
    /// make progress — on the inline executor that means the future would genuinely block.
    pub fn try_block_on<F: Future>(&self, future: F) -> DeltaResult<F::Output> {
        match self {
            ExecutorHandle::Tokio(handle) => Ok(handle.block_on(future)),
            ExecutorHandle::Inline(executor) => executor.try_block_on(future),
        }
    }
}

impl From<tokio::runtime::Handle> for ExecutorHandle {
    fn from(handle: tokio::runtime::Handle) -> Self {
        ExecutorHandle::Tokio(handle.into())
    }
}

impl From<TracedHandle> for ExecutorHandle {
    fn from(handle: TracedHandle) -> Self {
        ExecutorHandle::Tokio(handle)
    }
}

impl From<InlineExecutor> for ExecutorHandle {
    fn from(executor: InlineExecutor) -> Self {
        ExecutorHandle::Inline(executor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inline_executor_completes_ready_future() {
        let value = InlineExecutor.try_block_on(async { 42 }).unwrap();
        assert_eq!(value, 42);
    }

    #[test]
    fn inline_executor_handles_cooperative_yields() {
        // A future that yields (wake + Pending) a few times before resolving.
        struct Yield(usize);
        impl Future for Yield {
            type Output = ();
            fn poll(
                mut self: std::pin::Pin<&mut Self>,
                cx: &mut Context<'_>,
            ) -> Poll<Self::Output> {
                if self.0 == 0 {
                    Poll::Ready(())
                } else {
                    self.0 -= 1;
                    cx.waker().wake_by_ref();
                    Poll::Pending
                }
            }
        }
        InlineExecutor.try_block_on(Yield(5)).unwrap();
    }

    #[test]
    fn inline_executor_errors_on_would_block() {
        let err = InlineExecutor
            .try_block_on(futures::future::pending::<()>())
            .unwrap_err();
        assert!(err.to_string().contains("would block"), "got: {err}");
    }

    #[test]
    fn inline_executor_errors_on_wake_loop() {
        // Pathological future that always wakes and never resolves.
        struct Spin;
        impl Future for Spin {
            type Output = ();
            fn poll(self: std::pin::Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
                cx.waker().wake_by_ref();
                Poll::Pending
            }
        }
        let err = InlineExecutor.try_block_on(Spin).unwrap_err();
        assert!(err.to_string().contains("poll budget"), "got: {err}");
    }
}
