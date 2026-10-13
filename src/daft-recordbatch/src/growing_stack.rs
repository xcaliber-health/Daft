//! Polling a deeply recursive future without overflowing its thread's stack.
//!
//! A recursive async function nests one future per level, and polling the
//! outermost polls every level beneath it on the same stack. Each level is
//! wrapped so that its poll runs on a freshly allocated stack segment whenever
//! the current one runs low, with the same limits the synchronous recursions
//! use. Nothing lives on the stack across an await, so a grown segment is
//! released as soon as its poll returns.

use std::{
    future::Future,
    pin::Pin,
    task::{Context, Poll},
};

/// A future polled on a stack grown when the current one runs low.
pub(crate) struct GrowingStack<F>(pub(crate) F);

impl<F: Future + Unpin> Future for GrowingStack<F> {
    type Output = F::Output;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let inner = &mut self.0;
        stacker::maybe_grow(
            recursive::get_minimum_stack_size(),
            recursive::get_stack_allocation_size(),
            || Pin::new(inner).poll(cx),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Counts down through `depth` nested boxed futures, as a recursive
    /// async function does.
    fn count_down(depth: u32) -> Pin<Box<dyn Future<Output = u32> + Send>> {
        Box::pin(async move {
            if depth == 0 {
                0
            } else {
                GrowingStack(count_down(depth - 1)).await + 1
            }
        })
    }

    #[test]
    fn a_recursion_deeper_than_its_threads_stack_completes() {
        // Far more levels than a 256 KiB stack holds unguarded.
        let depth = 100_000;
        let completed = std::thread::Builder::new()
            .stack_size(256 * 1024)
            .spawn(move || futures::executor::block_on(GrowingStack(count_down(depth))))
            .unwrap()
            .join()
            .unwrap();

        assert_eq!(completed, depth);
    }
}
