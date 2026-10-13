//! Connections the server closes once its drain is over.
//!
//! A drain lets running queries finish, then cancels the rest. A client that
//! holds a result open without reading it keeps its connection's last stream
//! from ever finishing, and the transport would wait on that connection for
//! ever. Each accepted connection is wrapped so that, once the drain is over,
//! its next read or write fails and the transport lets it go.
//!
//! The close is noticed only when a read or write would otherwise wait, which
//! is when it matters: a connection moving data needs no check per call.

use std::{
    future::Future,
    io::{self, IoSlice},
    pin::Pin,
    task::{Context, Poll},
};

use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio_util::sync::{CancellationToken, WaitForCancellationFutureOwned};
use tonic::transport::server::Connected;

/// A connection whose reads and writes fail once `closed` trips, instead of
/// waiting on a peer that does not answer.
pub struct ClosableConnection<IO> {
    io: IO,
    closed: Pin<Box<WaitForCancellationFutureOwned>>,
}

impl<IO> ClosableConnection<IO> {
    /// Wraps `io`, which is closed from the moment `closed` trips.
    #[must_use]
    pub fn new(io: IO, closed: CancellationToken) -> Self {
        Self {
            io,
            closed: Box::pin(closed.cancelled_owned()),
        }
    }

    /// Settles a read or write of `io` that returned `polled`.
    ///
    /// One that would wait fails instead once the connection is closed, and
    /// otherwise wakes `cx` when it closes.
    fn settle<T>(
        &mut self,
        polled: Poll<io::Result<T>>,
        cx: &mut Context<'_>,
    ) -> Poll<io::Result<T>> {
        match polled {
            Poll::Pending if self.closed.as_mut().poll(cx).is_ready() => {
                Poll::Ready(Err(io::Error::new(
                    io::ErrorKind::ConnectionAborted,
                    "the server closed the connection after its drain",
                )))
            }
            polled => polled,
        }
    }
}

impl<IO: AsyncRead + Unpin> AsyncRead for ClosableConnection<IO> {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        let polled = Pin::new(&mut this.io).poll_read(cx, buf);
        this.settle(polled, cx)
    }
}

impl<IO: AsyncWrite + Unpin> AsyncWrite for ClosableConnection<IO> {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();
        let polled = Pin::new(&mut this.io).poll_write(cx, buf);
        this.settle(polled, cx)
    }

    fn poll_write_vectored(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        bufs: &[IoSlice<'_>],
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();
        let polled = Pin::new(&mut this.io).poll_write_vectored(cx, bufs);
        this.settle(polled, cx)
    }

    fn is_write_vectored(&self) -> bool {
        self.io.is_write_vectored()
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        let polled = Pin::new(&mut this.io).poll_flush(cx);
        this.settle(polled, cx)
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.get_mut().io).poll_shutdown(cx)
    }
}

impl<IO: Connected> Connected for ClosableConnection<IO> {
    type ConnectInfo = IO::ConnectInfo;

    fn connect_info(&self) -> Self::ConnectInfo {
        self.io.connect_info()
    }
}

#[cfg(test)]
mod tests {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    use super::*;

    #[tokio::test]
    async fn a_connection_moves_data_until_it_is_closed() {
        let (near, mut far) = tokio::io::duplex(64);
        let mut connection = ClosableConnection::new(near, CancellationToken::new());

        far.write_all(b"ping").await.unwrap();
        let mut read = [0u8; 4];
        connection.read_exact(&mut read).await.unwrap();
        connection.write_all(b"pong").await.unwrap();
        far.read_exact(&mut read).await.unwrap();

        assert_eq!(&read, b"pong");
    }

    #[rstest::rstest]
    #[case::a_read_waiting_on_a_silent_peer(true)]
    #[case::a_write_waiting_on_a_peer_not_reading(false)]
    #[tokio::test]
    async fn closing_ends_a_wait_on_the_peer(#[case] reading: bool) {
        let (near, _far) = tokio::io::duplex(8);
        let closed = CancellationToken::new();
        let mut connection = ClosableConnection::new(near, closed.clone());

        let waiting = tokio::spawn(async move {
            if reading {
                let mut buf = [0u8; 1];
                connection.read(&mut buf).await.map(|_| ())
            } else {
                // More than the peer's buffer holds, so the write waits on it.
                connection.write_all(&[0u8; 64]).await
            }
        });
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert!(!waiting.is_finished());
        closed.cancel();
        let ended = tokio::time::timeout(std::time::Duration::from_secs(5), waiting)
            .await
            .unwrap()
            .unwrap();

        assert_eq!(ended.unwrap_err().kind(), io::ErrorKind::ConnectionAborted);
    }
}
