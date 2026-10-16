//! A reader that records the bytes read through it as they are read.

use std::{
    io::{Read, Seek},
    pin::Pin,
    task::{Context, Poll},
};

use tokio::io::AsyncRead;

use crate::IOStatsRef;

/// Wraps a reader and adds every byte read through it to `io_stats`.
///
/// Bytes are recorded as each read returns, so a long read shows its
/// progress while it runs. Without `io_stats` it only passes reads through.
pub struct CountingReader<R> {
    reader: R,
    io_stats: Option<IOStatsRef>,
}

impl<R> CountingReader<R> {
    /// Reads from `reader`, recording the bytes read in `io_stats`.
    pub fn new(reader: R, io_stats: Option<IOStatsRef>) -> Self {
        Self { reader, io_stats }
    }

    fn record(&self, bytes: usize) {
        if let Some(io_stats) = &self.io_stats
            && bytes > 0
        {
            io_stats.mark_bytes_read(bytes);
        }
    }
}

impl<R> Read for CountingReader<R>
where
    R: Read,
{
    #[inline]
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let read = self.reader.read(buf)?;
        self.record(read);
        Ok(read)
    }
    #[inline]
    fn read_vectored(&mut self, bufs: &mut [std::io::IoSliceMut<'_>]) -> std::io::Result<usize> {
        let read = self.reader.read_vectored(bufs)?;
        self.record(read);
        Ok(read)
    }
    #[inline]
    fn read_to_end(&mut self, buf: &mut Vec<u8>) -> std::io::Result<usize> {
        let read = self.reader.read_to_end(buf)?;
        self.record(read);
        Ok(read)
    }
}

impl<R> Seek for CountingReader<R>
where
    R: Seek,
{
    #[inline]
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        self.reader.seek(pos)
    }
}

impl<R> AsyncRead for CountingReader<R>
where
    R: AsyncRead + Unpin,
{
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut tokio::io::ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        let before_len = buf.filled().len();
        let polled = Pin::new(&mut self.reader).poll_read(cx, buf);
        if matches!(&polled, Poll::Ready(Ok(()))) {
            self.record(buf.filled().len() - before_len);
        }
        polled
    }
}

#[cfg(test)]
mod tests {
    use std::io::Read;

    use super::CountingReader;
    use crate::IOStatsContext;

    #[test]
    fn every_byte_read_is_recorded_as_it_is_read() {
        let io_stats = IOStatsContext::new("counting");
        let mut reader = CountingReader::new(&b"0123456789"[..], Some(io_stats.clone()));
        let mut buf = [0u8; 4];

        reader.read_exact(&mut buf).unwrap();
        let after_four = io_stats.load_bytes_read();
        let mut rest = Vec::new();
        reader.read_to_end(&mut rest).unwrap();

        assert_eq!(after_four, 4);
        assert_eq!(io_stats.load_bytes_read(), 10);
    }

    #[tokio::test]
    async fn an_async_read_records_what_it_read() {
        let io_stats = IOStatsContext::new("counting");
        let mut reader = CountingReader::new(&b"abcdefgh"[..], Some(io_stats.clone()));
        let mut out = Vec::new();

        tokio::io::AsyncReadExt::read_to_end(&mut reader, &mut out)
            .await
            .unwrap();

        assert_eq!(out, b"abcdefgh");
        assert_eq!(io_stats.load_bytes_read(), 8);
    }
}
