//! [`FetchObjectStore`]: a read-only [`ObjectStore`] over plain HTTP(S), backed by the
//! browser's `fetch` (via reqwest's wasm backend).
//!
//! Wasm-only. Serves ranged `GET`s for table data and log priming; there is no listing
//! (plain HTTP has none — prime with [`LogSource::Manifest`](crate::LogSource)) and all
//! mutations return `NotSupported`.
//!
//! JS interop types are `!Send`, but `object_store`'s trait bounds require `Send`
//! futures; every request future is wrapped in [`SendWrapper`], which is sound on the
//! single-threaded wasm target (it panics if actually moved across threads).

use std::ops::Range;

use bytes::Bytes;
use chrono::{DateTime, Utc};
use futures::StreamExt;
use futures::stream::BoxStream;
use object_store::path::Path;
use object_store::{
    Attributes, CopyOptions, GetOptions, GetRange, GetResult, GetResultPayload, ListResult,
    MultipartUpload, ObjectMeta, ObjectStore, PutMultipartOptions, PutOptions, PutPayload,
    PutResult,
};
use send_wrapper::SendWrapper;
use url::Url;

const STORE: &str = "FetchObjectStore";

fn generic_err(source: impl Into<Box<dyn std::error::Error + Send + Sync>>) -> object_store::Error {
    object_store::Error::Generic {
        store: STORE,
        source: source.into(),
    }
}

fn not_supported(op: &str) -> object_store::Error {
    object_store::Error::NotSupported {
        source: format!("{STORE} is read-only over plain HTTP: {op} is not supported").into(),
    }
}

/// Read-only `ObjectStore` resolving paths against one HTTP(S) origin.
///
/// Registered for a table URL's scheme/authority; object paths become the request path.
/// A query string on the base URL (e.g. a static SAS token valid for the whole
/// container) is preserved on every request. Per-object presigned URLs are out of scope
/// for v1 (they cannot share one base query string).
#[derive(Debug)]
pub struct FetchObjectStore {
    client: SendWrapper<reqwest::Client>,
    base: Url,
}

impl std::fmt::Display for FetchObjectStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{STORE}({})", self.base)
    }
}

impl FetchObjectStore {
    /// Create a store for `base`'s origin, keeping `base`'s query string (if any) on
    /// every request.
    pub fn try_new(base: Url) -> object_store::Result<Self> {
        if !matches!(base.scheme(), "http" | "https") {
            return Err(generic_err(format!(
                "expected an http(s) base URL, got {base}"
            )));
        }
        if !base.has_host() {
            return Err(generic_err(format!("base URL has no host: {base}")));
        }
        Ok(Self {
            client: SendWrapper::new(reqwest::Client::new()),
            base,
        })
    }

    fn url_for(&self, location: &Path) -> Url {
        let mut url = self.base.clone();
        // Paths are absolute within the origin; `set_path` percent-encodes as needed
        // and leaves the (possibly presigned) query string untouched.
        url.set_path(&format!("/{location}"));
        url
    }

    async fn fetch(&self, location: &Path, options: GetOptions) -> object_store::Result<GetResult> {
        let url = self.url_for(location);
        let request = if options.head {
            self.client.head(url.clone())
        } else {
            let mut request = self.client.get(url.clone());
            if let Some(range) = &options.range {
                request = request.header("Range", range_header(range));
            }
            request
        };
        let mut response = request.send().await.map_err(generic_err)?;

        // Some static hosts reject HEAD; retry as a zero-length ranged GET.
        if options.head && matches!(response.status().as_u16(), 405 | 501) {
            response = self
                .client
                .get(url)
                .header("Range", "bytes=0-0")
                .send()
                .await
                .map_err(generic_err)?;
        }

        let status = response.status();
        if status.as_u16() == 404 {
            return Err(object_store::Error::NotFound {
                path: location.to_string(),
                source: "HTTP 404".into(),
            });
        }
        if !status.is_success() {
            return Err(generic_err(format!(
                "GET {} returned HTTP {status}",
                location
            )));
        }

        let headers = response.headers().clone();
        let content_range = headers
            .get("content-range")
            .and_then(|v| v.to_str().ok())
            .and_then(parse_content_range);
        let content_length = headers
            .get("content-length")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<u64>().ok());
        let e_tag = headers
            .get("etag")
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned);
        let last_modified = headers
            .get("last-modified")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| DateTime::parse_from_rfc2822(v).ok())
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or(DateTime::<Utc>::UNIX_EPOCH);

        let body = if options.head {
            Bytes::new()
        } else {
            response.bytes().await.map_err(generic_err)?
        };

        // Object size: a 206 carries it in Content-Range; otherwise it is the full body.
        let (size, range) = match (&options.range, content_range) {
            (Some(_), Some((start, end, total))) => {
                let size = total.or(content_length).unwrap_or(end + 1);
                (size, start..end + 1)
            }
            (Some(requested), None) => {
                // Server ignored the Range header (HTTP 200 with the full body):
                // honor the request by slicing locally so callers still get the range
                // semantics they asked for.
                let full = body.len() as u64;
                let range = resolve_range(requested, full)?;
                let sliced = body.slice(range.start as usize..range.end as usize);
                return Ok(make_result(
                    location,
                    sliced,
                    full,
                    range,
                    e_tag,
                    last_modified,
                ));
            }
            (None, _) => {
                let size = content_length.unwrap_or(body.len() as u64);
                (size, 0..size)
            }
        };
        Ok(make_result(
            location,
            body,
            size,
            range,
            e_tag,
            last_modified,
        ))
    }
}

fn make_result(
    location: &Path,
    body: Bytes,
    size: u64,
    range: Range<u64>,
    e_tag: Option<String>,
    last_modified: DateTime<Utc>,
) -> GetResult {
    GetResult {
        payload: GetResultPayload::Stream(
            futures::stream::once(futures::future::ready(Ok(body))).boxed(),
        ),
        meta: ObjectMeta {
            location: location.clone(),
            last_modified,
            size,
            e_tag,
            version: None,
        },
        range,
        attributes: Attributes::default(),
    }
}

fn range_header(range: &GetRange) -> String {
    match range {
        // HTTP ranges are inclusive of the end byte; `GetRange::Bounded` is exclusive.
        GetRange::Bounded(r) => format!("bytes={}-{}", r.start, r.end.saturating_sub(1)),
        GetRange::Offset(offset) => format!("bytes={offset}-"),
        GetRange::Suffix(n) => format!("bytes=-{n}"),
    }
}

/// Parse `Content-Range: bytes <start>-<end>/<total|*>` into `(start, end, total)`.
fn parse_content_range(value: &str) -> Option<(u64, u64, Option<u64>)> {
    let rest = value.trim().strip_prefix("bytes ")?;
    let (span, total) = rest.split_once('/')?;
    let (start, end) = span.split_once('-')?;
    Some((start.parse().ok()?, end.parse().ok()?, total.parse().ok()))
}

/// Resolve a requested range against a known object length (mirrors `GetRange` docs).
fn resolve_range(range: &GetRange, len: u64) -> object_store::Result<Range<u64>> {
    let out_of_bounds = |start: u64| {
        generic_err(format!(
            "requested range starting at {start} is beyond the object length {len}"
        ))
    };
    match range {
        GetRange::Bounded(r) => {
            if r.start >= len {
                return Err(out_of_bounds(r.start));
            }
            Ok(r.start..r.end.min(len))
        }
        GetRange::Offset(offset) => {
            if *offset >= len {
                return Err(out_of_bounds(*offset));
            }
            Ok(*offset..len)
        }
        GetRange::Suffix(n) => Ok(len.saturating_sub(*n)..len),
    }
}

#[async_trait::async_trait]
impl ObjectStore for FetchObjectStore {
    async fn put_opts(
        &self,
        _location: &Path,
        _payload: PutPayload,
        _opts: PutOptions,
    ) -> object_store::Result<PutResult> {
        Err(not_supported("put"))
    }

    async fn put_multipart_opts(
        &self,
        _location: &Path,
        _opts: PutMultipartOptions,
    ) -> object_store::Result<Box<dyn MultipartUpload>> {
        Err(not_supported("put_multipart"))
    }

    async fn get_opts(
        &self,
        location: &Path,
        options: GetOptions,
    ) -> object_store::Result<GetResult> {
        // The JS fetch future is !Send; SendWrapper makes it satisfy the trait bound,
        // sound on this single-threaded target.
        SendWrapper::new(self.fetch(location, options)).await
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, object_store::Result<Path>>,
    ) -> BoxStream<'static, object_store::Result<Path>> {
        locations.map(|_| Err(not_supported("delete"))).boxed()
    }

    fn list(&self, _prefix: Option<&Path>) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        // Plain HTTP has no listing; prime with a manifest instead.
        futures::stream::once(futures::future::ready(Err(not_supported("list")))).boxed()
    }

    async fn list_with_delimiter(
        &self,
        _prefix: Option<&Path>,
    ) -> object_store::Result<ListResult> {
        Err(not_supported("list_with_delimiter"))
    }

    async fn copy_opts(
        &self,
        _from: &Path,
        _to: &Path,
        _options: CopyOptions,
    ) -> object_store::Result<()> {
        Err(not_supported("copy"))
    }
}
