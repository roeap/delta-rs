//! URL grammar for Unity Catalog securables.

use std::str::FromStr;

use url::Url;

use crate::error::{Error, Result};

fn invalid(message: String) -> Error {
    Error::InvalidUrl(message)
}

/// A parsed reference to a Unity Catalog securable.
///
/// Construct one via [`UnityReference::parse`] (or its [`FromStr`] impl). The
/// variants map 1:1 to the three credential-vending endpoints exposed by
/// Unity Catalog.
///
/// # Scheme
///
/// | URL shape                                                  | Vending endpoint               |
/// |------------------------------------------------------------|--------------------------------|
/// | `uc:///Volumes/<catalog>/<schema>/<volume>[/<path>]`       | `temporary-volume-credentials` |
/// | `uc:///Tables/<catalog>/<schema>/<table>`                  | `temporary-table-credentials`  |
/// | `s3://`, `s3a://`, `gs://`, `abfs://`, `abfss://`, `az://`, `azure://`, `r2://` raw cloud URL | `temporary-path-credentials` |
///
/// For ecosystem compatibility, the parser also accepts:
///
/// - `vol+dbfs:/Volumes/<catalog>/<schema>/<volume>[/<path>]`, used by some
///   Databricks credential-vending samples — identical to `uc:///Volumes/...`;
/// - `uc://<catalog>.<schema>.<table>`, the table form `deltalake` has always
///   accepted — identical to `uc:///Tables/<catalog>/<schema>/<table>`.
///
/// ## Why the kind goes in the path
///
/// The capitalised `Volumes` / `Tables` segment mirrors the Databricks
/// workspace POSIX convention (`/Volumes/<catalog>/<schema>/<volume>/...`).
/// Names stay out of the URL authority because Unity Catalog names commonly
/// contain underscores, which hostnames do not allow (RFC 1123). The legacy
/// dotted form is therefore parsed from the raw string.
///
/// ## Case-insensitivity
///
/// Kind segments are matched **case-insensitively**, so `/Volumes/...`,
/// `/volumes/...`, and `/VOLUMES/...` all dispatch the same way. The
/// capitalised form is canonical in docs because it matches the Databricks
/// workspace path convention; the lowercase form matches the REST API.
/// Catalog, schema, volume, and table names are passed through verbatim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnityReference {
    /// A Unity Catalog volume, with an optional sub-path inside the volume.
    Volume {
        /// The catalog name.
        catalog: String,
        /// The schema name.
        schema: String,
        /// The volume name.
        volume: String,
        /// Sub-path inside the volume, e.g. `raw/2024/01/file.parquet`.
        ///
        /// Empty when the URL addresses the volume root.
        path: String,
    },
    /// A Unity Catalog table (referenced by its three-level name).
    Table {
        /// The catalog name.
        catalog: String,
        /// The schema name.
        schema: String,
        /// The table name.
        table: String,
    },
    /// A raw cloud URL (`s3://`, `gs://`, `abfss://`, ...). Credentials are
    /// vended via the `temporary-path-credentials` endpoint.
    Path(Url),
}

impl UnityReference {
    /// Parse a URL string into a [`UnityReference`].
    ///
    /// See the [type-level documentation](UnityReference) for the supported shapes.
    pub fn parse(input: &str) -> Result<Self> {
        // Handle the `vol+dbfs:` alias before passing through to `url::Url`,
        // because `vol+dbfs:/Volumes/...` is path-only (no authority) and we
        // want to share the rest of the volume-parsing logic with `uc://`.
        if let Some(rest) = input.strip_prefix("vol+dbfs:") {
            return parse_volume_form(rest)
                .map_err(|e| invalid(format!("invalid vol+dbfs URL `{input}`: {e}")));
        }

        if let Some(reference) = parse_legacy_table(input) {
            return Ok(reference);
        }

        let url =
            Url::parse(input).map_err(|e| invalid(format!("could not parse `{input}`: {e}")))?;

        match url.scheme() {
            "uc" => parse_uc(&url).map_err(|e| invalid(format!("invalid uc URL `{input}`: {e}"))),
            // Treat all known cloud schemes as raw external paths.
            "s3" | "s3a" | "gs" | "gcs" | "abfs" | "abfss" | "az" | "azure" | "adl" | "r2"
            | "file" | "http" | "https" => Ok(UnityReference::Path(url)),
            other => Err(invalid(format!(
                "unsupported URL scheme `{other}` (expected `uc`, `vol+dbfs`, or a cloud scheme like `s3`/`gs`/`abfss`)"
            ))),
        }
    }

    /// Convenience accessor: returns the three-level dotted name for
    /// [`Volume`] and [`Table`] references, [`None`] for raw paths.
    ///
    /// [`Volume`]: UnityReference::Volume
    /// [`Table`]: UnityReference::Table
    pub fn full_name(&self) -> Option<String> {
        match self {
            UnityReference::Volume {
                catalog,
                schema,
                volume,
                ..
            } => Some(format!("{catalog}.{schema}.{volume}")),
            UnityReference::Table {
                catalog,
                schema,
                table,
            } => Some(format!("{catalog}.{schema}.{table}")),
            UnityReference::Path(_) => None,
        }
    }
}

impl FromStr for UnityReference {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        Self::parse(s)
    }
}

fn parse_uc(url: &Url) -> std::result::Result<UnityReference, String> {
    // `uc:///Volumes/...` parses with `url::Url` as host == Some("") and
    // path == "/Volumes/..." — a leading slash on the path. We treat both
    // `uc:///` and `uc://` (no host) consistently by working off the path.
    if url.host_str().is_some_and(|h| !h.is_empty()) {
        return Err(format!(
            "`uc` URLs must use an empty authority (e.g. `uc:///Volumes/...`), got host `{}`",
            url.host_str().unwrap()
        ));
    }
    let path = url.path();
    if !path.starts_with('/') {
        return Err(format!("expected an absolute path, got `{path}`"));
    }
    let segments: Vec<&str> = path
        .trim_start_matches('/')
        .split('/')
        .filter(|s| !s.is_empty())
        .collect();
    parse_securable_segments(&segments)
}

/// Shared logic for `uc:///Volumes/...`, `uc:///Tables/...`, and the
/// `vol+dbfs:/Volumes/...` alias.
///
/// The first segment (the "kind") is matched case-insensitively so any of
/// `Volumes`, `volumes`, or `VOLUMES` work; all other segments are
/// preserved verbatim.
fn parse_securable_segments(segments: &[&str]) -> std::result::Result<UnityReference, String> {
    let kind = match segments.first() {
        None => return Err("empty path; expected `/Volumes/...` or `/Tables/...`".to_string()),
        Some(k) => *k,
    };
    let tail = &segments[1..];

    if kind.eq_ignore_ascii_case("Volumes") {
        return match tail {
            [catalog, schema, volume, rest @ ..] => Ok(UnityReference::Volume {
                catalog: percent_decode(catalog)?,
                schema: percent_decode(schema)?,
                volume: percent_decode(volume)?,
                path: rest.join("/"),
            }),
            _ => Err(
                "`uc:///Volumes/<catalog>/<schema>/<volume>[/<path>]` requires all three names"
                    .to_string(),
            ),
        };
    }

    if kind.eq_ignore_ascii_case("Tables") {
        return match tail {
            [catalog, schema, table] => Ok(UnityReference::Table {
                catalog: percent_decode(catalog)?,
                schema: percent_decode(schema)?,
                table: percent_decode(table)?,
            }),
            _ => Err(
                "`uc:///Tables/<catalog>/<schema>/<table>` requires exactly three name segments"
                    .to_string(),
            ),
        };
    }

    Err(format!(
        "unknown securable kind `{kind}`; expected `Volumes` or `Tables` (case-insensitive)"
    ))
}

/// Parse the legacy `uc://<catalog>.<schema>.<table>` form.
///
/// Works on the raw string rather than a parsed [`Url`] so the names reach the
/// caller exactly as written. As before, everything after the second `.` is the
/// table name.
fn parse_legacy_table(input: &str) -> Option<UnityReference> {
    let name = input.strip_prefix("uc://")?.trim_end_matches('/');
    if name.is_empty() || name.contains('/') {
        return None;
    }
    match name.splitn(3, '.').collect::<Vec<_>>().as_slice() {
        [catalog, schema, table]
            if !catalog.is_empty() && !schema.is_empty() && !table.is_empty() =>
        {
            Some(UnityReference::Table {
                catalog: (*catalog).to_string(),
                schema: (*schema).to_string(),
                table: (*table).to_string(),
            })
        }
        _ => None,
    }
}

/// Parse the path component of a `vol+dbfs:/Volumes/<c>/<s>/<v>/<p>` URL.
///
/// The `vol+dbfs` scheme is non-hierarchical (no `//` authority), so callers
/// pass the raw remainder after the `vol+dbfs:` prefix.
fn parse_volume_form(rest: &str) -> std::result::Result<UnityReference, String> {
    let path = rest.trim_start_matches('/');
    let segments: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
    match parse_securable_segments(&segments)? {
        volume @ UnityReference::Volume { .. } => Ok(volume),
        _ => Err("expected `vol+dbfs:/Volumes/<catalog>/<schema>/<volume>[/<path>]`".to_string()),
    }
}

fn percent_decode(s: &str) -> std::result::Result<String, String> {
    // The `url` crate already decodes most of the path for us; we just
    // surface invalid UTF-8 explicitly so callers see a clean error.
    percent_encoding::percent_decode_str(s)
        .decode_utf8()
        .map(|c| c.into_owned())
        .map_err(|e| format!("invalid percent-encoding in `{s}`: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_volume_with_path() {
        let r = UnityReference::parse("uc:///Volumes/main/default/landing/raw/2024/file.parquet")
            .unwrap();
        assert_eq!(
            r,
            UnityReference::Volume {
                catalog: "main".into(),
                schema: "default".into(),
                volume: "landing".into(),
                path: "raw/2024/file.parquet".into(),
            }
        );
        assert_eq!(r.full_name().as_deref(), Some("main.default.landing"));
    }

    #[test]
    fn parses_volume_root() {
        let r = UnityReference::parse("uc:///Volumes/main/default/landing").unwrap();
        assert!(matches!(&r, UnityReference::Volume { path, .. } if path.is_empty()));
    }

    #[test]
    fn parses_volume_root_trailing_slash() {
        let r = UnityReference::parse("uc:///Volumes/main/default/landing/").unwrap();
        assert!(matches!(&r, UnityReference::Volume { path, .. } if path.is_empty()));
    }

    #[test]
    fn parses_table() {
        let r = UnityReference::parse("uc:///Tables/main/default/orders").unwrap();
        assert_eq!(
            r,
            UnityReference::Table {
                catalog: "main".into(),
                schema: "default".into(),
                table: "orders".into(),
            }
        );
        assert_eq!(r.full_name().as_deref(), Some("main.default.orders"));
    }

    #[test]
    fn vol_dbfs_alias() {
        let r =
            UnityReference::parse("vol+dbfs:/Volumes/main/default/landing/sub/file.bin").unwrap();
        assert_eq!(
            r,
            UnityReference::Volume {
                catalog: "main".into(),
                schema: "default".into(),
                volume: "landing".into(),
                path: "sub/file.bin".into(),
            }
        );
    }

    #[test]
    fn raw_cloud_urls_are_paths() {
        for raw in [
            "s3://bucket/prefix/key",
            "s3a://bucket/prefix",
            "gs://bucket/x",
            "abfss://container@account.dfs.core.windows.net/path",
            "az://account/container/blob",
            "r2://bucket/key",
        ] {
            let r = UnityReference::parse(raw).unwrap();
            assert!(
                matches!(r, UnityReference::Path(_)),
                "expected Path for {raw}"
            );
            assert!(r.full_name().is_none());
        }
    }

    #[test]
    fn rejects_volume_with_missing_segments() {
        let err = UnityReference::parse("uc:///Volumes/main/default").unwrap_err();
        assert!(format!("{err}").contains("Volumes"));
    }

    #[test]
    fn rejects_table_with_extra_segments() {
        let err = UnityReference::parse("uc:///Tables/main/default/orders/extra").unwrap_err();
        assert!(format!("{err}").contains("Tables"));
    }

    #[test]
    fn rejects_unknown_securable_kind() {
        let err = UnityReference::parse("uc:///Functions/main/default/fn").unwrap_err();
        assert!(format!("{err}").contains("Functions"));
    }

    #[test]
    fn rejects_unknown_scheme() {
        let err = UnityReference::parse("ftp://server/path").unwrap_err();
        assert!(format!("{err}").contains("unsupported"));
    }

    #[test]
    fn rejects_uc_with_authority() {
        let err = UnityReference::parse("uc://some-host/Volumes/main/default/v").unwrap_err();
        assert!(format!("{err}").contains("empty authority"));
    }

    #[test]
    fn kind_segment_is_case_insensitive() {
        for url in [
            "uc:///Volumes/main/default/landing/raw/file.parquet",
            "uc:///volumes/main/default/landing/raw/file.parquet",
            "uc:///VOLUMES/main/default/landing/raw/file.parquet",
            "uc:///VoLuMeS/main/default/landing/raw/file.parquet",
        ] {
            let r = UnityReference::parse(url).unwrap();
            assert_eq!(
                r,
                UnityReference::Volume {
                    catalog: "main".into(),
                    schema: "default".into(),
                    volume: "landing".into(),
                    path: "raw/file.parquet".into(),
                },
                "parsing {url}",
            );
        }

        for url in [
            "uc:///Tables/main/default/orders",
            "uc:///tables/main/default/orders",
            "uc:///TABLES/main/default/orders",
        ] {
            let r = UnityReference::parse(url).unwrap();
            assert!(matches!(r, UnityReference::Table { .. }), "parsing {url}");
        }
    }

    #[test]
    fn vol_dbfs_alias_kind_is_case_insensitive() {
        for url in [
            "vol+dbfs:/Volumes/main/default/landing/file.bin",
            "vol+dbfs:/volumes/main/default/landing/file.bin",
            "vol+dbfs:/VOLUMES/main/default/landing/file.bin",
        ] {
            let r = UnityReference::parse(url).unwrap();
            assert!(matches!(r, UnityReference::Volume { .. }), "parsing {url}");
        }
    }

    #[test]
    fn legacy_dotted_table_form() {
        for url in ["uc://Main.Sales_DB.Orders", "uc://Main.Sales_DB.Orders/"] {
            assert_eq!(
                UnityReference::parse(url).unwrap(),
                UnityReference::Table {
                    catalog: "Main".into(),
                    schema: "Sales_DB".into(),
                    table: "Orders".into(),
                },
                "parsing {url}"
            );
        }
        // Only two names: not a table, and not a valid `uc` URL either.
        assert!(UnityReference::parse("uc://main.default").is_err());
    }

    #[test]
    fn percent_encoded_names_are_decoded() {
        let r = UnityReference::parse("uc:///Volumes/main/default/my%5Fvolume/file").unwrap();
        let UnityReference::Volume { volume, .. } = r else {
            panic!()
        };
        assert_eq!(volume, "my_volume");
    }
}
