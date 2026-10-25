# object_store_unity

[`object_store`](https://docs.rs/object_store) integration for
[Unity Catalog](https://www.unitycatalog.io/) credential vending.

Unity Catalog governs data in cloud storage and vends short-lived,
least-privilege credentials for it. This crate turns them into regular object
stores — usable by DataFusion, delta-rs, `parquet`, and anything else that
takes an `Arc<dyn ObjectStore>` — and refreshes them before they expire.

```rust
use object_store::ObjectStoreExt;
use object_store_unity::{Operation, UnityStoreFactoryBuilder};

// DATABRICKS_HOST plus any Databricks auth: a token, OAuth M2M, OIDC,
// or a ~/.databrickscfg profile.
let factory = UnityStoreFactoryBuilder::from_env().build()?;

let volume = factory
    .for_url("uc:///Volumes/main/default/landing/", Operation::Read)
    .await?;
let bytes = volume.as_dyn().get(&"raw/file.json".into()).await?.bytes().await?;
```

## Addressing securables

| URL                                                   | Credentials from               |
|-------------------------------------------------------|--------------------------------|
| `uc:///Volumes/<catalog>/<schema>/<volume>[/<path>]`  | `temporary-volume-credentials` |
| `uc:///Tables/<catalog>/<schema>/<table>`             | `temporary-table-credentials`  |
| `uc://<catalog>.<schema>.<table>` (deltalake's form)  | `temporary-table-credentials`  |
| `s3://`, `abfss://`, `gs://`, `r2://`, …              | `temporary-path-credentials`   |
| `file://`                                             | none — local storage           |

## What you get

- **Self-refreshing credentials.** Each store is bound to the securable and
  operation it was created for and re-vends exactly that near expiry, so a
  refresh never widens access. Concurrent callers share one refresh.
- **Every cloud Unity Catalog vends for.** AWS (incl. S3 access points),
  Cloudflare R2, Azure (user-delegation SAS and AAD), and GCP — each behind a
  feature (`aws`, `azure`, `gcp`; all default). `file://` and Azurite locations
  work for local development.
- **Databricks authentication** via [`olai-http`](https://docs.rs/olai-http):
  personal access tokens, OAuth M2M, OIDC token env/file (workload identity
  federation), Azure managed identity, GCP token exchange, and
  `.databrickscfg` profiles.
- **One registry, many securables.** `RoutingObjectStore` serves every
  table or volume in a bucket from a single registry entry.
- **Bring your own vending.** Implement `CredentialVendor` to back a store
  with another endpoint, such as the Unity Catalog Delta API.

## Configuration

`UnityStoreFactoryBuilder` follows `object_store`'s builder conventions:
`from_env()`, `with_config(UnityConfigKey, value)`, `get_config_value`, and a
lenient `with_options` that ignores keys it does not know, so a whole
storage-options map can be passed through.

| Key                              | Meaning                                              |
|----------------------------------|------------------------------------------------------|
| `unity_catalog_url`              | REST base URL (default `<host>/api/2.1/unity-catalog/`) |
| `databricks_host`                | Workspace host                                       |
| `databricks_token`               | Personal access token                                |
| `databricks_client_id` / `_secret` | OAuth M2M service principal                        |
| `databricks_*`                   | Any other Databricks auth or HTTP client option      |
| `aws_region`                     | Region for vended AWS credentials                    |
| `unity_allow_unauthenticated`    | Local development without auth                       |

## License

Apache-2.0
