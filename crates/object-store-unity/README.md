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

// UNITY_CATALOG_URL plus UNITY_* auth — or, on Databricks, DATABRICKS_HOST
// plus any Databricks auth or ~/.databrickscfg profile.
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
- **Any identity provider.** Unity Catalog runs wherever it's deployed, so
  authentication isn't tied to Databricks: a bearer token, OAuth client
  credentials against any IdP, Entra ID (service principal, managed or
  workload identity, Azure CLI), GCP ID tokens, tokens from files or env
  vars, and the RFC 8693 token exchange OSS Unity Catalog uses. Databricks
  (PATs, OAuth M2M, OIDC, Azure, `.databrickscfg` profiles) is supported via
  [`olai-http`](https://docs.rs/olai-http).
- **The workload's own identity.** The `azure_*` / `google_*` identity keys
  and `AZURE_*` / `GOOGLE_*` variables `object_store` reads also authenticate
  to Unity Catalog, once its scope or audience is known. `unity_*` keys
  select a different identity; the most specific configuration wins.
- **AWS regions discovered.** Without a configured region, a bucket's region
  comes from S3 itself (`x-amz-bucket-region`), once per bucket.
- **One registry, many securables.** `RoutingObjectStore` serves every
  table or volume in a bucket from a single registry entry.
- **Bring your own vending.** Implement `CredentialVendor` to back a store
  with another endpoint, such as the Unity Catalog Delta API.

## Configuration

`UnityStoreFactoryBuilder` follows `object_store`'s builder conventions:
`from_env()`, `with_config(UnityConfigKey, value)`, `get_config_value`, and a
lenient `with_options` that ignores keys it does not know, so a whole
storage-options map can be passed through.

| Key | Meaning |
|-----|---------|
| `unity_catalog_url` | REST base URL (default `<databricks host>/api/2.1/unity-catalog/`) |
| `unity_token` | Bearer token |
| `unity_client_id` / `unity_client_secret` | OAuth client, Entra service principal, or user-assigned managed identity |
| `unity_oauth_token_url` / `unity_oauth_scope` / `unity_oauth_audience` | Token endpoint, scope, and audience of your IdP |
| `unity_azure_tenant_id` | Entra tenant |
| `unity_token_file` / `unity_token_env` | Token read from a file or environment variable |
| `unity_token_exchange` | `true` (server's `/api/1.0/unity-control/auth/tokens`) or an endpoint URL |
| `unity_auth_type` | Force a flow: `bearer`, `oauth-client-credentials`, `azure-client-secret`, `azure-managed-identity`, `azure-workload-identity`, `azure-cli`, `gcp-id-token`, `token-file`, `token-env`, `databricks`, `none` |
| `azure_client_id` / `azure_client_secret` / `azure_tenant_id` / `azure_federated_token_file` / `azure_msi_endpoint` / `azure_object_id` / `azure_msi_resource_id` / `azure_use_azure_cli` | The workload's Entra ID identity, as for `object_store`; needs `unity_oauth_scope` (known for Azure Databricks) |
| `google_application_credentials` / `google_service_account` / `google_service_account_key` | The workload's Google service account; needs `unity_oauth_audience` |
| `databricks_*` | Databricks authentication, or HTTP client options (`allow_http`, `timeout`, …) |
| `aws_region` | Region of vended S3 locations (default: discovered, then `AWS_REGION`) |
| `unity_allow_unauthenticated` | Local development without auth |

Every key can also be set as an upper-case environment variable read by
`from_env()` (`UNITY_TOKEN`, `AZURE_CLIENT_ID`, …). A `unity_*` key wins over
its well-known counterpart, e.g. `unity_azure_tenant_id` over
`azure_tenant_id`; Unity-specific credentials win over Databricks, and
Databricks over the well-known cloud identity.

## License

Apache-2.0
