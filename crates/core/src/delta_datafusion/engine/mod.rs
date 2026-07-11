//! The kernel [`Engine`](delta_kernel::Engine) deltalake-core hands the kernel when it runs on a
//! DataFusion session, plus the expression conversions between DataFusion and the kernel.
//!
//! Both live in `deltalake-kernel-executor`: every [`DataFusionEngine`] handler routes
//! through the crate's plan executor, and the kernel's declarative plans run on DataFusion as well.

pub(crate) use deltalake_kernel_executor::expressions::*;
pub use deltalake_kernel_executor::{AsObjectStoreUrl, DataFusionEngine};
// Public DataFusion `Expr` → kernel `Predicate` converter (checkpoint row-group
// skipping); see `deltalake_kernel_executor::expressions::to_delta_predicate`.
pub use deltalake_kernel_executor::expressions::to_delta_predicate;
