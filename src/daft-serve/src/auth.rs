//! Bearer-token authentication for the serving endpoint.
//!
//! A server configured with a token rejects any request whose
//! `authorization` metadata is missing or does not match. Comparison is
//! constant-time with respect to the token contents so timing cannot be used
//! to recover it. Servers bound to non-loopback addresses must configure a
//! token unless explicitly opted out.

use crate::error::{ServeError, ServeResult};

/// Metadata key carrying the client credential.
pub const AUTHORIZATION_KEY: &str = "authorization";

/// Scheme prefix expected on the credential value.
pub const BEARER_PREFIX: &str = "Bearer ";

/// Identity resolved from a request's credential: the tenant name for a
/// tenant-scoped credential, or `None` for the anonymous single-credential
/// and unauthenticated modes.
pub type TenantId = Option<String>;

/// One named tenant and the credential that identifies it.
#[derive(Debug, Clone)]
pub struct TenantAuth {
    /// Tenant name; used for admission routing and query ownership, never
    /// secret.
    pub name: String,
    /// Bearer token identifying this tenant.
    pub token: String,
}

/// Server-side authentication policy.
#[derive(Debug, Clone)]
pub enum AuthPolicy {
    /// Every request must present the configured bearer token; callers are
    /// anonymous (no tenant identity).
    Token(String),
    /// Every request must present one of the tenant tokens; the matching
    /// tenant's name becomes the caller's identity.
    Tenants(Vec<TenantAuth>),
    /// No authentication; only permitted for loopback binds or explicit
    /// opt-out. Callers are anonymous.
    Insecure,
}

impl AuthPolicy {
    /// Validates the `authorization` metadata value for one request and
    /// resolves the caller's identity.
    ///
    /// `header` is the raw metadata value if present, e.g. `Bearer abc`.
    /// Every configured credential is compared in constant time, and all
    /// candidates are always examined, so response timing reveals neither
    /// token contents nor which tenant matched.
    ///
    /// # Errors
    /// Returns [`ServeError::Unauthenticated`] if a credential is required
    /// and the header is missing, malformed, or matches no configured token.
    pub fn check(&self, header: Option<&str>) -> ServeResult<TenantId> {
        match self {
            Self::Insecure => Ok(None),
            Self::Token(expected) => {
                let presented = extract_bearer(header)?;
                if constant_time_eq(presented.as_bytes(), expected.as_bytes()) {
                    Ok(None)
                } else {
                    Err(ServeError::Unauthenticated("invalid token".to_string()))
                }
            }
            Self::Tenants(tenants) => {
                let presented = extract_bearer(header)?;
                // Examine every candidate unconditionally so the number of
                // comparisons performed does not depend on which (if any)
                // tenant matched.
                let mut matched: Option<&str> = None;
                for tenant in tenants {
                    if constant_time_eq(presented.as_bytes(), tenant.token.as_bytes())
                        && matched.is_none()
                    {
                        matched = Some(tenant.name.as_str());
                    }
                }
                matched.map_or_else(
                    || Err(ServeError::Unauthenticated("invalid token".to_string())),
                    |name| Ok(Some(name.to_string())),
                )
            }
        }
    }
}

/// Extracts the bearer credential from a raw `authorization` value.
///
/// # Errors
/// Returns [`ServeError::Unauthenticated`] if the header is absent or does
/// not carry the expected scheme prefix.
fn extract_bearer(header: Option<&str>) -> ServeResult<&str> {
    let Some(header) = header else {
        return Err(ServeError::Unauthenticated(
            "missing authorization; pass the server token".to_string(),
        ));
    };
    header.strip_prefix(BEARER_PREFIX).ok_or_else(|| {
        ServeError::Unauthenticated(
            "malformed authorization; expected `Bearer <token>`".to_string(),
        )
    })
}

/// Compares two byte strings without early exit on mismatch.
///
/// The comparison touches every byte of `a` regardless of where the first
/// difference occurs, so response timing does not leak the matching prefix
/// length. Length inequality is folded into the same accumulator.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    let mut diff = u32::from(a.len() != b.len());
    for (i, &byte) in a.iter().enumerate() {
        let other = b.get(i).copied().unwrap_or(0);
        diff |= u32::from(byte ^ other);
    }
    diff == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insecure_policy_accepts_anything() {
        let policy = AuthPolicy::Insecure;
        assert!(policy.check(None).is_ok());
        assert!(policy.check(Some("Bearer whatever")).is_ok());
    }

    #[test]
    fn token_policy_accepts_matching_token() {
        let policy = AuthPolicy::Token("s3cr3t".to_string());
        assert!(policy.check(Some("Bearer s3cr3t")).is_ok());
    }

    #[test]
    fn token_policy_rejects_missing_header() {
        let policy = AuthPolicy::Token("s3cr3t".to_string());
        let err = policy.check(None).unwrap_err();
        assert!(matches!(err, ServeError::Unauthenticated(_)));
    }

    #[test]
    fn token_policy_rejects_malformed_header() {
        let policy = AuthPolicy::Token("s3cr3t".to_string());
        for header in ["s3cr3t", "bearer s3cr3t", "Token s3cr3t", ""] {
            let err = policy.check(Some(header)).unwrap_err();
            assert!(matches!(err, ServeError::Unauthenticated(_)), "{header}");
        }
    }

    #[test]
    fn token_policy_rejects_wrong_token() {
        let policy = AuthPolicy::Token("s3cr3t".to_string());
        for wrong in ["Bearer wrong", "Bearer s3cr3", "Bearer s3cr3tt", "Bearer "] {
            let err = policy.check(Some(wrong)).unwrap_err();
            assert!(matches!(err, ServeError::Unauthenticated(_)), "{wrong}");
        }
    }

    fn two_tenants() -> AuthPolicy {
        AuthPolicy::Tenants(vec![
            TenantAuth {
                name: "alpha".to_string(),
                token: "tok-alpha".to_string(),
            },
            TenantAuth {
                name: "beta".to_string(),
                token: "tok-beta".to_string(),
            },
        ])
    }

    #[test]
    fn tenant_policy_resolves_matching_tenant() {
        let policy = two_tenants();
        assert_eq!(
            policy.check(Some("Bearer tok-alpha")).unwrap(),
            Some("alpha".to_string())
        );
        assert_eq!(
            policy.check(Some("Bearer tok-beta")).unwrap(),
            Some("beta".to_string())
        );
    }

    #[test]
    fn tenant_policy_rejects_unknown_and_malformed() {
        let policy = two_tenants();
        for header in [None, Some("Bearer nope"), Some("tok-alpha"), Some("")] {
            let err = policy.check(header).unwrap_err();
            assert!(matches!(err, ServeError::Unauthenticated(_)), "{header:?}");
        }
    }

    #[test]
    fn empty_tenant_list_rejects_everything() {
        let policy = AuthPolicy::Tenants(vec![]);
        assert!(policy.check(Some("Bearer anything")).is_err());
    }

    #[test]
    fn single_token_policy_yields_anonymous_identity() {
        let policy = AuthPolicy::Token("s3cr3t".to_string());
        assert_eq!(policy.check(Some("Bearer s3cr3t")).unwrap(), None);
        assert_eq!(AuthPolicy::Insecure.check(None).unwrap(), None);
    }

    #[test]
    fn constant_time_eq_handles_lengths_and_content() {
        assert!(constant_time_eq(b"", b""));
        assert!(constant_time_eq(b"abc", b"abc"));
        assert!(!constant_time_eq(b"abc", b"abz"));
        assert!(!constant_time_eq(b"abc", b"ab"));
        assert!(!constant_time_eq(b"ab", b"abc"));
    }
}
