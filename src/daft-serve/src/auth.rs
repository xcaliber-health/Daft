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

/// Server-side authentication policy.
#[derive(Debug, Clone)]
pub enum AuthPolicy {
    /// Every request must present the configured bearer token.
    Token(String),
    /// No authentication; only permitted for loopback binds or explicit
    /// opt-out.
    Insecure,
}

impl AuthPolicy {
    /// Validates the `authorization` metadata value for one request.
    ///
    /// `header` is the raw metadata value if present, e.g. `Bearer abc`.
    ///
    /// # Errors
    /// Returns [`ServeError::Unauthenticated`] if a token is required and the
    /// header is missing, malformed, or does not match.
    pub fn check(&self, header: Option<&str>) -> ServeResult<()> {
        match self {
            Self::Insecure => Ok(()),
            Self::Token(expected) => {
                let Some(header) = header else {
                    return Err(ServeError::Unauthenticated(
                        "missing authorization; pass the server token".to_string(),
                    ));
                };
                let Some(presented) = header.strip_prefix(BEARER_PREFIX) else {
                    return Err(ServeError::Unauthenticated(
                        "malformed authorization; expected `Bearer <token>`".to_string(),
                    ));
                };
                if constant_time_eq(presented.as_bytes(), expected.as_bytes()) {
                    Ok(())
                } else {
                    Err(ServeError::Unauthenticated("invalid token".to_string()))
                }
            }
        }
    }
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

    #[test]
    fn constant_time_eq_handles_lengths_and_content() {
        assert!(constant_time_eq(b"", b""));
        assert!(constant_time_eq(b"abc", b"abc"));
        assert!(!constant_time_eq(b"abc", b"abz"));
        assert!(!constant_time_eq(b"abc", b"ab"));
        assert!(!constant_time_eq(b"ab", b"abc"));
    }
}
