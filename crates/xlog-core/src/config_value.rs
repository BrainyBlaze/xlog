//! Strict parsing for boolean configuration values.

use crate::{Result, XlogError};

const BOOLEAN_VALUES: &str = "one of 1/true/yes/on or 0/false/no/off";

/// Parse a boolean configuration value using the repository-wide spelling set.
pub fn parse_bool_value(name: &str, raw: &str) -> Result<bool> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => Err(XlogError::Configuration {
            name: name.to_owned(),
            value: raw.to_owned(),
            expected: BOOLEAN_VALUES,
        }),
    }
}

/// Read and strictly parse a boolean environment variable.
///
/// An unset variable returns `Ok(None)`. A present empty, unknown, or
/// non-Unicode value returns a typed configuration error.
pub fn read_bool_env(name: &str) -> Result<Option<bool>> {
    match std::env::var(name) {
        Ok(raw) => parse_bool_value(name, &raw).map(Some),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => Err(XlogError::Configuration {
            name: name.to_owned(),
            value: "<non-Unicode>".to_owned(),
            expected: BOOLEAN_VALUES,
        }),
    }
}

/// Resolve a boolean as explicit typed configuration, then environment, then default.
pub fn resolve_bool(explicit: Option<bool>, env_name: &str, default: bool) -> Result<bool> {
    match explicit {
        Some(value) => Ok(value),
        None => Ok(read_bool_env(env_name)?.unwrap_or(default)),
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_bool_value, read_bool_env, resolve_bool};
    use crate::XlogError;
    use std::sync::{Mutex, OnceLock};

    const TEST_ENV: &str = "XLOG_CORE_TEST_BOOLEAN";

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    struct EnvRestore(Option<std::ffi::OsString>);

    impl EnvRestore {
        fn capture() -> Self {
            Self(std::env::var_os(TEST_ENV))
        }
    }

    impl Drop for EnvRestore {
        fn drop(&mut self) {
            match self.0.take() {
                Some(value) => std::env::set_var(TEST_ENV, value),
                None => std::env::remove_var(TEST_ENV),
            }
        }
    }

    #[test]
    fn accepts_documented_true_and_false_values_case_insensitively() {
        for value in ["1", "true", "TRUE", " yes ", "On"] {
            assert!(parse_bool_value(TEST_ENV, value).unwrap(), "{value:?}");
        }
        for value in ["0", "false", "FALSE", " no ", "Off"] {
            assert!(!parse_bool_value(TEST_ENV, value).unwrap(), "{value:?}");
        }
    }

    #[test]
    fn rejects_empty_and_unknown_values_with_variable_diagnostics() {
        for raw in ["", " ", "enabled", "2"] {
            let error = parse_bool_value(TEST_ENV, raw).unwrap_err();
            assert!(matches!(
                error,
                XlogError::Configuration { ref name, ref value, .. }
                    if name == TEST_ENV && value == raw
            ));
            assert!(error.to_string().contains(TEST_ENV));
        }
    }

    #[test]
    fn unset_environment_value_is_absent() {
        let _guard = env_lock().lock().unwrap();
        let _restore = EnvRestore::capture();
        std::env::remove_var(TEST_ENV);
        assert_eq!(read_bool_env(TEST_ENV).unwrap(), None);
    }

    #[test]
    fn environment_value_uses_the_same_strict_parser() {
        let _guard = env_lock().lock().unwrap();
        let _restore = EnvRestore::capture();
        std::env::set_var(TEST_ENV, "yes");
        assert_eq!(read_bool_env(TEST_ENV).unwrap(), Some(true));
        std::env::set_var(TEST_ENV, "unknown");
        assert!(matches!(
            read_bool_env(TEST_ENV),
            Err(XlogError::Configuration { ref name, .. }) if name == TEST_ENV
        ));
    }

    #[test]
    fn resolution_precedence_is_explicit_then_environment_then_default() {
        let _guard = env_lock().lock().unwrap();
        let _restore = EnvRestore::capture();

        std::env::set_var(TEST_ENV, "invalid");
        assert!(!resolve_bool(Some(false), TEST_ENV, true).unwrap());

        std::env::set_var(TEST_ENV, "on");
        assert!(resolve_bool(None, TEST_ENV, false).unwrap());

        std::env::remove_var(TEST_ENV);
        assert!(resolve_bool(None, TEST_ENV, true).unwrap());
    }
}
