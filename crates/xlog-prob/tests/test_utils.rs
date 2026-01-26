#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ScanState {
    Normal,
    LineComment,
    BlockComment,
    String,
    Char,
    RawString { hashes: usize },
}

pub fn extract_fn_body(source: &str, fn_name: &str) -> Option<String> {
    let needle = format!("fn {}", fn_name);
    let start = source.find(&needle)?;
    let bytes = source.as_bytes();

    let mut i = start + needle.len();
    while i < bytes.len() && bytes[i] != b'{' {
        i += 1;
    }
    if i >= bytes.len() {
        return None;
    }

    let body_start = i;
    i += 1;

    let mut depth: i32 = 1;
    let mut state = ScanState::Normal;
    let mut end: Option<usize> = None;

    while i < bytes.len() {
        let b = bytes[i];
        match state {
            ScanState::Normal => {
                if b == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
                    state = ScanState::LineComment;
                    i += 1;
                } else if b == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
                    state = ScanState::BlockComment;
                    i += 1;
                } else if b == b'"' {
                    state = ScanState::String;
                } else if b == b'\'' {
                    state = ScanState::Char;
                } else if b == b'r' {
                    let mut j = i + 1;
                    let mut hashes = 0usize;
                    while j < bytes.len() && bytes[j] == b'#' {
                        hashes += 1;
                        j += 1;
                    }
                    if j < bytes.len() && bytes[j] == b'"' {
                        state = ScanState::RawString { hashes };
                        i = j;
                    }
                } else if b == b'{' {
                    depth += 1;
                } else if b == b'}' {
                    depth -= 1;
                    if depth == 0 {
                        end = Some(i);
                        break;
                    }
                }
            }
            ScanState::LineComment => {
                if b == b'\n' {
                    state = ScanState::Normal;
                }
            }
            ScanState::BlockComment => {
                if b == b'*' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
                    state = ScanState::Normal;
                    i += 1;
                }
            }
            ScanState::String => {
                if b == b'\\' && i + 1 < bytes.len() {
                    i += 1;
                } else if b == b'"' {
                    state = ScanState::Normal;
                }
            }
            ScanState::Char => {
                if b == b'\\' && i + 1 < bytes.len() {
                    i += 1;
                } else if b == b'\'' {
                    state = ScanState::Normal;
                }
            }
            ScanState::RawString { hashes } => {
                if b == b'"' {
                    let mut j = i + 1;
                    let mut seen = 0usize;
                    while j < bytes.len() && bytes[j] == b'#' && seen < hashes {
                        seen += 1;
                        j += 1;
                    }
                    if seen == hashes {
                        state = ScanState::Normal;
                        i = j;
                    }
                }
            }
        }
        i += 1;
    }

    let end = end?;
    Some(source[body_start..=end].to_string())
}
