use crate::tokenizer::vocab::{TokenID, Vocab};
use fancy_regex::Regex;
use once_cell::sync::Lazy;
use serde_json::*;

/*
This module sits between raw input text and the BPE engine.
It has one job: take a raw string and return a list of chunks that BPE will process independently.
*/

// using lazy as compiling regex strings are inefficient
static GPT2_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+").unwrap()
});

#[derive(Debug)]
pub enum Chunk {
    Text(String),
    Special(TokenID),
}

#[derive(Debug)]
pub enum PreToken {
    Text(String),     // ready for BPE — already byte-mapped
    Special(TokenID), // bypass BPE entirely
}

fn byte_to_unicode_table() -> [char; 256] {
    let mut table = [' '; 256];

    let printable: Vec<u8> = (b'!'..=b'~') // 33–126
        .chain(b'\xA1'..=b'\xAC') // 161–172
        .chain(b'\xAE'..=b'\xFF') // 174–255
        .collect();

    for &b in &printable {
        table[b as usize] = b as char;
    }

    // everything not in printable ranges gets mapped to
    // unicode codepoints starting at 256, in ascending order
    let mut n = 256u32;
    for i in 0u32..=255 {
        if table[i as usize] == ' ' {
            // not yet assigned
            table[i as usize] = char::from_u32(n).unwrap();
            n += 1;
        }
    }

    table
}

fn apply_byte_mapping(table: &[char; 256], s: &str) -> String {
    s.bytes().map(|b| table[b as usize]).collect()
}

fn split_special_tokens(text: &str, vocab: &Vocab) -> Vec<Chunk> {
    let mut pos = 0;
    let mut buf = String::new();
    let mut chunks = Vec::new();
    while pos < text.len() {
        if let Some((prefix, index)) = vocab.find_special_prefix(&text[pos..]) {
            if !buf.is_empty() {
                chunks.push(Chunk::Text(std::mem::take(&mut buf)));
            }
            chunks.push(Chunk::Special(index));
            pos += prefix.len();
        } else {
            let next_char = text[pos..].chars().next().unwrap();
            buf.push(next_char);
            pos += next_char.len_utf8();
        }
    }
    if !buf.is_empty() {
        chunks.push(Chunk::Text(buf));
    }
    chunks
}

fn apply_regex(chunks: Vec<Chunk>) -> Vec<Chunk> {
    chunks
        .into_iter()
        .flat_map(|chunk| match chunk {
            Chunk::Special(id) => vec![Chunk::Special(id)],
            Chunk::Text(s) => {
                let matches: Vec<Chunk> = GPT2_REGEX
                    .find_iter(&s)
                    .filter_map(|m| m.ok())
                    .map(|m| Chunk::Text(m.as_str().to_string()))
                    .collect();
                matches
            }
        })
        .collect()
}

fn apply_byte_mapping_to_chunks(chunks: Vec<Chunk>) -> Vec<PreToken> {
    let table = byte_to_unicode_table();
    chunks
        .into_iter()
        .flat_map(|chunk| match chunk {
            Chunk::Special(id) => vec![PreToken::Special(id)],
            Chunk::Text(s) => vec![PreToken::Text(apply_byte_mapping(&table, &s))],
        })
        .collect()
}

// orchestrates all 3 steps
pub fn pretokenize(text: &str, vocab: &Vocab) -> Vec<PreToken> {
    let chunks = split_special_tokens(text, vocab);
    let chunks = apply_regex(chunks);
    apply_byte_mapping_to_chunks(chunks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::vocab::{TokenID, Vocab};

    fn make_test_vocab() -> Vocab {
        let tokens = vec!["hello".to_string(), "world".to_string(), "foo".to_string()];
        let special_tokens = vec![
            ("<|begin_of_text|>".to_string(), 100),
            ("<|end_of_text|>".to_string(), 101),
            ("<|end|>".to_string(), 102), // shorter prefix of end_of_text — tests longest match
        ];
        Vocab::new(
            tokens,
            None,
            special_tokens,
            Some(100),
            Some(101),
            None,
            None,
        )
    }

    // helpers to make assertions readable
    fn is_text(chunk: &Chunk, expected: &str) -> bool {
        matches!(chunk, Chunk::Text(s) if s == expected)
    }

    fn is_special(chunk: &Chunk, expected_id: TokenID) -> bool {
        matches!(chunk, Chunk::Special(id) if *id == expected_id)
    }

    #[test]
    fn test_no_special_tokens() {
        let vocab = make_test_vocab();
        let chunks = split_special_tokens("hello world", &vocab);
        assert_eq!(chunks.len(), 1);
        assert!(is_text(&chunks[0], "hello world"));
    }

    #[test]
    fn test_only_special_token() {
        let vocab = make_test_vocab();
        let chunks = split_special_tokens("<|begin_of_text|>", &vocab);
        assert_eq!(chunks.len(), 1);
        assert!(is_special(&chunks[0], 100));
    }

    #[test]
    fn test_special_token_at_start() {
        let vocab = make_test_vocab();
        let chunks = split_special_tokens("<|begin_of_text|>hello world", &vocab);
        assert_eq!(chunks.len(), 2);
        assert!(is_special(&chunks[0], 100));
        assert!(is_text(&chunks[1], "hello world"));
    }

    #[test]
    fn test_special_token_at_end() {
        let vocab = make_test_vocab();
        let chunks = split_special_tokens("hello world<|end_of_text|>", &vocab);
        assert_eq!(chunks.len(), 2);
        assert!(is_text(&chunks[0], "hello world"));
        assert!(is_special(&chunks[1], 101));
    }

    #[test]
    fn test_special_token_in_middle() {
        let vocab = make_test_vocab();
        let chunks = split_special_tokens("hello<|begin_of_text|>world", &vocab);
        assert_eq!(chunks.len(), 3);
        assert!(is_text(&chunks[0], "hello"));
        assert!(is_special(&chunks[1], 100));
        assert!(is_text(&chunks[2], "world"));
    }

    #[test]
    fn test_multiple_special_tokens() {
        let vocab = make_test_vocab();
        let chunks = split_special_tokens("<|begin_of_text|>hello<|end_of_text|>", &vocab);
        assert_eq!(chunks.len(), 3);
        assert!(is_special(&chunks[0], 100));
        assert!(is_text(&chunks[1], "hello"));
        assert!(is_special(&chunks[2], 101));
    }

    #[test]
    fn test_adjacent_special_tokens() {
        let vocab = make_test_vocab();
        let chunks = split_special_tokens("<|begin_of_text|><|end_of_text|>", &vocab);
        assert_eq!(chunks.len(), 2);
        assert!(is_special(&chunks[0], 100));
        assert!(is_special(&chunks[1], 101));
    }

    #[test]
    fn test_longest_match_wins() {
        // "<|end|>" and "<|end_of_text|>" both start with "<|end"
        // longest must win — should match "<|end_of_text|>" not "<|end|>"
        let vocab = make_test_vocab();
        let chunks = split_special_tokens("<|end_of_text|>", &vocab);
        assert_eq!(chunks.len(), 1);
        assert!(is_special(&chunks[0], 101)); // 101 = end_of_text, not 102 = end
    }

    #[test]
    fn test_empty_string() {
        let vocab = make_test_vocab();
        let chunks = split_special_tokens("", &vocab);
        assert_eq!(chunks.len(), 0);
    }

    #[test]
    fn test_no_empty_text_chunks() {
        // when special tokens are adjacent, no empty Text chunks should appear
        let vocab = make_test_vocab();
        let chunks = split_special_tokens("<|begin_of_text|><|end_of_text|>", &vocab);
        for chunk in &chunks {
            if let Chunk::Text(s) = chunk {
                assert!(!s.is_empty(), "empty Text chunk found");
            }
        }
    }

    #[test]
    fn test_text_only_no_special_candidates() {
        let vocab = make_test_vocab();
        let chunks = split_special_tokens("just some plain text 123", &vocab);
        assert_eq!(chunks.len(), 1);
        assert!(is_text(&chunks[0], "just some plain text 123"));
    }

    #[test]
    fn test_regex_contraction() {
        let vocab = make_test_vocab();
        let chunks = split_special_tokens("don't", &vocab);
        let result = apply_regex(chunks);

        let texts: Vec<&str> = result
            .iter()
            .filter_map(|c| {
                if let Chunk::Text(s) = c {
                    Some(s.as_str())
                } else {
                    None
                }
            })
            .collect();

        assert!(texts.contains(&"don"));
        assert!(texts.contains(&"'t"));
    }

    #[test]
    fn test_regex_special_passthrough() {
        let vocab = make_test_vocab();
        let chunks = split_special_tokens("<|begin_of_text|>hello", &vocab);
        let result = apply_regex(chunks);

        // first chunk must still be Special(100), untouched by regex
        assert!(matches!(result[0], Chunk::Special(100)));
    }

    #[test]
    fn test_regex_digits_max_three() {
        let vocab = make_test_vocab();
        let chunks = split_special_tokens("12345", &vocab);
        let result = apply_regex(chunks);

        // 12345 should split into "123" and "45" — never more than 3 digits per chunk
        for chunk in &result {
            if let Chunk::Text(s) = chunk {
                if s.chars().all(|c| c.is_ascii_digit()) {
                    assert!(s.len() <= 3, "digit chunk too long: {}", s);
                }
            }
        }
    }

    #[test]
    fn test_regex_mixed_text() {
        let vocab = make_test_vocab();
        let chunks = split_special_tokens("hello, world!", &vocab);
        let result = apply_regex(chunks);

        let texts: Vec<&str> = result
            .iter()
            .filter_map(|c| {
                if let Chunk::Text(s) = c {
                    Some(s.as_str())
                } else {
                    None
                }
            })
            .collect();

        // letters and punctuation split apart
        assert!(texts.contains(&"hello"));
        assert!(texts.contains(&" world"));
        assert!(texts.iter().any(|s| s.contains('!')));
    }

    #[test]
    fn test_regex_special_between_text() {
        let vocab = make_test_vocab();
        let chunks = split_special_tokens("hello<|begin_of_text|>don't", &vocab);
        let result = apply_regex(chunks);

        // special token stays in the right position
        let special_pos = result.iter().position(|c| matches!(c, Chunk::Special(_)));
        assert!(special_pos.is_some());

        // text before special is present
        assert!(matches!(&result[0], Chunk::Text(s) if s == "hello"));

        // contraction after special is split
        let after_special: Vec<&str> = result[special_pos.unwrap() + 1..]
            .iter()
            .filter_map(|c| {
                if let Chunk::Text(s) = c {
                    Some(s.as_str())
                } else {
                    None
                }
            })
            .collect();
        assert!(after_special.contains(&"don"));
        assert!(after_special.contains(&"'t"));
    }

    #[test]
    fn test_regex_empty_text_chunk() {
        let vocab = make_test_vocab();
        let chunks = vec![Chunk::Text(String::new())];
        let result = apply_regex(chunks);
        // empty string produces no chunks
        assert!(result.is_empty());
    }

    #[test]
    fn test_regex_whitespace_handling() {
        let vocab = make_test_vocab();
        let chunks = split_special_tokens("hello world", &vocab);
        let result = apply_regex(chunks);

        let texts: Vec<&str> = result
            .iter()
            .filter_map(|c| {
                if let Chunk::Text(s) = c {
                    Some(s.as_str())
                } else {
                    None
                }
            })
            .collect();

        // "hello" and " world" — space attaches to the following word
        assert!(texts.contains(&"hello"));
        assert!(texts.contains(&" world"));
    }

    #[test]
    fn test_byte_mapping_special_passthrough() {
        let chunks = vec![Chunk::Special(100), Chunk::Special(101)];
        let result = apply_byte_mapping_to_chunks(chunks);
        assert_eq!(result.len(), 2);
        assert!(matches!(result[0], PreToken::Special(100)));
        assert!(matches!(result[1], PreToken::Special(101)));
    }

    #[test]
    fn test_byte_mapping_space_becomes_gmark() {
        // space (byte 32) must map to Ġ — this is the core GPT2 mapping
        let chunks = vec![Chunk::Text(" hello".to_string())];
        let result = apply_byte_mapping_to_chunks(chunks);
        assert_eq!(result.len(), 1);
        if let PreToken::Text(s) = &result[0] {
            assert!(s.starts_with('Ġ'), "space should map to Ġ, got: {}", s);
        } else {
            panic!("expected PreToken::Text");
        }
    }

    #[test]
    fn test_byte_mapping_newline_becomes_cmark() {
        // newline (byte 10) must map to Ċ
        let chunks = vec![Chunk::Text("\nhello".to_string())];
        let result = apply_byte_mapping_to_chunks(chunks);
        if let PreToken::Text(s) = &result[0] {
            assert!(s.starts_with('Ċ'), "newline should map to Ċ, got: {}", s);
        } else {
            panic!("expected PreToken::Text");
        }
    }

    #[test]
    fn test_byte_mapping_printable_ascii_unchanged() {
        // printable ASCII (33-126) maps to itself
        let chunks = vec![Chunk::Text("hello!".to_string())];
        let result = apply_byte_mapping_to_chunks(chunks);
        if let PreToken::Text(s) = &result[0] {
            assert_eq!(s, "hello!");
        } else {
            panic!("expected PreToken::Text");
        }
    }

    #[test]
    fn test_byte_mapping_mixed_special_and_text() {
        let chunks = vec![
            Chunk::Special(100),
            Chunk::Text("hi".to_string()),
            Chunk::Special(101),
        ];
        let result = apply_byte_mapping_to_chunks(chunks);
        assert_eq!(result.len(), 3);
        assert!(matches!(result[0], PreToken::Special(100)));
        assert!(matches!(result[2], PreToken::Special(101)));
        if let PreToken::Text(s) = &result[1] {
            assert_eq!(s, "hi");
        } else {
            panic!("expected PreToken::Text");
        }
    }

    #[test]
    fn test_byte_mapping_empty_text() {
        let chunks = vec![Chunk::Text(String::new())];
        let result = apply_byte_mapping_to_chunks(chunks);
        assert_eq!(result.len(), 1);
        if let PreToken::Text(s) = &result[0] {
            assert!(s.is_empty());
        } else {
            panic!("expected PreToken::Text");
        }
    }

    #[test]
    fn test_byte_mapping_preserves_order() {
        let chunks = vec![
            Chunk::Text("ab".to_string()),
            Chunk::Special(100),
            Chunk::Text("cd".to_string()),
        ];
        let result = apply_byte_mapping_to_chunks(chunks);
        assert_eq!(result.len(), 3);
        assert!(matches!(result[0], PreToken::Text(_)));
        assert!(matches!(result[1], PreToken::Special(100)));
        assert!(matches!(result[2], PreToken::Text(_)));
    }

    #[test]
    fn test_pretokenize_end_to_end() {
        let vocab = make_test_vocab();
        let result = pretokenize("<|begin_of_text|>don't jump", &vocab);

        // first token is special, untouched
        assert!(matches!(result[0], PreToken::Special(100)));

        // remaining are byte-mapped text chunks — no raw spaces
        for token in &result[1..] {
            if let PreToken::Text(s) = token {
                assert!(
                    !s.contains(' '),
                    "raw space found — byte mapping failed: {}",
                    s
                );
            }
        }
    }
}
