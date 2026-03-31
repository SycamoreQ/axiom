use crate::tokenizer::bpe::{Bpe, MergeMode};
use crate::tokenizer::loader::{LoadedTokenizer, Loader, TokenizerError};
use crate::tokenizer::pretokenize::{byte_to_unicode_table, pretokenize};
use crate::tokenizer::vocab::{TokenID, Vocab};
use std::collections::HashMap;
use std::sync::Arc;

pub type Result<T> = std::result::Result<T, TokenizerError>;

pub struct EncodeOptions {
    pub add_bos: bool,
    pub add_eos: bool,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        Self {
            add_bos: false,
            add_eos: false,
        }
    }
}

pub struct Tokenizer {
    vocab: Arc<Vocab>,
    bpe: Bpe,
    decode_table: HashMap<char, u8>,
}

impl Tokenizer {
    pub fn from_file(path: &str) -> Result<Self> {
        let loader = Loader {
            file: path.to_string(),
        };
        match loader.load()? {
            LoadedTokenizer::HfVocab(vocab, merges) => {
                let vocab = Arc::new(vocab);
                let bpe = Bpe::new(Arc::clone(&vocab), merges, MergeMode::Rank);
                let decode_table = build_decode_table();
                Ok(Self {
                    vocab,
                    bpe,
                    decode_table,
                })
            }
            LoadedTokenizer::GgufVocab(vocab) => {
                let vocab = Arc::new(vocab);
                let bpe = Bpe::new(Arc::clone(&vocab), vec![], MergeMode::Score);
                let decode_table = build_decode_table();
                Ok(Self {
                    vocab,
                    bpe,
                    decode_table,
                })
            }
        }
    }

    pub fn encode(&self, text: &str, opts: EncodeOptions) -> Vec<TokenID> {
        let pretokens = pretokenize(text, &self.vocab);
        let mut ids = self.bpe.encode(pretokens);

        if opts.add_bos {
            if let Some(bos) = self.vocab.bos_id() {
                ids.insert(0, bos);
            }
        }
        if opts.add_eos {
            if let Some(eos) = self.vocab.eos_id() {
                ids.push(eos);
            }
        }
        ids
    }

    pub fn decode(&self, ids: &[TokenID]) -> String {
        let mut bytes: Vec<u8> = Vec::new();

        for &id in ids {
            // skip special tokens in decode
            if self.vocab.is_special(id) {
                continue;
            }
            if let Some(token_str) = self.vocab.id_to_token(id) {
                for ch in token_str.chars() {
                    if let Some(&byte) = self.decode_table.get(&ch) {
                        bytes.push(byte);
                    }
                }
            }
        }

        String::from_utf8_lossy(&bytes).into_owned()
    }

    pub fn vocab(&self) -> &Vocab {
        &self.vocab
    }

    pub fn bos_id(&self) -> Option<TokenID> {
        self.vocab.bos_id()
    }

    pub fn eos_id(&self) -> Option<TokenID> {
        self.vocab.eos_id()
    }
}

fn build_decode_table() -> HashMap<char, u8> {
    let forward = byte_to_unicode_table();
    let mut reverse = HashMap::with_capacity(256);
    for (byte, ch) in forward.iter().enumerate() {
        reverse.insert(*ch, byte as u8);
    }
    reverse
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::OnceLock;

    const TOKENIZER_PATH: &str = "testdata/tokenizer.json";

    static TOKENIZER: OnceLock<Tokenizer> = OnceLock::new();

    fn get_tokenizer() -> &'static Tokenizer {
        TOKENIZER
            .get_or_init(|| Tokenizer::from_file(TOKENIZER_PATH).expect("failed to load tokenizer"))
    }

    #[test]
    fn test_encode_basic() {
        let tok = get_tokenizer();
        assert_eq!(
            tok.encode("Hello world", EncodeOptions::default()),
            vec![9906, 1917]
        );
    }

    #[test]
    fn test_encode_with_bos() {
        let tok = get_tokenizer();
        let ids = tok.encode(
            "Hello world",
            EncodeOptions {
                add_bos: true,
                ..Default::default()
            },
        );
        assert_eq!(ids[0], 128000);
        assert_eq!(&ids[1..], &[9906, 1917]);
    }

    #[test]
    fn test_encode_with_eos() {
        let tok = get_tokenizer();
        let ids = tok.encode(
            "Hello world",
            EncodeOptions {
                add_eos: true,
                ..Default::default()
            },
        );
        assert_eq!(*ids.last().unwrap(), 128001);
        assert_eq!(&ids[..ids.len() - 1], &[9906, 1917]);
    }

    #[test]
    fn test_encode_with_bos_and_eos() {
        let tok = get_tokenizer();
        let ids = tok.encode(
            "Hello world",
            EncodeOptions {
                add_bos: true,
                add_eos: true,
            },
        );
        assert_eq!(ids[0], 128000);
        assert_eq!(*ids.last().unwrap(), 128001);
        assert_eq!(&ids[1..ids.len() - 1], &[9906, 1917]);
    }

    #[test]
    fn test_decode_basic() {
        let tok = get_tokenizer();
        let decoded = tok.decode(&[9906, 1917]);
        assert_eq!(decoded, "Hello world");
    }

    #[test]
    fn test_decode_skips_special_tokens() {
        let tok = get_tokenizer();
        // BOS + "Hello world" + EOS — special tokens should not appear in output
        let decoded = tok.decode(&[128000, 9906, 1917, 128001]);
        assert_eq!(decoded, "Hello world");
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let tok = get_tokenizer();
        let cases = vec![
            "Hello world",
            "don't you think?",
            "The quick brown fox jumps over the lazy dog",
            "123456789",
        ];
        for s in cases {
            let ids = tok.encode(s, EncodeOptions::default());
            let decoded = tok.decode(&ids);
            assert_eq!(decoded, s, "round-trip failed for {:?}", s);
        }
    }

    #[test]
    fn test_encode_empty() {
        let tok = get_tokenizer();
        let ids = tok.encode("", EncodeOptions::default());
        assert!(ids.is_empty());
    }

    #[test]
    fn test_decode_empty() {
        let tok = get_tokenizer();
        assert_eq!(tok.decode(&[]), "");
    }

    #[test]
    fn test_bos_eos_accessors() {
        let tok = get_tokenizer();
        assert_eq!(tok.bos_id(), Some(128000));
        assert_eq!(tok.eos_id(), Some(128001));
    }
}
