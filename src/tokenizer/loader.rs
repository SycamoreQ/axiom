use crate::tokenizer::vocab::{TokenID, Vocab};
use serde_json::*;
use std::ffi::OsStr;
use std::path::Path;

/*
has one job: read a file from disk and produce a Vocab. It knows about file formats so that nothing else has to.
*/

pub type Result<T> = std::result::Result<T, TokenizerError>;

// this is for not using unwrap() too much so that we handle the errors explicitely
//
#[derive(Debug)]
pub enum TokenizerError {
    Io(std::io::Error),
    Json(serde_json::Error),
    MissingField(&'static str),
    FormatMismatch(String),
}

impl From<std::io::Error> for TokenizerError {
    fn from(e: std::io::Error) -> Self {
        TokenizerError::Io(e)
    }
}

impl From<serde_json::Error> for TokenizerError {
    fn from(e: serde_json::Error) -> Self {
        TokenizerError::Json(e)
    }
}

impl std::fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TokenizerError::Io(e) => write!(f, "IO error: {}", e),
            TokenizerError::Json(e) => write!(f, "JSON parse error: {}", e),
            TokenizerError::MissingField(s) => write!(f, "Missing field: {}", s),
            TokenizerError::FormatMismatch(s) => write!(f, "Format mismatch: {}", s),
        }
    }
}

impl std::error::Error for TokenizerError {}

pub enum LoadedTokenizer {
    GgufVocab(Vocab),
    HfVocab(Vocab, Vec<(String, String)>),
}

pub struct Loader {
    pub file: String,
}

impl Loader {
    pub fn load(&self) -> Result<LoadedTokenizer> {
        let ext = Path::new(&self.file).extension().and_then(OsStr::to_str);

        match ext {
            Some("gguf") => self.load_from_gguf(),
            Some("json") => self.load_from_json(),
            _ => Err(TokenizerError::FormatMismatch(
                "expected .gguf or .json".into(),
            )),
        }
    }

    fn load_from_gguf(&self) -> Result<LoadedTokenizer> {
        todo!();
    }

    fn load_from_json(&self) -> Result<LoadedTokenizer> {
        let file = std::fs::File::open(&self.file)?;
        let root: Value = serde_json::from_reader(file)?;

        let vocab_obj = root["model"]["vocab"]
            .as_object()
            .ok_or(TokenizerError::MissingField("model.vocab"))?;

        let added = root["added_tokens"]
            .as_array()
            .ok_or(TokenizerError::MissingField("added_tokens"))?;

        let mut token_pairs: Vec<(TokenID, String)> = vocab_obj
            .iter()
            .map(|(k, v)| (v.as_u64().unwrap() as TokenID, k.clone()))
            .collect();
        token_pairs.sort_by_key(|(id, _)| *id);
        let tokens: Vec<String> = token_pairs.into_iter().map(|(_, s)| s).collect();

        let special_tokens: Vec<(String, TokenID)> = added
            .iter()
            .filter(|entry| entry["special"].as_bool().unwrap_or(false))
            .map(|entry| {
                let content = entry["content"].as_str().unwrap().to_string();
                let id = entry["id"].as_u64().unwrap() as TokenID;
                (content, id)
            })
            .collect();

        let find_sentinel = |content: &str| -> Option<TokenID> {
            special_tokens
                .iter()
                .find(|(s, _)| s == content)
                .map(|(_, id)| *id)
        };

        let bos_id = find_sentinel("<|begin_of_text|>");
        let eos_id = find_sentinel("<|end_of_text|>");
        let pad_id = find_sentinel("<|pad|>");
        let unk_id = find_sentinel("<unk>");

        let empty = vec![];
        let merges_raw = root["model"]["merges"].as_array().unwrap_or(&empty);

        let merges: Vec<(String, String)> = merges_raw
            .iter()
            .filter_map(|v| {
                let s = v.as_str()?;
                let mut parts = s.splitn(2, ' ');
                let a = parts.next()?.to_string();
                let b = parts.next()?.to_string();
                Some((a, b))
            })
            .collect();

        let vocab = Vocab::new(tokens, None, special_tokens, bos_id, eos_id, pad_id, unk_id);
        Ok(LoadedTokenizer::HfVocab(vocab, merges))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOKENIZER_PATH: &str = "testdata/tokenizer.json";

    #[test]
    fn test_load_hf_json_vocab_size() {
        let loader = Loader {
            file: TOKENIZER_PATH.to_string(),
        };
        let result = loader.load().expect("failed to load tokenizer");

        match result {
            LoadedTokenizer::HfVocab(vocab, _) => {
                // LLaMA 3 base vocab is 128000 tokens
                assert_eq!(vocab.size(), 128000);
            }
            _ => panic!("expected HfVocab"),
        }
    }

    #[test]
    fn test_load_hf_json_sentinels() {
        let loader = Loader {
            file: TOKENIZER_PATH.to_string(),
        };
        let result = loader.load().expect("failed to load tokenizer");

        match result {
            LoadedTokenizer::HfVocab(vocab, _) => {
                assert!(vocab.bos_id().is_some());
                assert!(vocab.eos_id().is_some());
                // verify the actual known IDs for LLamA 3
                assert_eq!(vocab.bos_id(), Some(128000));
                assert_eq!(vocab.eos_id(), Some(128001));
            }
            _ => panic!("expected HfVocab"),
        }
    }

    #[test]
    fn test_load_hf_json_known_token() {
        let loader = Loader {
            file: TOKENIZER_PATH.to_string(),
        };
        let result = loader.load().expect("failed to load tokenizer");

        match result {
            LoadedTokenizer::HfVocab(vocab, _) => {
                // "hello" should be in the vocab
                assert!(vocab.token_to_id("hello").is_some());
                // round-trip: id -> string -> id
                let id = vocab.token_to_id("hello").unwrap();
                assert_eq!(vocab.id_to_token(id), Some("hello"));
            }
            _ => panic!("expected HfVocab"),
        }
    }

    #[test]
    fn test_load_hf_json_merges() {
        let loader = Loader {
            file: TOKENIZER_PATH.to_string(),
        };
        let result = loader.load().expect("failed to load tokenizer");

        match result {
            LoadedTokenizer::HfVocab(_, merges) => {
                // LLaMA 3 has a large merge table
                assert!(!merges.is_empty());
                // each merge is a valid pair of non-empty strings
                for (a, b) in &merges {
                    assert!(!a.is_empty());
                    assert!(!b.is_empty());
                }
            }
            _ => panic!("expected HfVocab"),
        }
    }

    #[test]
    fn test_load_hf_json_special_tokens() {
        let loader = Loader {
            file: TOKENIZER_PATH.to_string(),
        };
        let result = loader.load().expect("failed to load tokenizer");

        match result {
            LoadedTokenizer::HfVocab(vocab, _) => {
                assert!(vocab.is_special(128000)); // bos
                assert!(vocab.is_special(128001)); // eos
                assert!(!vocab.is_special(1)); // regular token
            }
            _ => panic!("expected HfVocab"),
        }
    }
}
