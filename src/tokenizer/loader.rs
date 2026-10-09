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
        let gguf = crate::weights::gguf::GgufFile::from_file(Path::new(&self.file))
            .map_err(|e| TokenizerError::FormatMismatch(e.to_string()))?;

        let tokens_val = gguf
            .metadata
            .get("tokenizer.ggml.tokens")
            .ok_or(TokenizerError::MissingField("tokenizer.ggml.tokens"))?;

        let tokens: Vec<String> = match tokens_val {
            crate::weights::gguf::GgufValue::Array(arr) => arr
                .iter()
                .filter_map(|v| match v {
                    crate::weights::gguf::GgufValue::String(s) => Some(s.clone()),
                    _ => None,
                })
                .collect(),
            _ => {
                return Err(TokenizerError::FormatMismatch(
                    "tokenizer.ggml.tokens must be array".into(),
                ))
            }
        };

        let scores: Option<Vec<f32>> = gguf.metadata.get("tokenizer.ggml.scores").and_then(|v| {
            if let crate::weights::gguf::GgufValue::Array(arr) = v {
                Some(
                    arr.iter()
                        .filter_map(|x| match x {
                            crate::weights::gguf::GgufValue::Float32(f) => Some(*f),
                            _ => None,
                        })
                        .collect(),
                )
            } else {
                None
            }
        });

        let mut special_tokens = Vec::new();
        if let Some(crate::weights::gguf::GgufValue::Array(types)) =
            gguf.metadata.get("tokenizer.ggml.token_type")
        {
            for (idx, t) in types.iter().enumerate() {
                let is_control = match t {
                    crate::weights::gguf::GgufValue::Int32(val) => *val == 3,
                    crate::weights::gguf::GgufValue::Uint32(val) => *val == 3,
                    _ => false,
                };
                if is_control && idx < tokens.len() {
                    special_tokens.push((tokens[idx].clone(), idx));
                }
            }
        }

        let get_u32 = |key: &str| -> Option<usize> {
            gguf.metadata.get(key).and_then(|v| match v {
                crate::weights::gguf::GgufValue::Uint32(n) => Some(*n as usize),
                crate::weights::gguf::GgufValue::Int32(n) => Some(*n as usize),
                crate::weights::gguf::GgufValue::Uint64(n) => Some(*n as usize),
                crate::weights::gguf::GgufValue::Int64(n) => Some(*n as usize),
                _ => None,
            })
        };

        let bos_id = get_u32("tokenizer.ggml.bos_token_id");
        let eos_id = get_u32("tokenizer.ggml.eos_token_id");
        let pad_id = get_u32("tokenizer.ggml.padding_token_id");
        let unk_id = get_u32("tokenizer.ggml.unknown_token_id");

        if let Some(bos) = bos_id {
            if bos < tokens.len() && !special_tokens.iter().any(|(_, id)| *id == bos) {
                special_tokens.push((tokens[bos].clone(), bos));
            }
        }
        if let Some(eos) = eos_id {
            if eos < tokens.len() && !special_tokens.iter().any(|(_, id)| *id == eos) {
                special_tokens.push((tokens[eos].clone(), eos));
            }
        }

        let vocab = Vocab::new(
            tokens,
            scores,
            special_tokens,
            bos_id,
            eos_id,
            pad_id,
            unk_id,
        );

        if let Some(crate::weights::gguf::GgufValue::Array(merges_arr)) =
            gguf.metadata.get("tokenizer.ggml.merges")
        {
            let mut merges = Vec::new();
            for m in merges_arr {
                if let crate::weights::gguf::GgufValue::String(s) = m {
                    let parts: Vec<&str> = s.split_whitespace().collect();
                    if parts.len() == 2 {
                        merges.push((parts[0].to_string(), parts[1].to_string()));
                    }
                }
            }
            if !merges.is_empty() {
                return Ok(LoadedTokenizer::HfVocab(vocab, merges));
            }
        }

        Ok(LoadedTokenizer::GgufVocab(vocab))
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

        let find_sentinel = |candidates: &[&str]| -> Option<TokenID> {
            for &c in candidates {
                if let Some((_, id)) = special_tokens.iter().find(|(s, _)| s == c) {
                    return Some(*id);
                }
            }
            None
        };

        let bos_id = find_sentinel(&["<|begin_of_text|>", "<s>", "<|im_start|>"]);
        let eos_id = find_sentinel(&["<|end_of_text|>", "<|endoftext|>", "</s>", "<|im_end|>"]);
        let pad_id = find_sentinel(&["<|pad|>", "<pad>"]);
        let unk_id = find_sentinel(&["<unk>"]);

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
                // LLaMA 3 base vocab is 128000, Qwen is 151643
                assert!(vocab.size() == 128000 || vocab.size() == 151643);
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
                if let Some(bos) = vocab.bos_id() {
                    assert!(vocab.is_special(bos));
                }
                if let Some(eos) = vocab.eos_id() {
                    assert!(vocab.is_special(eos));
                }
                assert!(!vocab.is_special(1)); // regular token
            }
            _ => panic!("expected HfVocab"),
        }
    }
}
