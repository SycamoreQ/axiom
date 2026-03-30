use crate::tokenizer::bpe::*;
use crate::tokenizer::pretokenize::*;
use crate::tokenizer::vocab::{TokenID, Vocab};
use serde_json::*;
use std::collections::*;
use std::sync::*;

/*
This is the public API of the entire axiom-tokenizer
*/

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
}


