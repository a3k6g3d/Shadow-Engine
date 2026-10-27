//! The model side of Shadow (docs/SHADOW_PLAN.md, stage 0 onwards): the GGUF reader now, the tokenizer and the
//! forward pass as the stages land.  Nothing in here reads Strata's formats; the GGUF is the only input.

pub mod chat;
pub mod gguf;
pub mod half;
pub mod tokenizer;
pub mod unicode_table;
