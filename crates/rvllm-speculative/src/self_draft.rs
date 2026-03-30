//! Self-draft model: uses the first N layers of the target model as the draft.
//!
//! Shares weights with the target model -- no separate model loading required.
//! The partial forward (embedding -> N layers -> norm -> LM head) produces
//! approximate token predictions at a fraction of the full model's cost.

use std::cell::RefCell;

use rvllm_core::prelude::{Result, TokenId};

use crate::draft::{DraftModel, DraftToken};

/// Callback type for running a partial forward pass through the target model.
/// Takes (token_ids, num_layers) and returns logits [num_tokens * vocab_size].
pub type PartialForwardFn = Box<dyn FnMut(&[TokenId], usize) -> Result<Vec<f32>> + Send>;

/// Self-draft model that runs the first N layers of the target model.
pub struct SelfDraftModel {
    num_draft_layers: usize,
    vocab_size: usize,
    partial_forward: RefCell<PartialForwardFn>,
}

impl SelfDraftModel {
    pub fn new(
        num_draft_layers: usize,
        vocab_size: usize,
        partial_forward: PartialForwardFn,
    ) -> Self {
        Self {
            num_draft_layers,
            vocab_size,
            partial_forward: RefCell::new(partial_forward),
        }
    }

    /// Greedy argmax from a logit slice.
    fn argmax(logits: &[f32]) -> u32 {
        let mut best_idx = 0u32;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_idx = i as u32;
            }
        }
        best_idx
    }

    /// Softmax in-place.
    fn softmax(logits: &mut [f32]) {
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in logits.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        if sum > 0.0 {
            for v in logits.iter_mut() {
                *v /= sum;
            }
        }
    }
}

impl DraftModel for SelfDraftModel {
    fn generate(&self, input_tokens: &[TokenId], num_tokens: usize) -> Result<Vec<DraftToken>> {
        if input_tokens.is_empty() || num_tokens == 0 {
            return Ok(Vec::new());
        }

        let mut forward = self.partial_forward.borrow_mut();
        let mut drafts = Vec::with_capacity(num_tokens);
        let mut context: Vec<TokenId> = input_tokens.to_vec();

        for _ in 0..num_tokens {
            let logits = forward(&context, self.num_draft_layers)?;

            let vs = self.vocab_size;
            let num_ctx = context.len();
            let offset = (num_ctx - 1) * vs;
            if logits.len() < offset + vs {
                return Err(rvllm_core::prelude::LLMError::ModelError(format!(
                    "self-draft: logits too short ({}) for context len {} vocab {}",
                    logits.len(),
                    num_ctx,
                    vs
                )));
            }

            let mut token_logits = logits[offset..offset + vs].to_vec();
            let selected = Self::argmax(&token_logits);
            let logprob = token_logits[selected as usize];
            Self::softmax(&mut token_logits);

            drafts.push(DraftToken {
                token_id: selected,
                logprob: logprob.ln().max(-100.0),
                draft_probs: token_logits,
            });

            context.push(selected);
        }

        Ok(drafts)
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn self_draft_generates_tokens() {
        let vocab = 100;
        let partial_fn: PartialForwardFn = Box::new(move |tokens: &[TokenId], _layers: usize| {
            let num_tokens = tokens.len();
            let mut logits = vec![0.0f32; num_tokens * vocab];
            for t in 0..num_tokens {
                let last = tokens[t] as usize;
                let pick = (last + 1) % vocab;
                logits[t * vocab + pick] = 10.0;
            }
            Ok(logits)
        });

        let model = SelfDraftModel::new(7, vocab, partial_fn);
        let drafts = model.generate(&[5], 3).unwrap();
        assert_eq!(drafts.len(), 3);
        assert_eq!(drafts[0].token_id, 6);
        assert_eq!(drafts[1].token_id, 7);
        assert_eq!(drafts[2].token_id, 8);
        for d in &drafts {
            assert_eq!(d.draft_probs.len(), vocab);
            let sum: f32 = d.draft_probs.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn self_draft_empty_input() {
        let partial_fn: PartialForwardFn = Box::new(|_, _| Ok(Vec::new()));
        let model = SelfDraftModel::new(7, 100, partial_fn);
        let drafts = model.generate(&[], 3).unwrap();
        assert!(drafts.is_empty());
    }

    #[test]
    fn self_draft_zero_tokens() {
        let partial_fn: PartialForwardFn = Box::new(|_, _| Ok(Vec::new()));
        let model = SelfDraftModel::new(7, 100, partial_fn);
        let drafts = model.generate(&[1, 2], 0).unwrap();
        assert!(drafts.is_empty());
    }
}
