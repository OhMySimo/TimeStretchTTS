# coding=utf-8
"""
Length Penalty LogitsProcessor for Qwen3-TTS
Prevents excessively long generation by boosting EOS token probability
when the generated sequence exceeds expected duration.
"""
import torch
from transformers import LogitsProcessor


class LengthPenaltyLogitsProcessor(LogitsProcessor):
    """
    Apply length penalty during autoregressive generation to prevent slow speech.
    
    How it works:
    - Estimates target frame count based on text length
    - Once generation exceeds target, increasingly boosts EOS token probability
    - Forces model to conclude generation at appropriate length
    
    Args:
        text_length: Number of text tokens in input
        frames_per_text_token: Expected codec frames per text token (default: 8.0)
                               Lower = expects faster speech, higher = slower speech
        penalty_alpha: Strength of penalty (default: 0.15)
                       Higher = more aggressive stopping, lower = gentler
        eos_token_id: Token ID for end-of-sequence
        start_penalty_ratio: Start applying penalty when length > target * ratio (default: 1.0)
    """
    def __init__(
        self, 
        text_length: int,
        frames_per_text_token: float = 8.0,
        penalty_alpha: float = 0.15,
        eos_token_id: int = None,
        start_penalty_ratio: float = 1.0,
    ):
        self.target_frames = int(text_length * frames_per_text_token)
        self.penalty_alpha = penalty_alpha
        self.eos_token_id = eos_token_id
        self.start_penalty_ratio = start_penalty_ratio
        self.step_count = 0
        
        # Track for debugging
        self.max_penalty_applied = 0.0
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Modify logits to apply length penalty.
        
        Args:
            input_ids: [batch_size, current_sequence_length] - not used, but required by interface
            scores: [batch_size, vocab_size] - logits before sampling
            
        Returns:
            Modified scores with EOS boost applied if over target length
        """
        self.step_count += 1
        
         # DEBUG
        if self.step_count % 10 == 0:
            print(f"[PENALTY] Step {self.step_count}/{self.target_frames}")
               
        # Only apply penalty after exceeding threshold
        threshold = self.target_frames * self.start_penalty_ratio
        if self.step_count <= threshold:
            return scores
        
        # Calculate how much we've exceeded the target
        excess_ratio = self.step_count / self.target_frames
        penalty = self.penalty_alpha * (excess_ratio - self.start_penalty_ratio)
        
        # Boost EOS token probability
        # Using exponential boost: small at first, grows rapidly
        # Factor of 10.0 is tuned to create significant probability shift
        # without completely forcing EOS (allows some flexibility)
        if self.eos_token_id is not None:
            eos_boost = penalty * 10.0
            scores[:, self.eos_token_id] += eos_boost
            self.max_penalty_applied = max(self.max_penalty_applied, eos_boost)
        
        return scores
    
    def reset(self):
        """Reset step counter for new generation."""
        self.step_count = 0
        self.max_penalty_applied = 0.0


class AdaptiveLengthPenaltyLogitsProcessor(LogitsProcessor):
    """
    Advanced version that adapts penalty based on reference audio duration.
    
    If you have access to the reference audio duration, this can provide
    more accurate length control by directly targeting that duration.
    
    Args:
        target_duration_seconds: Desired output duration in seconds
        fps: Frames per second (12.5 for 12Hz tokenizer)
        penalty_alpha: Strength of penalty
        eos_token_id: Token ID for end-of-sequence
        tolerance: Allow generation to be within ±tolerance before applying penalty (default: 0.1 = 10%)
    """
    def __init__(
        self,
        target_duration_seconds: float,
        fps: float = 12.5,
        penalty_alpha: float = 0.15,
        eos_token_id: int = None,
        tolerance: float = 0.1,
    ):
        self.target_frames = int(target_duration_seconds * fps)
        self.penalty_alpha = penalty_alpha
        self.eos_token_id = eos_token_id
        self.tolerance = tolerance
        self.step_count = 0
        self.max_penalty_applied = 0.0
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.step_count += 1
        
        # Allow some tolerance before applying penalty
        threshold = self.target_frames * (1.0 + self.tolerance)
        if self.step_count <= threshold:
            return scores
        
        # Calculate excess
        excess_ratio = (self.step_count - threshold) / self.target_frames
        penalty = self.penalty_alpha * excess_ratio
        
        if self.eos_token_id is not None:
            eos_boost = penalty * 10.0
            scores[:, self.eos_token_id] += eos_boost
            self.max_penalty_applied = max(self.max_penalty_applied, eos_boost)
        
        return scores
    
    def reset(self):
        self.step_count = 0
        self.max_penalty_applied = 0.0
