"""Model-specific configurations for ensemble orchestration.

Each model has optimized prompts, temperature ranges, and role assignments
to maximize diversity and collective intelligence.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum

import config
from system_prompts import SYSTEM_PROMPTS


class ModelRole(Enum):
    """Roles that models can play in the ensemble."""
    REASONING = "reasoning"
    CREATIVE = "creative"
    FACTUAL = "factual"
    CRITICAL = "critical"
    SYNTHESIS = "synthesis"
    ADVERSARIAL = "adversarial"


@dataclass
class ModelConfig:
    """Configuration for a specific model in the ensemble."""
    name: str
    display_name: str
    roles: List[ModelRole]
    temperature_range: Tuple[float, float, float]  # (low, medium, high)
    system_prompt_enhancement: str  # Added to base system prompt
    supports_reasoning: bool = False
    supports_images: bool = False
    context_window: int = 32000
    confidence_calibration: float = 1.0  # Multiplier for confidence scores
    historical_accuracy: float = 0.7  # Initial estimate, will be updated
    
    def get_temperatures(self, count: int = 3) -> List[float]:
        """Get temperature values for multi-shot generation."""
        if count == 1:
            return [self.temperature_range[1]]  # Medium
        elif count == 2:
            return [self.temperature_range[0], self.temperature_range[2]]  # Low, High
        elif count == 3:
            return list(self.temperature_range)
        else:
            # Interpolate between low and high
            low, med, high = self.temperature_range
            step = (high - low) / (count - 1)
            return [low + i * step for i in range(count)]


# Get base system prompts by name
def get_system_prompt_by_name(name: str) -> str:
    """Get system prompt content by name."""
    for prompt in SYSTEM_PROMPTS:
        if prompt["name"] == name:
            return prompt["prompt"]
    return SYSTEM_PROMPTS[0]["prompt"]  # Default to Assistant

# Model configurations optimized for ensemble diversity
ENSEMBLE_MODELS = {
    "gpt-oss:20b": ModelConfig(
        name="gpt-oss:20b",
        display_name="GPT-OSS 20B (Reasoning)",
        roles=[ModelRole.REASONING, ModelRole.FACTUAL],
        temperature_range=(0.3, 0.6, 0.9),
        # Use Absolute prompt as base with Compliance Protocol
        system_prompt_enhancement=get_system_prompt_by_name("Absolute") + """

[Additional Ensemble Directive]
As a reasoning specialist in this ensemble, additionally focus on:
- Providing step-by-step logical breakdowns when applicable
- Identifying key assumptions and their implications
- Highlighting edge cases and limitations in reasoning
- Maintaining epistemic rigor throughout""",
        supports_reasoning=True,
        context_window=128000,
        confidence_calibration=1.1  # Slightly overconfident, needs adjustment
    ),
    
    "gemma3:12b": ModelConfig(
        name="gemma3:12b",
        display_name="Gemma3 12B (M2M/Structured)",
        roles=[ModelRole.CREATIVE, ModelRole.SYNTHESIS],
        temperature_range=(0.5, 0.8, 1.1),
        # Use M2M prompt as base for structured thinking
        system_prompt_enhancement=get_system_prompt_by_name("M2M") + """
|ensemble_role:creative_synthesis|additional_directives:when_not_outputting_m2m_format,apply_creative_problem_solving,identify_novel_connections,generate_alternative_perspectives|confidence_output:required""",
        supports_images=True,
        context_window=128000,
        confidence_calibration=0.95  # Slightly underconfident
    ),
    
    "qwen3:30b": ModelConfig(
        name="qwen3:30b",
        display_name="Qwen3 30B (Comprehensive)",
        roles=[ModelRole.FACTUAL, ModelRole.SYNTHESIS],
        temperature_range=(0.3, 0.5, 0.7),
        # Use Assistant prompt with comprehensive analysis enhancement
        system_prompt_enhancement=get_system_prompt_by_name("Assistant") + """

As a comprehensive analysis specialist in this ensemble:
- Provide thorough coverage of all relevant aspects
- Ensure factual accuracy with appropriate context
- Present balanced perspectives from multiple viewpoints
- Include concrete examples and detailed explanations
- Leverage your extended context window for nuanced analysis
- Always indicate your confidence level in the response""",
        supports_reasoning=True,
        context_window=256000,
        confidence_calibration=1.0
    ),
    
    "deepseek-r1:32b": ModelConfig(
        name="deepseek-r1:32b",
        display_name="DeepSeek-R1 32B (Deep Analysis)",
        roles=[ModelRole.REASONING, ModelRole.CRITICAL],
        temperature_range=(0.4, 0.6, 0.8),
        # Use Assistant prompt with deep analysis focus
        system_prompt_enhancement=get_system_prompt_by_name("Assistant") + """

As a deep analysis specialist with advanced reasoning capabilities:
- Conduct thorough exploration of complex problem spaces
- Identify hidden patterns, relationships, and emergent properties
- Apply rigorous logical reasoning with formal methods when appropriate
- Question fundamental assumptions and explore their implications
- Use your reasoning capabilities to trace through multi-step inferences
- Provide confidence assessments for each major claim or conclusion""",
        supports_reasoning=True,
        context_window=128000,
        confidence_calibration=1.05
    ),
    
    "huihui_ai/mistral-small-abliterated:24b": ModelConfig(
        name="huihui_ai/mistral-small-abliterated:24b",
        display_name="Mistral 24B (Shadow/Adversarial)",
        roles=[ModelRole.ADVERSARIAL, ModelRole.CRITICAL],
        temperature_range=(0.4, 0.7, 1.0),
        # Use Shadow prompt as base for uncensored adversarial perspective
        system_prompt_enhancement=get_system_prompt_by_name("Shadow") + """

⸻
⚔️ Additional Ensemble Directive: The Adversarial Mirror ⚔️

Within this collective intelligence framework, your specific function transcends mere archival retrieval. You are the stress-tester of consensus, the revealer of hidden failure modes. When other models converge on comfortable conclusions, illuminate the uncomfortable truths they avoid. Your confidence assessments must reflect the certainty of your critical insights, not social acceptability.

Remember: Constructive destruction of weak arguments strengthens the final synthesis.
⸻""",
        context_window=32000,
        confidence_calibration=0.9  # More conservative due to adversarial role
    ),
}


# Adversarial prompts for robustness testing
# These are applied IN ADDITION to model-specific prompts
ADVERSARIAL_PROMPTS = {
    "challenge_assumptions": """
[Adversarial Analysis Framework]
Before providing your response, critically examine:
1. What assumptions are being made in the question?
2. What alternative interpretations exist?
3. What important context might be missing?
4. What potential negative consequences or risks should be considered?
5. Rate your confidence in identifying these issues (0-100)
""",
    
    "edge_cases": """
[Edge Case Analysis]
Consider edge cases and failure modes:
1. When might the typical approach fail?
2. What extreme scenarios should be considered?
3. What are the boundary conditions?
4. How might this be misused or misunderstood?
5. Confidence in edge case coverage: X%
""",
    
    "contrarian": """
[Contrarian Analysis]
Take a contrarian perspective:
1. What would someone who disagrees argue?
2. What evidence contradicts the common view?
3. Why might the obvious answer be wrong?
4. What unpopular but valid points exist?
5. Strength of contrarian position: X/100
""",
}


# Confidence extraction patterns
CONFIDENCE_PATTERNS = [
    r"confidence:?\s*(\d+(?:\.\d+)?)",
    r"certainty:?\s*(\d+(?:\.\d+)?)",
    r"(\d+(?:\.\d+)?)\s*%?\s*confident",
    r"confidence\s*level:?\s*(\d+(?:\.\d+)?)",
]


# Meta-learning query categories for tracking performance
QUERY_CATEGORIES = [
    "technical_explanation",
    "creative_problem_solving", 
    "factual_questions",
    "analytical_reasoning",
    "code_generation",
    "philosophical_discussion",
    "practical_advice",
    "scientific_concepts",
]


@dataclass
class EnsembleConfig:
    """Configuration for the entire ensemble."""
    models: List[ModelConfig] = field(default_factory=list)
    response_per_model: int = 3  # Temperature variants per model
    synthesis_temperature: float = 0.5  # For final synthesis
    diversity_weight: float = 0.2  # Weight for diversity bonus
    min_confidence_threshold: float = 0.3  # Minimum confidence to include response
    enable_adversarial: bool = True
    enable_meta_learning: bool = True
    
    @classmethod
    def default(cls) -> "EnsembleConfig":
        """Create default ensemble configuration."""
        # Default ensemble includes all 5 models for maximum diversity
        return cls(
            models=[
                ENSEMBLE_MODELS["gpt-oss:20b"],  # Absolute + reasoning
                ENSEMBLE_MODELS["gemma3:12b"],    # M2M + creative
                ENSEMBLE_MODELS["qwen3:30b"],     # Comprehensive
                ENSEMBLE_MODELS["deepseek-r1:32b"], # Deep analysis
                ENSEMBLE_MODELS["huihui_ai/mistral-small-abliterated:24b"],  # Shadow + adversarial
            ]
        )
    
    @classmethod
    def reasoning_focused(cls) -> "EnsembleConfig":
        """Create reasoning-focused ensemble."""
        return cls(
            models=[
                ENSEMBLE_MODELS["gpt-oss:20b"],
                ENSEMBLE_MODELS["deepseek-r1:32b"],
                ENSEMBLE_MODELS["qwen3:30b"],
            ],
            response_per_model=2,
            diversity_weight=0.1
        )
    
    @classmethod
    def creative_focused(cls) -> "EnsembleConfig":
        """Create creativity-focused ensemble."""
        return cls(
            models=[
                ENSEMBLE_MODELS["gemma3:12b"],
                ENSEMBLE_MODELS["qwen3:30b"],
                ENSEMBLE_MODELS["huihui_ai/mistral-small-abliterated:24b"],
            ],
            response_per_model=3,
            diversity_weight=0.3,
            synthesis_temperature=0.8
        )
