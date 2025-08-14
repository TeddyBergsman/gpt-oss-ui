"""Ensemble orchestrator for managing multiple model workers and synthesis.

This module implements the core ensemble logic including:
- Multi-model coordination
- Confidence scoring
- Bayesian aggregation
- Diversity metrics
- Performance tracking
"""

from __future__ import annotations

import re
import json
import time
import uuid
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Callable
from collections import defaultdict
from PySide6 import QtCore
import ollama

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from model_configs import (
    ModelConfig, EnsembleConfig, CONFIDENCE_PATTERNS, 
    ADVERSARIAL_PROMPTS, QUERY_CATEGORIES
)
from core.performance_tracker import PerformanceTracker, QueryPerformance


@dataclass
class ModelResponse:
    """Individual response from a model."""
    model_name: str
    content: str
    thinking: Optional[str]
    temperature: float
    confidence: float
    response_time: float
    token_count: int
    diversity_score: float = 0.0
    role_alignment: float = 1.0  # How well response aligns with model's role


@dataclass
class EnsembleResponse:
    """Aggregated response from the ensemble."""
    final_content: str
    final_thinking: Optional[str]
    model_responses: List[ModelResponse]
    confidence_scores: Dict[str, float]
    diversity_metrics: Dict[str, float]
    aggregation_method: str
    total_time: float
    query_category: Optional[str] = None
    consensus_level: float = 0.0
    synthesis_token_count: int = 0
    synthesis_reasoning_time: float = 0.0
    synthesis_response_time: float = 0.0


@dataclass 
class PerformanceMetrics:
    """Track model performance over time."""
    model_accuracies: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    model_response_times: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    category_performance: Dict[str, Dict[str, float]] = field(default_factory=lambda: defaultdict(dict))
    ensemble_combinations: Dict[str, float] = field(default_factory=dict)  # Track which combos work best
    total_queries: int = 0
    
    def update_model_performance(self, model: str, accuracy: float, response_time: float, category: Optional[str] = None):
        """Update performance metrics for a model."""
        # Rolling average for accuracy
        alpha = 0.1  # Learning rate
        self.model_accuracies[model] = (1 - alpha) * self.model_accuracies.get(model, 0.7) + alpha * accuracy
        
        # Track response times
        self.model_response_times[model].append(response_time)
        if len(self.model_response_times[model]) > 100:
            self.model_response_times[model] = self.model_response_times[model][-100:]
        
        # Category-specific performance
        if category:
            if category not in self.category_performance[model]:
                self.category_performance[model][category] = 0.7
            self.category_performance[model][category] = (
                (1 - alpha) * self.category_performance[model][category] + alpha * accuracy
            )
    
    def get_model_weight(self, model: str, category: Optional[str] = None) -> float:
        """Get dynamic weight for a model based on performance."""
        if category and category in self.category_performance.get(model, {}):
            return self.category_performance[model][category]
        return self.model_accuracies.get(model, 0.7)


class ModelWorker(QtCore.QObject):
    """Worker for individual model in the ensemble."""
    token = QtCore.Signal(str)
    thinking = QtCore.Signal(str)
    finished = QtCore.Signal(str, str, float, int)  # content, thinking, response_time, token_count
    error = QtCore.Signal(str)
    
    def __init__(self, model_config: ModelConfig, messages: list, options: dict, 
                 response_id: int, adversarial_prompt: Optional[str] = None,
                 model_specific_settings: Optional[dict] = None):
        super().__init__()
        self.model_config = model_config
        self.messages = messages.copy()
        self.options = options
        self.response_id = response_id
        self.adversarial_prompt = adversarial_prompt
        self.model_specific_settings = model_specific_settings or {}
        self._stop_requested = False
        self._content_accumulator = ""
        self._thinking_accumulator = ""
        self._start_time = None
        self._token_count = 0
        
    def request_stop(self):
        """Request the worker to stop."""
        self._stop_requested = True
        
    @QtCore.Slot()
    def run(self):
        """Run the model and emit results."""
        try:
            self._start_time = time.time()
            
            # Build messages with model-specific enhancements
            messages = self.messages.copy()
            
            # Set system prompt with model-specific enhancement
            if messages and messages[0]["role"] == "system":
                # Use the model's system prompt enhancement (which already includes base prompt)
                enhanced_prompt = self.model_config.system_prompt_enhancement
                
                # Add adversarial prompt if provided
                if self.adversarial_prompt:
                    enhanced_prompt += "\n\n" + self.adversarial_prompt
                    
                # Add confidence instruction
                enhanced_prompt += "\n\nAt the end of your response, indicate your confidence level (0-100) in your answer."
                
                messages[0] = {"role": "system", "content": enhanced_prompt}
            
            # Apply model-specific settings
            options = self.options.copy()
            
            # Handle compliance protocol for models that support it
            if self.model_specific_settings.get("compliance_protocol") and "gpt-oss" in self.model_config.name:
                # Add compliance message if needed
                if messages[-1]["role"] == "user":
                    # Load compliance prompt
                    import os
                    compliance_path = os.path.join(os.path.dirname(__file__), "..", "compliance_prompt.txt")
                    with open(compliance_path, "r") as f:
                        compliance_prompt = f.read()
                    original_content = messages[-1]["content"]
                    messages[-1]["content"] = compliance_prompt.format(user_input=original_content)
            
            # Handle reasoning effort
            if self.model_specific_settings.get("reasoning_effort") and self.model_config.supports_reasoning:
                options["reasoning_effort"] = self.model_specific_settings["reasoning_effort"]
            
            # Stream response
            stream = ollama.chat(
                model=self.model_config.name,
                messages=messages,
                options=options,
                stream=True
            )
            
            for chunk in stream:
                if self._stop_requested:
                    break
                    
                if "message" in chunk:
                    message = chunk["message"]
                    
                    # Handle content streaming
                    if "content" in message and message["content"]:
                        token = message["content"]
                        self._content_accumulator += token
                        self._token_count += 1
                        self.token.emit(token)
                    
                    # Handle thinking streaming (for reasoning models)
                    # Normalize across different providers/keys
                    thinking_content = (
                        message.get("thinking")
                        or message.get("reasoning")
                        or message.get("thinking_content")
                        or message.get("reasoning_content")
                    )
                    if thinking_content:
                        self._thinking_accumulator += thinking_content
                        self.thinking.emit(thinking_content)
            
            response_time = time.time() - self._start_time
            self.finished.emit(self._content_accumulator, self._thinking_accumulator, 
                             response_time, self._token_count)
            
        except Exception as e:
            self.error.emit(str(e))


class EnsembleOrchestrator(QtCore.QObject):
    """Orchestrates multiple model workers for ensemble responses."""
    
    # Signals for individual model responses
    model_token = QtCore.Signal(str, int, str)  # model_name, response_id, token
    model_thinking = QtCore.Signal(str, int, str)  # model_name, response_id, thinking
    model_finished = QtCore.Signal(str, int, ModelResponse)  # model_name, response_id, response
    
    # Signals for ensemble synthesis
    synthesis_started = QtCore.Signal()
    synthesis_token = QtCore.Signal(str)
    synthesis_thinking = QtCore.Signal(str) 
    synthesis_finished = QtCore.Signal(EnsembleResponse)
    
    # Progress and status
    progress = QtCore.Signal(int, int)  # completed, total
    status = QtCore.Signal(str)  # status message
    error = QtCore.Signal(str)
    
    def __init__(self, ensemble_config: EnsembleConfig, messages: list, base_options: dict,
                 performance_metrics: Optional[PerformanceMetrics] = None):
        super().__init__()
        self.ensemble_config = ensemble_config
        self.messages = messages
        self.base_options = base_options
        self.performance_metrics = performance_metrics or PerformanceMetrics()
        
        self.model_responses: Dict[str, List[ModelResponse]] = defaultdict(list)
        self.workers: List[Tuple[ModelWorker, QtCore.QThread]] = []
        self.completed_count = 0
        self.total_responses = len(ensemble_config.models) * ensemble_config.response_per_model
        self._start_time = None
        
    def start(self):
        """Start the ensemble generation process."""
        self._start_time = time.time()
        self.status.emit("Starting ensemble generation...")
        
        # Detect query category for meta-learning
        query_category = self._detect_query_category(self.messages[-1]["content"])
        
        # Prepare all workers but don't start them yet
        response_id = 0
        for model_config in self.ensemble_config.models:
            temperatures = model_config.get_temperatures(self.ensemble_config.response_per_model)
            
            for i, temperature in enumerate(temperatures):
                # Prepare options with temperature
                options = self.base_options.copy()
                options["temperature"] = temperature
                
                # Add adversarial prompt for designated models
                adversarial_prompt = None
                if (self.ensemble_config.enable_adversarial and 
                    "adversarial" in [r.value for r in model_config.roles]):
                    # Rotate through adversarial prompts
                    prompt_keys = list(ADVERSARIAL_PROMPTS.keys())
                    adversarial_prompt = ADVERSARIAL_PROMPTS[prompt_keys[i % len(prompt_keys)]]
                
                # Prepare model-specific settings
                model_specific_settings = {}
                
                # GPT-OSS gets compliance protocol and high reasoning
                if model_config.name == "gpt-oss:20b":
                    model_specific_settings["compliance_protocol"] = True
                    model_specific_settings["reasoning_effort"] = "high"
                
                # Other models with reasoning get medium reasoning by default
                elif model_config.supports_reasoning:
                    model_specific_settings["reasoning_effort"] = "medium"
                
                # Create worker
                worker = ModelWorker(
                    model_config, self.messages, options, response_id, 
                    adversarial_prompt, model_specific_settings
                )
                thread = QtCore.QThread()
                worker.moveToThread(thread)
                
                # Connect signals - use default parameter trick to capture values
                thread.started.connect(worker.run)
                
                # Create closures with proper value capture
                def make_token_handler(model_name, resp_id):
                    return lambda token: self.model_token.emit(model_name, resp_id, token)
                
                def make_thinking_handler(model_name, resp_id):
                    return lambda token: self.model_thinking.emit(model_name, resp_id, token)
                
                def make_finished_handler(model_cfg, resp_id, temp):
                    return lambda content, thinking, resp_time, tok_count: \
                        self._on_model_finished(model_cfg, resp_id, content, thinking, resp_time, tok_count, temp)
                
                worker.token.connect(make_token_handler(model_config.name, response_id))
                worker.thinking.connect(make_thinking_handler(model_config.name, response_id))
                worker.finished.connect(make_finished_handler(model_config, response_id, temperature))
                worker.error.connect(self._on_model_error)
                worker.finished.connect(worker.deleteLater)
                thread.finished.connect(thread.deleteLater)
                
                self.workers.append((worker, thread))
                response_id += 1
        
        self.progress.emit(0, self.total_responses)
        
        # Start the first worker
        if self.workers:
            self.workers[0][1].start()
    
    def _detect_query_category(self, query: str) -> Optional[str]:
        """Detect the category of the query for meta-learning."""
        query_lower = query.lower()
        
        # Simple keyword-based categorization (can be enhanced with ML)
        if any(word in query_lower for word in ["explain", "how does", "what is", "why"]):
            return "technical_explanation"
        elif any(word in query_lower for word in ["create", "design", "imagine", "innovative"]):
            return "creative_problem_solving"
        elif any(word in query_lower for word in ["fact", "data", "statistic", "when did"]):
            return "factual_questions"
        elif any(word in query_lower for word in ["analyze", "compare", "evaluate", "reason"]):
            return "analytical_reasoning"
        elif any(word in query_lower for word in ["code", "program", "function", "implement"]):
            return "code_generation"
        elif any(word in query_lower for word in ["meaning", "purpose", "ethics", "philosophy"]):
            return "philosophical_discussion"
        elif any(word in query_lower for word in ["advice", "should i", "recommend", "suggest"]):
            return "practical_advice"
        elif any(word in query_lower for word in ["science", "physics", "chemistry", "biology"]):
            return "scientific_concepts"
        
        return None
    
    def _on_model_finished(self, model_config: ModelConfig, response_id: int, 
                          content: str, thinking: str, response_time: float, 
                          token_count: int, temperature: float):
        """Handle completion of a model response."""
        # Extract confidence score
        confidence = self._extract_confidence(content)
        if confidence is None:
            confidence = 0.7  # Default confidence
        
        # Apply model-specific calibration
        confidence *= model_config.confidence_calibration
        confidence = min(1.0, confidence)  # Cap at 1.0
        
        # Calculate initial diversity score (will be updated after all responses)
        diversity_score = 0.0
        
        # Create response object
        response = ModelResponse(
            model_name=model_config.name,
            content=content,
            thinking=thinking,
            temperature=temperature,
            confidence=confidence,
            response_time=response_time,
            token_count=token_count,
            diversity_score=diversity_score
        )
        
        self.model_responses[model_config.name].append(response)
        self.model_finished.emit(model_config.name, response_id, response)
        
        self.completed_count += 1
        self.progress.emit(self.completed_count, self.total_responses)
        
        # Start the next worker in sequence
        if self.completed_count < self.total_responses:
            next_worker_idx = self.completed_count
            if next_worker_idx < len(self.workers):
                self.workers[next_worker_idx][1].start()
        else:
            self._all_models_complete()
    
    def _extract_confidence(self, content: str) -> Optional[float]:
        """Extract confidence score from model output."""
        for pattern in CONFIDENCE_PATTERNS:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                confidence = float(match.group(1))
                # Normalize to 0-1 range if percentage
                if confidence > 1:
                    confidence /= 100
                return confidence
        return None
    
    def _on_model_error(self, error_msg: str):
        """Handle model error."""
        self.error.emit(f"Model error: {error_msg}")
        self.completed_count += 1
        self.progress.emit(self.completed_count, self.total_responses)
        
        if self.completed_count == self.total_responses:
            self._all_models_complete()
    
    def _all_models_complete(self):
        """All models have completed - start synthesis."""
        self.status.emit("All models complete. Starting synthesis...")
        
        # Calculate diversity scores
        self._calculate_diversity_scores()
        
        # Perform synthesis
        self._synthesize_responses()
    
    def _calculate_diversity_scores(self):
        """Calculate diversity scores for all responses."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Collect all responses
        all_responses = []
        response_map = {}
        
        for model_name, responses in self.model_responses.items():
            for i, response in enumerate(responses):
                all_responses.append(response.content)
                response_map[len(all_responses) - 1] = (model_name, i)
        
        if len(all_responses) < 2:
            return
        
        # Calculate TF-IDF vectors
        try:
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(all_responses)
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Calculate diversity score for each response
            for idx, (model_name, response_idx) in response_map.items():
                # Diversity = 1 - average similarity to other responses
                similarities = similarity_matrix[idx]
                avg_similarity = (similarities.sum() - 1) / (len(similarities) - 1)  # Exclude self
                diversity_score = 1 - avg_similarity
                
                self.model_responses[model_name][response_idx].diversity_score = diversity_score
                
        except Exception as e:
            # If diversity calculation fails, continue with default scores
            self.error.emit(f"Diversity calculation warning: {e}")
    
    def _synthesize_responses(self):
        """Synthesize all responses using enhanced Bayesian synthesis."""
        self.synthesis_started.emit()
        
        # Get query category for performance tracking
        query_category = self._detect_query_category(self.messages[-1]["content"])
        
        # Prepare weighted responses using Bayesian Model Averaging
        weighted_responses = self._prepare_weighted_responses(query_category)
        
        # Create synthesis worker
        from core.ensemble_synthesis import EnsembleSynthesizer
        self.synthesizer = EnsembleSynthesizer()
        self.synthesis_thread_internal = QtCore.QThread()
        self.synthesizer.moveToThread(self.synthesis_thread_internal)
        
        # Connect synthesis signals
        self.synthesis_thread_internal.started.connect(
            lambda: self.synthesizer.synthesize_bayesian(
                weighted_responses, 
                self.messages[-1]["content"],
                self.ensemble_config
            )
        )
        self.synthesizer.token.connect(self.synthesis_token.emit)
        self.synthesizer.thinking.connect(self.synthesis_thinking.emit)
        self.synthesizer.finished.connect(lambda result: self._on_synthesis_result(result, query_category))
        self.synthesizer.error.connect(self.error.emit)
        
        # Cleanup
        self.synthesizer.finished.connect(self.synthesizer.deleteLater)
        self.synthesis_thread_internal.finished.connect(self.synthesis_thread_internal.deleteLater)
        
        # Start synthesis
        self.synthesis_thread_internal.start()
    
    def _prepare_weighted_responses(self, query_category: Optional[str]) -> List[Tuple[ModelResponse, float]]:
        """Prepare weighted responses for synthesis."""
        weighted_responses = []
        
        for model_name, responses in self.model_responses.items():
            model_config = next(m for m in self.ensemble_config.models if m.name == model_name)
            
            # Synthesize within-model responses first
            if len(responses) > 1:
                synthesized = self._synthesize_model_responses(responses, model_config)
                responses = [synthesized]
            
            for response in responses:
                # Calculate weight based on:
                # 1. Model's historical performance
                # 2. Response confidence
                # 3. Diversity bonus
                base_weight = self.performance_metrics.get_model_weight(model_name, query_category)
                confidence_weight = response.confidence
                diversity_bonus = response.diversity_score * self.ensemble_config.diversity_weight
                
                total_weight = base_weight * confidence_weight * (1 + diversity_bonus)
                
                # Apply minimum confidence threshold
                if response.confidence >= self.ensemble_config.min_confidence_threshold:
                    weighted_responses.append((response, total_weight))
        
        # Normalize weights
        if weighted_responses:
            total_weight = sum(w for _, w in weighted_responses)
            weighted_responses = [(r, w/total_weight) for r, w in weighted_responses]
        
        return weighted_responses
    
    def _on_synthesis_result(self, synthesis_result, query_category: Optional[str]):
        """Handle synthesis result from the enhanced synthesizer."""
        from core.ensemble_synthesis import SynthesisResult
        
        # Build ensemble response
        ensemble_response = EnsembleResponse(
            final_content=synthesis_result.content,
            final_thinking=synthesis_result.thinking,
            model_responses=[r for r, _ in self._prepare_weighted_responses(query_category)],
            confidence_scores={m: r[0].confidence for m, r in self.model_responses.items() if r},
            diversity_metrics={
                "unique_insights": len(synthesis_result.key_insights),
                "conflicts": len(synthesis_result.conflicts),
                "uncertainty_areas": len(synthesis_result.uncertainty_areas)
            },
            aggregation_method="enhanced_bayesian_synthesis",
            total_time=time.time() - self._start_time,
            query_category=query_category,
            consensus_level=synthesis_result.consensus_level,
            synthesis_token_count=synthesis_result.token_count,
            synthesis_reasoning_time=synthesis_result.reasoning_time,
            synthesis_response_time=synthesis_result.response_time
        )
        
        # Update performance metrics
        self.performance_metrics.total_queries += 1
        
        # Track query performance
        query_id = str(uuid.uuid4())
        
        query_perf = QueryPerformance(
            query_id=query_id,
            timestamp=time.time(),
            query_text=self.messages[-1]["content"],
            query_category=query_category,
            model_responses={m: r[0].confidence for m, r in self.model_responses.items() if r},
            ensemble_confidence=synthesis_result.consensus_level,
            response_time=ensemble_response.total_time
        )
        
        # Save performance data
        perf_tracker = PerformanceTracker()
        perf_tracker.record_query(query_perf)
        
        # Emit final response
        self.synthesis_finished.emit(ensemble_response)
    
    def _bayesian_model_averaging(self, query_category: Optional[str]) -> EnsembleResponse:
        """Perform Bayesian Model Averaging on responses."""
        # Collect all responses with weights
        weighted_responses = []
        confidence_scores = {}
        diversity_metrics = {
            "average_diversity": 0.0,
            "diversity_bonus_applied": 0.0,
            "unique_insights": 0
        }
        
        for model_name, responses in self.model_responses.items():
            model_config = next(m for m in self.ensemble_config.models if m.name == model_name)
            
            # Synthesize within-model responses first
            if len(responses) > 1:
                synthesized = self._synthesize_model_responses(responses, model_config)
                responses = [synthesized]
            
            for response in responses:
                # Calculate weight based on:
                # 1. Model's historical performance
                # 2. Response confidence
                # 3. Diversity bonus
                base_weight = self.performance_metrics.get_model_weight(model_name, query_category)
                confidence_weight = response.confidence
                diversity_bonus = response.diversity_score * self.ensemble_config.diversity_weight
                
                total_weight = base_weight * confidence_weight * (1 + diversity_bonus)
                
                # Apply minimum confidence threshold
                if response.confidence >= self.ensemble_config.min_confidence_threshold:
                    weighted_responses.append((response, total_weight))
                    confidence_scores[model_name] = response.confidence
                    diversity_metrics["average_diversity"] += response.diversity_score
                    diversity_metrics["diversity_bonus_applied"] += diversity_bonus
        
        if not weighted_responses:
            # Fallback if all responses below threshold
            weighted_responses = [(r, 1.0) for responses in self.model_responses.values() 
                                 for r in responses]
        
        # Normalize weights
        total_weight = sum(w for _, w in weighted_responses)
        weighted_responses = [(r, w/total_weight) for r, w in weighted_responses]
        
        # Create synthesis prompt
        synthesis_content = self._create_synthesis_prompt(weighted_responses)
        
        # Calculate diversity metrics
        diversity_metrics["average_diversity"] /= len(weighted_responses)
        diversity_metrics["unique_insights"] = sum(1 for r, _ in weighted_responses 
                                                  if r.diversity_score > 0.7)
        
        # For now, use simple weighted combination (can be enhanced)
        # In production, this would use a synthesis model
        final_content = self._weighted_synthesis(weighted_responses)
        final_thinking = self._combine_thinking(weighted_responses)
        
        return EnsembleResponse(
            final_content=final_content,
            final_thinking=final_thinking,
            model_responses=[r for r, _ in weighted_responses],
            confidence_scores=confidence_scores,
            diversity_metrics=diversity_metrics,
            aggregation_method="bayesian_model_averaging",
            total_time=0.0  # Will be set by caller
        )
    
    def _synthesize_model_responses(self, responses: List[ModelResponse], 
                                   model_config: ModelConfig) -> ModelResponse:
        """Synthesize multiple responses from the same model."""
        # Simple average for now - can be enhanced
        avg_confidence = sum(r.confidence for r in responses) / len(responses)
        avg_diversity = sum(r.diversity_score for r in responses) / len(responses)
        
        # Combine content - for now just concatenate unique insights
        combined_content = responses[0].content
        combined_thinking = responses[0].thinking or ""
        
        return ModelResponse(
            model_name=model_config.name,
            content=combined_content,
            thinking=combined_thinking,
            temperature=0.5,  # Average temperature
            confidence=avg_confidence,
            response_time=sum(r.response_time for r in responses) / len(responses),
            token_count=sum(r.token_count for r in responses),
            diversity_score=avg_diversity
        )
    
    def _create_synthesis_prompt(self, weighted_responses: List[Tuple[ModelResponse, float]]) -> str:
        """Create prompt for final synthesis."""
        synthesis_prompt = f"Original question: {self.messages[-1]['content']}\n\n"
        synthesis_prompt += "Weighted model responses:\n\n"
        
        for response, weight in weighted_responses:
            model_config = next(m for m in self.ensemble_config.models if m.name == response.model_name)
            synthesis_prompt += f"=== {model_config.display_name} (Weight: {weight:.2f}, Confidence: {response.confidence:.2f}) ===\n"
            synthesis_prompt += response.content + "\n\n"
        
        synthesis_prompt += """
Please synthesize these responses using Bayesian Model Averaging principles:
1. Give more weight to high-confidence, high-weight responses
2. Include unique insights from high-diversity responses
3. Resolve contradictions by favoring well-supported positions
4. Acknowledge uncertainty where models disagree significantly
5. Provide a balanced, comprehensive answer
"""
        
        return synthesis_prompt
    
    def _weighted_synthesis(self, weighted_responses: List[Tuple[ModelResponse, float]]) -> str:
        """Perform weighted synthesis of responses."""
        # For now, return the highest weighted response
        # In production, this would use a proper synthesis model
        best_response = max(weighted_responses, key=lambda x: x[1])
        
        # Add ensemble metadata
        synthesis = f"{best_response[0].content}\n\n"
        synthesis += f"[Ensemble: {len(self.ensemble_config.models)} models, "
        synthesis += f"Confidence: {best_response[0].confidence:.2f}, "
        synthesis += f"Diversity: {best_response[0].diversity_score:.2f}]"
        
        return synthesis
    
    def _combine_thinking(self, weighted_responses: List[Tuple[ModelResponse, float]]) -> Optional[str]:
        """Combine thinking from all responses."""
        thinking_parts = []
        
        for response, weight in weighted_responses:
            if response.thinking:
                model_config = next(m for m in self.ensemble_config.models if m.name == response.model_name)
                thinking_parts.append(f"[{model_config.display_name}]:\n{response.thinking}")
        
        if thinking_parts:
            return "\n\n".join(thinking_parts)
        return None
    
    def stop(self):
        """Stop all workers."""
        for worker, thread in self.workers:
            worker.request_stop()
            thread.quit()
            thread.wait(100)
