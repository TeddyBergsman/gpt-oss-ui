"""Enhanced synthesis system for ensemble responses.

This module implements sophisticated synthesis strategies including:
- Bayesian Model Averaging
- Diversity-aware aggregation  
- Conflict resolution
- Uncertainty quantification
"""

from __future__ import annotations

import re
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Set
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PySide6 import QtCore
import ollama

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from model_configs import ModelConfig, EnsembleConfig


@dataclass
class SynthesisResult:
    """Result of synthesis operation."""
    content: str
    thinking: Optional[str]
    consensus_level: float  # 0-1, how much models agree
    key_insights: List[str]
    conflicts: List[Dict[str, str]]
    uncertainty_areas: List[str]
    dominant_perspective: Optional[str]


class EnsembleSynthesizer(QtCore.QObject):
    """Advanced synthesizer for ensemble responses."""
    
    # Streaming signals
    token = QtCore.Signal(str)
    thinking = QtCore.Signal(str)
    finished = QtCore.Signal(SynthesisResult)
    error = QtCore.Signal(str)
    
    def __init__(self, synthesis_model: str = "gemma3:12b"):
        super().__init__()
        self.synthesis_model = synthesis_model
        self._stop_requested = False
        
    def request_stop(self):
        """Request synthesis to stop."""
        self._stop_requested = True
    
    def synthesize_bayesian(self, weighted_responses: List[Tuple], 
                           original_question: str,
                           ensemble_config: EnsembleConfig) -> None:
        """Perform Bayesian synthesis of weighted responses."""
        # Extract insights and identify conflicts
        analysis = self._analyze_responses(weighted_responses)
        
        # Build comprehensive synthesis prompt
        synthesis_prompt = self._build_bayesian_prompt(
            weighted_responses, original_question, analysis, ensemble_config
        )
        
        # Add synthesis system prompt
        system_prompt = """You are an expert synthesis system using Bayesian Model Averaging principles.

Your task is to create a unified response that:
1. Weighs evidence based on model confidence and historical accuracy
2. Preserves unique valuable insights even from lower-weight responses  
3. Explicitly handles contradictions using probabilistic reasoning
4. Quantifies uncertainty where models disagree
5. Provides a coherent, well-structured final answer

Use these Bayesian principles:
- Prior: Model weights represent prior belief in each model's accuracy
- Likelihood: Confidence scores represent likelihood of correctness
- Posterior: Your synthesis should reflect updated beliefs after considering all evidence

Structure your response to be clear and actionable for the user."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": synthesis_prompt}
        ]
        
        # Stream synthesis
        self._stream_synthesis(messages, analysis)
    
    def _analyze_responses(self, weighted_responses: List[Tuple]) -> Dict:
        """Analyze responses for patterns, conflicts, and insights."""
        from core.ensemble_orchestrator import ModelResponse
        
        analysis = {
            "key_insights": [],
            "conflicts": [],
            "consensus_topics": [],
            "uncertainty_areas": [],
            "fact_claims": [],
            "unique_perspectives": []
        }
        
        # Extract key sentences and claims
        all_sentences = []
        response_sentences = {}
        
        for response, weight in weighted_responses:
            sentences = self._extract_key_sentences(response.content)
            all_sentences.extend(sentences)
            response_sentences[response.model_name] = sentences
        
        # Find consensus using TF-IDF similarity
        if len(all_sentences) > 5:
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            try:
                tfidf_matrix = vectorizer.fit_transform(all_sentences)
                similarity_matrix = cosine_similarity(tfidf_matrix)
                
                # Identify high-consensus sentences
                consensus_threshold = 0.7
                for i, sentence in enumerate(all_sentences):
                    similar_count = np.sum(similarity_matrix[i] > consensus_threshold) - 1
                    if similar_count >= len(weighted_responses) * 0.6:
                        analysis["consensus_topics"].append(sentence)
                
            except Exception:
                pass  # Continue without consensus analysis
        
        # Identify conflicts through contradiction detection
        analysis["conflicts"] = self._detect_conflicts(weighted_responses)
        
        # Extract unique high-value insights
        for response, weight in weighted_responses:
            if response.diversity_score > 0.7:
                # High diversity indicates unique perspective
                unique_sentences = self._extract_unique_insights(
                    response.content, all_sentences
                )
                analysis["unique_perspectives"].extend(unique_sentences[:2])
        
        # Identify areas of uncertainty
        confidence_by_topic = self._analyze_confidence_distribution(weighted_responses)
        for topic, confidences in confidence_by_topic.items():
            if np.std(confidences) > 0.2:  # High variance in confidence
                analysis["uncertainty_areas"].append(topic)
        
        return analysis
    
    def _extract_key_sentences(self, text: str) -> List[str]:
        """Extract key sentences from text."""
        # Simple sentence extraction - can be enhanced with NLP
        sentences = re.split(r'[.!?]+', text)
        
        # Filter and clean
        key_sentences = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 20 and not sent.lower().startswith(('however', 'therefore', 'thus')):
                key_sentences.append(sent)
        
        return key_sentences[:10]  # Limit to top sentences
    
    def _detect_conflicts(self, weighted_responses: List[Tuple]) -> List[Dict]:
        """Detect conflicting statements between responses."""
        conflicts = []
        
        # Simple conflict detection based on negation patterns
        negation_words = {'not', 'no', 'never', 'neither', 'nor', 'cannot', "can't", "won't", "don't"}
        
        responses_content = [(r.model_name, r.content.lower()) for r, _ in weighted_responses]
        
        for i, (model1, content1) in enumerate(responses_content):
            for j, (model2, content2) in enumerate(responses_content[i+1:], i+1):
                # Check for explicit contradictions
                if any(word in content1 for word in negation_words):
                    # Look for opposite claims
                    for topic in self._extract_topics(content1):
                        if topic in content2 and any(word in content2 for word in negation_words):
                            conflicts.append({
                                "topic": topic,
                                "model1": model1,
                                "position1": self._extract_position(content1, topic),
                                "model2": model2, 
                                "position2": self._extract_position(content2, topic)
                            })
        
        return conflicts[:5]  # Limit conflicts
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract main topics from text."""
        # Simple noun phrase extraction
        # In production, use spaCy or similar
        common_topics = ['accuracy', 'performance', 'speed', 'efficiency', 'quality',
                        'reliability', 'cost', 'complexity', 'scalability', 'safety']
        
        found_topics = []
        for topic in common_topics:
            if topic in text:
                found_topics.append(topic)
        
        return found_topics
    
    def _extract_position(self, text: str, topic: str) -> str:
        """Extract position on a topic from text."""
        # Find sentence containing topic
        sentences = text.split('.')
        for sent in sentences:
            if topic in sent.lower():
                return sent.strip()[:100] + "..."
        return "Position unclear"
    
    def _extract_unique_insights(self, content: str, all_sentences: List[str]) -> List[str]:
        """Extract insights unique to this response."""
        response_sentences = self._extract_key_sentences(content)
        unique = []
        
        for sent in response_sentences:
            # Check if similar sentence exists in other responses
            is_unique = True
            for other in all_sentences:
                if sent != other and self._sentence_similarity(sent, other) > 0.8:
                    is_unique = False
                    break
            
            if is_unique:
                unique.append(sent)
        
        return unique
    
    def _sentence_similarity(self, sent1: str, sent2: str) -> float:
        """Calculate similarity between two sentences."""
        # Simple Jaccard similarity
        words1 = set(sent1.lower().split())
        words2 = set(sent2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _analyze_confidence_distribution(self, weighted_responses: List[Tuple]) -> Dict[str, List[float]]:
        """Analyze confidence distribution by topic."""
        topic_confidences = {}
        
        for response, _ in weighted_responses:
            topics = self._extract_topics(response.content.lower())
            for topic in topics:
                if topic not in topic_confidences:
                    topic_confidences[topic] = []
                topic_confidences[topic].append(response.confidence)
        
        return topic_confidences
    
    def _build_bayesian_prompt(self, weighted_responses: List[Tuple], 
                              original_question: str,
                              analysis: Dict,
                              ensemble_config: EnsembleConfig) -> str:
        """Build comprehensive prompt for Bayesian synthesis."""
        prompt = f"Original Question: {original_question}\n\n"
        prompt += "## Weighted Model Responses\n\n"
        
        # Add each response with metadata
        for response, weight in weighted_responses:
            model_config = next((m for m in ensemble_config.models if m.name == response.model_name), None)
            if model_config:
                prompt += f"### {model_config.display_name}\n"
                prompt += f"- Weight: {weight:.3f} (Prior: {response.confidence:.2f}, Diversity: {response.diversity_score:.2f})\n"
                prompt += f"- Role: {', '.join(r.value for r in model_config.roles)}\n"
                prompt += f"- Response:\n{response.content}\n\n"
        
        # Add analysis results
        prompt += "## Response Analysis\n\n"
        
        if analysis["consensus_topics"]:
            prompt += "### Consensus Points:\n"
            for topic in analysis["consensus_topics"][:5]:
                prompt += f"- {topic}\n"
            prompt += "\n"
        
        if analysis["conflicts"]:
            prompt += "### Identified Conflicts:\n"
            for conflict in analysis["conflicts"]:
                prompt += f"- **{conflict['topic']}**: {conflict['model1']} vs {conflict['model2']}\n"
            prompt += "\n"
        
        if analysis["unique_perspectives"]:
            prompt += "### Unique Insights:\n"
            for insight in analysis["unique_perspectives"][:5]:
                prompt += f"- {insight}\n"
            prompt += "\n"
        
        if analysis["uncertainty_areas"]:
            prompt += "### Areas of Uncertainty:\n"
            for area in analysis["uncertainty_areas"]:
                prompt += f"- {area}\n"
            prompt += "\n"
        
        # Synthesis instructions
        prompt += """## Synthesis Task

Create a Bayesian synthesis that:

1. **Weighs Evidence**: Give more credence to high-weight responses while preserving valuable insights from all models
2. **Resolves Conflicts**: Use the weights and confidence scores to adjudicate between conflicting positions
3. **Preserves Diversity**: Include unique valuable insights even from lower-weight models
4. **Quantifies Uncertainty**: Clearly indicate confidence levels and areas where models disagree
5. **Provides Clarity**: Structure the response to be maximally useful to the user

Format: Provide a clear, comprehensive answer that represents the best collective intelligence of the ensemble."""
        
        return prompt
    
    def _stream_synthesis(self, messages: List[Dict], analysis: Dict) -> None:
        """Stream the synthesis response."""
        try:
            content_accumulator = ""
            thinking_accumulator = ""
            
            # Run synthesis model
            stream = ollama.chat(
                model=self.synthesis_model,
                messages=messages,
                options={"temperature": 0.5},  # Lower temperature for synthesis
                stream=True
            )
            
            for chunk in stream:
                if self._stop_requested:
                    break
                    
                if "message" in chunk:
                    message = chunk["message"]
                    
                    if "content" in message and message["content"]:
                        token = message["content"]
                        content_accumulator += token
                        self.token.emit(token)
                    
                    thinking_content = message.get("thinking_content", message.get("reasoning_content"))
                    if thinking_content:
                        thinking_accumulator += thinking_content
                        self.thinking.emit(thinking_content)
            
            # Calculate consensus level
            consensus_level = self._calculate_consensus_level(analysis)
            
            # Extract key insights from synthesis
            key_insights = analysis.get("consensus_topics", [])[:5]
            
            # Build synthesis result
            result = SynthesisResult(
                content=content_accumulator,
                thinking=thinking_accumulator if thinking_accumulator else None,
                consensus_level=consensus_level,
                key_insights=key_insights,
                conflicts=analysis.get("conflicts", []),
                uncertainty_areas=analysis.get("uncertainty_areas", []),
                dominant_perspective=self._identify_dominant_perspective(analysis)
            )
            
            self.finished.emit(result)
            
        except Exception as e:
            self.error.emit(f"Synthesis error: {str(e)}")
    
    def _calculate_consensus_level(self, analysis: Dict) -> float:
        """Calculate overall consensus level from analysis."""
        # Simple heuristic based on consensus vs conflicts
        consensus_count = len(analysis.get("consensus_topics", []))
        conflict_count = len(analysis.get("conflicts", []))
        uncertainty_count = len(analysis.get("uncertainty_areas", []))
        
        if consensus_count + conflict_count + uncertainty_count == 0:
            return 0.5
        
        consensus_score = consensus_count / (consensus_count + conflict_count + uncertainty_count)
        return min(1.0, consensus_score * 1.2)  # Slight boost for consensus
    
    def _identify_dominant_perspective(self, analysis: Dict) -> Optional[str]:
        """Identify the dominant perspective from the analysis."""
        # For now, return None - can be enhanced to detect primary viewpoint
        return None
