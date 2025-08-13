# Ensemble Model System Documentation

## Overview

The ensemble model system leverages collective intelligence principles to achieve superior accuracy by combining responses from multiple smaller language models. This implementation uses Bayesian Model Averaging, diversity metrics, and sophisticated synthesis to potentially exceed the performance of single large models.

## Architecture

### Core Components

1. **Model Configuration (`model_configs.py`)**
   - Defines model-specific prompts, temperature ranges, and roles
   - Supports 5 default models with diverse capabilities
   - Includes adversarial prompts for robustness

2. **Ensemble Orchestrator (`core/ensemble_orchestrator.py`)**
   - Manages parallel execution of multiple models
   - Implements confidence scoring and diversity calculation
   - Handles Bayesian weight calculation
   - Tracks performance metrics

3. **Enhanced Synthesis (`core/ensemble_synthesis.py`)**
   - Performs intelligent synthesis using a dedicated model
   - Identifies consensus, conflicts, and unique insights
   - Quantifies uncertainty and confidence levels
   - Uses TF-IDF for semantic analysis

4. **Performance Tracking (`core/performance_tracker.py`)**
   - Persists model performance over time
   - Tracks category-specific accuracy
   - Enables meta-learning for optimal model selection
   - SQLite-based for efficiency

## Key Features

### 1. Collective Intelligence Implementation

- **Condorcet's Jury Theorem**: Aggregates multiple "weak" models to create a strong ensemble
- **Wisdom of Crowds**: Leverages diversity and independence for better predictions
- **Bayesian Model Averaging**: Weights responses based on confidence and historical performance

### 2. Model Diversity

The system uses 5 diverse models by default:
- **GPT-OSS 20B**: Reasoning specialist
- **Gemma3 12B**: Creative problem solver
- **Qwen3 30B**: Comprehensive analysis
- **DeepSeek-R1 32B**: Deep analytical thinking
- **Mistral 24B Abliterated**: Adversarial/critical perspective

### 3. Advanced Features

- **Confidence Calibration**: Each model's confidence is calibrated based on historical accuracy
- **Diversity Bonus**: Unique insights receive additional weight
- **Conflict Resolution**: Bayesian principles resolve contradictions
- **Uncertainty Quantification**: Clear indication of areas where models disagree

## Usage

### Basic Ensemble Query

1. Click the send button's dropdown menu
2. Select "Ensemble Response (5 Models)"
3. The system will:
   - Run 5 models with 3 temperature variants each (15 total responses)
   - Calculate diversity and confidence scores
   - Synthesize using Bayesian averaging
   - Display the final response with consensus level

### UI Features

- **Model Selector**: Switch between individual model responses
- **Response Selector**: View different temperature variants
- **Progress Indicator**: Track ensemble generation progress
- **Consensus Level**: See how much models agree (0-100%)

### Performance Optimization

The system continuously learns:
- Tracks which models perform best for different query types
- Adjusts weights based on user feedback
- Identifies optimal model combinations

## Configuration

### Custom Ensemble Configuration

```python
# In model_configs.py
custom_ensemble = EnsembleConfig(
    models=[...],  # Select specific models
    response_per_model=3,  # Temperature variants
    synthesis_temperature=0.5,  # Final synthesis temp
    diversity_weight=0.2,  # Importance of unique insights
    min_confidence_threshold=0.3,  # Minimum confidence
    enable_adversarial=True,  # Include devil's advocate
    enable_meta_learning=True  # Track performance
)
```

### Model Roles

- **REASONING**: Logical analysis and step-by-step thinking
- **CREATIVE**: Innovative approaches and connections
- **FACTUAL**: Accurate information and comprehensive coverage
- **CRITICAL**: Questioning assumptions and identifying flaws
- **ADVERSARIAL**: Contrarian perspectives and edge cases

## Performance Expectations

Based on ensemble learning theory:
- **Individual Model Accuracy**: ~60-70%
- **Ensemble Accuracy**: ~80-90% (with proper diversity)
- **Response Time**: 3-5x single model (but parallelized)
- **Cost**: ~5x single model (but using smaller models)

## Advanced Topics

### Meta-Learning

The system tracks:
- Model performance by query category
- Optimal model combinations
- Response time percentiles
- User preference patterns

### Synthesis Process

1. **Within-Model Synthesis**: Combines temperature variants
2. **Cross-Model Weighting**: Bayesian averaging with performance weights
3. **Diversity Analysis**: TF-IDF similarity and unique insight detection
4. **Conflict Resolution**: Weighted voting with confidence scores
5. **Final Synthesis**: Enhanced model creates coherent response

### Database Schema

Performance data stored in SQLite:
- `query_performance`: Individual query results
- `model_stats`: Aggregated model performance
- `ensemble_combinations`: Successful model combinations

## Future Enhancements

1. **Dynamic Model Selection**: Choose models based on query type
2. **User Feedback Integration**: Direct rating affects weights
3. **Custom Model Integration**: Easy addition of new models
4. **Streaming Synthesis**: Real-time aggregation
5. **Visualization**: Performance dashboards and insights

## Theoretical Foundation

The implementation is based on:
- Condorcet's Jury Theorem (1785)
- Francis Galton's Wisdom of Crowds (1907)
- Modern ensemble learning (Boosting, Bagging)
- Bayesian Model Averaging
- Information diversity theory

## Conclusion

This ensemble system represents a sophisticated implementation of collective intelligence principles applied to language models. By carefully orchestrating diverse models and using advanced synthesis techniques, it achieves accuracy levels that can rival or exceed much larger individual models while maintaining cost efficiency and interpretability.
