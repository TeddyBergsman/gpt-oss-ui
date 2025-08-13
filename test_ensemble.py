#!/usr/bin/env python3
"""Test script for the ensemble model system.

This script demonstrates how to use the ensemble orchestrator
independently of the UI for testing and validation.
"""

import sys
import asyncio
from PySide6.QtCore import QCoreApplication, QThread, QObject, Slot
from PySide6.QtCore import QEventLoop

from model_configs import EnsembleConfig, ENSEMBLE_MODELS
from core.ensemble_orchestrator import EnsembleOrchestrator, EnsembleResponse
from core.performance_tracker import PerformanceTracker


class EnsembleTestRunner(QObject):
    """Test runner for ensemble system."""
    
    def __init__(self):
        super().__init__()
        self.results = []
        self.loop = QEventLoop()
        
    @Slot(str, int, str)
    def on_model_token(self, model_name: str, response_id: int, token: str):
        """Handle streaming tokens from models."""
        # In real usage, you'd update UI here
        pass
        
    @Slot(str)
    def on_synthesis_token(self, token: str):
        """Handle synthesis tokens."""
        print(token, end='', flush=True)
        
    @Slot(EnsembleResponse)
    def on_synthesis_finished(self, response: EnsembleResponse):
        """Handle completed ensemble response."""
        print("\n\n" + "="*80)
        print("ENSEMBLE COMPLETE")
        print("="*80)
        print(f"Consensus Level: {response.consensus_level:.1%}")
        print(f"Total Time: {response.total_time:.1f}s")
        print(f"Models Used: {len(response.model_responses)}")
        print(f"Aggregation Method: {response.aggregation_method}")
        
        print("\nConfidence Scores:")
        for model, confidence in response.confidence_scores.items():
            print(f"  {model}: {confidence:.2f}")
            
        print("\nDiversity Metrics:")
        for metric, value in response.diversity_metrics.items():
            print(f"  {metric}: {value}")
            
        self.results.append(response)
        self.loop.quit()
        
    @Slot(str)
    def on_error(self, error_msg: str):
        """Handle errors."""
        print(f"\nERROR: {error_msg}")
        self.loop.quit()
        
    @Slot(str)
    def on_status(self, status: str):
        """Handle status updates."""
        print(f"\nSTATUS: {status}")
        
    def test_ensemble(self, question: str):
        """Run ensemble test with a question."""
        print(f"Testing ensemble with question: {question}")
        print("-" * 80)
        
        # Create ensemble configuration
        ensemble_config = EnsembleConfig.default()
        
        # For testing, use fewer models and responses
        ensemble_config.models = [
            ENSEMBLE_MODELS["gpt-oss:20b"],
            ENSEMBLE_MODELS["gemma3:12b"],
            ENSEMBLE_MODELS["huihui_ai/mistral-small-abliterated:24b"],
        ]
        ensemble_config.response_per_model = 2  # Reduce for faster testing
        
        # Create messages
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": question}
        ]
        
        # Model options
        options = {"temperature": 0.7}
        
        # Create orchestrator
        orchestrator = EnsembleOrchestrator(
            ensemble_config,
            messages,
            options,
            PerformanceTracker()
        )
        
        # Move to thread
        thread = QThread()
        orchestrator.moveToThread(thread)
        
        # Connect signals
        thread.started.connect(orchestrator.start)
        orchestrator.model_token.connect(self.on_model_token)
        orchestrator.synthesis_token.connect(self.on_synthesis_token)
        orchestrator.synthesis_finished.connect(self.on_synthesis_finished)
        orchestrator.error.connect(self.on_error)
        orchestrator.status.connect(self.on_status)
        
        # Cleanup
        orchestrator.synthesis_finished.connect(thread.quit)
        orchestrator.error.connect(thread.quit)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(orchestrator.deleteLater)
        
        # Start
        thread.start()
        
        # Wait for completion
        self.loop.exec()
        thread.wait()


def main():
    """Main test function."""
    # Create Qt application (required for signals/slots)
    app = QCoreApplication(sys.argv)
    
    # Create test runner
    runner = EnsembleTestRunner()
    
    # Test questions
    test_questions = [
        "What are the key principles of ensemble learning and how do they apply to language models?",
        # "Explain the mathematical foundation behind Condorcet's Jury Theorem.",
        # "Compare and contrast different approaches to model aggregation.",
    ]
    
    # Run tests
    for question in test_questions:
        runner.test_ensemble(question)
        print("\n" + "="*80 + "\n")
    
    # Print summary
    print("ENSEMBLE TEST SUMMARY")
    print("="*80)
    print(f"Total tests run: {len(runner.results)}")
    
    if runner.results:
        avg_consensus = sum(r.consensus_level for r in runner.results) / len(runner.results)
        avg_time = sum(r.total_time for r in runner.results) / len(runner.results)
        print(f"Average consensus level: {avg_consensus:.1%}")
        print(f"Average response time: {avg_time:.1f}s")
    
    # Export performance metrics
    tracker = PerformanceTracker()
    tracker.export_metrics("ensemble_test_metrics.json")
    print("\nPerformance metrics exported to ensemble_test_metrics.json")


if __name__ == "__main__":
    main()
