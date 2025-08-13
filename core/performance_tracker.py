"""Performance tracking and persistence for ensemble models.

This module tracks model performance over time and persists metrics
for continuous improvement and meta-learning.
"""

import json
import sqlite3
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np


@dataclass
class QueryPerformance:
    """Performance data for a single query."""
    query_id: str
    timestamp: float
    query_text: str
    query_category: Optional[str]
    model_responses: Dict[str, float]  # model_name -> confidence
    ensemble_confidence: float
    user_rating: Optional[float] = None  # 1-5 rating
    response_time: float = 0.0
    selected_response: Optional[str] = None  # Which response user preferred


@dataclass 
class ModelPerformanceStats:
    """Aggregated performance statistics for a model."""
    model_name: str
    total_queries: int
    average_confidence: float
    average_accuracy: float  # Based on user ratings
    category_performance: Dict[str, float]
    response_time_p50: float
    response_time_p95: float
    last_updated: float


class PerformanceTracker:
    """Track and persist model performance metrics."""
    
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            # Default to user's data directory
            data_dir = Path.home() / ".gpt-oss-ui" / "performance"
            data_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(data_dir / "performance.db")
        
        self.db_path = db_path
        self._init_database()
        
    def _init_database(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS query_performance (
                    query_id TEXT PRIMARY KEY,
                    timestamp REAL,
                    query_text TEXT,
                    query_category TEXT,
                    model_responses TEXT,  -- JSON
                    ensemble_confidence REAL,
                    user_rating REAL,
                    response_time REAL,
                    selected_response TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_stats (
                    model_name TEXT PRIMARY KEY,
                    total_queries INTEGER,
                    average_confidence REAL,
                    average_accuracy REAL,
                    category_performance TEXT,  -- JSON
                    response_time_p50 REAL,
                    response_time_p95 REAL,
                    last_updated REAL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ensemble_combinations (
                    combination_hash TEXT PRIMARY KEY,
                    model_names TEXT,  -- JSON list
                    success_rate REAL,
                    usage_count INTEGER,
                    last_used REAL
                )
            """)
            
            # Create indices
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON query_performance(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_category ON query_performance(query_category)")
    
    def record_query(self, query_perf: QueryPerformance):
        """Record a query performance."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO query_performance 
                (query_id, timestamp, query_text, query_category, model_responses,
                 ensemble_confidence, user_rating, response_time, selected_response)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                query_perf.query_id,
                query_perf.timestamp,
                query_perf.query_text,
                query_perf.query_category,
                json.dumps(query_perf.model_responses),
                query_perf.ensemble_confidence,
                query_perf.user_rating,
                query_perf.response_time,
                query_perf.selected_response
            ))
    
    def update_user_rating(self, query_id: str, rating: float, selected_response: Optional[str] = None):
        """Update user rating for a query."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE query_performance 
                SET user_rating = ?, selected_response = ?
                WHERE query_id = ?
            """, (rating, selected_response, query_id))
    
    def get_model_stats(self, model_name: str, 
                       time_window: Optional[timedelta] = None) -> ModelPerformanceStats:
        """Get aggregated stats for a model."""
        with sqlite3.connect(self.db_path) as conn:
            # Set time boundary
            min_timestamp = 0.0
            if time_window:
                min_timestamp = time.time() - time_window.total_seconds()
            
            # Get basic stats
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total,
                    AVG(CAST(json_extract(model_responses, '$.' || ?) AS REAL)) as avg_conf,
                    AVG(CASE WHEN user_rating IS NOT NULL THEN user_rating / 5.0 ELSE NULL END) as avg_acc
                FROM query_performance
                WHERE timestamp > ?
                AND json_extract(model_responses, '$.' || ?) IS NOT NULL
            """, (model_name, min_timestamp, model_name))
            
            row = cursor.fetchone()
            total_queries = row[0] or 0
            avg_confidence = row[1] or 0.0
            avg_accuracy = row[2] or 0.7  # Default if no ratings
            
            # Get category performance
            cursor = conn.execute("""
                SELECT 
                    query_category,
                    AVG(CASE WHEN user_rating IS NOT NULL THEN user_rating / 5.0 ELSE 0.7 END) as cat_acc
                FROM query_performance
                WHERE timestamp > ?
                AND json_extract(model_responses, '$.' || ?) IS NOT NULL
                AND query_category IS NOT NULL
                GROUP BY query_category
            """, (min_timestamp, model_name))
            
            category_performance = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Get response time percentiles
            cursor = conn.execute("""
                SELECT response_time
                FROM query_performance
                WHERE timestamp > ?
                AND json_extract(model_responses, '$.' || ?) IS NOT NULL
                ORDER BY response_time
            """, (min_timestamp, model_name))
            
            response_times = [row[0] for row in cursor.fetchall()]
            if response_times:
                p50 = np.percentile(response_times, 50)
                p95 = np.percentile(response_times, 95)
            else:
                p50 = p95 = 0.0
            
            return ModelPerformanceStats(
                model_name=model_name,
                total_queries=total_queries,
                average_confidence=avg_confidence,
                average_accuracy=avg_accuracy,
                category_performance=category_performance,
                response_time_p50=p50,
                response_time_p95=p95,
                last_updated=time.time()
            )
    
    def get_ensemble_combination_stats(self, model_names: List[str]) -> Tuple[float, int]:
        """Get performance stats for a specific model combination."""
        combination_hash = self._hash_combination(model_names)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT success_rate, usage_count
                FROM ensemble_combinations
                WHERE combination_hash = ?
            """, (combination_hash,))
            
            row = cursor.fetchone()
            if row:
                return row[0], row[1]
            return 0.7, 0  # Default success rate
    
    def update_ensemble_combination(self, model_names: List[str], success: bool):
        """Update stats for an ensemble combination."""
        combination_hash = self._hash_combination(model_names)
        
        with sqlite3.connect(self.db_path) as conn:
            # Get current stats
            cursor = conn.execute("""
                SELECT success_rate, usage_count
                FROM ensemble_combinations
                WHERE combination_hash = ?
            """, (combination_hash,))
            
            row = cursor.fetchone()
            if row:
                # Update with exponential moving average
                alpha = 0.1
                new_rate = (1 - alpha) * row[0] + alpha * (1.0 if success else 0.0)
                new_count = row[1] + 1
            else:
                new_rate = 1.0 if success else 0.0
                new_count = 1
            
            conn.execute("""
                INSERT OR REPLACE INTO ensemble_combinations
                (combination_hash, model_names, success_rate, usage_count, last_used)
                VALUES (?, ?, ?, ?, ?)
            """, (
                combination_hash,
                json.dumps(sorted(model_names)),
                new_rate,
                new_count,
                time.time()
            ))
    
    def get_best_models_for_category(self, category: str, top_k: int = 3) -> List[str]:
        """Get best performing models for a category."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    model_name,
                    category_performance
                FROM model_stats
                WHERE json_extract(category_performance, '$.' || ?) IS NOT NULL
                ORDER BY json_extract(category_performance, '$.' || ?) DESC
                LIMIT ?
            """, (category, category, top_k))
            
            return [row[0] for row in cursor.fetchall()]
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old performance data."""
        cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                DELETE FROM query_performance
                WHERE timestamp < ?
            """, (cutoff_time,))
            
            # Update model stats after cleanup
            self._refresh_model_stats()
    
    def _refresh_model_stats(self):
        """Refresh aggregated model statistics."""
        # Get all unique model names
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT DISTINCT key
                FROM query_performance, json_each(model_responses)
            """)
            
            model_names = [row[0] for row in cursor.fetchall()]
            
            # Update stats for each model
            for model_name in model_names:
                stats = self.get_model_stats(model_name)
                
                conn.execute("""
                    INSERT OR REPLACE INTO model_stats
                    (model_name, total_queries, average_confidence, average_accuracy,
                     category_performance, response_time_p50, response_time_p95, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    stats.model_name,
                    stats.total_queries,
                    stats.average_confidence,
                    stats.average_accuracy,
                    json.dumps(stats.category_performance),
                    stats.response_time_p50,
                    stats.response_time_p95,
                    stats.last_updated
                ))
    
    def _hash_combination(self, model_names: List[str]) -> str:
        """Create a hash for a model combination."""
        import hashlib
        sorted_names = sorted(model_names)
        combination_str = "|".join(sorted_names)
        return hashlib.md5(combination_str.encode()).hexdigest()
    
    def export_metrics(self, output_path: str):
        """Export performance metrics to JSON."""
        with sqlite3.connect(self.db_path) as conn:
            # Export recent queries
            cursor = conn.execute("""
                SELECT * FROM query_performance
                ORDER BY timestamp DESC
                LIMIT 1000
            """)
            
            queries = []
            for row in cursor.fetchall():
                queries.append({
                    "query_id": row[0],
                    "timestamp": row[1],
                    "query_text": row[2],
                    "query_category": row[3],
                    "model_responses": json.loads(row[4]),
                    "ensemble_confidence": row[5],
                    "user_rating": row[6],
                    "response_time": row[7],
                    "selected_response": row[8]
                })
            
            # Export model stats
            cursor = conn.execute("SELECT * FROM model_stats")
            model_stats = []
            for row in cursor.fetchall():
                model_stats.append({
                    "model_name": row[0],
                    "total_queries": row[1],
                    "average_confidence": row[2],
                    "average_accuracy": row[3],
                    "category_performance": json.loads(row[4]),
                    "response_time_p50": row[5],
                    "response_time_p95": row[6],
                    "last_updated": row[7]
                })
            
            # Export ensemble combinations
            cursor = conn.execute("SELECT * FROM ensemble_combinations")
            combinations = []
            for row in cursor.fetchall():
                combinations.append({
                    "combination_hash": row[0],
                    "model_names": json.loads(row[1]),
                    "success_rate": row[2],
                    "usage_count": row[3],
                    "last_used": row[4]
                })
            
            export_data = {
                "export_timestamp": time.time(),
                "queries": queries,
                "model_stats": model_stats,
                "ensemble_combinations": combinations
            }
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
