#!/usr/bin/env python3
"""
Caching System for Hyperparameter Optimization

Provides persistent caching of optimization results to avoid redundant computations.
"""

import os
import json
import pickle
import sqlite3
import hashlib
import logging
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cached optimization result"""
    params: Dict[str, Any]
    score: float
    model_class: str
    data_hash: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


class OptimizationCache:
    """
    Persistent cache for optimization results using SQLite backend.
    
    Stores and retrieves model evaluation results based on:
    - Model parameters
    - Input data characteristics
    - Model class/type
    """
    
    def __init__(self, cache_dir: str = ".optimization_cache"):
        """
        Initialize the optimization cache.
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = cache_dir
        self.db_path = os.path.join(cache_dir, "optimization_cache.db")
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize the SQLite database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create cache table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cache_key TEXT UNIQUE NOT NULL,
                    params_json TEXT NOT NULL,
                    score REAL NOT NULL,
                    model_class TEXT NOT NULL,
                    data_hash TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    metadata_json TEXT
                )
            """)
            
            # Create indices for faster lookups
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_key 
                ON cache_entries(cache_key)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_model_class 
                ON cache_entries(model_class)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_data_hash 
                ON cache_entries(data_hash)
            """)
            
            conn.commit()
    
    def _generate_cache_key(self, 
                           params: Dict[str, Any], 
                           data_hash: str, 
                           model_class: str) -> str:
        """Generate a unique cache key."""
        params_str = json.dumps(params, sort_keys=True)
        combined = f"{model_class}_{params_str}_{data_hash}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _hash_data(self, data: pd.DataFrame) -> str:
        """Generate a hash for the input data."""
        # Use data shape, column names, and a sample of values for hash
        shape_str = f"{data.shape}"
        columns_str = "_".join(sorted(data.columns))
        
        # Sample some values to include in hash (for sensitivity to data changes)
        sample_data = data.iloc[::max(1, len(data)//100)].values  # Sample every 100th row
        sample_str = str(sample_data.tobytes())
        
        combined = f"{shape_str}_{columns_str}_{sample_str}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get(self, 
            params: Dict[str, Any], 
            data: pd.DataFrame, 
            model_class: str) -> Optional[float]:
        """
        Retrieve cached result for given parameters and data.
        
        Args:
            params: Model parameters
            data: Input data
            model_class: Name of the model class
            
        Returns:
            Cached score if found, None otherwise
        """
        data_hash = self._hash_data(data)
        cache_key = self._generate_cache_key(params, data_hash, model_class)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT score FROM cache_entries 
                WHERE cache_key = ?
            """, (cache_key,))
            
            result = cursor.fetchone()
            
            if result:
                logger.debug(f"Cache hit for key: {cache_key[:8]}...")
                return result[0]
            
            logger.debug(f"Cache miss for key: {cache_key[:8]}...")
            return None
    
    def put(self, 
            params: Dict[str, Any], 
            score: float,
            data: pd.DataFrame, 
            model_class: str,
            metadata: Optional[Dict[str, Any]] = None):
        """
        Store result in cache.
        
        Args:
            params: Model parameters
            score: Evaluation score
            data: Input data
            model_class: Name of the model class
            metadata: Optional metadata to store
        """
        data_hash = self._hash_data(data)
        cache_key = self._generate_cache_key(params, data_hash, model_class)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Insert or replace cache entry
            cursor.execute("""
                INSERT OR REPLACE INTO cache_entries 
                (cache_key, params_json, score, model_class, data_hash, timestamp, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                cache_key,
                json.dumps(params),
                score,
                model_class,
                data_hash,
                datetime.now().isoformat(),
                json.dumps(metadata) if metadata else None
            ))
            
            conn.commit()
            
        logger.debug(f"Cached result for key: {cache_key[:8]}...")
    
    def get_entries_by_model(self, model_class: str) -> List[CacheEntry]:
        """Get all cache entries for a specific model class."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT params_json, score, model_class, data_hash, timestamp, metadata_json
                FROM cache_entries 
                WHERE model_class = ?
                ORDER BY timestamp DESC
            """, (model_class,))
            
            results = cursor.fetchall()
            
            entries = []
            for row in results:
                params_json, score, mc, data_hash, timestamp, metadata_json = row
                
                entry = CacheEntry(
                    params=json.loads(params_json),
                    score=score,
                    model_class=mc,
                    data_hash=data_hash,
                    timestamp=datetime.fromisoformat(timestamp),
                    metadata=json.loads(metadata_json) if metadata_json else None
                )
                entries.append(entry)
            
            return entries
    
    def get_best_entry(self, 
                      model_class: str, 
                      direction: str = 'maximize') -> Optional[CacheEntry]:
        """Get the best cached entry for a model class."""
        entries = self.get_entries_by_model(model_class)
        
        if not entries:
            return None
        
        if direction == 'maximize':
            best_entry = max(entries, key=lambda e: e.score)
        else:
            best_entry = min(entries, key=lambda e: e.score)
        
        return best_entry
    
    def clear_cache(self, model_class: Optional[str] = None):
        """
        Clear cache entries.
        
        Args:
            model_class: If provided, only clear entries for this model class
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if model_class:
                cursor.execute("DELETE FROM cache_entries WHERE model_class = ?", (model_class,))
                logger.info(f"Cleared cache for model class: {model_class}")
            else:
                cursor.execute("DELETE FROM cache_entries")
                logger.info("Cleared all cache entries")
            
            conn.commit()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total entries
            cursor.execute("SELECT COUNT(*) FROM cache_entries")
            total_entries = cursor.fetchone()[0]
            
            # Entries by model class
            cursor.execute("""
                SELECT model_class, COUNT(*) 
                FROM cache_entries 
                GROUP BY model_class
            """)
            entries_by_model = dict(cursor.fetchall())
            
            # Cache size on disk
            cache_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
            
            # Date range
            cursor.execute("""
                SELECT MIN(timestamp), MAX(timestamp) 
                FROM cache_entries
            """)
            date_range = cursor.fetchone()
            
            return {
                'total_entries': total_entries,
                'entries_by_model': entries_by_model,
                'cache_size_bytes': cache_size,
                'cache_size_mb': cache_size / (1024 * 1024),
                'date_range': date_range,
                'db_path': self.db_path
            }
    
    def export_cache(self, output_path: str):
        """Export cache to JSON file."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT cache_key, params_json, score, model_class, 
                       data_hash, timestamp, metadata_json
                FROM cache_entries
                ORDER BY timestamp
            """)
            
            results = cursor.fetchall()
            
            export_data = []
            for row in results:
                cache_key, params_json, score, model_class, data_hash, timestamp, metadata_json = row
                
                export_data.append({
                    'cache_key': cache_key,
                    'params': json.loads(params_json),
                    'score': score,
                    'model_class': model_class,
                    'data_hash': data_hash,
                    'timestamp': timestamp,
                    'metadata': json.loads(metadata_json) if metadata_json else None
                })
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Exported {len(export_data)} cache entries to {output_path}")
    
    def import_cache(self, input_path: str, overwrite: bool = False):
        """Import cache from JSON file."""
        if not overwrite:
            # Check if we would overwrite existing entries
            with open(input_path, 'r') as f:
                import_data = json.load(f)
            
            existing_keys = set()
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT cache_key FROM cache_entries")
                existing_keys = {row[0] for row in cursor.fetchall()}
            
            import_keys = {entry['cache_key'] for entry in import_data}
            conflicts = existing_keys & import_keys
            
            if conflicts:
                logger.warning(f"Found {len(conflicts)} conflicting cache keys. "
                              f"Use overwrite=True to replace existing entries.")
                return
        
        with open(input_path, 'r') as f:
            import_data = json.load(f)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for entry in import_data:
                if overwrite:
                    cursor.execute("""
                        INSERT OR REPLACE INTO cache_entries 
                        (cache_key, params_json, score, model_class, data_hash, timestamp, metadata_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        entry['cache_key'],
                        json.dumps(entry['params']),
                        entry['score'],
                        entry['model_class'],
                        entry['data_hash'],
                        entry['timestamp'],
                        json.dumps(entry['metadata']) if entry['metadata'] else None
                    ))
                else:
                    cursor.execute("""
                        INSERT OR IGNORE INTO cache_entries 
                        (cache_key, params_json, score, model_class, data_hash, timestamp, metadata_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        entry['cache_key'],
                        json.dumps(entry['params']),
                        entry['score'],
                        entry['model_class'],
                        entry['data_hash'],
                        entry['timestamp'],
                        json.dumps(entry['metadata']) if entry['metadata'] else None
                    ))
            
            conn.commit()
        
        logger.info(f"Imported {len(import_data)} cache entries from {input_path}")


class MemoryCache:
    """
    Simple in-memory cache for temporary optimization results.
    Useful for caching results within a single optimization run.
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize memory cache.
        
        Args:
            max_size: Maximum number of entries to keep in memory
        """
        self.max_size = max_size
        self.cache = {}
        self.access_order = []  # For LRU eviction
    
    def _generate_key(self, params: Dict[str, Any], data_hash: str) -> str:
        """Generate cache key."""
        params_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(f"{params_str}_{data_hash}".encode()).hexdigest()
    
    def get(self, params: Dict[str, Any], data_hash: str) -> Optional[float]:
        """Get cached result."""
        key = self._generate_key(params, data_hash)
        
        if key in self.cache:
            # Update access order for LRU
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        
        return None
    
    def put(self, params: Dict[str, Any], data_hash: str, score: float):
        """Store result in cache."""
        key = self._generate_key(params, data_hash)
        
        # Evict if at capacity
        if len(self.cache) >= self.max_size and key not in self.cache:
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]
        
        # Store new entry
        if key not in self.cache:
            self.access_order.append(key)
        else:
            # Update access order
            self.access_order.remove(key)
            self.access_order.append(key)
        
        self.cache[key] = score
    
    def clear(self):
        """Clear all cached entries."""
        self.cache.clear()
        self.access_order.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)