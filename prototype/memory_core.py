"""
Arrow-based memory storage engine

Core abstraction for storing and retrieving memories using Apache Arrow format.
"""
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import json


class MemoryStore:
    """Arrow-based memory storage"""
    
    # Schema for experiences
    EXPERIENCE_SCHEMA = pa.schema([
        ('id', pa.string()),
        ('timestamp', pa.timestamp('us')),
        ('context', pa.string()),
        ('content', pa.string()),
        ('embedding', pa.list_(pa.float32(), 1536)),
        ('metadata', pa.string()),  # JSON
    ])
    
    def __init__(self, storage_path: str = "data/memories"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.experiences_file = self.storage_path / "experiences.parquet"
    
    def store(
        self,
        memory_id: str,
        content: str,
        context: str = "",
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Store a memory"""
        
        # Prepare data
        data = {
            'id': [memory_id],
            'timestamp': [pa.scalar(datetime.now(), type=pa.timestamp('us'))],
            'context': [context],
            'content': [content],
            'embedding': [embedding or [0.0] * 1536],
            'metadata': [json.dumps(metadata or {})],
        }
        
        table = pa.table(data, schema=self.EXPERIENCE_SCHEMA)
        
        # Append to existing file or create new
        if self.experiences_file.exists():
            existing = pq.read_table(self.experiences_file)
            table = pa.concat_tables([existing, table])
        
        pq.write_table(table, self.experiences_file)
    
    def retrieve(
        self,
        memory_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Retrieve memories"""
        
        if not self.experiences_file.exists():
            return []
        
        table = pq.read_table(self.experiences_file)
        
        # Filter by ID if provided
        if memory_id:
            mask = pa.compute.equal(table['id'], memory_id)
            table = table.filter(mask)
        
        # Convert to list of dicts
        results = []
        for i in range(min(len(table), limit)):
            results.append({
                'id': table['id'][i].as_py(),
                'timestamp': table['timestamp'][i].as_py(),
                'context': table['context'][i].as_py(),
                'content': table['content'][i].as_py(),
                'embedding': table['embedding'][i].as_py(),
                'metadata': json.loads(table['metadata'][i].as_py()),
            })
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        
        if not self.experiences_file.exists():
            return {
                'total_memories': 0,
                'storage_size_bytes': 0,
            }
        
        table = pq.read_table(self.experiences_file)
        file_size = self.experiences_file.stat().st_size
        
        return {
            'total_memories': len(table),
            'storage_size_bytes': file_size,
            'storage_size_mb': file_size / (1024 * 1024),
        }
