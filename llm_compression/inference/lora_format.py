
"""
AI-OS Arrow-Native LoRA Format Definition.

Defines the data structure for LoRA cards that can be zero-copy loaded.
Uses PyArrow for efficient storage and retrieval.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np
import pyarrow as pa
import io

@dataclass
class LoRACard:
    """
    Represents a loaded LoRA adapter card.
    Ready for injection into the inference core.
    """
    name: str
    rank: int
    alpha: float
    target_modules: List[str]  # e.g. ["attention.q_proj", "attention.v_proj"]
    weights_A: Dict[str, np.ndarray] # {module_name: weight_matrix}
    weights_B: Dict[str, np.ndarray] # {module_name: weight_matrix}
    metadata: Dict[str, str] = field(default_factory=dict) # version, author, description

class LoRAFormat:
    """
    Handles serialization/deserialization of LoRA cards to/from Arrow format.
    """

    @staticmethod
    def save(card: LoRACard, path: str):
        """Unified Schema Saving."""
        # Metadata repeated for every row is inefficient but simple for Arrow IPC
        # Better: Just write metadata to the file custom metadata footer? 
        # Yes, Arrow file key-value metadata.
        
        custom_meta = {
            "lora_name": card.name,
            "lora_rank": str(card.rank),
            "lora_alpha": str(card.alpha),
            "target_modules": ",".join(card.target_modules)
        }
        # Add extra metadata (all values must be strings for Arrow)
        for k, v in card.metadata.items():
            custom_meta[str(k)] = str(v)
        
        modules = []
        w_a_list = []
        w_b_list = []
        s_a_list = []
        s_b_list = []
        dtype_list = []
        
        all_keys = sorted(card.weights_A.keys())
        for key in all_keys:
            wa = card.weights_A[key]
            wb = card.weights_B[key]
            
            modules.append(key)
            w_a_list.append(wa.tobytes())
            w_b_list.append(wb.tobytes())
            s_a_list.append(wa.shape)
            s_b_list.append(wb.shape)
            dtype_list.append(str(wa.dtype))

        schema = pa.schema([
            ("module", pa.string()),
            ("weight_a", pa.binary()),
            ("weight_b", pa.binary()),
            ("shape_a", pa.list_(pa.int32())),
            ("shape_b", pa.list_(pa.int32())),
            ("dtype", pa.string())
        ], metadata=custom_meta)
        
        batch = pa.RecordBatch.from_arrays(
            [
                pa.array(modules),
                pa.array(w_a_list),
                pa.array(w_b_list),
                pa.array(s_a_list),
                pa.array(s_b_list),
                pa.array(dtype_list)
            ],
            schema=schema
        )
        
        with pa.OSFile(path, 'wb') as sink:
            with pa.ipc.new_file(sink, schema) as writer:
                writer.write(batch)

    @staticmethod
    def load(path: str) -> LoRACard:
        """Load LoRACard from Arrow file (Zero-copy where possible)."""
        with pa.memory_map(path, 'r') as source:
            reader = pa.ipc.open_file(source)
            table = reader.read_all()
            
            # Read Metadata
            meta = table.schema.metadata
            name = meta.get(b'lora_name', b'unknown').decode('utf-8')
            rank = int(meta.get(b'lora_rank', b'0').decode('utf-8'))
            alpha = float(meta.get(b'lora_alpha', b'1.0').decode('utf-8'))
            targets = meta.get(b'target_modules', b'').decode('utf-8').split(',')
            
            # Reconstruct Weights
            # Arrow table columns
            modules = table.column("module").to_pylist()
            w_a_bins = table.column("weight_a")
            w_b_bins = table.column("weight_b")
            s_a_lists = table.column("shape_a").to_pylist()
            s_b_lists = table.column("shape_b").to_pylist()
            dtypes = table.column("dtype").to_pylist()
            
            weights_A = {}
            weights_B = {}
            
            for i, mod_key in enumerate(modules):
                dtype = np.dtype(dtypes[i])
                
                # Zero-copy view from Arrow buffer? 
                # to_numpy() on binary column might copy?
                # Using memoryview on buffer is safer for zero-copy
                
                buf_a = w_a_bins[i].as_buffer()
                buf_b = w_b_bins[i].as_buffer()
                
                arr_a = np.frombuffer(buf_a, dtype=dtype).reshape(s_a_lists[i])
                arr_b = np.frombuffer(buf_b, dtype=dtype).reshape(s_b_lists[i])
                
                weights_A[mod_key] = arr_a
                weights_B[mod_key] = arr_b
            
            # Extract generic metadata
            generic_meta = {}
            for k, v in meta.items():
                k_str = k.decode('utf-8')
                if k_str not in ["lora_name", "lora_rank", "lora_alpha", "target_modules"]:
                    generic_meta[k_str] = v.decode('utf-8')

            return LoRACard(
                name=name,
                rank=rank,
                alpha=alpha,
                target_modules=targets,
                weights_A=weights_A,
                weights_B=weights_B,
                metadata=generic_meta
            )
