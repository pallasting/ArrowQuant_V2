"""
Phase 3 Integration Diagnostic
Tests: SemanticIndexDB, SemanticIndexer, VectorSearch, MemorySearch, BackgroundQueue
All output written to file to avoid Windows encoding issues.
"""
import asyncio
import traceback
import tempfile
import os
import numpy as np

os.environ["ARROW_MODEL_PATH"] = r"m:\Documents\ai-os-memory\models\minilm"

LOG = r"m:\Documents\ai-os-memory\diag_out.txt"

with open(LOG, "w", encoding="utf-8") as f:
    def log(msg):
        f.write(msg + "\n")
        f.flush()

    log("=== Phase 3: Semantic Index Modules Diagnostic ===")
    log("")

    # --- 1. SemanticIndexDB ---
    log("--- 1. SemanticIndexDB ---")
    try:
        from llm_compression.semantic_index_db import SemanticIndexDB
        log("  Import OK")

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SemanticIndexDB(index_path=tmpdir)
            log("  Created OK")

            # Add entries
            from datetime import datetime
            emb1 = np.random.rand(384).astype(np.float32)
            emb2 = np.random.rand(384).astype(np.float32)
            db.add_entry("mem_1", "knowledge", emb1, datetime.now())
            db.add_entry("mem_2", "knowledge", emb2, datetime.now())
            log(f"  add_entry OK, categories={db.get_categories()}")
            log(f"  category size={db.get_category_size('knowledge')}")

            # Query
            query_emb = emb1.copy()  # should match mem_1 exactly
            results = db.query("knowledge", query_emb, top_k=2, threshold=0.0)
            log(f"  query OK: {len(results)} results, top={results[0]['memory_id']} sim={results[0]['similarity']:.4f}")
            assert results[0]["memory_id"] == "mem_1", f"Expected mem_1, got {results[0]['memory_id']}"

            log("  SemanticIndexDB: PASS")
    except Exception as e:
        log(f"  FAIL: {e}")
        log(traceback.format_exc())

    log("")

    # --- 2. SemanticIndexer ---
    log("--- 2. SemanticIndexer ---")
    try:
        from llm_compression.semantic_indexer import SemanticIndexer
        from llm_compression.semantic_index_db import SemanticIndexDB
        from llm_compression.embedding_provider import ArrowEngineProvider
        from llm_compression.arrow_storage import ArrowStorage
        log("  Imports OK")

        with tempfile.TemporaryDirectory() as tmpdir:
            provider = ArrowEngineProvider(model_path=r"m:\Documents\ai-os-memory\models\minilm")
            storage = ArrowStorage(storage_path=tmpdir)
            index_db = SemanticIndexDB(index_path=os.path.join(tmpdir, "index"))
            indexer = SemanticIndexer(
                embedding_provider=provider,
                storage=storage,
                index_db=index_db
            )
            log("  Created OK")

            # Index a memory dict
            memory = {
                "memory_id": "test_mem_1",
                "category": "knowledge",
                "context": "Python is a high-level programming language.",
                "timestamp": "2024-01-01T00:00:00"
            }
            indexer.index_memory(memory)
            log(f"  index_memory OK, index size={index_db.get_category_size('knowledge')}")

            # Batch index
            memories = [
                {"memory_id": f"mem_{i}", "category": "knowledge",
                 "context": f"Memory content {i}", "timestamp": "2024-01-01T00:00:00"}
                for i in range(5)
            ]
            indexer.batch_index(memories, batch_size=3)
            log(f"  batch_index OK, index size={index_db.get_category_size('knowledge')}")

            log("  SemanticIndexer: PASS")
    except Exception as e:
        log(f"  FAIL: {e}")
        log(traceback.format_exc())

    log("")

    # --- 3. VectorSearch ---
    log("--- 3. VectorSearch ---")
    try:
        from llm_compression.vector_search import VectorSearch
        from llm_compression.semantic_index_db import SemanticIndexDB
        from llm_compression.semantic_indexer import SemanticIndexer
        from llm_compression.embedding_provider import ArrowEngineProvider
        from llm_compression.arrow_storage import ArrowStorage
        log("  Imports OK")

        with tempfile.TemporaryDirectory() as tmpdir:
            provider = ArrowEngineProvider(model_path=r"m:\Documents\ai-os-memory\models\minilm")
            storage = ArrowStorage(storage_path=tmpdir)
            index_db = SemanticIndexDB(index_path=os.path.join(tmpdir, "index"))
            indexer = SemanticIndexer(provider, storage, index_db)

            # Index some memories
            memories = [
                {"memory_id": "py_mem", "category": "knowledge",
                 "context": "Python is a programming language.", "timestamp": "2024-01-01T00:00:00"},
                {"memory_id": "ml_mem", "category": "knowledge",
                 "context": "Machine learning uses neural networks.", "timestamp": "2024-01-01T00:00:00"},
                {"memory_id": "db_mem", "category": "knowledge",
                 "context": "Databases store structured data.", "timestamp": "2024-01-01T00:00:00"},
            ]
            indexer.batch_index(memories)

            vs = VectorSearch(embedding_provider=provider, storage=storage, index_db=index_db)
            log("  Created OK")

            results = vs.search("programming language", category="knowledge", top_k=2)
            log(f"  search OK: {len(results)} results")
            for r in results:
                log(f"    {r.memory_id}: sim={r.similarity:.4f}")
            assert len(results) > 0, "Expected at least 1 result"

            log("  VectorSearch: PASS")
    except Exception as e:
        log(f"  FAIL: {e}")
        log(traceback.format_exc())

    log("")

    # --- 4. MemorySearch ---
    log("--- 4. MemorySearch ---")
    try:
        from llm_compression.memory_search import MemorySearch, SearchMode
        from llm_compression.vector_search import VectorSearch
        from llm_compression.semantic_index_db import SemanticIndexDB
        from llm_compression.semantic_indexer import SemanticIndexer
        from llm_compression.embedding_provider import ArrowEngineProvider
        from llm_compression.arrow_storage import ArrowStorage
        log("  Imports OK")

        with tempfile.TemporaryDirectory() as tmpdir:
            provider = ArrowEngineProvider(model_path=r"m:\Documents\ai-os-memory\models\minilm")
            storage = ArrowStorage(storage_path=tmpdir)
            index_db = SemanticIndexDB(index_path=os.path.join(tmpdir, "index"))
            indexer = SemanticIndexer(provider, storage, index_db)
            memories = [
                {"memory_id": "py_mem", "category": "knowledge",
                 "context": "Python is a programming language.", "timestamp": "2024-01-01T00:00:00"},
                {"memory_id": "ml_mem", "category": "knowledge",
                 "context": "Machine learning uses neural networks.", "timestamp": "2024-02-01T00:00:00"},
            ]
            indexer.batch_index(memories)

            vs = VectorSearch(provider, storage, index_db)
            ms = MemorySearch(vector_search=vs, storage=storage)
            log("  Created OK")

            # Semantic search
            results = ms.search("programming", category="knowledge", mode=SearchMode.SEMANTIC, top_k=2)
            log(f"  SEMANTIC search OK: {len(results)} results")

            log("  MemorySearch: PASS")
    except Exception as e:
        log(f"  FAIL: {e}")
        log(traceback.format_exc())

    log("")

    # --- 5. BackgroundQueue ---
    log("--- 5. BackgroundQueue ---")
    try:
        from llm_compression.background_queue import BackgroundQueue
        from llm_compression.semantic_indexer import SemanticIndexer
        from llm_compression.semantic_index_db import SemanticIndexDB
        from llm_compression.embedding_provider import ArrowEngineProvider
        from llm_compression.arrow_storage import ArrowStorage
        log("  Imports OK")

        async def test_bg_queue():
            with tempfile.TemporaryDirectory() as tmpdir:
                provider = ArrowEngineProvider(model_path=r"m:\Documents\ai-os-memory\models\minilm")
                storage = ArrowStorage(storage_path=tmpdir)
                index_db = SemanticIndexDB(index_path=os.path.join(tmpdir, "index"))
                indexer = SemanticIndexer(provider, storage, index_db)

                queue = BackgroundQueue(indexer=indexer, batch_size=2)
                await queue.start()
                log("  Started OK")

                memories = [
                    {"memory_id": f"bg_mem_{i}", "category": "knowledge",
                     "context": f"Background memory {i}", "timestamp": "2024-01-01T00:00:00"}
                    for i in range(4)
                ]
                await queue.submit_batch(memories)
                log(f"  Submitted {len(memories)} items, queue_size={queue.get_queue_size()}")

                await queue.wait_until_empty(timeout=30.0)
                await queue.stop()
                log(f"  Processed OK, index_size={index_db.get_category_size('knowledge')}")

        asyncio.run(test_bg_queue())
        log("  BackgroundQueue: PASS")
    except Exception as e:
        log(f"  FAIL: {e}")
        log(traceback.format_exc())

    log("")
    log("=== Phase 3 Diagnostic Complete ===")
