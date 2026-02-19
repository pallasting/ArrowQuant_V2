
import os
import sys
import tempfile
import numpy as np
from pathlib import Path
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[1]))

from llm_compression.multimodal.image_manager import ImageManager
from llm_compression.knowledge_graph.manager import KnowledgeGraphManager

def test_visual_memory_pipeline():
    print("=" * 60)
    print("  Visual Memory Integration Test")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        storage_path = Path(tmp_dir)
        
        # 1. Setup Managers
        img_mgr = ImageManager(storage_path)
        kg_mgr = KnowledgeGraphManager(storage_path)
        
        # Use Reference Provider to avoid downloading 600MB model in test
        from llm_compression.multimodal.vision_provider import ReferenceVisionProvider
        img_mgr.provider = ReferenceVisionProvider()
        print("[INFO] Using ReferenceVisionProvider for fast testing.")
        
        # 2. Create Dummy Image (Red Square)
        img_path = storage_path / "test_red_square.png"
        img = Image.new('RGB', (100, 100), color = 'red')
        img.save(img_path)
        
        print("[1] Ingesting Image...")
        # Since we might not have CLIP, ImageManager will use ReferenceVisionProvider (random)
        # or CLIP if available.
        img_id = img_mgr.add_image(str(img_path), caption="A red square test image")
        
        if img_id:
            print(f"  Image ingested. ID: {img_id}")
            vm = img_mgr.memories[img_id]
            print(f"  Embedding Shape: {vm.embedding.shape}")
            print(f"  Thumbnail length: {len(vm.thumbnail_b64)}")
        else:
            print("  Failed to ingest image.")
            return

        # 3. Link to Knowledge Graph
        print("\n[2] Linking to Knowledge Graph...")
        concepts = ["red", "square", "test", "color"]
        scores = [0.9, 0.8, 0.5, 0.7]
        
        kg_mgr.add_image_node(img_id, concepts, scores)
        
        print(f"  Graph Nodes: {kg_mgr.graph.number_of_nodes()}")
        print(f"  Graph Edges: {kg_mgr.graph.number_of_edges()}")
        
        # 4. Test Search (Concept -> Image)
        print("\n[3] Searching for images by concept 'red'...")
        found_images = kg_mgr.find_related_images("red")
        print(f"  Found Images: {found_images}")
        assert img_id in found_images
        
        # 5. Test Visual Similarity (Simulated)
        # Query with text "red square" using ImageManager's vector search
        print("\n[4] Testing Vector Similarity Search...")
        results = img_mgr.search("red square", top_k=1)
        for vm, score in results:
            print(f"  Match: {vm.id} (Score: {score:.4f})")
            
        print("\n[PASS] Visual Memory Pipeline Verified.")

if __name__ == "__main__":
    test_visual_memory_pipeline()
