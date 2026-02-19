
"""
AI-OS Skill Factory.

The "Industrial Revolution" of AI-OS.
A proactive, automated system for mass-producing LoRA skills from:
1. Raw datasets (HuggingFace Parquet)
2. Large Foundation Models (e.g. Llama-3-70B)
3. Aggregated user interaction logs

Features:
- **Nightly Batch Processing**: Runs heavy training jobs when system is idle.
- **Model Rotation**: Sequentially loads massive models, extracts specific domain skills, then unloads them.
- **Skill Queue**: Persistent backlog of skills to acquire.

Usage:
    factory = SkillFactory(arrow_engine, "./factory_workspace")
    factory.add_task("learn_medical_imaging", dataset_path="...")
    factory.start_worker()
"""

import time
import json
import threading
import logging
from typing import Dict, List, Optional
from pathlib import Path
from dataclasses import dataclass, asdict

from llm_compression.evolution.skill_distiller import SkillDistiller, NodeTier
from llm_compression.evolution.cloud_distiller import CloudDistiller
from llm_compression.evolution.lora_trainer import LoRATrainer

logger = logging.getLogger(__name__)


@dataclass
class SkillTask:
    """A unit of work for the factory."""
    id: str
    name: str
    type: str  # "distill_cloud", "train_dataset", "extract_model"
    status: str  # "pending", "running", "completed", "failed"
    priority: int  # 1 (low) to 10 (high)
    params: Dict
    created_at: float
    completed_at: Optional[float] = None
    error: Optional[str] = None


class SkillFactory:
    """
    Manager for automated skill production.
    """
    
    def __init__(
        self,
        engine,
        workspace_dir: str = "./skill_factory",
        max_workers: int = 1
    ):
        self.engine = engine
        self.workspace = Path(workspace_dir)
        self.workspace.mkdir(parents=True, exist_ok=True)
        
        self.tasks_file = self.workspace / "tasks.jsonl"
        self.tasks: Dict[str, SkillTask] = self._load_tasks()
        
        self.max_workers = max_workers
        self._stop_event = threading.Event()
        self._worker_thread = None
        
        # Tools
        self.trainer = LoRATrainer(engine, output_dir=str(self.workspace / "products"))
        # Distiller reused from engine if available, else new one
        self.distiller = getattr(engine, 'distiller', None) or SkillDistiller(engine)
        
    def add_task(
        self,
        name: str,
        task_type: str,
        priority: int = 5,
        **params
    ) -> str:
        """Add a new production task to the queue."""
        import uuid
        task_id = str(uuid.uuid4())[:8]
        
        task = SkillTask(
            id=task_id,
            name=name,
            type=task_type,
            status="pending",
            priority=priority,
            params=params,
            created_at=time.time()
        )
        
        self.tasks[task_id] = task
        self._save_task(task)
        logger.info(f"Factory task added: {name} (ID: {task_id})")
        return task_id

    def start_worker(self):
        """Start the background worker thread."""
        if self._worker_thread and self._worker_thread.is_alive():
            logger.warning("Factory worker already running.")
            return
            
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._process_queue,
            name="skill-factory-worker",
            daemon=True
        )
        self._worker_thread.start()
        logger.info("Skill Factory worker started.")

    def stop_worker(self):
        """Stop the background worker."""
        self._stop_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
            logger.info("Skill Factory worker stopped.")

    def _process_queue(self):
        """Main worker loop."""
        while not self._stop_event.is_set():
            # Find highest priority pending task
            pending = [
                t for t in self.tasks.values() 
                if t.status == "pending"
            ]
            
            if not pending:
                # Sleep and wait
                time.sleep(5)
                continue
                
            # Sort by priority (desc), then creation time (asc)
            pending.sort(key=lambda x: (-x.priority, x.created_at))
            task = pending[0]
            
            self._run_task(task)
            
    def _run_task(self, task: SkillTask):
        """Execute a single task."""
        logger.info(f"Factory executing task: {task.name} ({task.type})")
        task.status = "running"
        self._save_task(task)  # Checkpoint status
        
        try:
            start_time = time.time()
            
            if task.type == "distill_cloud":
                self._run_distill_cloud(task)
            elif task.type == "train_dataset":
                self._run_train_dataset(task)
            elif task.type == "extract_model":
                self._run_extract_model(task)
            else:
                raise ValueError(f"Unknown task type: {task.type}")
                
            task.status = "completed"
            task.completed_at = time.time()
            duration = task.completed_at - start_time
            logger.info(f"Task {task.id} COMPLETED in {duration:.1f}s")
            
        except Exception as e:
            logger.error(f"Task {task.id} FAILED: {e}")
            task.status = "failed"
            task.error = str(e)
            
        self._save_task(task)

    # ──────────────────────────────────────────
    # Task Implementation Handlers
    # ──────────────────────────────────────────

    def _run_distill_cloud(self, task: SkillTask):
        """Execute cloud distillation."""
        topic = task.params.get("topic")
        provider_config = task.params.get("provider", {})
        
        # Determine provider
        # For simplicity, we assume engine has configured providers
        # or we instantiate a mock one for testing
        if "mock" in provider_config:
            from llm_compression.evolution.cloud_distiller import MockCloudProvider
            provider = MockCloudProvider()
        else:
            # TODO: instantiate real provider from config
            raise NotImplementedError("Real cloud provider instantiation pending")
            
        cd = CloudDistiller(
            self.engine,
            output_dir=str(self.workspace / "products")
        )
        cd.distill_topic(
            topic=topic, 
            provider=provider,
            skill_name=task.name
        )

    def _run_train_dataset(self, task: SkillTask):
        """Execute dataset training."""
        dataset_path = task.params.get("dataset_path")
        # Load dataset (mock jsonl loading)
        data = []
        with open(dataset_path, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        
        self.trainer.train_qa(
            qa_pairs=data,
            skill_name=task.name,
            epochs=task.params.get("epochs", 3)
        )

    def _run_extract_model(self, task: SkillTask):
        """Execute model extraction (Model Rotation)."""
        # This is the "big one":
        # 1. Unload current engine model
        # 2. detailed extraction logic
        # 3. Reload original model
        # For Phase 9 MVP: Just use current engine
        queries = task.params.get("queries", [])
        self.distiller.extract_skill_from_engine(
            name=task.name,
            test_queries=queries,
            description=f"Extracted from {self.engine.metadata.get('model_name')}"
        )

    # ──────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────

    def _load_tasks(self) -> Dict[str, SkillTask]:
        """Load tasks from disk."""
        tasks = {}
        if not self.tasks_file.exists():
            return tasks
            
        try:
            with open(self.tasks_file, 'r') as f:
                for line in f:
                    if not line.strip(): continue
                    data = json.loads(line)
                    task = SkillTask(**data)
                    tasks[task.id] = task
        except Exception as e:
            logger.error(f"Failed to load tasks: {e}")
            
        return tasks

    def check_autonomic_triggers(self):
        """
        Check for autonomic triggers (Sleep Consolidation).
        
        Logic:
        1. Check system load (is it idle?).
        2. Check for pending QA logs (is there something to learn?).
        3. If both true, create a task to distill the logs.
        """
        # 1. Check Load: If factory is busy, don't interrupt
        pending_count = sum(1 for t in self.tasks.values() if t.status in ("pending", "running"))
        if pending_count > 0:
            return

        # 2. Check pending QA logs
        # SkillDistiller saves to [lora_output_dir]/qa_log.jsonl
        qa_log_path = self.distiller.output_dir / "qa_log.jsonl"
        if not qa_log_path.exists():
            return

        try:
            # Check line count
            with open(qa_log_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Threshold: Accumulate at least 10 observations before dreaming
            if len(lines) < 10:
                return

            logger.info(f"Autonomic System: Consolidating {len(lines)} memories...")

            # 3. Rotate Log: Move to processing area to prevent race conditions
            timestamp = int(time.time())
            processing_path = self.workspace / f"memory_consolidation_{timestamp}.jsonl"
            
            # Atomic move (rename)
            import shutil
            shutil.move(str(qa_log_path), str(processing_path))

            # 4. Create "Dream" Task
            task_name = f"autonomic_consolidation_{timestamp}"
            self.add_task(
                name=task_name,
                task_type="train_dataset",
                priority=2, # Low priority (background dreaming)
                dataset_path=str(processing_path),
                epochs=3
            )
            logger.info(f"Autonomic System: Created consolidation task '{task_name}'")

        except Exception as e:
            logger.error(f"Autonomic Check Failed: {e}")

    def _save_task(self, task: SkillTask):
        """Append task state to log (append-only for safety)."""
        # We append status updates. The loader will just read sequential updates
        # But for simplistic "current state" view, we might want a rewrite or DB.
        # MVP: Append to jsonl. Reader logic assumes last entry prevails?
        # Actually _load_tasks reads sequentially, so last entry for an ID overwrites.
        with open(self.tasks_file, 'a') as f:
            f.write(json.dumps(asdict(task)) + "\n")
