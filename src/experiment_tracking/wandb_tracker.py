"""
Weights & Biases integration for experiment tracking
"""

import wandb
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class WandbTracker:
    """Weights & Biases experiment tracker"""

    def __init__(
        self,
        project: str,
        entity: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[list] = None,
    ):
        """
        Initialize W&B tracker

        Args:
            project: W&B project name
            entity: W&B entity (team/username)
            name: Run name
            config: Configuration dictionary
            tags: List of tags
        """
        self.run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            config=config,
            tags=tags,
        )
        logger.info(f"Initialized W&B run: {self.run.name}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to W&B"""
        wandb.log(metrics, step=step)

    def log_artifacts(self, artifact_path: str, artifact_type: str = "model"):
        """Log artifacts (models, datasets, etc.)"""
        artifact = wandb.Artifact(name=f"{artifact_type}_{wandb.run.id}", type=artifact_type)
        artifact.add_file(artifact_path)
        wandb.log_artifact(artifact)

    def finish(self):
        """Finish the W&B run"""
        wandb.finish()


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None):
    """Helper function to log metrics"""
    wandb.log(metrics, step=step)


def log_artifacts(artifact_path: str, artifact_type: str = "model"):
    """Helper function to log artifacts"""
    artifact = wandb.Artifact(name=f"{artifact_type}_{wandb.run.id}", type=artifact_type)
    artifact.add_file(artifact_path)
    wandb.log_artifact(artifact)
