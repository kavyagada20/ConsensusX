"""Data loader for the FEVER dataset supporting streaming, validation, and stratified sampling."""

import json
import random
from pathlib import Path
from typing import Generator, List, Optional, Union

from consensus_x.data.schema import FEVERLabel, FEVERRecord
from consensus_x.utils.logging import get_logger

logger = get_logger(__name__)


class FEVERDataLoader:
    """Loader and sampler for FEVER jsonl files."""

    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"FEVER data file not found at: {self.file_path}")

    def stream_records(self, limit: Optional[int] = None) -> Generator[FEVERRecord, None, None]:
        """Streams FEVER records one-by-one from the jsonl file.
        
        Args:
            limit: Maximum number of records to yield.
            
        Yields:
            FEVERRecord: Parsed and validated FEVER record.
        """
        count = 0
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    record = FEVERRecord(**data)
                    yield record
                    count += 1
                    if limit and count >= limit:
                        break
                except Exception as e:
                    logger.warning("Failed parsing line %d in %s: %s", line_no, self.file_path.name, e)

    def load_all(self, limit: Optional[int] = None) -> List[FEVERRecord]:
        """Loads records into a list."""
        return list(self.stream_records(limit=limit))

    def create_stratified_sample(
        self,
        sample_size: int = 300,
        random_seed: int = 42,
        output_path: Optional[Union[str, Path]] = None
    ) -> List[FEVERRecord]:
        """Extracts a balanced, stratified subset across all 3 FEVER labels.
        
        Args:
            sample_size: Total number of samples (divided evenly among 3 classes).
            random_seed: Random seed for deterministic reproducibility.
            output_path: Optional destination path to save sample as jsonl.
            
        Returns:
            List[FEVERRecord]: The sampled records.
        """
        target_per_class = sample_size // 3
        pools: dict[FEVERLabel, List[FEVERRecord]] = {
            FEVERLabel.SUPPORTS: [],
            FEVERLabel.REFUTES: [],
            FEVERLabel.NOT_ENOUGH_INFO: []
        }

        # Collect records by class
        for record in self.stream_records():
            label = record.label
            if label in pools:
                pools[label].append(record)

        rng = random.Random(random_seed)
        sampled: List[FEVERRecord] = []
        for label, records in pools.items():
            if len(records) < target_per_class:
                logger.warning(
                    "Only %d samples found for %s, requested %d",
                    len(records), label.value, target_per_class
                )
                sampled.extend(records)
            else:
                sampled.extend(rng.sample(records, target_per_class))

        # Shuffle sampled set deterministically so classes are intermixed
        rng.shuffle(sampled)
        logger.info("Sampled %d balanced records from %s", len(sampled), self.file_path.name)

        if output_path:
            out_file = Path(output_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, "w", encoding="utf-8") as f:
                for rec in sampled:
                    f.write(rec.model_dump_json() + "\n")
            logger.info("Saved stratified sample to: %s", out_file)

        return sampled

