"""Data loader for the FEVER dataset supporting streaming, validation, stratified sampling, and leakage checks."""

import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Optional, Set, Tuple, Union

from src.data.schema import DatasetManifest, FEVERLabel, FEVERRecord
from src.utils.logging import get_logger

logger = get_logger(__name__)


def check_data_leakage(
    set_a_ids: Union[Set[int], Iterable[FEVERRecord]],
    set_b_ids: Union[Set[int], Iterable[FEVERRecord]],
    set_a_name: str = "Split A",
    set_b_name: str = "Split B"
) -> Set[int]:
    """Verifies that two splits or samples have zero overlapping claim IDs.
    
    Args:
        set_a_ids: Set of claim IDs or iterable of FEVERRecords.
        set_b_ids: Set of claim IDs or iterable of FEVERRecords.
        set_a_name: Descriptive name for split A.
        set_b_name: Descriptive name for split B.
        
    Returns:
        Set[int]: Set of intersecting IDs (empty if no leakage).
        
    Raises:
        ValueError: If data leakage (overlapping IDs) is detected.
    """
    ids_a = {r.id if isinstance(r, FEVERRecord) else r for r in set_a_ids}
    ids_b = {r.id if isinstance(r, FEVERRecord) else r for r in set_b_ids}

    leakage = ids_a.intersection(ids_b)
    if leakage:
        err_msg = (
            f"DATA LEAKAGE DETECTED between {set_a_name} and {set_b_name}! "
            f"Found {len(leakage)} overlapping claim IDs (e.g. {list(leakage)[:5]})."
        )
        logger.error(err_msg)
        raise ValueError(err_msg)

    logger.info(
        "Zero data leakage verified between %s (%d items) and %s (%d items).",
        set_a_name, len(ids_a), set_b_name, len(ids_b)
    )
    return leakage


class FEVERDataLoader:
    """Loader, sampler, and split manager for FEVER jsonl files."""

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
        """Loads records into an in-memory list."""
        return list(self.stream_records(limit=limit))

    def get_label_distribution(self, limit: Optional[int] = None) -> Dict[str, int]:
        """Counts occurrences of each label in the dataset."""
        counter = Counter()
        for record in self.stream_records(limit=limit):
            counter[record.label.value] += 1
        return dict(counter)

    def create_stratified_sample(
        self,
        sample_size: int = 300,
        random_seed: int = 42,
        output_path: Optional[Union[str, Path]] = None,
        create_manifest: bool = True
    ) -> List[FEVERRecord]:
        """Extracts a balanced, stratified subset across all 3 FEVER labels.
        
        Args:
            sample_size: Total number of samples (divided evenly among 3 classes).
            random_seed: Random seed for deterministic reproducibility.
            output_path: Optional destination path to save sample as jsonl.
            create_manifest: Whether to generate a provenance manifest JSON file.
            
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
            # Sort deterministically by record id before sampling
            records.sort(key=lambda r: r.id)
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

            if create_manifest:
                dist = dict(Counter(r.label.value for r in sampled))
                manifest = DatasetManifest(
                    source_file=str(self.file_path),
                    output_file=str(out_file),
                    sample_size=len(sampled),
                    random_seed=random_seed,
                    label_distribution=dist,
                    claim_ids=[r.id for r in sampled],
                    description=f"Stratified sample of {len(sampled)} claims from {self.file_path.name}"
                )
                manifest_path = out_file.with_suffix(".manifest.json")
                with open(manifest_path, "w", encoding="utf-8") as mf:
                    mf.write(manifest.model_dump_json(indent=2))
                logger.info("Saved dataset provenance manifest to: %s", manifest_path)

        return sampled
