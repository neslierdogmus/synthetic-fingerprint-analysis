from __future__ import annotations

from collections import defaultdict


class HPSelector:
    """Ranks HP configurations and selects top-k or best."""

    @staticmethod
    def rank_fc_foe(
        scores: dict[tuple[int, int], float],
        n_configs: int,
        top_k: int = 3,
    ) -> list[int]:
        """
        Returns list of top_k hp_config indices, best first.
        scores: {(fold_idx, hp_idx): metric_value}
        Higher metric = better.
        """
        sums: dict[int, float] = defaultdict(float)
        counts: dict[int, int] = defaultdict(int)
        for (fold, hp_idx), score in scores.items():
            sums[hp_idx] += score
            counts[hp_idx] += 1
        means = {hp_idx: sums[hp_idx] / counts[hp_idx] for hp_idx in sums}
        ranked = sorted(means.keys(), key=lambda k: means[k], reverse=True)
        return ranked[:top_k]

    @staticmethod
    def rank_fmd(
        scores: dict[tuple[int, int, float, int], float],
        top_k: int = 3,
    ) -> list[dict]:
        """
        Ranks all (hp_idx, threshold, radius) triplets jointly.
        scores: {(fold_idx, hp_idx, threshold, radius): f1_score}
        Returns top_k triplets as list of dicts:
          [{"hp_config": idx, "nms_threshold": t, "nms_radius": r}, ...]
        """
        sums: dict[tuple, float] = defaultdict(float)
        counts: dict[tuple, int] = defaultdict(int)
        for (fold, hp_idx, t, r), score in scores.items():
            key = (hp_idx, t, r)
            sums[key] += score
            counts[key] += 1
        means = {key: sums[key] / counts[key] for key in sums}
        ranked = sorted(means.keys(), key=lambda k: means[k], reverse=True)
        return [
            {"hp_config": hp_idx, "nms_threshold": t, "nms_radius": r}
            for (hp_idx, t, r) in ranked[:top_k]
        ]

    @staticmethod
    def select_best_fc_foe(scores: dict, n_configs: int) -> int:
        """Returns single best hp_config index."""
        return HPSelector.rank_fc_foe(scores, n_configs, top_k=1)[0]

    @staticmethod
    def select_best_fmd(scores: dict) -> dict:
        """Returns single best triplet dict."""
        return HPSelector.rank_fmd(scores, top_k=1)[0]
