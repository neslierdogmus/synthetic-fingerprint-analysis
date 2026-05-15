import itertools
import json
import random
import sys
import argparse
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_SRC_ROOT = Path(__file__).resolve().parent

for import_path in (PROJECT_ROOT, TASK_SRC_ROOT):
    import_path_str = str(import_path)
    if import_path_str not in sys.path:
        sys.path.insert(0, import_path_str)

from fc_experiment import FCExperiment
from fc_ground_truth import FCGroundTruth


class _TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _float_to_token(value: float) -> str:
    return format(value, '.0e').replace('+', '').replace('-', 'm')


def _merge_overrides(*overrides):
    merged = OmegaConf.create({})
    for override in overrides:
        merged = OmegaConf.merge(merged, OmegaConf.create(override or {}))
    merged_container = OmegaConf.to_container(merged, resolve=True)
    if not isinstance(merged_container, dict):
        raise ValueError('Merged overrides must resolve to a mapping.')
    return merged_container


def _save_json(file_path: Path, payload):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open('w', encoding='utf-8') as file_handle:
        json.dump(payload, file_handle, indent=2, sort_keys=True)


def _load_json(file_path: Path):
    with file_path.open('r', encoding='utf-8') as file_handle:
        return json.load(file_handle)


def _load_fc_primary_group_labels(dataset_root: str, group_ids):
    dataset_root_path = Path(dataset_root)
    group_labels = {}
    for group_id in group_ids:
        fp_id = int(group_id)
        fp_folder = dataset_root_path / 'png_txt' / f'figs_{fp_id // 250}'
        fp_file_id = str(fp_id + 1).zfill(4)
        gt_paths = sorted(fp_folder.glob(f'*{fp_file_id}*.txt'))
        if not gt_paths:
            raise FileNotFoundError(f'Could not find ground-truth file for fingerprint group {group_id}.')
        group_labels[group_id] = FCGroundTruth.from_file(gt_paths[0]).fp_class
    return group_labels


def _sample_group_subset(group_ids, group_labels, fraction: float, seed: int):
    if fraction >= 1.0:
        return sorted(group_ids)

    rng = random.Random(seed)
    grouped_ids = {}
    for group_id in group_ids:
        grouped_ids.setdefault(group_labels[group_id], []).append(group_id)

    selected_group_ids = []
    for label, label_group_ids in grouped_ids.items():
        del label
        label_group_ids = sorted(label_group_ids)
        rng.shuffle(label_group_ids)
        subset_size = max(1, int(round(len(label_group_ids) * fraction)))
        selected_group_ids.extend(label_group_ids[:subset_size])

    return sorted(selected_group_ids)


def _aggregate_fold_results(fold_results):
    accuracies = [float(result['accuracy']) for result in fold_results]
    losses = [float(result['loss']) for result in fold_results]
    return {
        'fold_results': fold_results,
        'mean_accuracy': float(np.mean(accuracies)),
        'std_accuracy': float(np.std(accuracies)),
        'mean_loss': float(np.mean(losses)),
        'std_loss': float(np.std(losses)),
    }


class FCHyperparameterSearch:
    def __init__(self, base_config_path: str, hp_search_config_path: str):
        self.base_config_path = str(base_config_path)
        self.hp_search_config_path = str(hp_search_config_path)
        self.hp_search_config = OmegaConf.load(hp_search_config_path)
        hp_search_config_data = OmegaConf.to_container(self.hp_search_config, resolve=True)
        if not isinstance(hp_search_config_data, dict):
            raise ValueError('HP search config must resolve to a mapping.')

        base_overrides_data = hp_search_config_data.get('base_overrides', {})
        if not isinstance(base_overrides_data, dict):
            raise ValueError('base_overrides must resolve to a mapping.')

        self.base_overrides = dict(base_overrides_data)
        self.search_config = self.hp_search_config.search
        self.base_experiment = FCExperiment(self.base_config_path, config_overrides=self.base_overrides)
        self.base_experiment.load_or_create_split_definition(force_rebuild=False)

        self.group_labels = _load_fc_primary_group_labels(
            self.base_experiment.config.dataset.root,
            self.base_experiment.fp_group_ids,
        )
        self.artifacts_root = Path(
            self.search_config.get(
                'artifacts_root',
                str(Path(self.base_experiment.output_dir) / 'hp_search'),
            )
        )
        self.results_path = self.base_experiment.results_dir / str(
            self.search_config.get('results_filename', 'hp_search.json')
        )
        self.log_path = self.base_experiment.output_dir / 'logs' / str(
            self.search_config.get('log_filename', 'hp_search.log')
        )
        self.subset_splits_path = self.base_experiment.splits_dir / str(
            self.search_config.get('subset_splits_filename', 'hp_search_splits.json')
        )
        self.legacy_results_root = self.base_experiment.results_dir / 'tuning'
        self.legacy_artifacts_root = Path(self.base_experiment.output_dir) / 'tuning'
        self.subset_groups_by_fold = self._create_subset_groups_by_fold()

    def _create_results_payload(self):
        return {
            'base_config_path': self.base_config_path,
            'hp_search_config_path': self.hp_search_config_path,
            'base_split_file': str(self.base_experiment.get_split_file_path()),
            'subset_split_file': str(self.subset_splits_path),
            'artifacts_root': str(self.artifacts_root),
            'search_space': {
                'optimizer_name': str(self.search_config.get('optimizer_name', 'AdamW')),
                'learning_rates': [float(value) for value in self.search_config.learning_rates],
                'weight_decays': [float(value) for value in self.search_config.weight_decays],
                'use_cosine_annealing': [bool(value) for value in self.search_config.use_cosine_annealing],
                'cosine_annealing': OmegaConf.to_container(
                    self.search_config.get('cosine_annealing', {}),
                    resolve=True,
                ),
                'subset': OmegaConf.to_container(self.search_config.subset, resolve=True),
                'rerun': OmegaConf.to_container(self.search_config.rerun, resolve=True),
            },
            'subset_results': [],
            'full_results': [],
            'selected_config_id': None,
            'selected_test_metrics': None,
        }

    def _load_results_payload(self):
        if self.results_path.exists():
            payload = _load_json(self.results_path)
        else:
            payload = self._create_results_payload()

        payload.setdefault('subset_results', [])
        payload.setdefault('full_results', [])
        payload.setdefault('selected_config_id', None)
        payload.setdefault('selected_test_metrics', None)
        return payload

    def _save_results_payload(self, payload):
        _save_json(self.results_path, payload)

    @staticmethod
    def _stage_results_key(stage: str) -> str:
        if stage == 'subset':
            return 'subset_results'
        if stage == 'full':
            return 'full_results'
        raise ValueError(f'Unsupported search stage: {stage}')

    def _find_stage_result(self, payload, stage: str, config_id: str):
        for result in payload.get(self._stage_results_key(stage), []):
            if result.get('config_id') == config_id:
                return result
        return None

    def _upsert_stage_result(self, payload, stage: str, result):
        stage_key = self._stage_results_key(stage)
        existing_results = [
            item for item in payload.get(stage_key, [])
            if item.get('config_id') != result['config_id']
        ]
        existing_results.append(result)
        payload[stage_key] = existing_results

    def _legacy_candidate_result_paths(self, candidate, stage: str):
        config_id = candidate['config_id']
        return [
            self.legacy_results_root / stage / f'{config_id}.json',
            self.legacy_artifacts_root / stage / config_id / 'search_result.json',
        ]

    def _load_cached_candidate_result(self, candidate, stage: str):
        payload = self._load_results_payload()
        cached_result = self._find_stage_result(payload, stage, candidate['config_id'])
        if cached_result is not None:
            print(f"Reusing cached {stage} result for {candidate['config_id']}")
            return cached_result

        for legacy_result_path in self._legacy_candidate_result_paths(candidate, stage):
            if not legacy_result_path.exists():
                continue

            cached_result = _load_json(legacy_result_path)
            self._save_candidate_result(candidate, stage, cached_result)
            print(f"Migrated legacy {stage} result for {candidate['config_id']}")
            return cached_result

        return None

    def _save_candidate_result(self, candidate, stage: str, result):
        payload = self._load_results_payload()
        self._upsert_stage_result(payload, stage, result)
        self._save_results_payload(payload)

    def _save_final_test_result(self, candidate, test_metrics):
        payload = self._load_results_payload()
        payload['selected_config_id'] = candidate['config_id']
        payload['selected_test_metrics'] = test_metrics
        self._save_results_payload(payload)

    def _load_cached_final_test_result(self, candidate):
        payload = self._load_results_payload()
        if (
            payload.get('selected_config_id') == candidate['config_id'] and
            payload.get('selected_test_metrics') is not None
        ):
            print(f"Reusing cached final test result for {candidate['config_id']}")
            return payload['selected_test_metrics']

        legacy_paths = [
            self.legacy_results_root / 'final' / f"{candidate['config_id']}.json",
            self.legacy_artifacts_root / 'final' / candidate['config_id'] / 'results' / 'results_test.json',
        ]
        for legacy_path in legacy_paths:
            if not legacy_path.exists():
                continue
            legacy_payload = _load_json(legacy_path)
            legacy_metrics = legacy_payload.get('test_metrics', legacy_payload)
            self._save_final_test_result(candidate, legacy_metrics)
            print(f"Migrated legacy final test result for {candidate['config_id']}")
            return legacy_metrics

        return None

    def _save_subset_splits(self):
        _save_json(
            self.subset_splits_path,
            {
                'base_split_file': str(self.base_experiment.get_split_file_path()),
                'seed': self.base_experiment.seed,
                'subset': {
                    'train_fraction': float(self.search_config.subset.get('train_fraction', 1.0)),
                    'val_fraction': float(self.search_config.subset.get('val_fraction', 1.0)),
                },
                'subset_groups_by_fold': self.subset_groups_by_fold,
            },
        )

    def _create_subset_groups_by_fold(self):
        subset_groups = {}
        subset_config = self.search_config.subset
        train_fraction = float(subset_config.get('train_fraction', 1.0))
        val_fraction = float(subset_config.get('val_fraction', 1.0))

        for fold_idx in range(self.base_experiment.num_folds):
            train_group_ids, val_group_ids = self.base_experiment.get_fold_group_ids(fold_idx)
            subset_groups[str(fold_idx)] = {
                'train': _sample_group_subset(
                    train_group_ids,
                    self.group_labels,
                    train_fraction,
                    self.base_experiment.seed + fold_idx,
                ),
                'val': _sample_group_subset(
                    val_group_ids,
                    self.group_labels,
                    val_fraction,
                    self.base_experiment.seed + 1000 + fold_idx,
                ),
            }

        return subset_groups

    def _build_candidates(self):
        optimizer_name = str(self.search_config.get('optimizer_name', 'AdamW'))
        learning_rates = [float(value) for value in self.search_config.learning_rates]
        weight_decays = [float(value) for value in self.search_config.weight_decays]
        use_cosine_values = [bool(value) for value in self.search_config.use_cosine_annealing]
        eta_min = float(self.search_config.cosine_annealing.get('eta_min', 1e-6))

        candidates = []
        for learning_rate, weight_decay, use_cosine in itertools.product(
            learning_rates,
            weight_decays,
            use_cosine_values,
        ):
            config_id = (
                f"{optimizer_name.lower()}_"
                f"lr{_float_to_token(learning_rate)}_"
                f"wd{_float_to_token(weight_decay)}_"
                f"cos{int(use_cosine)}"
            )
            scheduler_config = None
            if use_cosine:
                scheduler_config = {
                    'name': 'CosineAnnealingLR',
                    'T_max': int(self.base_experiment.config.train.num_epochs),
                    'eta_min': eta_min,
                }
            candidates.append(
                {
                    'config_id': config_id,
                    'optimizer_name': optimizer_name,
                    'learning_rate': learning_rate,
                    'weight_decay': weight_decay,
                    'use_cosine_annealing': use_cosine,
                    'scheduler': scheduler_config,
                }
            )

        return candidates

    def _build_experiment_overrides(self, candidate, stage: str):
        stage_overrides: dict[str, object] = {
            'train': {
                'optimizer': {
                    'name': candidate['optimizer_name'],
                    'learning_rate': candidate['learning_rate'],
                    'weight_decay': candidate['weight_decay'],
                },
                'scheduler': candidate['scheduler'],
            },
        }
        if stage in {'full', 'final'}:
            stage_overrides['output_dir'] = str(self.artifacts_root / stage / candidate['config_id'])
        return _merge_overrides(self.base_overrides, stage_overrides)

    def _evaluate_candidate(self, candidate, stage: str):
        cached_result = self._load_cached_candidate_result(candidate, stage)
        if cached_result is not None:
            return cached_result

        fold_results = []
        for fold_idx in range(self.base_experiment.num_folds):
            if stage == 'subset':
                train_group_ids = self.subset_groups_by_fold[str(fold_idx)]['train']
                val_group_ids = self.subset_groups_by_fold[str(fold_idx)]['val']
                checkpoint_name = None
                split_label = 'Subset'
            else:
                train_group_ids, val_group_ids = self.base_experiment.get_fold_group_ids(fold_idx)
                checkpoint_name = f'fold_{fold_idx}.pt'
                split_label = 'Full'

            experiment = FCExperiment(
                self.base_config_path,
                config_overrides=self._build_experiment_overrides(candidate, stage),
            )
            fold_metrics = experiment.run_group_split(
                train_group_ids,
                val_group_ids,
                split_label=split_label,
                split_idx=fold_idx,
                checkpoint_name=checkpoint_name,
            )
            fold_results.append(fold_metrics)

        aggregated_results = _aggregate_fold_results(fold_results)
        aggregated_results.update(
            {
                'config_id': candidate['config_id'],
                'optimizer_name': candidate['optimizer_name'],
                'learning_rate': candidate['learning_rate'],
                'weight_decay': candidate['weight_decay'],
                'use_cosine_annealing': candidate['use_cosine_annealing'],
                'scheduler': candidate['scheduler'],
            }
        )
        self._save_candidate_result(candidate, stage, aggregated_results)
        return aggregated_results

    @staticmethod
    def _rank_results(results):
        return sorted(
            results,
            key=lambda result: (-result['mean_accuracy'], result['mean_loss']),
        )

    def _run_final_test(self, candidate):
        cached_test_metrics = self._load_cached_final_test_result(candidate)
        if cached_test_metrics is not None:
            return cached_test_metrics

        final_overrides = self._build_experiment_overrides(candidate, 'final')
        experiment = FCExperiment(self.base_config_path, config_overrides=final_overrides)
        experiment.split_definition = self.base_experiment.split_definition
        test_metrics = experiment.run_test_evaluation()
        self._save_final_test_result(candidate, test_metrics)
        return test_metrics

    def run(self):
        candidates = self._build_candidates()
        candidates_by_id = {candidate['config_id']: candidate for candidate in candidates}
        self._save_subset_splits()

        subset_results = [self._evaluate_candidate(candidate, 'subset') for candidate in candidates]
        ranked_subset_results = self._rank_results(subset_results)
        results_payload = self._load_results_payload()
        results_payload['subset_results'] = ranked_subset_results
        self._save_results_payload(results_payload)

        top_k = min(int(self.search_config.rerun.get('top_k', 3)), len(ranked_subset_results))
        top_candidates = [
            candidates_by_id[result['config_id']]
            for result in ranked_subset_results[:top_k]
        ]
        full_results = [
            self._evaluate_candidate(candidate, 'full')
            for candidate in top_candidates
        ]
        ranked_full_results = self._rank_results(full_results)
        best_candidate = ranked_full_results[0]
        test_metrics = self._run_final_test(best_candidate)

        results_payload = self._load_results_payload()
        results_payload['subset_results'] = ranked_subset_results
        results_payload['full_results'] = ranked_full_results
        results_payload['selected_config_id'] = best_candidate['config_id']
        results_payload['selected_test_metrics'] = test_metrics
        self._save_results_payload(results_payload)
        return results_payload


def _configure_main_logging(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open('a', encoding='utf-8')
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = _TeeStream(original_stdout, log_file)
    sys.stderr = _TeeStream(original_stderr, log_file)
    return log_file, original_stdout, original_stderr


def _restore_main_logging(log_file, original_stdout, original_stderr):
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_file.close()


def _resolve_main_log_path(base_config_path: Path, hp_search_config_path: Path) -> Path:
    log_filename = 'hp_search.log'
    output_dir = Path(__file__).resolve().parents[1] / 'outputs'
    base_config = OmegaConf.load(base_config_path)
    base_config_data = OmegaConf.to_container(base_config, resolve=True)
    if isinstance(base_config_data, dict):
        configured_output_dir = base_config_data.get('output_dir')
        if configured_output_dir:
            output_dir = Path(str(configured_output_dir))

    hp_search_config = OmegaConf.load(hp_search_config_path)
    hp_search_config_data = OmegaConf.to_container(hp_search_config, resolve=True)
    if isinstance(hp_search_config_data, dict):
        search_config_data = hp_search_config_data.get('search', {})
        if isinstance(search_config_data, dict):
            log_filename = str(search_config_data.get('log_filename', log_filename))
    return output_dir / 'logs' / log_filename


def main(base_config_path: str, hp_search_config_path: str):
    search = FCHyperparameterSearch(base_config_path, hp_search_config_path)
    summary = search.run()
    print(
        json.dumps(
            {
                'selected_config_id': summary['selected_config_id'],
                'selected_test_metrics': summary['selected_test_metrics'],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run FC hyperparameter search.')
    parser.add_argument(
        '--base-config',
        default=str(Path(__file__).resolve().parents[1] / 'configs' / 'base.yaml'),
        help='Path to base experiment config YAML.',
    )
    parser.add_argument(
        '--hp-search-config',
        default=str(Path(__file__).resolve().parents[1] / 'configs' / 'hp_search.yaml'),
        help='Path to hp-search config YAML.',
    )
    args = parser.parse_args()

    base_config = Path(args.base_config)
    hp_search_config = Path(args.hp_search_config)
    log_file, original_stdout, original_stderr = _configure_main_logging(
        _resolve_main_log_path(base_config, hp_search_config)
    )
    try:
        main(str(base_config), str(hp_search_config))
    finally:
        _restore_main_logging(log_file, original_stdout, original_stderr)