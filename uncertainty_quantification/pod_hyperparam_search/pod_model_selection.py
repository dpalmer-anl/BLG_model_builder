"""CSV/cache-only POD model-selection compatibility wrapper."""

from blg_model_builder.pod_model_selection import (
    POD_SEARCH_RESULTS_CSV,
    find_pod_energy_ensemble_folder,
    load_pod_hyperparam_search_fit,
    load_pod_search_results,
    pod_energy_ensemble_names_from_csv,
    pod_hyperparam_search_cache_paths,
    pod_hyperparams_for_index,
    pod_hyperparams_from_row,
    pod_row_for_index,
    write_pod_energy_best_fit_caches_from_tightened_csv,
)

__all__ = [
    "POD_SEARCH_RESULTS_CSV",
    "find_pod_energy_ensemble_folder",
    "load_pod_hyperparam_search_fit",
    "load_pod_search_results",
    "pod_energy_ensemble_names_from_csv",
    "pod_hyperparam_search_cache_paths",
    "pod_hyperparams_for_index",
    "pod_hyperparams_from_row",
    "pod_row_for_index",
    "write_pod_energy_best_fit_caches_from_tightened_csv",
]

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Inspect POD hyperparameter-search CSV rows or export best-fit caches."
    )
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--export-best-fit-caches", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.export_best_fit_caches:
        paths = write_pod_energy_best_fit_caches_from_tightened_csv(
            csv_path=args.csv,
            force=args.force,
        )
        print(f"Exported {len(paths)} POD best-fit cache(s).")
    else:
        rows = load_pod_search_results(args.csv)
        for pod_index, row in rows.iterrows():
            print(
                f"POD_index={pod_index} hash={row['hash']} "
                f"pass={row['meets_all_criteria']}"
            )
