"""Offline DCK training pipeline (Spec 6).

prepare_train_questions -> collect_trajectories -> build_stop_tables ->
materialize_labels -> train_head. Every stage is deterministic given its
inputs and writes a manifest recording input hashes and code version.
"""
