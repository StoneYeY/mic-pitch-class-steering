"""When-to-Steer (w2s): timestep-adaptive probe guidance utilities for the ICASSP 2027 sprint.

Modules
-------
reliability : target-free probe reliability R_t from multi-label sigmoid outputs
schedules   : fixed / calibrated / online guidance schedules with matched budget K
data        : prompt and melody sets (dev / test split) and frame-level pitch-class targets
metrics     : chroma cosine, CLAP score, FAD on CLAP embeddings, paired statistics
probe_eval  : per-timestep probe evaluation on saved diffusion trajectories
"""
__version__ = "0.1.0"
