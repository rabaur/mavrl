# Expert policies (external reference, not our outputs)

Pre-trained external policies used as **Boltzmann-rational simulators**
for generating synthetic training feedback (demonstrations,
preferences, ratings, stops). These are *inputs* to our pipeline, not
outputs of our research.

Specifically:
- `dqn/`, `ppo/` — best-effort policies trained with stable-baselines3's
  default DQN/PPO on each environment. Used at three temperatures
  (`β_traj`) to simulate demonstrators of varying optimality, as
  described in Appendix B.
- `expert_models_acrobot_asymmetry/` — per-asymmetry optimal policies
  used in the Acrobot transfer experiments (Section 5.3).
