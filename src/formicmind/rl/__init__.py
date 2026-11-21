"""FOR-42: from-scratch PPO / self-play reinforcement learning on top of the unchanged FOR-1/FOR-2 engine.

Nothing in this package changes the simulator, ``ObservationV2``, ``ActionV2`` or the shared decoder: a sampled
*latent* action is converted into the existing 17 finite float32 logits and decoded by ``decode_action_logits``.
Importing the package needs PyTorch (``python -m pip install -e ".[ml]"``); the core simulator stays dependency-free.
"""
