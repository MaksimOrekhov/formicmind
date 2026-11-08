"""Ecosystem v1 learning stack (FOR-49): a trainable actor-critic for ``ObservationE1 -> ActionE1`` and a time-aware PPO trainer.

This subpackage is the **only** part of ``formicmind.eco`` that may import PyTorch; the core (rules, engine, observation, masks, decoder, replay)
never imports it, and importing ``formicmind.eco`` does not load torch (``tests/test_eco_boundary.py``). The network sees exactly an
``ObservationE1`` and its masks, nothing else.
"""
