"""Speed measurement for the simulator and the self-play pipeline.

Section 24 of the design guide puts simulation speed first, ahead of network sophistication,
and section 6 asks for this to be the number the work is optimised against. Kept out of
``training/`` because it measures the engine as much as the learner, and out of ``catan/``
because the engine contains no AI code and no measurement of one.
"""
