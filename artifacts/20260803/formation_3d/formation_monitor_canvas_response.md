# Canvas response

Your proposed runtime monitor can be demonstrated directly with a 3D formation benchmark. We replay the same delayed communication schedule for a delay-blind controller and for a monitored controller that checks local separation, formation error, and message age. When the monitor detects risk, it switches to a conservative fallback that caps motion and prioritizes separation over mission progress.

The animation is intentionally a kinematic demonstration rather than a claim about real aircraft dynamics. Its point is to make the assurance tradeoff visible: stale messages can preserve confident motion while degrading formation structure, whereas fallback can reduce unsafe overlap at the cost of slower progress and more conservative behavior.

Artifact: `formation_3d_delay_monitor_comparison.mp4`.
