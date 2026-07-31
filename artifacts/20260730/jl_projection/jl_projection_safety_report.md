# JL-Style Projection: Geometry versus Coordinate Safety

This diagnostic is inspired by Yingru Li's *Simple, unified analysis of Johnson-Lindenstrauss with applications* (arXiv:2402.10232). It tests a narrow engineering question: a random projection may preserve pairwise geometry while a projected-norm monitor fails to preserve a coordinatewise safety condition.

Each synthetic multi-agent feature vector has 64 coordinates, eight of which are treated as safety channels. Violations are single-coordinate spikes. The coordinate oracle checks those channels directly. The compressed monitor projects the vector to dimension `k` and thresholds the projected norm using the 99.5th percentile of normal projected samples.

This is not a JL theorem, and the projected-norm detector is only one possible compressed monitor. The result should be read as a warning against treating distance preservation as automatic preservation of axis-aligned safety constraints. Any real projection-based monitor would need a task-specific proof or empirical certification of the safety predicate.
