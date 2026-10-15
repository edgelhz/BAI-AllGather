# From Two-Node BAI to Multi-Node Scheduling

BAI-AllGather was first developed on a two-node
hierarchical GPU network, where the shared inter-node
link makes communication bottlenecks and dependency
interleaving particularly visible.

The multi-node extension generalizes remote seeding
through parameterized node-pair planning, local forwarding,
and remaining-delivery completion.

The central design remains unchanged: minimize logical
inter-node data deliveries while organizing dependencies
to expose ready communication and improve overlap.

The main two-node BAI completion time changed from
6,964,388 to 6,962,484 cycles following schedule
construction refinements. Improvement over Hierarchical
Ring changed from 9.52% to 9.55%.

The Strict Deferred-Inter control was reconstructed
using the refined BAI operation set. Its reported
improvement changed from 11.49% to 14.60%.

The reconstruction changes the dependency-controlled
comparison, while preserving the main scheduling principle.
