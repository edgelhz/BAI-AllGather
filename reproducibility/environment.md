# Experiment environment

See `versions.json` for the actual local git commits, dirty tracked-file
status, and SHA256 of the scheduler, simulation executable, and ET converter.
For the default evaluated configurations: 400 Gbps GPU access links at
0.00025 ms and 100 Gbps inter-node links/uplinks at 0.0015 ms; the network
configurations use PACKET_PAYLOAD_SIZE 1000 and CC_MODE 12. The precise
meaning of CC_MODE 12, endpoint model, and fixed startup costs must be
verified against the pinned simulator source before claiming them.
The third-party simulator and tooling are NOT vendored in this repository.
The supplied system/network JSON and TXT files are original snapshots, not
portable templates; update paths using the included runners and inspect them.
