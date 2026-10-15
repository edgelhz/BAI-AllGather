# R1: Verified Execution Model

## Physical topology

The reference 2x4 system contains eight GPU ranks and two
local switches. GPU-switch links provide 400 Gbps per
direction with 250 ns propagation delay. The inter-switch
link provides 100 Gbps per direction with 1500 ns delay.

Links are full duplex. Flows in the same direction compete
for the corresponding egress capacity; opposite directions
have independent transmission resources.

## GPU endpoints and concurrency

Each GPU maps to one communication rank. Ready SEND operations
create ns-3 RDMA flows with associated queue pairs.
Multiple queue pairs may be active on the same GPU endpoint.
Their packets share the NIC egress queue and are serialized
by the sending device. RECV events are matched separately.

The system uses FIFO scheduling and
active-chunks-per-dimension=1. This is an ASTRA-sim chunk
scheduling setting, not a one-queue-pair hardware limit.

## Startup-cost treatment

The custom Chakra executor passes delay=0 to both
front_end_sim_send and front_end_sim_recv.

The reference system JSON does not override endpoint-delay.
Sys.cc initializes communication_delay=10 and supplies
this parameter to the MemBus model. It is not an explicit
network SEND startup alpha.

Network packet serialization, queueing, and propagation
are represented by the ns-3 execution path.

## Complete settings and provenance

The byte-preserved reference inputs are:

- configs/observed/r1_2x4_network.txt
- configs/observed/r1_2x4_system.json
- configs/topology/r1_2x4_topology.txt

Their SHA256 values are recorded in inputs.sha256.

The network file records packetization, CC_MODE=12,
QCN, PFC, buffer settings, error rate and trace options.

Source locations: common.h, qbb-channel.cc,
qbb-net-device.cc, rdma-hw.cc, AstraSimNetwork.cc,
entry.h, CustomAlgorithm.cc, and Sys.cc.

ASTRA-sim: 28f18ea0d58a96fd812e49b71bdac330a1e3c965
ns-3 backend: 40d9153caefb9ac1a6ec316aedad3068cfc0fbc9
MSCCL-tools: 8d735800a87bf4426816e2c10a9834a42ab64fe6
