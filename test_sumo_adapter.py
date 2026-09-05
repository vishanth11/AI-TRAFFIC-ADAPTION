import traci

from integration.sumo_topology_adapter import SUMOTopologyAdapter


SUMO_CMD = [
    "sumo",
    "-c",
    "simulation/configs/corridor.sumocfg"
]


try:
    traci.start(SUMO_CMD)

    adapter = SUMOTopologyAdapter()

    topologies = adapter.discover_all_junctions()

    for junction_id, topology in topologies.items():

        print("\n" + "=" * 60)
        print("JUNCTION:", junction_id)
        print("=" * 60)

        print("\nROADS:")

        for road_id, road in topology.roads.items():
            print(
                " ",
                road_id,
                "lanes:",
                road.lanes
            )

        print("\nMOVEMENTS:")

        for movement_id, movement in topology.movements.items():

            print(
                " ",
                movement_id,
                ":",
                movement.from_road,
                "->",
                movement.to_road
            )

            print(
                "    SUMO:",
                adapter.sumo_metadata[movement_id]
            )

finally:
    traci.close()