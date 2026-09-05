import traci

from integration.topology import JunctionTopology


class SUMOTopologyAdapter:
    """
    Converts SUMO/TraCI topology into the project's
    generic JunctionTopology representation.
    """

    def __init__(self, traci_connection=None):
        self.traci = traci_connection or traci

        # Stores SUMO-specific information for every movement.
        #
        # Generic topology.py should not need to know about
        # SUMO link indices, lanes, or via lanes.
        self.sumo_metadata = {}

    # ============================================================
    # DISCOVER ONE JUNCTION
    # ============================================================

    def discover_junction(self, junction_id):

        topology = JunctionTopology(junction_id)

        controlled_links = (
            self.traci.trafficlight.getControlledLinks(
                junction_id
            )
        )

        movement_index = 0

        # --------------------------------------------------------
        # First pass:
        # collect all roads used by this junction.
        # --------------------------------------------------------

        roads = set()

        for links in controlled_links:

            if not links:
                continue

            for link in links:

                if len(link) < 2:
                    continue

                from_lane = link[0]
                to_lane = link[1]

                if not from_lane or not to_lane:
                    continue

                from_edge = self._lane_to_edge(from_lane)
                to_edge = self._lane_to_edge(to_lane)

                # Ignore same-edge movement.
                if from_edge == to_edge:
                    continue

                roads.add(from_edge)
                roads.add(to_edge)

        # --------------------------------------------------------
        # Add all roads BEFORE adding movements.
        #
        # JunctionTopology.add_movement() requires both roads
        # to already exist.
        # --------------------------------------------------------

        for road_id in sorted(roads):

            lane_count = self._get_lane_count(road_id)

            topology.add_road(
                road_id,
                lanes=lane_count
            )

        # --------------------------------------------------------
        # Verify roads were actually registered.
        # --------------------------------------------------------

        for road_id in roads:

            if road_id not in topology.roads:
                raise RuntimeError(
                    f"Failed to register road: {road_id}"
                )

        # --------------------------------------------------------
        # Second pass:
        # create movements.
        # --------------------------------------------------------

        for link_index, links in enumerate(controlled_links):

            if not links:
                continue

            for link in links:

                if len(link) < 2:
                    continue

                from_lane = link[0]
                to_lane = link[1]
                via_lane = (
                    link[2]
                    if len(link) > 2
                    else None
                )

                if not from_lane or not to_lane:
                    continue

                from_edge = self._lane_to_edge(
                    from_lane
                )

                to_edge = self._lane_to_edge(
                    to_lane
                )

                if from_edge == to_edge:
                    continue

                movement_id = (
                    f"{junction_id}_M{movement_index}"
                )

                topology.add_movement(
                    movement_id,
                    from_edge,
                    to_edge,
                    lanes=1,
                    allowed=True
                )

                # Store SUMO-specific data separately.
                self.sumo_metadata[movement_id] = {

                    "junction_id": junction_id,

                    "link_index": link_index,

                    "from_lane": from_lane,

                    "to_lane": to_lane,

                    "via_lane": via_lane
                }

                movement_index += 1

        # --------------------------------------------------------
        # Final validation.
        # --------------------------------------------------------

        topology.validate()

        return topology

    # ============================================================
    # DISCOVER ALL TRAFFIC LIGHTS
    # ============================================================

    def discover_all_junctions(self):

        junction_ids = (
            self.traci.trafficlight.getIDList()
        )

        topologies = {}

        for junction_id in junction_ids:

            topologies[junction_id] = (
                self.discover_junction(
                    junction_id
                )
            )

        return topologies

    # ============================================================
    # LANE -> EDGE
    # ============================================================

    def _lane_to_edge(self, lane_id):

        """
        Convert a SUMO lane ID to its edge ID.

        Example:

            top0A0_0
                ↓
            top0A0

            A0B0_0
                ↓
            A0B0
        """

        # Ask SUMO directly first.
        try:

            edge_id = self.traci.lane.getEdgeID(
                lane_id
            )

            if edge_id:
                return edge_id

        except Exception:
            pass

        # Fallback for normal SUMO lane naming.
        if "_" in lane_id:

            return lane_id.rsplit(
                "_",
                1
            )[0]

        return lane_id

    # ============================================================
    # GET NUMBER OF LANES
    # ============================================================

    def _get_lane_count(self, edge_id):

        """
        Get the number of lanes on a SUMO edge.
        """

        try:

            lane_count = (
                self.traci.edge.getLaneNumber(
                    edge_id
                )
            )

            return max(
                1,
                int(lane_count)
            )

        except Exception:

            return 1


# ================================================================
# STANDALONE TEST
# ================================================================

if __name__ == "__main__":

    SUMO_CMD = [
        "sumo",
        "-c",
        "simulation/configs/corridor.sumocfg"
    ]

    try:

        traci.start(SUMO_CMD)

        adapter = SUMOTopologyAdapter()

        topologies = (
            adapter.discover_all_junctions()
        )

        for junction_id, topology in topologies.items():

            print()
            print("=" * 70)
            print("JUNCTION:", junction_id)
            print("=" * 70)

            print()
            print("ROADS:")

            for road_id, road in topology.roads.items():

                print(
                    f"  {road_id}"
                    f" | lanes={road.lanes}"
                )

            print()
            print("MOVEMENTS:")

            for movement_id, movement in (
                topology.movements.items()
            ):

                print(
                    f"  {movement_id}: "
                    f"{movement.from_road} "
                    f"-> "
                    f"{movement.to_road}"
                )

                metadata = (
                    adapter.sumo_metadata[
                        movement_id
                    ]
                )

                print(
                    f"      link_index="
                    f"{metadata['link_index']}"
                )

                print(
                    f"      from_lane="
                    f"{metadata['from_lane']}"
                )

                print(
                    f"      to_lane="
                    f"{metadata['to_lane']}"
                )

    finally:

        traci.close()