import traci


SUMO_CMD = [
    "sumo",
    "-c",
    "simulation/configs/corridor.sumocfg"
]


def inspect_junction(tls_id):
    print("\n" + "=" * 60)
    print("TRAFFIC LIGHT:", tls_id)
    print("=" * 60)

    controlled_links = traci.trafficlight.getControlledLinks(tls_id)

    for link_index, links in enumerate(controlled_links):
        print(f"\nLink Index {link_index}")

        for link in links:
            if link is None:
                continue

            incoming_lane = link[0]
            outgoing_lane = link[1]
            via_lane = link[2]

            print("  incoming:", incoming_lane)
            print("  outgoing:", outgoing_lane)
            print("  via:", via_lane)

    phases = traci.trafficlight.getAllProgramLogics(tls_id)

    for program in phases:
        print("\nPROGRAM:", program.programID)

        for index, phase in enumerate(program.phases):
            print(
                f"  Phase {index}: "
                f"duration={phase.duration}, "
                f"state={phase.state}"
            )


try:
    traci.start(SUMO_CMD)

    for tls_id in ["A0", "B0", "C0"]:
        inspect_junction(tls_id)

finally:
    traci.close()