from entities import TNC, MT, MaaS, Travelers, distribute_travelers
import numpy as np
import matplotlib.pyplot as plt


def plot_total_allocations(services, allocation_history, number_days):
    """Plot total allocations per service over time."""
    plt.figure(figsize=(8, 5))
    for service in services:
        plt.plot(range(number_days),
                 allocation_history[service.name],
                 label=service.name,
                 linewidth=2)
    plt.title("Evolution of Total Service Allocations")
    plt.xlabel("Day")
    plt.ylabel("Number of Travelers (total)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_per_type_allocations(services, allocation_by_type, travelers, number_days):
    """Plot allocations per traveler type for each service over time."""
    fig, axes = plt.subplots(len(travelers), 1, figsize=(8, 4 * len(travelers)), sharex=True)
    if len(travelers) == 1:
        axes = [axes]  # Ensure iterable

    for t_idx, ax in enumerate(axes):
        for service in services:
            # Rebuild y-values per day for this traveler type
            y_vals = [day_vals[t_idx] for day_vals in zip(*allocation_by_type[service.name])]
            ax.plot(range(number_days), y_vals, label=service.name, linewidth=2)
        ax.set_title(f"Traveler Type {t_idx + 1}")
        ax.set_ylabel("Number of Travelers")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)

    axes[-1].set_xlabel("Day")
    plt.tight_layout()
    plt.show()

def main():
    # --------------------------
    # 0. Initialization
    # --------------------------
    number_days = 50

    # --------------------------
    # 1. Define services
    # --------------------------
    tnc = TNC(
        ASC=1.0, 
        fare=2.5, # monetary units per km
        detour_ratio=1.3, # 1.3 times the direct distance
        average_speed=40, # in km/h
        average_veh_travel_dist_per_day=8*40, # 320 km per veh per day
        capacity_ratio_to_MaaS=0.4, # TNC gives 40% of its capacity to MaaS
        total_service_capacity=32000, # in veh * km per day
        cost_purchasing_capacity_TNC= 100, # monetary units per veh (MM : check this value later)
        operating_cost= 90 # monetary units per veh (MM : check this value later)
    )

    mt = MT(
        ASC=0.0, 
        fare=1, # monetary units per segment (* n_transfer_per_length (eg. 0.3) = monetary units per km)
        detour_ratio=1.8,
        average_speed=15,
        n_transfer_per_length=0.3, # per km
        access_time=1/6, # hours
        transit_time=1/12 # hours
    )
    
    maas = MaaS(
        ASC=0.5, 
        fare=1.05 * (0.2 * tnc.fare + (1 - 0.2) * mt.fare * mt.n_transfer_per_length), # additional maas operation cost * (...) monetary units per km 
        share_TNC=0.2, # share of TNC inside MaaS (first and last kilometers)
        detour_ratio_TNC=tnc.detour_ratio,
        average_speed_TNC=tnc.average_speed,
        capacity_ratio_from_TNC=tnc.capacity_ratio_to_MaaS,
        total_service_capacity_TNC=tnc.total_service_capacity,
        average_veh_travel_dist_per_day_TNC=tnc.average_veh_travel_dist_per_day,
        cost_purchasing_capacity_TNC=tnc.cost_purchasing_capacity_TNC,
        detour_ratio_MT=mt.detour_ratio,
        average_speed_MT=mt.average_speed,
        transit_time_MT=mt.transit_time,
        n_transfer_per_length_MT=mt.n_transfer_per_length,
        cost_purchasing_capacity_MT=80 # MM : check this value later
        )

    services = [tnc, mt, maas]

    # --------------------------
    # 2. Define traveler groups
    # --------------------------
    travelers = [
        Travelers(number_traveler=200, trip_length=30, value_time=25, value_wait=25), # count, km, monetary unit/h, monetary unit/h
        Travelers(number_traveler=150, trip_length=10, value_time=25, value_wait=25),
        Travelers(number_traveler=100, trip_length=5, value_time=25, value_wait=25)
    ]

    # --------------------------
    # 3. Uniform initial allocation
    # --------------------------
    allocation = {service.name: [0] * len(travelers) for service in services}

    for type_i, traveler in enumerate(travelers):
        for service in services:
            allocation[service.name][type_i] += traveler.number_traveler / len(services)
    
    # Pass trip info to services (km and monetary units per time)
    tnc.trip_length_per_traveler_type = [traveler.trip_length for traveler in travelers] 
    tnc.value_waiting_time_per_traveler_type = [traveler.value_wait for traveler in travelers]
    maas.trip_length_per_traveler_type = [traveler.trip_length for traveler in travelers] 
    maas.value_travel_time_per_traveler_type = [traveler.value_time for traveler in travelers]
    maas.value_waiting_time_per_traveler_type = [traveler.value_wait for traveler in travelers]
    
    # --------------------------
    # Storage for plotting
    # --------------------------
    # Total travelers per service per day
    allocation_history = {service.name: [] for service in services}
    # Per traveler type per service
    allocation_by_type = {service.name: [[] for _ in travelers] for service in services}

    # --------------------------
    # 4. Simulation loop
    # --------------------------
    for day in range(number_days):
        tnc.get_allocation(allocation)
        maas.get_allocation(allocation)
        allocation = distribute_travelers(travelers, services)


        # Store allocations
        for service in services:
            total_travelers = sum(allocation[service.name])
            allocation_history[service.name].append(total_travelers)
            for t_idx in range(len(travelers)):
                if len(allocation_by_type[service.name][t_idx]) < day + 1:
                    allocation_by_type[service.name][t_idx].append(allocation[service.name][t_idx])
                else:
                    allocation_by_type[service.name][t_idx][day] = allocation[service.name][t_idx]

        # example of objectives computation for MaaS or TNC 
        utilities = np.ones((len(travelers), len(services)))
        for idx_m, service in enumerate(services):
            for idx_i, traveler in enumerate(travelers):
                utilities[idx_i, idx_m] = service.compute_utility(trip_length = traveler.trip_length, value_time = traveler.value_time, value_wait = traveler.value_wait)
        print(maas.compute_objective_function(U = utilities)) # or tnc.
        print(maas.gradient_objective(U = utilities)) # or tnc.

    print("\nFinal allocation:")
    print(f"{', '.join([f'{k}: {[round(v) for v in vals]}' for k, vals in allocation.items()])}")

    # Plot results using helper functions
    # plot_total_allocations(services, allocation_history, number_days)
    # plot_per_type_allocations(services, allocation_by_type, travelers, number_days)


if __name__ == "__main__":
    main()