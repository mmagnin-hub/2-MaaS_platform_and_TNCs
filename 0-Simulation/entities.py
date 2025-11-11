from abc import ABC, abstractmethod
import numpy as np


# --------------------------
# Base Service Class
# --------------------------

class Service(ABC):
    """
    Abstract base class representing a generic transportation service.
    """

    def __init__(self, name: str, ASC: float, fare: float):
        self.name = name
        self.ASC = ASC
        self.fare = fare

    @abstractmethod
    def trip_fare(self, trip_length: float) -> float:
        """Compute total fare for a trip of given length."""
        raise NotImplementedError

    @abstractmethod
    def trip_time(self, trip_length: float) -> float:
        """Compute total travel time for a trip of given length."""
        raise NotImplementedError

    @abstractmethod
    def waiting_time(self, trip_length: float) -> float:
        """Compute expected waiting time for a trip of given length."""
        raise NotImplementedError


# --------------------------
# TNC Service
# --------------------------

class TNC(Service):
    """
    Transportation Network Company (TNC) service model, e.g., ride-hailing.
    """

    def __init__(self,
                 detour_ratio: float,
                 average_speed: float,
                 average_veh_travel_dist_per_day: float,
                 capacity_ratio_to_MaaS: float,
                 total_service_capacity: float,):
        super().__init__(name="TNC", ASC=0.0, fare=0.0)  # placeholder values; !! CHANGE THEM !!

        self.detour_ratio = detour_ratio
        self.average_speed = average_speed
        self.average_veh_travel_dist_per_day = average_veh_travel_dist_per_day
        self.capacity_ratio_to_MaaS = capacity_ratio_to_MaaS
        self.total_service_capacity = total_service_capacity
        
        # self.wholesale_price = wholesale_price # not use now
        # self.operating_cost = operating_cost # not use now
        
        self.trip_length_per_traveler_type: list[float] | None = None 
        self.demand_per_traveler_type: list[float] | None = None 
        self.vacant_veh_available: float | None = None  

    def trip_fare(self, trip_length: float) -> float:
        return self.fare * self.detour_ratio * trip_length # monetary units

    def trip_time(self, trip_length: float) -> float:
        return self.detour_ratio * trip_length / self.average_speed # hours

    def waiting_time(self) -> float:
        """
        A is a parameter that counts the exogenous factors in the matching process.
        A = 2.5 (Zhou et al. (2022), Competition ND third-party platform-integration in ride-sourcing markets. 
            Transportation Research Part. B: Methodological, 159, 76-103.)
        sensitivity_param = 0.5 in regular e-hailing service without passenger competition in the matching process.
        """
        A = 2.5
        sensitivity_param = 0.5
        vacant_veh_available = self.find_vacant_veh_available()
        if vacant_veh_available <= 0:
            print(f"[Warning] {self.name}: No vacant vehicles available — using minimum threshold.")
            vacant_veh_available = max(self.find_vacant_veh_available(), 1e-6) # make it so expensive that no one will choose it 
        return A * (vacant_veh_available ** (-sensitivity_param)) # hours (empirical formula)

    def find_vacant_veh_available(self) -> float:
        """
        Compute the number of idle vehicles in the TNC fleet (in veh per day).
        """
        total_demand = np.sum(np.array(self.trip_length_per_traveler_type) * np.array(self.demand_per_traveler_type))
        return ((1-self.capacity_ratio_to_MaaS) * self.total_service_capacity - total_demand) / self.average_veh_travel_dist_per_day

    def compute_objective_function(self) -> float:
        """
        Compute operator objective function (e.g., profit).

        Returns
        -------
        float
            Objective value (e.g., profit in monetary units).
        TODO: Use lagragian formulation ? 
        """
        return

    def optimize(self):
        """
        Optimize TNC operational variables:
        - fare
        - capacity_ratio_to_MaaS

        Subject to service and capacity constraints.
        TODO: Define optimization logic. How to maximize considering the travelers distribution ?
        """
        return

    def get_allocation(self, allocation: dict[str, list[float]]) -> None:
        self.demand_per_traveler_type = allocation[self.name]
        return

# --------------------------
# Mass Transit Service
# --------------------------

class MT(Service):
    """
    Mass Transit (MT) service model.
    """

    def __init__(self,
                 n_transfer_per_length: float,
                 detour_ratio: float,
                 average_speed: float,
                 access_time: float,
                 transit_time: float):
        super().__init__(name="MT", ASC=0.0, fare=0.0)  # placeholder values; !! CHANGE THEM !!
        self.n_transfer_per_length = n_transfer_per_length
        self.detour_ratio = detour_ratio 
        self.average_speed = average_speed
        self.access_time = access_time
        self.transit_time = transit_time

    def trip_fare(self, trip_length: float) -> float:
        return self.fare * (self.n_transfer_per_length * trip_length + 1)

    def trip_time(self, trip_length: float) -> float:
        return self.detour_ratio * trip_length / self.average_speed

    def waiting_time(self, trip_length: float) -> float:
        return 2 * self.access_time + self.transit_time * (self.n_transfer_per_length * trip_length)


# --------------------------
# MaaS Service
# --------------------------

class MaaS(Service):
    """
    Mobility-as-a-Service (MaaS) platform combining TNC and MT modes.
    """

    def __init__(self,
                 share_TNC: float,
                 detour_ratio_TNC: float,
                 average_speed_TNC: float,
                 capacity_ratio_to_MaaS: float,
                 total_service_capacity_TNC: float,
                 average_veh_travel_dist_per_day_TNC: float,
                 detour_ratio_MT: float,
                 average_speed_MT: float,
                 transit_time_MT: float,
                 n_transfer_per_length_MT: float):
        super().__init__(name="MaaS", ASC=0.0, fare=0.0)  # placeholder values; !! CHANGE THEM !!

        # Share between sub-services
        self.share_TNC = share_TNC

        # TNC parameters
        self.detour_ratio_TNC = detour_ratio_TNC
        self.average_speed_TNC = average_speed_TNC
        self.capacity_ratio_to_MaaS = capacity_ratio_to_MaaS
        self.total_service_capacity_TNC = total_service_capacity_TNC
        self.average_veh_travel_dist_per_day_TNC = average_veh_travel_dist_per_day_TNC

        self.trip_length_per_traveler_type: list[float] | None = None 
        self.demand_per_traveler_type: list[float] | None = None 
        self.vacant_veh_available: float | None = None  

        # MT parameters
        self.detour_ratio_MT = detour_ratio_MT
        self.average_speed_MT = average_speed_MT
        self.transit_time_MT = transit_time_MT
        self.n_transfer_per_length_MT = n_transfer_per_length_MT

    def trip_fare(self, trip_length: float) -> float:
        return self.fare * trip_length

    def trip_time(self, trip_length: float) -> float:
        time_TNC = self.detour_ratio_TNC * self.share_TNC * trip_length / self.average_speed_TNC
        time_MT = self.detour_ratio_MT * (1 - self.share_TNC) * trip_length / self.average_speed_MT
        return time_TNC + time_MT

    def waiting_time(self, trip_length: float) -> float:
        """
        Same as TNC : 
        A is a parameter that counts the exogenous factors in the matching process.
        A = 2.5 (Zhou et al. (2022), Competition ND third-party platform-integration in ride-sourcing markets. 
            Transportation Research Part. B: Methodological, 159, 76-103.)
        sensitivity_param = 0.5 in regular e-hailing service without passenger competition in the matching process.
        """
        A = 2.5
        sensitivity_param = 0.5
        vacant_veh_available = self.find_vacant_veh_available()
        if vacant_veh_available <= 0:
            print(f"[Warning] {self.name}: No vacant vehicles available — using minimum threshold.")
            vacant_veh_available = max(self.find_vacant_veh_available(), 1e-6) # make it so expensive that no one will choose it 
        TNC_waiting_time = A * (vacant_veh_available ** (- sensitivity_param))
        MT_waiting_time = self.transit_time_MT * (self.n_transfer_per_length_MT * (1 - self.share_TNC) * trip_length)
        return TNC_waiting_time + MT_waiting_time

    def find_vacant_veh_available(self) -> float:
        """
        Compute the number of idle vehicles in the MaaS fleet (in veh per day).
        """
        total_demand = self.share_TNC * np.sum(np.array(self.trip_length_per_traveler_type) * np.array(self.demand_per_traveler_type)) 
        return (self.capacity_ratio_to_MaaS * self.total_service_capacity_TNC - total_demand) / self.average_veh_travel_dist_per_day_TNC
    
    def optimize(self):
        """
        Optimize MaaS parameters:
        - fare
        - TNC wholesale price
        - TNC/MT split (share_TNC)
        - MT capacity usage
        TODO: Define MaaS optimization logic.
        """
        return
    
    def get_allocation(self, allocation: dict[str, list[float]]) -> None:
        self.demand_per_traveler_type = allocation[self.name]
        return

# --------------------------
# Group of Travelers 
# --------------------------
class Travelers: 
    '''
    Represents a group of travelers characterized by their size, trip attributes,
    and sensitivity to travel time and waiting time.
    '''

    def __init__(self, number_traveler: int, trip_length: float, 
                 value_time: float, value_wait: float):
        self.number_traveler = number_traveler
        self.trip_length = trip_length
        self.value_time = value_time
        self.value_wait = value_wait
        self.utilities: list[float] | None = None
        self.travelers_per_service: list[float] | None = None

    def compute_utilities(self, services: list[Service]) -> None: 
        '''
        Compute the utility of each service for this traveler group.
        '''
        
        if not self.utilities:
            self.utilities = [0]*len(services)

        for idx, service in enumerate(services):
            asc = service.ASC
            fare = service.trip_fare(self.trip_length)
            time = service.trip_time(self.trip_length)
            if service.name == "TNC":
                wait = service.waiting_time()
            else:
                wait = service.waiting_time(self.trip_length)
            U = asc - fare - self.value_time * time - self.value_wait * wait
            self.utilities[idx] = 0.5 * (self.utilities[idx] + U)  # smoothing 

            # Print debug info
            print(
            f"{service.name}: "
            f"trip_length={self.trip_length:.4f}, value_time={self.value_time:.4f}, "
            f"value_wait={self.value_wait:.4f}, U={U:.4f}, "
            f"fare={fare:.4f}, time={time:.4f}, wait={wait:.4f}"
            )
        return
    
    def choose_service(self, services: list[Service]) -> None:
        '''
        Choose service distribution based on a logit model of choice probabilities.
        '''
        self.compute_utilities(services)
        exp_utilities = np.exp(self.utilities) 
        probabilities = exp_utilities / np.sum(exp_utilities)
        self.travelers_per_service = probabilities * self.number_traveler 
        '''
        Here I could smooth too:
        self.travelers_per_service = (
            0.5 * self.travelers_per_service + 
            0.5 * probabilities * self.number_traveler
        )
        '''
        return

# --------------------------
# Functions
# --------------------------
def distribute_travelers(travelers: list[Travelers], services: list[Service]) -> dict[str, list[float]]: 
    '''
    Allocate groups of travelers among services based on their utility preferences.
    allocation = {"TNC": [n_traveler_type_0, n_traveler_type_1, n_traveler_type_2],
                    "MT": [n_traveler_type_0, n_traveler_type_1, n_traveler_type_2],
                    "MaaS": [n_traveler_type_0, n_traveler_type_1, n_traveler_type_2]}
    '''
    allocation = {service.name: [0] * len(travelers) for service in services}

    for type_i, traveler in enumerate(travelers):
        traveler.choose_service(services)
        for index, service in enumerate(services):
            allocation[service.name][type_i] += traveler.travelers_per_service[index]
    return allocation 