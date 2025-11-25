from abc import ABC, abstractmethod
import numpy as np


# --------------------------
# Base Service Class
# --------------------------

class Service(ABC):
    """
    Abstract base class representing a generic transportation service.
    """

    def __init__(self, name: str):
        self.name = name

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

    def compute_utility(self, trip_length: float, value_time: float, value_wait: float) -> float:
        """
        Generic utility computation (can be overridden).
        """
        # Default behavior (valid for MT and MaaS)
        fare = self.trip_fare(trip_length)
        time = self.trip_time(trip_length)
        wait = self.waiting_time(trip_length)  
        return self.ASC - fare - value_time * time - value_wait * wait
    
    def get_allocation(self, allocation: dict[str, list[float]]) -> None:
        self.demand_per_traveler_type = allocation
        return

# --------------------------
# TNC Service
# --------------------------

class TNC(Service):
    """
    Transportation Network Company (TNC) service model, e.g., ride-hailing.
    """

    def __init__(self,
                 ASC: float,
                 fare: float,
                 detour_ratio: float,
                 average_speed: float,
                 average_veh_travel_dist_per_day: float,
                 capacity_ratio_to_MaaS: float,
                 total_service_capacity: float,
                 cost_purchasing_capacity_TNC: float,
                 operating_cost: float):
        super().__init__(name="TNC") 
        
        self.ASC = ASC
        self.fare = fare
        self.detour_ratio = detour_ratio
        self.average_speed = average_speed
        self.average_veh_travel_dist_per_day = average_veh_travel_dist_per_day
        self.capacity_ratio_to_MaaS = capacity_ratio_to_MaaS
        self.total_service_capacity = total_service_capacity
        
        # Parameters for objective
        self.cost_purchasing_capacity_TNC = cost_purchasing_capacity_TNC         
        self.operating_cost = operating_cost   
        self.lambda_T = 0      
        
        self.trip_length_per_traveler_type: list[float] | None = None 
        self.demand_per_traveler_type: dict[str, list[float]] = None
        self.value_waiting_time_per_traveler_type: list[float] | None = None 
        self.vacant_veh_available: float | None = None  

    def trip_fare(self, trip_length: float) -> float:
        return self.fare * self.detour_ratio * trip_length # monetary units

    def trip_time(self, trip_length: float) -> float:
        return self.detour_ratio * trip_length / self.average_speed # hours

    def waiting_time(self, trip_length = None) -> float:
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
        total_demand = np.sum(np.array(self.trip_length_per_traveler_type) * np.array(self.demand_per_traveler_type[self.name])) 
        return ((1-self.capacity_ratio_to_MaaS) * self.total_service_capacity - total_demand) / self.average_veh_travel_dist_per_day
    
    def compute_objective_function(self, U: np.ndarray, service_index_T: int = 0) -> float:
        l = np.asarray(self.trip_length_per_traveler_type)
        Q = np.sum(list(self.demand_per_traveler_type.values()), axis=0) 

        # get P_im from U_im
        P = np.exp(U)
        P /= np.sum(P, axis=1, keepdims=True)
        P_iT = P[:, service_index_T]   # Column for TNC

        sum_l_PiT_Qi = np.sum(l * P_iT * Q)
        sum_PiT_Qi   = np.sum(P_iT * Q)

        # Build 4-term objective
        term1 = -self.fare * self.detour_ratio * sum_l_PiT_Qi
        term2 = -self.cost_purchasing_capacity_TNC * self.capacity_ratio_to_MaaS * self.total_service_capacity
        term3 = self.operating_cost * self.total_service_capacity
        term4 = self.lambda_T * (self.average_veh_travel_dist_per_day * sum_PiT_Qi - (1 - self.capacity_ratio_to_MaaS) * self.total_service_capacity)

        return float(term1 + term2 + term3 + term4)
    
    def gradient_objective(
            self,
            U: np.ndarray,
            service_index_T: int = 0,
            service_index_M: int = 2
        ) -> np.ndarray:
        """
        Compute gradient of TNC objective wrt:
        - f_T
        - y_T   (capacity ratio to MaaS)
        - lambda_T
        
        Returns:
            grad_vector: ndarray shape (3,)
        """

        # Extract demand inputs
        l = np.asarray(self.trip_length_per_traveler_type)   # l_i    
        Q = np.sum(list(self.demand_per_traveler_type.values()), axis=0)  # Q_i
        
        # Softmax for probabilities
        P = np.exp(U)
        P /= np.sum(P, axis=1, keepdims=True)

        P_iT = P[:, service_index_T]    # Choice prob for TNC
        P_iM = P[:, service_index_M]    # for MT (needed in y_T gradient)

        # Partial derivatives of utility:
        dUdf = -self.detour_ratio * l     

        A, s = 2.5, 0.5
        vacant = self.find_vacant_veh_available()

        dUdy = - np.asarray(self.value_waiting_time_per_traveler_type) * s * A * (vacant)**(-(s + 1)) * (self.total_service_capacity /
                    self.average_veh_travel_dist_per_day)

        # Terms needed for gradient
        sum_l_PiT_Qi = np.sum(l * P_iT * Q)
        sum_PiT_Qi   = np.sum(P_iT * Q)

        # ========= GRAD w.r.t. f_T ==========
        grad_fT = (
            -self.detour_ratio * sum_l_PiT_Qi
            - self.fare * self.detour_ratio * np.sum(l * Q * P_iT * (1 - P_iT) * dUdf)
            + self.lambda_T * self.average_veh_travel_dist_per_day * np.sum(Q * P_iT * (1 - P_iT) * dUdf)
        )

        # ========= GRAD w.r.t. y_T ==========
        grad_yT = (
            -self.fare * self.detour_ratio * np.sum(l * Q * P_iT *
                ((1 - P_iT) * dUdy - P_iM * dUdy))  
            - self.fare * self.total_service_capacity
            + self.lambda_T * self.average_veh_travel_dist_per_day * np.sum(Q * P_iT *
                ((1 - P_iT) * dUdy - P_iM * dUdy))
            + self.lambda_T * self.total_service_capacity
        )

        # ========= GRAD w.r.t. λ_T ==========
        grad_lambdaT = self.average_veh_travel_dist_per_day * sum_PiT_Qi - (1 - self.capacity_ratio_to_MaaS) * self.total_service_capacity

        return np.array([grad_fT, grad_yT, grad_lambdaT])



    def optimize(self):
        """
        Optimize TNC operational variables:
        - fare
        - capacity_ratio_to_MaaS

        Subject to service and capacity constraints.
        TODO: Define optimization logic. How to maximize considering the travelers distribution ?
        """
        return

# --------------------------
# Mass Transit Service
# --------------------------

class MT(Service):
    """
    Mass Transit (MT) service model.
    """

    def __init__(self,
                 ASC: float,
                 fare: float,
                 n_transfer_per_length: float,
                 detour_ratio: float,
                 average_speed: float,
                 access_time: float,
                 transit_time: float):
        super().__init__(name="MT") 
        
        self.ASC = ASC
        self.fare = fare
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
                 ASC: float,
                 fare: float,
                 share_TNC: float,
                 detour_ratio_TNC: float,
                 average_speed_TNC: float,
                 capacity_ratio_from_TNC: float,
                 total_service_capacity_TNC: float,
                 average_veh_travel_dist_per_day_TNC: float,
                 cost_purchasing_capacity_TNC: float,
                 detour_ratio_MT: float,
                 average_speed_MT: float,
                 transit_time_MT: float,
                 n_transfer_per_length_MT: float,
                 cost_purchasing_capacity_MT: float):
        super().__init__(name="MaaS") 

        self.ASC = ASC
        self.fare = fare
        self.share_TNC = share_TNC
        self.lambda_M = 0
        
        # TNC parameters
        self.detour_ratio_TNC = detour_ratio_TNC
        self.average_speed_TNC = average_speed_TNC
        self.capacity_ratio_from_TNC = capacity_ratio_from_TNC
        self.total_service_capacity_TNC = total_service_capacity_TNC
        self.average_veh_travel_dist_per_day_TNC = average_veh_travel_dist_per_day_TNC
        self.cost_purchasing_capacity_TNC = cost_purchasing_capacity_TNC

        self.trip_length_per_traveler_type: list[float] | None = None 
        self.value_travel_time_per_traveler_type: list[float] | None = None 
        self.value_waiting_time_per_traveler_type: list[float] | None = None
        self.demand_per_traveler_type: dict[str, list[float]] = None
        self.vacant_veh_available: float | None = None  

        # MT parameters
        self.detour_ratio_MT = detour_ratio_MT
        self.average_speed_MT = average_speed_MT
        self.transit_time_MT = transit_time_MT
        self.n_transfer_per_length_MT = n_transfer_per_length_MT
        self.cost_purchasing_capacity_MT = cost_purchasing_capacity_MT

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
        total_demand = self.share_TNC * np.sum(np.array(self.trip_length_per_traveler_type) * np.array(self.demand_per_traveler_type[self.name])) # MM : demand per traveler type for MaaS !!!
        return (self.capacity_ratio_from_TNC * self.total_service_capacity_TNC - total_demand) / self.average_veh_travel_dist_per_day_TNC

    def compute_objective_function(self, U: np.ndarray, service_index_M: int = 2) -> float:
        l = np.asarray(self.trip_length_per_traveler_type)
        Q = np.sum(list(self.demand_per_traveler_type.values()), axis=0) 

        # get P_im from U_im
        P = np.exp(U)
        P /= np.sum(P, axis=1, keepdims=True)
        P_iM = P[:, service_index_M]   # Column for MaaS

        sum_l_PiM_Qi = np.sum(l * P_iM * Q)
        sum_PiM_Qi   = np.sum(P_iM * Q)

        # Build 4-term objective
        term1 = -self.fare * sum_l_PiM_Qi
        term2 = self.cost_purchasing_capacity_MT * (1 - self.share_TNC) * sum_PiM_Qi
        term3 = self.cost_purchasing_capacity_TNC * self.capacity_ratio_from_TNC * self.total_service_capacity_TNC
        term4 = self.lambda_M * (self.share_TNC * sum_PiM_Qi -  self.capacity_ratio_from_TNC * self.total_service_capacity_TNC)

        return float(term1 + term2 + term3 + term4)

    def gradient_objective(
            self,
            U: np.ndarray,
            service_index_T: int = 0,
            service_index_M: int = 2
        ) -> np.ndarray:
        """
        Compute gradient of MaaS objective wrt:
        - f_M
        - p_T
        - alpha (share_TNC)
        - lambda_M
        
        Returns:
            grad_vector: ndarray shape (4,)
        """

        # Extract demand inputs
        l = np.asarray(self.trip_length_per_traveler_type)   # l_i    
        Q = np.sum(list(self.demand_per_traveler_type.values()), axis=0)  # Q_i
        
        # Softmax for probabilities
        P = np.exp(U)
        P /= np.sum(P, axis=1, keepdims=True)

        P_iM = P[:, service_index_M]  

        # Terms needed for gradient
        sum_l_PiM_Qi = np.sum(l * P_iM * Q)
        sum_PiM_Qi   = np.sum(P_iM * Q) 

        # Partial derivatives of utility:
        dUdf = - l     

        A, s = 2.5, 0.5
        vacant = self.find_vacant_veh_available()


        dUdalph = - np.asarray(self.value_travel_time_per_traveler_type) * (self.detour_ratio_TNC / self.average_speed_TNC * l - self.detour_ratio_MT / self.average_speed_MT * l) \
                  - np.asarray(self.value_waiting_time_per_traveler_type) * (s * A * (vacant)**(-(s + 1)) * (sum_l_PiM_Qi /
                    self.average_veh_travel_dist_per_day_TNC) - self.transit_time_MT * self.n_transfer_per_length_MT * l)

        # ========= GRAD w.r.t. f_M ==========
        grad_fM = (
            - sum_l_PiM_Qi
            - self.fare * np.sum(l * Q * P_iM * (1 - P_iM) * dUdf)
            + (self.cost_purchasing_capacity_MT * (1 - self.share_TNC) + self.lambda_M * self.share_TNC) * np.sum(Q * P_iM * (1 - P_iM) * dUdf)
        )

        # ========= GRAD w.r.t. p_T ==========
        grad_pT = self.capacity_ratio_from_TNC * self.total_service_capacity_TNC

        # ========= GRAD w.r.t. alpha ========
        grad_alpha = (- self.fare * np.sum(l * Q * P_iM * (1 - P_iM) * dUdalph)
                      + (self.lambda_M - self.cost_purchasing_capacity_MT) * sum_PiM_Qi
                      + (self.cost_purchasing_capacity_MT * (1 - self.share_TNC) + self.lambda_M * self.share_TNC) 
                      * (np.sum(Q * P_iM * (1 - P_iM) * dUdalph))
        )

        # ========= GRAD w.r.t. λ_M ==========
        grad_lambdaM = self.share_TNC * sum_PiM_Qi - self.capacity_ratio_from_TNC * self.total_service_capacity_TNC

        return np.array([grad_fM, grad_pT ,grad_alpha, grad_lambdaM])

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
        if not self.utilities:
            self.utilities = [0]*len(services)

        for idx, service in enumerate(services):
            U = service.compute_utility(
                trip_length=self.trip_length,
                value_time=self.value_time,
                value_wait=self.value_wait,
            )
            self.utilities[idx] = 0.5 * (self.utilities[idx] + U)  # smoothing

            # print(
            #     f"{service.name}: trip_length={self.trip_length:.4f}, "
            #     f"value_time={self.value_time:.4f}, value_wait={self.value_wait:.4f}, "
            #     f"U={U:.4f}"
            # )
        return
    
    def choose_service(self, services: list[Service]) -> None:
        '''
        Choose service distribution based on a logit model of choice probabilities.
        '''
        self.compute_utilities(services)
        exp_utilities = np.exp(self.utilities) 
        probabilities = exp_utilities / np.sum(exp_utilities)
        self.travelers_per_service = probabilities * self.number_traveler 
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