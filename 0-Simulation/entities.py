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
        """Compute total fare for a trip of given length.

        Parameters
        - trip_length (float): trip distance in km.

        Returns
        - float: monetary fare for that trip.
        """
        raise NotImplementedError

    @abstractmethod
    def trip_time(self, trip_length: float) -> float:
        """Compute total travel time for a trip of given length.

        Parameters
        - trip_length (float): trip distance in km.

        Returns
        - float: expected in-vehicle travel time in hours.
        """
        raise NotImplementedError

    @abstractmethod
    def waiting_time(self, trip_length: float) -> float:
        """Compute expected waiting time for a trip of given length.

        Parameters
        - trip_length (float): trip distance in km (may be unused by some
            services such as TNC that compute waiting from fleet state).

        Returns
        - float: expected waiting time in hours.
        """
        raise NotImplementedError

    def compute_utility(self, trip_length: float, value_time: float, value_wait: float) -> float:
        """
        Generic utility computation (can be overridden).
        """
        # Default behavior (valid for MT and MaaS)
        fare = self.trip_fare(trip_length)
        time = self.trip_time(trip_length)
        wait = self.waiting_time(trip_length)
        # Utility is ASC minus monetary fare and time/disutility terms.
        return self.ASC - fare - value_time * time - value_wait * wait
    
    def get_allocation(self, allocation: dict[str, list[float]]) -> None:
        """Provide the service with the current allocation (demand) per traveler type.

        Parameters
        - allocation: dict mapping service name -> list of counts per traveler type.
            Example: {"TNC": [n0, n1, n2], "MT": [..], "MaaS": [...]}

        Side effects
        - sets `self.demand_per_traveler_type` used by capacity and waiting-time
            calculations elsewhere in the class.
        """
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
                 trip_length_per_traveler_type: list[float],
                 value_waiting_time_per_traveler_type: list[float],
                 cost_purchasing_capacity_TNC: float,
                 operating_cost: float):
                super().__init__(name="TNC") 
                
                """Initialize a TNC service model.

                Parameters
                - ASC (float): alternative-specific constant (utility intercept).
                - fare (float): base monetary fare per unit distance.
                - detour_ratio (float): multiplicative factor for distance/time due to
                    detours (route inefficiency).
                - average_speed (float): average vehicle speed in km/h.
                - average_veh_travel_dist_per_day (float): km covered by an idle/used
                    vehicle per day (used to convert capacity in veh*km to vehicle counts).
                - capacity_ratio_to_MaaS (float): fraction of TNC capacity allocated to MaaS.
                - total_service_capacity (float): total service capacity in veh*km per day.
                - trip_length_per_traveler_type (list[float]): trip km per traveler group.
                - value_waiting_time_per_traveler_type (list[float]): monetary value per hour
                    of waiting time, per traveler group.
                - cost_purchasing_capacity_TNC (float): capital cost per unit of capacity.
                - operating_cost (float): operating cost per unit of capacity.

                Attributes set on the instance mirror these parameters and also
                initialize internal variables used by other methods (e.g. `lambda_T`).
                """
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
                
                self.trip_length_per_traveler_type: list[float] = trip_length_per_traveler_type 
                self.demand_per_traveler_type: dict[str, list[float]] | None = None
                self.value_waiting_time_per_traveler_type: list[float] = value_waiting_time_per_traveler_type 
                self.vacant_veh_available: float | None = None  

    def trip_fare(self, trip_length: float) -> float:
        """Return monetary fare for a given trip length (km).

        Variables used:
        - self.fare: base fare per km
        - self.detour_ratio: multiplies distance to reflect detours
        - trip_length: trip distance in km
        Returns monetary units (float).
        """
        return self.fare * self.detour_ratio * trip_length 

    def trip_time(self, trip_length: float) -> float:
        """Return expected in-vehicle travel time (hours) for trip_length.

        Variables used:
        - self.detour_ratio: increases nominal distance/time due to detours
        - trip_length: distance in km
        - self.average_speed: speed in km/h
        Returns hours (float).
        """
        return self.detour_ratio * trip_length / self.average_speed 

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
            # If zero or negative (not enough veh available), fall back to a tiny positive value 
            # so the waiting time becomes very large and effectively deters choice.
            print(f"[Warning] {self.name}: No vacant vehicles available — using minimum threshold.")
            vacant_veh_available = max(self.find_vacant_veh_available(), 1e-6)
        # Empirical relationship: waiting time grows as a negative power of
        # vacant vehicles available.
        return A * (vacant_veh_available ** (-sensitivity_param)) 

    def find_vacant_veh_available(self) -> float:
        """
        Compute the number of idle vehicles in the TNC fleet (in veh per day).
        """
        # total_demand (veh*km per day) computed as sum over traveler types of
        # (trip_length_i * allocated_travelers_i). 
        total_demand = np.sum(
            np.array(self.trip_length_per_traveler_type)
            * np.array(self.demand_per_traveler_type[self.name])
        )
        # Convert available capacity (veh*km) into number of vehicles by
        # dividing by average km per vehicle per day.
        return ((1 - self.capacity_ratio_to_MaaS) * self.total_service_capacity - total_demand) / self.average_veh_travel_dist_per_day
    
    def compute_objective_function(self, U: np.ndarray, service_index_T: int = 0) -> float:
        """Compute the TNC objective function given utilities matrix U.

        Parameters
        - U: ndarray shape (n_types, n_services) of utility values U_im for each
            traveler type i and service m.
        - service_index_T: int index of the TNC column in U (default 0).

        Internal variables
        - l: array of trip lengths per traveler type (l_i).
        - Q: array of total travelers per type aggregated across services (Q_i).
        - P: softmax probabilities over services from U (shape n_types x n_services).
        - P_iT: column vector of probabilities of choosing TNC for each traveler type.
        - sum_l_PiT_Qi: sum_i l_i * P_iT * Q_i (veh*km demand assigned to TNC).
        - sum_PiT_Qi: sum_i P_iT * Q_i (number of travelers choosing TNC weighted by Q).

        Objective terms (economic interpretation):
        - term1: TNC fare revenue (negative cost here because we minimize cost-like objective).
        - term2: cost of purchasing capacity allocated to MaaS from TNC.
        - term3: operating cost of running the TNC fleet.
        - term4: Lagrangian term enforcing vehicle capacity constraint (lambda_T).

        Returns
        - float: scalar value of the objective.
        """
        l = np.asarray(self.trip_length_per_traveler_type)
        Q = np.sum(list(self.demand_per_traveler_type.values()), axis=0)
        P = np.exp(U)
        P /= np.sum(P, axis=1, keepdims=True)
        P_iT = P[:, service_index_T]   # Column for TNC

        sum_l_PiT_Qi = np.sum(l * P_iT * Q)
        sum_PiT_Qi = np.sum(P_iT * Q)

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
        """Compute gradient of the TNC objective w.r.t. decision variables.

        Variables and shapes
        - U: ndarray (n_types x n_services) utility matrix.
        - l: 1d array of trip lengths per traveler type (l_i).
        - Q: 1d array sum of demands across services per traveler type (Q_i).
        - P: softmax probabilities from U (n_types x n_services).
        - P_iT: probabilities of choosing TNC for each traveler type.
        - P_iM: probabilities for the MT or MaaS column used in cross-derivatives.

        Returns
        - ndarray: gradient vector [dObj/df_T, dObj/dy_T, dObj/dlambda_T].
        """
        l = np.asarray(self.trip_length_per_traveler_type)
        Q = np.sum(list(self.demand_per_traveler_type.values()), axis=0)
        P = np.exp(U)
        P /= np.sum(P, axis=1, keepdims=True)

        P_iT = P[:, service_index_T]    # Choice prob for TNC
        P_iM = P[:, service_index_M]    # for MT (needed in y_T gradient)

        # Partial derivatives of utility wrt fare f_T
        dUdf = -self.detour_ratio * l

        # Waiting-time sensitivity terms (A and s match waiting_time implementation)
        A, s = 2.5, 0.5
        vacant = self.find_vacant_veh_available()
        dUdy = - np.asarray(self.value_waiting_time_per_traveler_type) * s * A * (vacant)**(-(s + 1)) * (self.total_service_capacity / self.average_veh_travel_dist_per_day)

        sum_l_PiT_Qi = np.sum(l * P_iT * Q)
        sum_PiT_Qi = np.sum(P_iT * Q)

        # ========= GRAD w.r.t. f_T ==========
        grad_fT = (
            -self.detour_ratio * sum_l_PiT_Qi
            - self.fare * self.detour_ratio * np.sum(l * Q * P_iT * (1 - P_iT) * dUdf)
            + self.lambda_T * self.average_veh_travel_dist_per_day * np.sum(Q * P_iT * (1 - P_iT) * dUdf))

        # ========= GRAD w.r.t. y_T ==========
        grad_yT = (
            -self.fare * self.detour_ratio * np.sum(l * Q * P_iT *
                ((1 - P_iT) * dUdy - P_iM * dUdy))
            - self.fare * self.total_service_capacity
            + self.lambda_T * self.average_veh_travel_dist_per_day * np.sum(Q * P_iT *
                ((1 - P_iT) * dUdy - P_iM * dUdy))
            + self.lambda_T * self.total_service_capacity)

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
                
                """Initialize MT service parameters.

                Parameters
                - ASC: alternative-specific constant for MT utility.
                - fare: base fare (monetary units).
                - n_transfer_per_length: number of transfers per km (affects fare/time).
                - detour_ratio: detour factor for travel time calculations.
                - average_speed: average travel speed (km/h).
                - access_time: access/egress time per access leg in hours.
                - transit_time: in-vehicle transit time per transfer in hours.
                """
                self.ASC = ASC
                self.fare = fare
                self.n_transfer_per_length = n_transfer_per_length
                self.detour_ratio = detour_ratio 
                self.average_speed = average_speed
                self.access_time = access_time
                self.transit_time = transit_time

    def trip_fare(self, trip_length: float) -> float:
        """Compute MT fare for a given trip length.

        Variables used:
        - self.fare: base fare component
        - self.n_transfer_per_length: number of transfers per km,
            multiplied by `trip_length` and added to the base 1 fare component.
        Returns monetary units (float).
        """
        return self.fare * (self.n_transfer_per_length * trip_length + 1)

    def trip_time(self, trip_length: float) -> float:
        """Return in-vehicle travel time for MT (hours).

        Uses `detour_ratio` and `average_speed` to convert distance to time.
        """
        return self.detour_ratio * trip_length / self.average_speed

    def waiting_time(self, trip_length: float) -> float:
        """Compute expected waiting/access time for MT trips.

        Variables used:
        - self.access_time: time for access/egress legs (hours); multiplied by 2
            for both access and egress.
        - self.transit_time: base transit time per transfer.
        - self.n_transfer_per_length: transfers per km used to scale transit_time.
        Returns hours (float).
        """
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
                 trip_length_per_traveler_type: list[float],
                 value_travel_time_per_traveler_type: list[float],
                 value_waiting_time_per_traveler_type: list[float],
                 detour_ratio_MT: float,
                 average_speed_MT: float,
                 transit_time_MT: float,
                 n_transfer_per_length_MT: float,
                 cost_purchasing_capacity_MT: float
                 ):
                super().__init__(name="MaaS") 

                """Initialize a MaaS platform combining TNC and MT.

                Parameters mirror those of the TNC and MT models and include:
                - ASC, fare: platform-level utility intercept and fare per km.
                - share_TNC: fraction of each MaaS trip performed by TNC (first/last km).
                - detour_ratio_TNC, average_speed_TNC: TNC travel characteristics.
                - capacity_ratio_from_TNC: fraction of TNC capacity provided to MaaS.
                - total_service_capacity_TNC: TNC capacity in veh*km/day accessible to MaaS.
                - average_veh_travel_dist_per_day_TNC: km per vehicle per day for TNC fleet.
                - cost_purchasing_capacity_TNC/MT: cost parameters for capacity acquisition.
                - trip_length_per_traveler_type: list of trip km by traveler type.
                - value_travel_time_per_traveler_type / value_waiting_time_per_traveler_type:
                    monetary values per hour used in utility sensitivity.

                The instance stores these parameters as attributes for later computation.
                """
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

                self.trip_length_per_traveler_type: list[float] = trip_length_per_traveler_type
                self.value_travel_time_per_traveler_type: list[float] = value_travel_time_per_traveler_type
                self.value_waiting_time_per_traveler_type: list[float] = value_waiting_time_per_traveler_type
                self.demand_per_traveler_type: dict[str, list[float]] | None = None
                self.vacant_veh_available: float | None = None  

                # MT parameters
                self.detour_ratio_MT = detour_ratio_MT
                self.average_speed_MT = average_speed_MT
                self.transit_time_MT = transit_time_MT
                self.n_transfer_per_length_MT = n_transfer_per_length_MT
                self.cost_purchasing_capacity_MT = cost_purchasing_capacity_MT

    def trip_fare(self, trip_length: float) -> float:
        """Return MaaS monetary fare for a given trip length (monetary units).

        Variables used:
        - self.fare: per-km platform fare (may include markup/wholesale blending).
        - trip_length: km of the trip.
        """
        return self.fare * trip_length

    def trip_time(self, trip_length: float) -> float:
        """Return the expected in-vehicle time (hours) when using the MaaS bundle.

        Variables used:
        - time_TNC: contribution from the portion of the trip performed by TNC.
        - time_MT: contribution from the portion of the trip performed by MT.
        The total is the sum of both components.
        """
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
        Compute the number of idle vehicles in the MaaS fleet (in veh).
        """
        total_demand = self.share_TNC * np.sum(np.array(self.trip_length_per_traveler_type) * np.array(self.demand_per_traveler_type[self.name])) # MM : demand per traveler type for MaaS !!!
        return (self.capacity_ratio_from_TNC * self.total_service_capacity_TNC - total_demand) / self.average_veh_travel_dist_per_day_TNC

    def compute_objective_function(self, U: np.ndarray, service_index_M: int = 2) -> float:
        """Compute MaaS objective function from utilities U.

        Parameters
        - U: ndarray (n_types x n_services) utility matrix.
        - service_index_M: index of the MaaS column in U (default 2).

        Variables
        - l: trip lengths per traveler type.
        - Q: total demand per traveler type aggregated across services.
        - P: softmax probabilities.
        - P_iM: probabilities of choosing MaaS per traveler type.

        Objective terms:
        - term1: fare revenue from MaaS users (negative sign as cost/revenue term).
        - term2: MT capacity purchasing cost for MaaS users.
        - term3: TNC capacity purchasing cost associated with MaaS.
        - term4: Lagrangian penalty enforcing capacity constraints (lambda_M).
        """
        l = np.asarray(self.trip_length_per_traveler_type)
        Q = np.sum(list(self.demand_per_traveler_type.values()), axis=0) 

        # get P_im from U_im
        P = np.exp(U)
        P /= np.sum(P, axis=1, keepdims=True)
        P_iM = P[:, service_index_M]   # Column for MaaS

        sum_l_PiM_Qi = np.sum(l * P_iM * Q)
        sum_PiM_Qi = np.sum(P_iM * Q)

        # Build 4-term objective
        term1 = -self.fare * sum_l_PiM_Qi
        term2 = self.cost_purchasing_capacity_MT * (1 - self.share_TNC) * sum_PiM_Qi
        term3 = self.cost_purchasing_capacity_TNC * self.capacity_ratio_from_TNC * self.total_service_capacity_TNC
        term4 = self.lambda_M * (self.share_TNC * sum_PiM_Qi - self.capacity_ratio_from_TNC * self.total_service_capacity_TNC)

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

        """Compute gradient of the MaaS objective w.r.t. platform decision vars.

        Returns gradient vector in the order: [dObj/df_M, dObj/dp_T, dObj/dalpha, dObj/dlambda_M]

        Internal variables
        - l: trip lengths per traveler type
        - Q: demand per traveler type aggregated across services
        - P: softmax probabilities
        - P_iM: probabilities of MaaS per traveler type
        - dUdf: partial derivative of utility with respect to MaaS fare (vector)
        - dUdalph: partial derivative of utility w.r.t the MaaS share parameter
        - vacant: vacant vehicles available for MaaS TNC portion
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
        sum_PiM_Qi = np.sum(P_iM * Q)

        # Partial derivatives of utility:
        dUdf = - l

        A, s = 2.5, 0.5
        vacant = self.find_vacant_veh_available()


        dUdalph = - np.asarray(self.value_travel_time_per_traveler_type) * (self.detour_ratio_TNC / self.average_speed_TNC * l - self.detour_ratio_MT / self.average_speed_MT * l) \
                  - np.asarray(self.value_waiting_time_per_traveler_type) * (s * A * (vacant)**(-(s + 1)) * (sum_l_PiM_Qi / self.average_veh_travel_dist_per_day_TNC) - self.transit_time_MT * self.n_transfer_per_length_MT * l)

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
                """Represents a homogeneous group of travelers.

                Attributes
                - number_traveler (int): group size.
                - trip_length (float): typical trip distance for the group (km).
                - value_time (float): monetary value of in-vehicle time (monetary units/hour).
                - value_wait (float): monetary value of waiting time (monetary units/hour).
                - utilities (list[float] | None): smoothed utilities per service computed
                    by `compute_utilities`.
                - travelers_per_service (list[float] | None): expected number of
                    travelers in this group assigned to each service after `choose_service`.
                """
                self.number_traveler = number_traveler
                self.trip_length = trip_length
                self.value_time = value_time
                self.value_wait = value_wait
                self.utilities: list[float] | None = None
                self.travelers_per_service: list[float] | None = None

    def compute_utilities(self, services: list[Service]) -> None:
        """Compute (and smooth) utilities of available services for this group.

        Parameters
        - services: list of Service instances available to choose from.

        Side effects
        - sets/updates `self.utilities`, a list with one utility per service.

        Local variables
        - idx: index of the current service in `services`.
        - U: utility computed by `service.compute_utility` for this traveler group.

        The implementation smooths utilities by averaging previous utility and
        current computed value (simple exponential smoothing with alpha=0.5).
        """
        if not self.utilities:
            self.utilities = [0] * len(services)

        for idx, service in enumerate(services):
            U = service.compute_utility(
                trip_length=self.trip_length,
                value_time=self.value_time,
                value_wait=self.value_wait,
            )
            self.utilities[idx] = 0.5 * (self.utilities[idx] + U)  # smoothing
        return
    
    def choose_service(self, services: list[Service]) -> None:
        '''
        Choose service distribution based on a logit model of choice probabilities.
        '''
        # Compute (smoothed) utilities first
        self.compute_utilities(services)
        # Convert utilities into choice probabilities via softmax/logit
        exp_utilities = np.exp(self.utilities)
        probabilities = exp_utilities / np.sum(exp_utilities)
        # Allocate the group's travellers across services according to probabilities
        self.travelers_per_service = probabilities * self.number_traveler
        return

# --------------------------
# Functions
# --------------------------
def distribute_travelers(travelers: list[Travelers], services: list[Service]) -> dict[str, list[float]]: 
        """Allocate groups of travelers among services using each group's choice model.

        Parameters
        - travelers: list of Travelers groups.
        - services: list of Service instances (their .name attributes determine keys).

        Returns
        - allocation: dict mapping service name -> list of floats giving the number
            of travelers of each traveler group assigned to that service.

        Example structure:
            allocation = {
                    "TNC": [n_type0, n_type1, n_type2],
                    "MT":  [..],
                    "MaaS":[..]
            }
        """
        allocation = {service.name: [0] * len(travelers) for service in services}

        for type_i, traveler in enumerate(travelers):
            traveler.choose_service(services)
            for index, service in enumerate(services):
                allocation[service.name][type_i] += traveler.travelers_per_service[index]
        return allocation 