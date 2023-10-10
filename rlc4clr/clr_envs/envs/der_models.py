import numpy as np

from scipy.stats import truncnorm

from clr_envs.envs.DEFAULT_CONFIG import *


class MicroTurbine(object):
    """ A simple Mirco Turbine model.

    Only active power is considered. For reactive power, we assume the MT can
    always support if the power factor angle is within the designed limit.

    """

    def __init__(self):
        self.remaining_fuel_in_kwh = None
        self.original_fuel_in_kwh = 1200.0
        self.reset()

    def reset(self):
        """ Resetting the status of the MT back to its original status.
        """

        self.remaining_fuel_in_kwh = self.original_fuel_in_kwh

    def control(self, power):
        """ Implement a single step of control.

        Args:
          power: A float number, generated power output in kW.
        """

        self.remaining_fuel_in_kwh -= power * STEP_INTERVAL_IN_HOUR

        # For numerical reason, feasibility range might be slightly violated,
        # following two steps bound it.
        self.remaining_fuel_in_kwh = max([0.0, self.remaining_fuel_in_kwh])
        self.remaining_fuel_in_kwh = min([self.remaining_fuel_in_kwh,
                                          self.original_fuel_in_kwh])

    def validate_power(self, power):
        """ Validate if such power can be provided given the fuel availability.

        Args:
          power: A float number, power in kW.
        Returns:
          power: A float number, feasibility validated power in kW
        """
        if self.remaining_fuel_in_kwh <= 0:
            power = 0.0

        # Still energy remaining, but not enough
        if power * STEP_INTERVAL_IN_HOUR > self.remaining_fuel_in_kwh:
            power = self.remaining_fuel_in_kwh / STEP_INTERVAL_IN_HOUR

        return power


class BatteryStorage(object):
    """ A simple Battery Storage model.

    Only active power is considered. For reactive power, we assume the
    inverter can always support
    if the power factor angle is within the designed limit.

    """

    def __init__(self):

        self.storage_range = [160.0, 1250.0]

        self.initial_storage_mean = 1000.0  # kWh
        self.initial_storage_std = 250.0
        self.charge_efficiency = 0.95
        self.discharge_efficiency = 0.9

        self.current_storage = None

    def reset(self, init_storage=None):
        """ Reset the battery storage at the beginning of an episode.

        Args:
          init_storage: A float within the self.storage_range limit.

        """
        if init_storage is None:
            # Initial battery storage is sampled from a truncated normal
            # distribution.
            # On HPC, the output is a numpy array, so adding a float operator
            # to force converting the type.
            self.current_storage = float(truncnorm(-1, 1).rvs()
                                         * self.initial_storage_std
                                         + self.initial_storage_mean)
        else:
            try:
                init_storage = float(init_storage)
                init_storage = np.clip(init_storage, self.storage_range[0],
                                       self.storage_range[1])
            except (TypeError, ValueError) as e:
                print(e)
                print("init_storage value needs to be a float,"
                      "use default value instead")
                init_storage = self.initial_storage_mean

            self.current_storage = init_storage

        if DEBUG:
            print("The initial storage for the battery system is %f."
                  % self.current_storage)

    def control(self, power):
        """ Implement control to the storage.

        Args:
          power: A float, the controlled power to the storage. It discharges
            if the value is positive, else it is negative.
        """

        if power < 0:
            self.current_storage -= (self.charge_efficiency * power 
                                     * STEP_INTERVAL_IN_HOUR)
        elif power > 0:
            self.current_storage -= (power * STEP_INTERVAL_IN_HOUR 
                                     / self.discharge_efficiency)

    def validate_power(self, power):
        """ Validate if such power can be provided based on current SOC.

        Args:
          power: A float number, power in kW.
        Returns:
          power: A float number, feasibility validated power in kW
        """

        if power > 0:
            # ensure the discharging power is within the range.
            if (self.current_storage 
                - power * STEP_INTERVAL_IN_HOUR / self.discharge_efficiency 
                < self.storage_range[0]):
                power = max(self.current_storage - self.storage_range[0],
                            0.0) / STEP_INTERVAL_IN_HOUR

        elif power < 0:
            # ensure charging does not exceed the limit
            if (self.current_storage 
                - self.charge_efficiency * power * STEP_INTERVAL_IN_HOUR 
                > self.storage_range[1]):
                power = (-max(self.storage_range[1] 
                             - self.current_storage, 0.0) 
                             / STEP_INTERVAL_IN_HOUR)

        return power
