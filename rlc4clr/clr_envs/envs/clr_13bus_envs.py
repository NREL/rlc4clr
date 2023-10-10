import copy
import os

import gymnasium as gym
import numpy as np
import pandas as pd

from gymnasium import spaces

from clr_envs.envs.der_models import BatteryStorage
from clr_envs.envs.der_models import MicroTurbine
from clr_envs.envs.DEFAULT_CONFIG import *


# Local config
DATA_DIR = os.path.dirname(os.path.abspath(__file__))


class LoadRestoration13BusBaseEnv(gym.Env):
    """ Base gym environment for the 13-bus load restoration problem.

    """

    def __init__(self):

        import opendssdirect as dss
        self.dss = dss

        # Vector z in paper
        self.importance_factor = [1.0, 1.0, 0.9, 0.85, 0.8, 0.8, 0.75,
                                  0.7, 0.65, 0.5, 0.45, 0.4, 0.3, 0.3, 0.2]
        self.load_name = ['671', '634a', '634b', '634c', '645', '646',
                          '692', '675a', '675b', '675c', '611', '652',
                          '670a', '670b', '670c']

        self.main_dss_subdir = 'system_data/13BusUnbalanced/main.dss'

        # self.load_name is provided in the children class.
        self.num_of_load = len(self.load_name)
        # Minimum sustaining steps after load pickup.
        self.epsilon = np.diag([100] * self.num_of_load)

        project_path = os.path.join(DATA_DIR, 'data/')
        exo_data_path = os.path.join(project_path,
                                     'exogenous_data/'
                                     'five_min_renewable_profile.csv')

        self.exo_data = pd.read_csv(exo_data_path)

        self.dss_data_path = os.path.join(project_path, self.main_dss_subdir)
        self.dss.run_command('Redirect ' + self.dss_data_path)

        # Gather critical load information
        self.base_load = []
        for name in self.load_name:
            self.dss.Loads.Name(name)
            self.base_load.append([self.dss.Loads.kW(), self.dss.Loads.kvar()])

        self.base_load = np.array(self.base_load)

        # Define DERs in the system
        self.pv_max_gen = 300
        self.wind_max_gen = 400
        self.mt_max_gen = 400
        self.st_max_gen = 250
        self.wt_profile = None
        self.pv_profile = None
        # Create ST/MT instances.
        self.st = BatteryStorage()
        self.mt = MicroTurbine()

        self.simulation_step = 0
        self.terminated = False
        self.load_pickup_decision_last_step = [0.0] * self.num_of_load
        self.v_lambda = DEFAULT_V_LAMBDA

        self.debug = None
        self.history = None
        self.enable_debug(DEBUG)

        self.time_of_day_list = None
        self.forecasts_len_in_hours = 1

    def enable_debug(self, debug_flag):
        """ If debug mode is enabled or disabled.

        Args:
          debug_flag: A Boolean. Indicate if the run is in debug mode.
            If in debug mode, additional info will be printed and control
            history will be logged.
        """

        self.debug = debug_flag
        if debug_flag:
            self.history = copy.deepcopy(CONTROL_HISTORY_DICT)

    def reset(self, seed=None, options={}):
        """ Resetting the control episode.

        Args:
          seed: [Optional] An integer number. Random seed.
          options: [Optional] A dictionary. Additional configuration on how the
            environment will be initialized. 
            start_index: [Optional] An integer number. The index number
                indicating which time to start the episode. By default, this 
                value is set as None, and an index is randomly selected in the 
                July time frame. When testing using scenarios from August time
                frame, this start_index can be specified.
            init_storage: [Optional] A float number. Initial storage amount 
                (kWh)

        Returns:
          state: A Numpy array. The RL state of the system at the initial step.
          info: A dictionary. Additional information.
        """

        super().reset(seed=seed)

        start_index = (options['start_index'] 
                       if 'start_index' in options.keys() else None)
        init_storage = (options['init_storage'] 
                        if 'init_storage' in options.keys() else None)


        self.simulation_step = 0
        self.terminated = False
        if self.debug:
            self.history = copy.deepcopy(CONTROL_HISTORY_DICT)
        self.load_pickup_decision_last_step = [0.0] * self.num_of_load

        self.mt.reset()
        self.st.reset(init_storage)

        # start_index = 180
        if not start_index:
            # Training scenarios are from July, during training scenarios are
            # randomly chosen.
            # 52128: Index for time 07/01 00:00
            # 60984: Index for time 07/31 18:00
            start_index = np.random.randint(52128, 60984)

        if self.debug:
            print("Scenario index used here is %d" % start_index)

        self.wt_profile = self.exo_data['wind_gen'][start_index:
                                                    start_index
                                                    + CONTROL_HORIZON_LEN
                                                    + STEPS_PER_HOUR]
        self.wt_profile = self.wt_profile.to_numpy() * self.wind_max_gen
        self.pv_profile = self.exo_data['pv_gen'][start_index:
                                                  start_index
                                                  + CONTROL_HORIZON_LEN
                                                  + STEPS_PER_HOUR]
        self.pv_profile = self.pv_profile.to_numpy() * self.pv_max_gen

        self.time_of_day_list = []
        step_time = pd.to_datetime(self.exo_data['Time'][start_index])
        for idx in range(CONTROL_HORIZON_LEN):
            self.time_of_day_list.append(step_time)
            step_time += pd.Timedelta('5min')

        state = self.get_state()

        # Update info
        info = {}

        return state, info

    def update_opendss_pickup_load(self, load_pickup_decision):
        """ Update how loads will be restored at this step in OpenDSS.

        Args:
          load_pickup_decision: List of floats [0.0, 1.0] with length of
            self.num_of_loads. Each element represents how much load is
            restored.
        """

        for load_idx, name in enumerate(self.load_name):
            self.dss.Loads.Name(name)
            self.dss.Loads.kW(self.base_load[load_idx, 0]
                              * load_pickup_decision[load_idx])
            self.dss.Loads.kvar(self.base_load[load_idx, 1]
                                * load_pickup_decision[load_idx])

    def update_opendss_generation(self, p_gen, q_gen):
        """ Updating the generators' settings in OpenDSS

        Micro turbine p_mt/q_mt is not set as it is on the "slack bus" to
        balance line losses because power (gen-load) balance has been enforced.
        We have verified that the p_mt/q_mt is approximately equal to those
        from OpenDSS, with the system losses as the difference.

        Args:
          p_gen: List of floats, Active power generated by four DERs
            [p_pv, p_wt, p_st, p_mt]
          q_gen: List of floats, Reactive power generated by four DERs
            [q_pv, q_wt, q_st, q_mt]

        """

        p_pv, p_wt, p_st, p_mt = p_gen
        q_pv, q_wt, q_st, q_mt = q_gen

        self.dss.Generators.Name('pv')
        self.dss.Generators.kW(p_pv)
        self.dss.Generators.kvar(q_pv)

        self.dss.Generators.Name('wt')
        self.dss.Generators.kW(p_wt)
        self.dss.Generators.kvar(q_wt)

        # Energy Storage (ES) at node 632
        if p_st > 0.0:
            # Discharing: treat as a generator
            self.dss.Generators.Name('esg')
            self.dss.Generators.kW(p_st)
            self.dss.Generators.kvar(q_st)

            self.dss.Loads.Name('esl')
            self.dss.Loads.kW(0.0)
            self.dss.Loads.kvar(0.0)
        else:
            # Charging: treat as a pure resistance load.
            # This node doesn't have other load,
            # otherwise need to sum up the load.

            self.dss.Generators.Name('esg')
            self.dss.Generators.kW(0.0)
            self.dss.Generators.kvar(0.0)

            self.dss.Loads.Name('esl')
            self.dss.Loads.kW(-p_st)
            self.dss.Loads.kvar(0.0)

    @staticmethod
    def get_trigonomical_representation(pd_datetime):
        """ Generate the trigonomical encoding of the time.

        Args:
          pd_datetime: A pandas datetime object.

        Returns:
          A two elements tuple with sine/cosine encoding of the time.
        """

        daily_five_min_position = (STEPS_PER_HOUR * pd_datetime.hour
                                   + pd_datetime.minute / 5)
        degree = daily_five_min_position / 288.0 * 2 * np.pi

        return np.sin(degree), np.cos(degree)

    def get_state(self):
        """ Gather the system state at current step.

        - State dimension is 44, these elements are:
        0 -11: PV generation forecast for the next 1 hour
        12-23: Wind generation forecast for the next 1 hour
        24-38: Load pickup level for all 15 loads
        39-43: [ST SOC, mt_remaining_fuel, current_timestep, sinT, cosT]

        """

        sim_step = self.simulation_step

        # In this version we use the perfect one-hour ahead forecasts for
        # renewables.
        pv_forecast = list(self.pv_profile[
                           sim_step:
                           sim_step + STEPS_PER_HOUR] / self.pv_max_gen)
        wt_forecast = list(self.wt_profile[
                           sim_step:
                           sim_step + STEPS_PER_HOUR] / self.wind_max_gen)

        current_load_status = self.load_pickup_decision_last_step

        battery_soc = self.st.current_storage / self.st.storage_range[1]
        mt_fuel_remain_percentage = \
            self.mt.remaining_fuel_in_kwh / self.mt.original_fuel_in_kwh
        current_step = sim_step / CONTROL_HORIZON_LEN

        sin_t, cos_t = self.get_trigonomical_representation(
            self.time_of_day_list[sim_step])

        return np.array(pv_forecast + wt_forecast + current_load_status +
                        [battery_soc, mt_fuel_remain_percentage, current_step,
                         sin_t, cos_t])

    def render(self, mode='human'):
        pass

    def step(self, action):
        pass

    def get_reward_and_voltages(self, load_pickup_decision):
        """ Calculate reward and get voltages from the simulation.

        Reward consists of three parts: voltage violation, load restored reward
        and load shed penalty.

        Args:
          load_pickup_decision: A list. Dimension equals the number of critical
            loads. Elements are the fraction of load being restored.
            E.g, 0.5 -> 50% restored.

        Returns:
          reward: A float number. Scalar reward calculated.
          voltages: A list of float numbers. Numbers are the P.U. values of the
            voltages.
          voltage_bus_name: A list of strings. Strings are the single phase bus
            ID, e.g., '675.1'.
        """

        # Voltage Violation
        # Skip the first three as they are additional Vsource added by OpenDSS.
        voltages = self.dss.Circuit.AllBusMagPu()[3:]
        voltage_bus_name = self.dss.Circuit.AllNodeNames()[3:]
        voltage_violations = [max((0.0, v - V_MAX, V_MIN - v))
                              for v in voltages]
        voltage_violation_penalty = np.sum(
            [v ** 2 for v in voltage_violations]) * self.v_lambda

        # Load Restoration Reward
        load_restored = [self.base_load[idx, 0] * load_pickup_decision[idx]
                         for idx in range(self.num_of_load)]
        load_restore_reward = np.dot(self.importance_factor, load_restored)

        load_shed = [self.base_load[idx, 0]
                     * max(0.0, self.load_pickup_decision_last_step[idx]
                           - load_pickup_decision[idx])
                     for idx in range(self.num_of_load)]

        load_by_epsilon = np.matmul(self.epsilon,
                                    np.array(load_shed).reshape([-1, 1]))
        load_shed_penalty = float(np.dot(self.importance_factor,
                                         load_by_epsilon))

        self.load_pickup_decision_last_step = load_pickup_decision

        load_only_reward = ((load_restore_reward - load_shed_penalty)
                            * REWARD_SCALING_FACTOR)
        reward = (load_only_reward
                  - voltage_violation_penalty * REWARD_SCALING_FACTOR)

        return reward, load_only_reward, voltages, voltage_bus_name

    def step_gen_only(self, action):
        """ Implementing one step of control considering only DERs dispatch.

        In this simpler version of the problem, reactive power is not actively
        monitored.

        This function consists of four parts:
          1. Action pre-processing: convert the normalized control to their
             original range.
          2. Power balance: according to the available generation at this step,
             we determine how many load to pick up according to their priority.
          3. Control implementation: a. run the power flow; b. update ST and MT
             status.
          4. Post-control process: gather next state, reward and done signal.

        Args:
          action: Numpy array with dimention (6,), Same for the balanced and
            unbalanced case.. These elements are shown as follow:
            Action items: [P_st, alpha_st, P_mt, alpha_mt, alpha_wt, alpha_pv],
              all in [-1.0, 1.0] range.
        """

        # Step 1: Preprocess actions
        action = np.clip(action, self.action_lower, self.action_upper)

        p_st, st_angle, p_mt, mt_angle, wt_angle, pv_angle = action

        # Project p_mt and the angle value from [-1, 1] to [0, 1]
        p_mt = (p_mt + 1) / 2.0
        st_angle = (st_angle + 1) / 2.0
        mt_angle = (mt_angle + 1) / 2.0
        wt_angle = (wt_angle + 1) / 2.0
        pv_angle = (pv_angle + 1) / 2.0

        # Active power
        p_pv = self.pv_profile[self.simulation_step]
        p_wt = self.wt_profile[self.simulation_step]

        p_st *= self.st_max_gen
        p_mt *= self.mt_max_gen

        # validate the storage/MT can really have this power.
        p_st = self.st.validate_power(p_st)
        p_mt = self.mt.validate_power(p_mt)

        # Reactive power
        # Assuming the power factor angle for inverter is 0~45 degree.
        q_pv = p_pv * np.tan(np.pi / 4 * pv_angle)
        q_wt = p_wt * np.tan(np.pi / 4 * wt_angle)
        q_mt = p_mt * np.tan(np.pi / 4 * mt_angle)
        if p_st > 0:
            q_st = p_st * np.tan(np.pi / 4 * st_angle)
        else:
            q_st = 0.0

        # Step 2: Power balance.
        total_gen = [p_pv + p_wt + p_st + p_mt,
                     q_pv + q_wt + q_st + q_mt]

        load_pickup_decision = [0.0 for _ in range(self.num_of_load)]
        total_gen_p = total_gen[0]

        # very large generation, even larger than total load
        if total_gen_p > sum(self.base_load[:, 0]):
            shrinking_ratio = sum(self.base_load[:, 0]) / total_gen_p
            p_pv, p_wt, p_st, p_mt = [x * shrinking_ratio
                                      for x in [p_pv, p_wt, p_st, p_mt]]
            q_pv, q_wt, q_st, q_mt = [x * shrinking_ratio
                                      for x in [q_pv, q_wt, q_st, q_mt]]

            total_gen = [p_pv + p_wt + p_st + p_mt,
                         q_pv + q_wt + q_st + q_mt]
            total_gen_p = total_gen[0]

        # Sequentially determining load restoration according to their priority
        # The base load is ranked from higher priority to lower.
        load_idx = 0
        while total_gen_p > 1e-3:
            load_pickup_decision[load_idx] \
                = min([1.0, total_gen_p / self.base_load[load_idx][0]])
            total_gen_p -= self.base_load[load_idx][0]
            load_idx += 1

        # Step 3: Control implementation and compute power flow

        # Setting updated picked up load
        self.update_opendss_pickup_load(load_pickup_decision)

        # Update generators set points
        p_gen = [p_pv, p_wt, p_st, p_mt]
        q_gen = [q_pv, q_wt, q_st, q_mt]

        # Note the reactive power is not balanced, so the excessive Q or
        # shortfall will be compensate by the Vsource.
        self.update_opendss_generation(p_gen, q_gen)

        # Solve the power flow using the DSS
        self.dss.run_command('Solve mode=snap')

        # Update the actual micro-turbine power generation, to include line
        # loss.
        self.dss.Circuit.SetActiveElement('Vsource.mt')
        mt_power = [-x for x in self.dss.CktElement.Powers()[:6]]
        p_mt = sum([mt_power[i] for i in range(6) if i % 2 == 0])
        q_mt = sum([mt_power[i] for i in range(6) if i % 2 == 1])

        # Implement control to the DERs model
        self.mt.control(p_mt)
        self.st.control(p_st)

        # calculate reward
        (reward, load_only_reward, voltages,
         voltage_bus_name) = self.get_reward_and_voltages(load_pickup_decision)

        self.simulation_step += 1
        state = self.get_state()
        self.terminated = (True 
                           if self.simulation_step >= (CONTROL_HORIZON_LEN - 1) 
                           else False)

        if self.debug:

            # TODO: confirm this is the right way to get slack power.
            Sslack = self.dss.Circuit.TotalPower()

            self.history['load_status'].append(load_pickup_decision)
            self.history['pv_power'].append([p_pv, q_pv])
            self.history['wt_power'].append([p_wt, q_wt])
            self.history['mt_power'].append([p_mt, q_mt])
            self.history['st_power'].append([p_st, q_st])
            self.history['slack_power'].append([Sslack[0], Sslack[1]])
            self.history['voltages'].append(voltages)
            self.history['mt_remaining_fuel']. \
                append(
                self.mt.remaining_fuel_in_kwh / self.mt.original_fuel_in_kwh)
            self.history['st_soc'].\
                append(self.st.current_storage / self.st.storage_range[1])

            if self.terminated:
                self.history['voltage_bus_names'] = voltage_bus_name

        return state, reward, self.terminated, False, {}

    def step_gen_load(self, action):
        """ Implementing one step of control considering both generator
            dispatch and load pick-up.

        This function consists of four parts:
          1. Action pre-processing: convert the normalized control to their
               original range.
          2. Power balance: enforcing power balance constraints: we use some
             pre-defined rules to balance generation and load.
          3. Control implementation: a. run the power flow; b. update ST and
             MT status.
          4. Post-control process: gather next state, reward and done signal.

        Args:
          action: Numpy array with dimention (19,). First 15 elements are load
            pickup decisions (-1, 1)->(0, 100%). The next element is the
            normalized control signal for battery storage,
            (-1, 1)->(Max Charge, Max Discharge). The next three are power
            factor angle for the storage, wind turbine and the PV generation,
            (-1, 1) -> (0, Max angle).
        """

        # Step 1: Pre-process actions

        action = np.clip(action, self.action_lower, self.action_upper)

        # Get total load RL agent plans to pick up this step.
        # (-1, 1) -> (0, 100%) load pickup
        load_pickup_decision = [(x + 1) / 2.0
                                for x in action[:self.num_of_load]]
        (load_picked_up_p, load_picked_up_q) = np.sum(
            [self.base_load[idx] * load_pickup_decision[idx]
             for idx in range(self.num_of_load)], axis=0)

        # Get total generation RL agent plans to have.
        # here we assume storage has max charge = max discharge.
        p_st = action[self.num_of_load] * self.st_max_gen
        # (-1, 1) -> (0, 1)
        st_angle = (action[self.num_of_load + 1] + 1) / 2.0
        wt_angle = (action[self.num_of_load + 2] + 1) / 2.0
        pv_angle = (action[self.num_of_load + 3] + 1) / 2.0

        p_pv = self.pv_profile[self.simulation_step]
        p_wt = self.wt_profile[self.simulation_step]

        # validate the storage can really have this power.
        p_st = self.st.validate_power(p_st)

        # Step 2: Power balance (This step is important because RL itself
        # cannot balance power.)
        # Update MT's generation capacity
        mt_max_gen_pq = [self.mt.validate_power(self.mt_max_gen),
                         self.mt.validate_power(self.mt_max_gen) * 0.75]

        st_gen_contribution = p_st if p_st > 0.0 else 0.0
        # the load part of p_st, only activated when charging
        p_st_load = max(0.0, -1 * p_st)

        # Possible range for total generation
        p_gen_range = [p_pv + p_wt + st_gen_contribution,
                       p_pv + p_wt + st_gen_contribution + mt_max_gen_pq[0]]

        def power_reduction(gen_excess, gen_pv, gen_wt, gen_st, p_st_load):

            if gen_excess <= (gen_pv + gen_wt):
                pv_ratio = gen_pv / (gen_pv + gen_wt)
                gen_pv -= pv_ratio * gen_excess
                gen_wt -= (1 - pv_ratio) * gen_excess
            else:
                # in this case, the storage must be in discharging mode,
                # reducing its power output.
                assert p_st_load == 0.0
                gen_excess -= (gen_pv + gen_wt)
                gen_pv = 0.0
                gen_wt = 0.0
                gen_st -= gen_excess

            return gen_pv, gen_wt, gen_st

        # Implement the logic for balancing load and generation.
        # See GM Paper Algorithm 1.

        if load_picked_up_p + p_st_load >= p_gen_range[1]:  # not enough gen
            # load pickup decision update (cannot pick up that much load,
            # discard less important load)
            if p_st_load > p_gen_range[1]:
                # Usually this does not happen since p_mt can cover p_st's
                # charging. This happens when mt's fuel is running out.
                p_st_load = p_gen_range[1]
                p_mt = mt_max_gen_pq[0]
                # Cannot pick up any other load
                load_pickup_decision = [0.0] * self.num_of_load
                load_picked_up_p = 0.0
                load_picked_up_q = 0.0
            else:
                load_picked_up_p = 0.0
                load_picked_up_q = 0.0
                load_pickup_decision_new = []
                # adjust load pickup decision: not pick up lower priority load.
                for idx, decision in enumerate(load_pickup_decision):
                    if load_picked_up_p + p_st_load < p_gen_range[1]:
                        if (load_picked_up_p + p_st_load
                                + self.base_load[idx, 0] * decision
                                <= p_gen_range[1]):
                            load_picked_up_p += (self.base_load[idx, 0]
                                                 * decision)
                            load_picked_up_q += (self.base_load[idx, 1]
                                                 * decision)
                        else:
                            decision = 0.0
                    else:
                        decision = 0.0

                    load_pickup_decision_new.append(decision)

                load_pickup_decision = load_pickup_decision_new

                if load_picked_up_p + p_st_load < p_gen_range[0]:
                    # this is most likely mt is already out of fuel now.
                    p_mt = 0.0
                    gen_excess = p_gen_range[0] - load_picked_up_p - p_st_load
                    p_pv, p_wt, p_st = power_reduction(
                        gen_excess, p_pv, p_wt, p_st, p_st_load)
                else:
                    p_mt = load_picked_up_p + p_st_load - p_gen_range[0]
        elif load_picked_up_p + p_st_load >= p_gen_range[0]:
            # Enough generation, all generator are used.
            p_mt = load_picked_up_p + p_st_load - p_gen_range[0]
        elif max(p_st, 0.0) < load_picked_up_p + p_st_load < p_gen_range[0]:
            # Too much generation, cut renewable
            p_mt = 0.0
            gen_excess = p_gen_range[0] - load_picked_up_p - p_st_load
            p_pv, p_wt, p_st = power_reduction(
                gen_excess, p_pv, p_wt, p_st, p_st_load)
        else:
            # Still to much generation after renewable are totally curtailed.
            p_pv = 0.0
            p_wt = 0.0
            p_mt = 0.0
            p_st = load_picked_up_p

        if DEBUG:
            print("Gym storage power: %f" % p_st)
            print("Gym mt power: %f" % p_mt)

        # Assuming the maximum angle for inverter is 45 degree (pi/4).
        q_pv = p_pv * np.tan(np.pi / 4 * pv_angle)
        q_wt = p_wt * np.tan(np.pi / 4 * wt_angle)
        if p_st > 0:
            q_st = p_st * np.tan(np.pi / 4 * st_angle)
        else:
            q_st = 0.0

        q_gen_range = [q_pv + q_wt + q_st,
                       q_pv + q_wt + q_st + mt_max_gen_pq[1]]

        q_vsource = 0.0
        if load_picked_up_q > q_gen_range[1]:
            if self.debug:
                print("Reactive power shortage, Vsource is compensating."
                      " %f, %f" % (load_picked_up_q, q_gen_range[1]))
                # TODO: Add penalty for using Vsource Q.
            q_vsource = load_picked_up_q - q_gen_range[1]
            q_mt = mt_max_gen_pq[1]
        elif load_picked_up_q > q_gen_range[0]:
            q_mt = load_picked_up_q - q_gen_range[0]
        elif q_st < load_picked_up_q < q_gen_range[0]:
            q_mt = 0.0
            q_excess = q_gen_range[0] - load_picked_up_q
            q_pv, q_wt, q_st = power_reduction(q_excess, q_pv, q_wt,
                                               q_st, p_st_load)
        else:
            q_mt = 0.0
            q_pv = 0.0
            q_wt = 0.0
            q_st = load_picked_up_q

        # Step 3: Control implementation
        # Compute power flow
        self.update_opendss_pickup_load(load_pickup_decision)

        # Add Generators
        p_gen = [p_pv, p_wt, p_st, p_mt]
        q_gen = [q_pv, q_wt, q_st, q_mt]
        self.update_opendss_generation(p_gen, q_gen)

        # Solve the power flow using the DSS.
        self.dss.run_command('Solve mode=snap')

        # Update the actual MT power generation, to include line loss.
        self.dss.Circuit.SetActiveElement('Vsource.mt')
        mt_power = [-x for x in self.dss.CktElement.Powers()[:6]]
        p_mt = sum([mt_power[i] for i in range(6) if i % 2 == 0])
        q_mt = sum([mt_power[i] for i in range(6) if i % 2 == 1])

        if DEBUG:
            loss_p, loss_q = self.dss.Circuit.Losses()  # These come in Watt.
            print("Loss at this step is: %f kW and %f kvar" % (loss_p / 1000.,
                                                               loss_q / 1000.))
            print("Lost energy is %f kWh" %
                  (loss_p / 1000.0 * STEP_INTERVAL_IN_HOUR))

        self.mt.control(p_mt)
        self.st.control(p_st)

        # Step 4: Post-control process

        (reward, load_only_reward, voltages,
         voltage_bus_name) = self.get_reward_and_voltages(load_pickup_decision)

        q_vsource_penalty = 0.1 * q_vsource  # TODO: Update this unit penalty
        reward -= q_vsource_penalty * REWARD_SCALING_FACTOR

        self.simulation_step += 1
        state = self.get_state()
        self.terminated = (True 
                           if self.simulation_step >= (CONTROL_HORIZON_LEN - 1) 
                           else False)

        if self.debug:
            Sslack = self.dss.Circuit.TotalPower()

            self.history['load_status'].append(load_pickup_decision)
            self.history['pv_power'].append([p_pv, q_pv])
            self.history['wt_power'].append([p_wt, q_wt])
            self.history['mt_power'].append([p_mt, q_mt])
            self.history['st_power'].append([p_st, q_st])
            self.history['slack_power'].append([Sslack[0], Sslack[1]])
            self.history['voltages'].append(voltages)
            self.history['mt_remaining_fuel'].append(
                self.mt.remaining_fuel_in_kwh / self.mt.original_fuel_in_kwh)
            self.history['st_soc'].append(
                self.st.current_storage / self.st.storage_range[1])

            if self.terminated:
                self.history['voltage_bus_names'] = voltage_bus_name

        info = {'load_only_reward': load_only_reward}

        return state, reward, self.terminated, False, info

    def get_control_history(self):

        results = {
            'pv_power': np.array(self.history['pv_power']),
            'wt_power': np.array(self.history['wt_power']),
            'mt_power': np.array(self.history['mt_power']),
            'st_power': np.array(self.history['st_power']),
            'slack_power': np.array(self.history['slack_power']),
            'mt_remaining_fuel': np.array(self.history['mt_remaining_fuel']),
            'st_soc': np.array(self.history['st_soc']),
            # 'total_power': total_power,
            'voltages': np.array(self.history['voltages']).transpose(),
            'voltage_bus_names': np.array(self.history['voltage_bus_names']),
            'load_status': np.array(self.history['load_status']),
            'time_stamps': self.time_of_day_list,
        }

        return results


class LoadRestoration13BusUnbalancedSimplified(LoadRestoration13BusBaseEnv):
    """ A greedy load pickup controller that only learned to control
        dispatchable DERs for the 13 bus system.

    Load pickup decision is based on a greedy rule: Picking up higher priority
      load first.

    """

    def __init__(self):

        super(LoadRestoration13BusUnbalancedSimplified, self).__init__()

        # See self.get_state() for details on the observation dimension.
        dim_obs = 44  # 12 wind_forecast + 12 pv forecast + 15 load + 5 scalar.
        scalar_obs_upper = np.array([1.0] * dim_obs)
        # last two are for sinT and cosT, using -1 as lower bound.
        scalar_obs_lower = np.array([0.0] * (dim_obs - 2) + [-1.0] * 2)
        self.observation_space = spaces.Box(scalar_obs_lower, scalar_obs_upper,
                                            dtype=np.float64)

        self.action_upper = np.array([1.0] * 6)
        self.action_lower = np.array([-1.0] * 6)
        self.action_space = spaces.Box(self.action_lower, self.action_upper,
                                       dtype=np.float64)

    def step(self, action):
        return self.step_gen_only(action)


class LoadRestoration13BusUnbalancedFull(LoadRestoration13BusBaseEnv):
    """ A load restoration controller that control both generation and load
        pickup for the unbalanced 13 bus system.

    Compared with V0, the controller trained with this environment will
      determine both generation and load.

    """

    def __init__(self):

        super(LoadRestoration13BusUnbalancedFull, self).__init__()

        # See self.get_state() for details on the observation dimension.
        dim_obs = 44  # 12 wind_forecast + 12 pv forecast + 15 load + 5 scalar.
        scalar_obs_upper = np.array([1.0] * dim_obs)
        # last two are for sinT and cosT, using -1 as lower bound.
        scalar_obs_lower = np.array([0.0] * (dim_obs - 2) + [-1.0] * 2)
        self.observation_space = spaces.Box(scalar_obs_lower, scalar_obs_upper,
                                            dtype=np.float64)

        self.action_upper = np.array([1.0] * 19)
        self.action_lower = np.array([-1.0] * 19)
        self.action_space = spaces.Box(self.action_lower, self.action_upper,
                                       dtype=np.float64)

    def step(self, action):
        return self.step_gen_load(action)


