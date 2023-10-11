

DEBUG = True

CONTROL_HORIZON_LEN = 72
STEPS_PER_HOUR = 12
STEP_INTERVAL_IN_HOUR = 1. / STEPS_PER_HOUR

REWARD_SCALING_FACTOR = 0.001

DEFAULT_V_LAMBDA = 1e8
DEFAULT_ERROR_LEVEL = 0.1

V_MAX = 1.05
V_MIN = 0.95

CONTROL_HISTORY_DICT = {"load_status": [],
                        "pv_power": [],
                        "wt_power": [],
                        "mt_power": [],
                        "st_power": [],
                        "slack_power": [],
                        "voltages": [],
                        "mt_remaining_fuel": [],
                        "st_soc": []}

# Local config
# absolute path for pseudo forecasts files.
PSEUDO_FORECASTS_DIR = '/Users/xzhang2/CodeRepo/agm_dataset/json_data'
