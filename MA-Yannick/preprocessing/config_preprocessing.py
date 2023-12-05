import config_computer
# noinspection PyUnresolvedReferences
from config_influx import *
from config_computer import *
from enum import IntEnum

if config_computer.computer == ComputerID.YANNICK_LAPTOP:
    SD_IMAGE_PATH = "/home/yannick/Documents/SD/S01/"
    SD_IMAGE_FILE = "/home/yannick/Documents/SD/S01/2023-02-01_S01.img"

    CSV_WORKING_DIR = "/home/yannick/Documents/SD/"
    CSV_RESULT_DIR = "/home/yannick/Documents/SD/"
    LSTM_INPUT_DIR = "/home/yannick/Documents/SD/"
    # CSV_FILENAME_01_SD_UPTIME = "01.SDByteUptime.csv"
    # CSV_FILENAME_01_SD_UPTIME_PEAKS = "partSD.csv"
    # CSV_FILENAME_02_INFLUX_UPTIME = "uptime%sInfluxDB.csv"
    # CSV_FILENAME_02_INFLUX_UPTIME_PEAKS = "partInflux.csv"
    # CSV_FILENAME_03_TIMESTAMP_AND_SD_BLOCK_ID_PEAKS = "3.exportTimeToByteBlock.csv"
    # CSV_FILENAME_04_SD_BLOCK_ID_UPTIME_TIMESTAMP = "temp2.csv"

elif config_computer.computer == ComputerID.MATZE_IPE_PC:
    SD_IMAGE_PATH = "F:\\Battery_Cycler\\bck\\2023-02-01_BACKUP_LG_HG2\\SD\\"
    CSV_WORKING_DIR = "F:\\Battery_Cycler\\analysis\\preprocessing\\doodle\\"
    CSV_RESULT_DIR = "F:\\Battery_Cycler\\analysis\\preprocessing\\result\\"
    LSTM_INPUT_DIR = "F:\\Battery_Cycler\\analysis\\preprocessing\\lstm_input\\"

elif config_computer.computer == ComputerID.MATZE_IPE_LAPTOP:
    SD_IMAGE_PATH = "C:\\Users\\zc2932\\batcyc\\2023-02-01_BACKUP_LG_HG2\\SD_copy\\"
    CSV_WORKING_DIR = "C:\\Users\\zc2932\\batcyc\\analysis\\preprocessing\\doodle\\"
    CSV_RESULT_DIR = "C:\\Users\\zc2932\\batcyc\\analysis\\preprocessing\\result\\"
    LSTM_INPUT_DIR = "C:\\Users\\zc2932\\batcyc\\analysis\\preprocessing\\lstm_input\\"

elif config_computer.computer == ComputerID.IPE_AVT_SIM:
    SD_IMAGE_PATH = "D:\\Luh\\bat\\img\\"
    CSV_WORKING_DIR = "D:\\Luh\\bat\\analysis\\preprocessing\\doodle\\"
    CSV_RESULT_DIR = "D:\\Luh\\bat\\analysis\\preprocessing\\result\\"
    LSTM_INPUT_DIR = "D:\\Luh\\bat\\analysis\\preprocessing\\lstm_input\\"

CSV_FILENAME_01_SD_UPTIME = "01_SD_block_index_and_uptime_%s%02u.csv"
CSV_FILENAME_01_SD_UPTIME_PEAKS = "01_SD_block_index_and_uptime_peaks_%s%02u.csv"
CSV_FILENAME_02_INFLUX_UPTIME_RAW = "02_InfluxDB_timestamp_and_uptime_raw_%s%02u.csv"
CSV_FILENAME_02_INFLUX_UPTIME = "02_InfluxDB_timestamp_and_uptime_%s%02u.csv"
CSV_FILENAME_02_INFLUX_UPTIME_PEAKS = "02_InfluxDB_timestamp_and_uptime_peaks_%s%02u.csv"
CSV_FILENAME_03_TIMESTAMP_AND_SD_BLOCK_ID_PEAKS = "03_InfluxDB_timestamp_and_SD_block_index_%s%02u.csv"
CSV_FILENAME_04_SD_BLOCK_ID_UPTIME_TIMESTAMP = "04_SD_block_index_uptime_timestamp_%s%02u.csv"
CSV_FILENAME_05_RESULT_BASE_SLAVE = f"slave_%s_%s%02u.csv"  # slave + thermal management
CSV_FILENAME_05_RESULT_BASE_CELL = f"cell_%s_P%03u_%u_S%02u_C%02u.csv"  # cell_log_P017_2_S04_C07.csv
CSV_FILENAME_05_RESULT_BASE_CELL_RE = f"cell_(\w)_P(\d+)_(\d+)_S(\d+)_C(\d+).csv"  # same as above, but for "re" library
CSV_FILENAME_05_RESULT_BASE_POOL = f"pool_%s_T%02u_P%u.csv"
CSV_FILENAME_05_TYPE_CONFIG = "cfg"
CSV_FILENAME_05_TYPE_LOG = "log"
CSV_FILENAME_05_TYPE_LOG_EXT = "logext"
CSV_FILENAME_05_TYPE_EIS = "eis"
CSV_FILENAME_05_TYPE_EOC = "eoc"
CSV_FILENAME_05_TYPE_EOC_FIXED = "eocv2"
CSV_FILENAME_05_TYPE_PULSE = "pls"

# SD file format
SD_IMAGE_FILE_REGEX_SLAVE = "(\S+)_S(\d+).img"
SD_IMAGE_FILE_REGEX_TMGMT = "(\S+)_S(\d+)_TM.img"

SD_BLOCK_SIZE_BYTES = 512
UPTIME_MIN_PEAK_THRESHOLD = 900000  # 900000 = ca. 2.6 h, ignore peaks below this threshold

# if the uptime falls by at least UPTIME_MIN_REBOOT_THRESHOLD, assume there as a reboot in between (new session)
UPTIME_MIN_REBOOT_THRESHOLD = 5000  # ca. 50 seconds, default threshold
UPTIME_MIN_REBOOT_SMALL_THRESHOLD = 1000  # ca. 10 seconds, smaller threshold, used in some gap filling algorithms

# if True, plots are opened for each step - you need to close the plots in order for the program to continue!
PLOT = True


# CSV row names
# CSV_FILE_HEADER_SD_BLOCK_ID = "SD_block_ID"
# CSV_FILE_HEADER_STM_TIM_UPTIME = "stm_tim_uptime"
# CSV_FILE_HEADER_UNIX_TIMESTAMP = "unixtimestamp"
CSV_SEP = ";"
CSV_NEWLINE = "\n"
DELTA_T_LOG = 1.99950336  # CM0 * 2^16 / f_STM = 3051 * 65536 / 100 000 000
DELTA_T_STM_TICK = 0.01048576  # seconds per STM tick
DELTA_T_REBOOT = 4  # assume a reboot takes 4 seconds

# slave general configuration
NUM_SLAVES_MAX = 20
NUM_CELLS_PER_SLAVE = 12
NUM_POOLS_PER_TMGMT = 4
NUM_PELTIERS_PER_POOL = 4


# enum type definitions, gathered comparing LOG/TLOG from SD & Influx. Other packets receive timestamp of last LOG/TLOG.
class TimestampOrigin(IntEnum):
    UNKNOWN = 0  # timestamp unknown (this should only be used in the scripts, not in final data output)
    INFLUX_EXACT = 1  # data from SD, timestamp: exact match with SD data and influx timestamp
    INFLUX_ESTIMATED = 2  # data from SD, timestamp: no exact (only a rough) match with SD data and influx timestamp
    INTERPOLATED = 3  # data from SD, timestamp: no influx data, timestamp interpolated using uptime or fixed steps
    EXTRAPOLATED = 4  # data from SD, timestamp: no influx data, timestamp extrapolated (likely to be wrong)
    INSERTED = 5  # newly inserted timestamp, no data from cycler, measurements: interpolate linearly, status: use last
    ERROR = 6  # timestamp that is known to be false, but this is still the best estimation possible


class SdBox(IntEnum):  # used for battery cycler file system encoding
    SD_BOX_UNUSED = 0  # unused data block
    SD_BOX_LOG_DATA = 1  # slave (cycler) log data
    SD_BOX_EIS_DATA = 2  # electrochemical impedance spectroscopy (EIS) data
    SD_BOX_EOC_DATA = 3  # end of charge/discharge (EOC) data
    SD_BOX_CFG_DATA = 4  # cycler configuration data (defines how the cycler or thermal management is operated)
    SD_BOX_TLOG_DATA = 5  # thermal management log data
    SD_BOX_TCFG_DATA = 6  # thermal management config data (defines how the cycler or thermal management is operated)
    SD_BOX_SPACER = 7  # signalizes that data block is potentially corrupted but more data is coming afterward
    SD_BOX_PULSE_DATA = 8  # pulse pattern data
# 9...15: not used/invalid


class sch_state_ph(IntEnum):  # scheduler phase state, used to extend LOG file (_06_fix_issues_mc.py)
    NONE = 0
    CYCLING = 1
    CHECKUP = 2


class sch_state_cu(IntEnum):  # scheduler check-up state, used to extend LOG file (_06_fix_issues_mc.py)
    NONE = 0
    WAIT_FOR_RT = 1
    PREPARE_SET = 2
    PREPARE_DISCHG = 3
    CAP_MEAS_CHG = 4
    CAP_MEAS_DISCHG = 5
    RT_EIS_PULSE = 6
    WAIT_FOR_OT = 7
    OT_EIS_PULSE = 8
    FOLLOW_UP = 9
    TMGMT_OT = 10
    TMGMT_RT = 11


class sch_state_sub(IntEnum):  # scheduler sub state, used to extend LOG file (_06_fix_issues_mc.py)
    IDLE_PREPARE = 0  # idle or prepare
    WAIT_FOR_START = 1
    START = 2
    RESTART = 3
    RUNNING = 4
    FINISH = 10
    FINISHED = 9
    PAUSE = 14
    PAUSED = 13


class sch_state_chg(IntEnum):  # scheduler charging state, used to extend LOG file (_06_fix_issues_mc.py)
    IDLE = 0
    CHARGE = 1
    DISCHARGE = 2
    DYNAMIC = 3
    CALENDAR_AGING = 4


SD_BOX_MAX_VALID = SdBox.SD_BOX_PULSE_DATA

# parameter set id / nr from slave + cell id
PARAMETER_SET_ID_FROM_SXX_CXX = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # slave 00 --> not used for cell cycling --> use param ID 0
    [20, 24, 67, 65, 19, 27, 18, 26, 17, 25, 1, 3],  # slave 01
    [20, 28, 67, 66, 19, 27, 18, 26, 21, 25, 2, 3],  # slave 02
    [20, 28, 65, 66, 23, 27, 22, 26, 21, 25, 2, 4],  # slave 03
    [24, 28, 65, 66, 23, 18, 22, 17, 21, 1, 2, 4],  # slave 04
    [24, 67, 19, 23, 22, 17, 1, 3, 4, 36, 39, 37],  # slave 05
    [32, 40, 70, 68, 31, 39, 30, 38, 29, 37, 6, 7],  # slave 06
    [32, 40, 70, 68, 31, 39, 34, 38, 33, 5, 6, 8],  # slave 07
    [32, 40, 69, 68, 35, 30, 34, 29, 33, 5, 6, 8],  # slave 08
    [36, 70, 69, 31, 35, 30, 34, 29, 33, 5, 7, 8],  # slave 09
    [36, 69, 35, 38, 37, 7, 48, 72, 51, 50, 49, 11],  # slave 10
    [44, 52, 73, 71, 43, 51, 42, 50, 41, 49, 10, 11],  # slave 11
    [44, 52, 73, 71, 43, 51, 46, 50, 45, 9, 10, 12],  # slave 12
    [44, 52, 72, 71, 47, 42, 46, 41, 45, 9, 10, 12],  # slave 13
    [48, 73, 72, 43, 47, 42, 46, 41, 45, 9, 11, 12],  # slave 14
    [48, 47, 49, 60, 76, 55, 59, 58, 53, 13, 15, 16],  # slave 15
    [56, 60, 76, 75, 55, 63, 54, 62, 53, 61, 13, 15],  # slave 16
    [56, 64, 76, 74, 55, 63, 54, 62, 57, 61, 14, 15],  # slave 17
    [56, 64, 75, 74, 59, 63, 58, 62, 57, 61, 14, 16],  # slave 18
    [60, 64, 75, 74, 59, 54, 58, 53, 57, 13, 14, 16]]  # slave 19

# a "1" at PARAMETER_SET_CELL_NR_FROM_SXX_CXX[x][y] means, that this is the first index with the parameter set ID from
# PARAMETER_SET_ID_FROM_SXX_CXX[x][y]. Filling order: The lowest slave has the lowest cell PARAMETER_SET_CELL_NR.
PARAMETER_SET_CELL_NR_FROM_SXX_CXX = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # slave 00 --> not used for cell cycling --> use index 0
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # slave 01
    [2, 1, 2, 1, 2, 2, 2, 2, 1, 2, 1, 2],  # slave 02
    [3, 2, 2, 2, 1, 3, 1, 3, 2, 3, 2, 1],  # slave 03
    [2, 3, 3, 3, 2, 3, 2, 2, 3, 2, 3, 2],  # slave 04
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1],  # slave 05
    [1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1],  # slave 06
    [2, 2, 2, 2, 2, 3, 1, 2, 1, 1, 2, 1],  # slave 07
    [3, 3, 1, 3, 1, 2, 2, 2, 2, 2, 3, 2],  # slave 08
    [2, 3, 2, 3, 2, 3, 3, 3, 3, 3, 2, 3],  # slave 09
    [3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1],  # slave 10
    [1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2],  # slave 11
    [2, 2, 2, 2, 2, 3, 1, 3, 1, 1, 2, 1],  # slave 12
    [3, 3, 2, 3, 1, 2, 2, 2, 2, 2, 3, 2],  # slave 13
    [2, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3],  # slave 14
    [3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # slave 15
    [1, 2, 2, 1, 2, 1, 1, 1, 2, 1, 2, 2],  # slave 16
    [2, 1, 3, 1, 3, 2, 2, 2, 1, 2, 1, 3],  # slave 17
    [3, 2, 2, 2, 2, 3, 2, 3, 2, 3, 2, 2],  # slave 18
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]]  # slave 19

# SD card file system
BATCYC_FS_INFO_STR = "BatCycFS v1.0 - Number of FS Blocks: \{(\d+)\}, number of data blocks: \[(\d+)\]"
BATCYC_FS_BACKUP_STR = "BatCycFS v1.0 - BACKUP of FS Blocks"
IMAGE_ENCODING = "unicode_escape"  # "utf-8"
NUM_BOXES_PER_FS_BLOCK = 1000
OFFSET_BOX_START_IN_FS_BLOCK = 12

# pulse pattern definition
NUM_PULSE_LOG_POINTS = 61  # 9 * 3 + 11 * 3 + 1 = 61
PULSE_TIME_OFFSET_S = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3,  # short pulse (10s) high (9 points)
                       10, 10.001, 10.003, 10.01, 10.03, 10.1, 10.3, 11, 13,  # short pulse low (9)
                       20, 20.001, 20.003, 20.01, 20.03, 20.1, 20.3, 21, 23,  # short pulse relax (9)
                       30, 30.001, 30.003, 30.01, 30.03, 30.1, 30.3, 31, 33, 40, 60,  # long pulse (30s) high (11)
                       90, 90.001, 90.003, 90.01, 90.03, 90.1, 90.3, 91, 93, 100, 120,  # long pulse low (11)
                       150, 150.001, 150.003, 150.01, 150.03, 150.1, 150.3, 151, 153, 160, 180,  # long pulse relax (11)
                       210]  # end (1)
PULSE_TIME_OFFSET_S_MAX_DECIMALS = 3  # how many numbers after decimal point in PULSE_TIME_OFFSET_S

# electrochemical impedance spectroscopy (EIS) definition
NUM_EIS_SOCS_MAX = 10
NUM_EIS_POINTS_MAX_EXPECTED_ON_SD = 29  # typically, a maximum of 29 points are written to the SD card
NUM_EIS_POINTS_MAX_FITTING_ON_SD = 39  # max. num. of points that fit into SD block (if someone extends the experiment)
# NUM_EIS_POINTS_MAX = 37
# EIS_FREQ_LIST_HZ = [50000, 31250, 20833.3333, 14705.8824, 10000,
#                     6756.7568, 5000, 3125, 2083.3333, 1470.5882, 1000,
#                     675.6757, 500, 312.5000, 208.3333, 147.0588, 100,
#                     67.5676, 50, 31.2500, 20.8333, 14.7059, 10,
#                     6.7568, 5, 3.1250, 2.0833, 1.4706, 1,
#                     0.6757, 0.5000, 0.3125, 0.2083, 0.1471, 0.1000,
#                     0.0676, 0.0500]
# EIS_USE_LIST = [False, False, False, True, True,
#                 True, True, True, True, True, True,
#                 True, True, True, True, True, True,
#                 True, True, True, True, True, True,
#                 True, True, True, True, False, True,
#                 False, True, False, True, False, True,
#                 False, True]
