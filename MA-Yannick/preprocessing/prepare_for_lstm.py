import gc
# import time
import pandas as pd
import config_preprocessing as cfg
from config_preprocessing import sch_state_ph as state_ph
from config_preprocessing import sch_state_cu as state_cu
from config_preprocessing import sch_state_sub as state_sub
from config_preprocessing import sch_state_chg as state_chg
import multiprocessing
from datetime import datetime
import os
import re
import config_logging as logging
# import numpy as np
import warnings
from influxdb_client.client.warnings import MissingPivotFunction
from influxdb_client import InfluxDBClient
import math


CSV_LABEL_TIMESTAMP = "timestamp_s"
CSV_LABEL_TIMESTAMP_ORIGIN = "timestamp_origin"
CSV_LABEL_SD_BLOCK_ID = "sd_block_id"
CSV_LABEL_V_CELL = "v_raw_V"
CSV_LABEL_I_CELL = "i_raw_A"
CSV_LABEL_P_CELL = "p_raw_W"
CSV_LABEL_T_CELL = "t_cell_degC"
CSV_LABEL_DELTA_Q = "delta_q_Ah"
CSV_LABEL_DELTA_E = "delta_e_Wh"
CSV_LABEL_SOC_EST = "soc_est"
CSV_LABEL_OCV_EST = "ocv_est_V"
CSV_LABEL_MEAS_CONDITION = "meas_condition"
CSV_LABEL_BMS_CONDITION = "bms_condition"
CSV_LABEL_V_STATE = "v_state"
CSV_LABEL_I_STATE = "i_state"
CSV_LABEL_T_STATE = "t_state"
CSV_LABEL_OVER_Q = "over_q"
CSV_LABEL_UNDER_Q = "under_q"
CSV_LABEL_OVER_SOC = "over_soc"
CSV_LABEL_UNDER_SOC = "under_soc"
CSV_LABEL_EOL_CAP = "eol_cap"
CSV_LABEL_EOL_IMP = "eol_imp"
CSV_LABEL_SCH_STATE_DEC = "scheduler_state_dec"
CSV_LABEL_SCH_STATE_HEX = "scheduler_state_hex"
CSV_LABEL_SCH_STATE_PH = "scheduler_state_phase"
CSV_LABEL_SCH_STATE_CU = "scheduler_state_checkup"
CSV_LABEL_SCH_STATE_CHG = "scheduler_state_charge"
CSV_LABEL_SCH_STATE_TIMEOUT = "scheduler_timeout"
CSV_LABEL_SCH_STATE_CU_PENDING = "scheduler_cu_pending"
CSV_LABEL_SCH_STATE_PAUSE_PENDING = "scheduler_pause_pending"
CSV_LABEL_SCH_STATE_SUB = "scheduler_state_sub"
CSV_LABEL_DELTA_Q_CHG = "delta_q_chg_Ah"
CSV_LABEL_DELTA_Q_DISCHG = "delta_q_dischg_Ah"
CSV_LABEL_DELTA_E_CHG = "delta_e_chg_Wh"
CSV_LABEL_DELTA_E_DISCHG = "delta_e_dischg_Wh"
CSV_LABEL_TOTAL_Q_CHG_CU_RT = "total_q_chg_CU_RT_Ah"
CSV_LABEL_TOTAL_Q_DISCHG_CU_RT = "total_q_dischg_CU_RT_Ah"
CSV_LABEL_TOTAL_Q_CHG_CYC_OT = "total_q_chg_cyc_OT_Ah"
CSV_LABEL_TOTAL_Q_DISCHG_CYC_OT = "total_q_dischg_cyc_OT_Ah"
CSV_LABEL_TOTAL_Q_CHG_OTHER_RT = "total_q_chg_other_RT_Ah"
CSV_LABEL_TOTAL_Q_DISCHG_OTHER_RT = "total_q_dischg_other_RT_Ah"
CSV_LABEL_TOTAL_Q_CHG_OTHER_OT = "total_q_chg_other_OT_Ah"
CSV_LABEL_TOTAL_Q_DISCHG_OTHER_OT = "total_q_dischg_other_OT_Ah"
CSV_LABEL_TOTAL_Q_CHG_SUM = "total_q_chg_sum_Ah"
CSV_LABEL_TOTAL_Q_DISCHG_SUM = "total_q_dischg_sum_Ah"
CSV_LABEL_TOTAL_E_CHG_CU_RT = "total_e_chg_CU_RT_Wh"
CSV_LABEL_TOTAL_E_DISCHG_CU_RT = "total_e_dischg_CU_RT_Wh"
CSV_LABEL_TOTAL_E_CHG_CYC_OT = "total_e_chg_cyc_OT_Wh"
CSV_LABEL_TOTAL_E_DISCHG_CYC_OT = "total_e_dischg_cyc_OT_Wh"
CSV_LABEL_TOTAL_E_CHG_OTHER_RT = "total_e_chg_other_RT_Wh"
CSV_LABEL_TOTAL_E_DISCHG_OTHER_RT = "total_e_dischg_other_RT_Wh"
CSV_LABEL_TOTAL_E_CHG_OTHER_OT = "total_e_chg_other_OT_Wh"
CSV_LABEL_TOTAL_E_DISCHG_OTHER_OT = "total_e_dischg_other_OT_Wh"
CSV_LABEL_TOTAL_E_CHG_SUM = "total_e_chg_sum_Wh"
CSV_LABEL_TOTAL_E_DISCHG_SUM = "total_e_dischg_sum_Wh"
CSV_LABEL_DELTA_DQ_CHG = "delta_dQ_chg"
CSV_LABEL_DELTA_DQ_DISCHG = "delta_dQ_dischg"
CSV_LABEL_DELTA_DE_CHG = "delta_dE_chg"
CSV_LABEL_DELTA_DE_DISCHG = "delta_dE_dischg"

CSV_LABEL_TEST_NAME = "test_name"  # -> TEST_NAME
CSV_LABEL_RECORD_ID = "record_id"  # -> in final data output, increment from 1 on (index + 1)
CSV_LABEL_TIME = "time"  # -> in hours, starts at 0 --> use (unix timestamp - first unix timestamp) / 3600.0
CSV_LABEL_STEP_TIME = "step_time"  # timestamp - timestamp.shift(+/-1), first: 0
CSV_LABEL_LINE = "line"  # 37 for charging, 40 for discharging
CSV_LABEL_VOLTAGE = "voltage"  # CSV_LABEL_V_CELL
CSV_LABEL_CURRENT = "current"  # CSV_LABEL_I_CELL, discharging current is negative (as in our data)
CSV_LABEL_CHARGING_CAPACITY = "charging_capacity"  # delta_q for charging, positive, hold until discharging complete
CSV_LABEL_DISCHARGING_CAPACITY = "discharging_capacity"  # delta_q for discharging, positive
CSV_LABEL_WH_CHARGING = "wh_charging"  # see charging_capacity, but with delta_e
CSV_LABEL_WH_DISCHARGE = "wh_discharging"  # see discharging_capacity, but with delta_e
CSV_LABEL_TEMPERATURE = "temperature"  # CSV_LABEL_T_CELL
CSV_LABEL_CYCLE_COUNT = "cycle_count"  # ...?

# cycle_count
#   charge/discharge cycle count. It goes from 0 to 100 for the main cycle and from 0 to 1000 for the resistance cycle.
#   The two cycles are interleaved i.e. 0->100->0->1000->0->100...


CSV_LABELS_DELETE_IMMEDIATELY = [CSV_LABEL_TIMESTAMP_ORIGIN,
                                 CSV_LABEL_SD_BLOCK_ID,
                                 CSV_LABEL_P_CELL,
                                 CSV_LABEL_DELTA_Q,
                                 CSV_LABEL_DELTA_E,
                                 CSV_LABEL_SOC_EST,
                                 CSV_LABEL_OCV_EST,
                                 CSV_LABEL_MEAS_CONDITION,
                                 CSV_LABEL_BMS_CONDITION,
                                 CSV_LABEL_V_STATE,
                                 CSV_LABEL_I_STATE,
                                 CSV_LABEL_T_STATE,
                                 CSV_LABEL_OVER_Q,
                                 CSV_LABEL_UNDER_Q,
                                 CSV_LABEL_OVER_SOC,
                                 CSV_LABEL_UNDER_SOC,
                                 CSV_LABEL_EOL_CAP,
                                 CSV_LABEL_EOL_IMP,
                                 CSV_LABEL_SCH_STATE_DEC,
                                 CSV_LABEL_SCH_STATE_HEX,
                                 CSV_LABEL_SCH_STATE_CU,
                                 CSV_LABEL_SCH_STATE_TIMEOUT,
                                 CSV_LABEL_SCH_STATE_CU_PENDING,
                                 CSV_LABEL_SCH_STATE_PAUSE_PENDING,
                                 CSV_LABEL_TOTAL_Q_CHG_CU_RT,
                                 CSV_LABEL_TOTAL_Q_DISCHG_CU_RT,
                                 CSV_LABEL_TOTAL_Q_CHG_CYC_OT,
                                 CSV_LABEL_TOTAL_Q_DISCHG_CYC_OT,
                                 CSV_LABEL_TOTAL_Q_CHG_OTHER_RT,
                                 CSV_LABEL_TOTAL_Q_DISCHG_OTHER_RT,
                                 CSV_LABEL_TOTAL_Q_CHG_OTHER_OT,
                                 CSV_LABEL_TOTAL_Q_DISCHG_OTHER_OT,
                                 CSV_LABEL_TOTAL_Q_CHG_SUM,
                                 CSV_LABEL_TOTAL_Q_DISCHG_SUM,
                                 CSV_LABEL_TOTAL_E_CHG_CU_RT,
                                 CSV_LABEL_TOTAL_E_DISCHG_CU_RT,
                                 CSV_LABEL_TOTAL_E_CHG_CYC_OT,
                                 CSV_LABEL_TOTAL_E_DISCHG_CYC_OT,
                                 CSV_LABEL_TOTAL_E_CHG_OTHER_RT,
                                 CSV_LABEL_TOTAL_E_DISCHG_OTHER_RT,
                                 CSV_LABEL_TOTAL_E_CHG_OTHER_OT,
                                 CSV_LABEL_TOTAL_E_DISCHG_OTHER_OT,
                                 CSV_LABEL_TOTAL_E_CHG_SUM,
                                 CSV_LABEL_TOTAL_E_DISCHG_SUM,
                                 CSV_LABEL_DELTA_DQ_CHG,
                                 CSV_LABEL_DELTA_DQ_DISCHG,
                                 CSV_LABEL_DELTA_DE_CHG,
                                 CSV_LABEL_DELTA_DE_DISCHG]

CSV_LABELS_DELETE_LATER = [CSV_LABEL_SCH_STATE_PH, CSV_LABEL_SCH_STATE_SUB]

NUMBER_OF_PROCESSORS_TO_USE = math.ceil(multiprocessing.cpu_count() / 4)

logext_csv_task_queue = multiprocessing.Queue()


TEST_NAME = "%03u-LE-3.0-4820-H"
# Produktionsdatum Zellen: wahrscheinlich 26.11.2020 --> week 48
# Test name convention
# "000-XW-Y.Y-AABB-T", with
#    000 = sample serial number --> in our case parameter set * 3 + parameter
#    X = lecter related to cell manufacturer
#    W = cell type | P: powertool, M: mid power, E: e-byke
#    Y.Y = cell capacity e.g. 2.5Ah -> 2.5
#    AABB = samples delivery date (AA: week, bb: year)
#    T = test type | S: Standard (5A discharge), H: High Current (8A discharge),
#                    P: Pre-conditioned (90 days storing at 45C degree before testing)

CELLS_PER_PARAMETER = max(max(cfg.PARAMETER_SET_CELL_NR_FROM_SXX_CXX))


def run():
    start_timestamp = datetime.now()
    logging.log.info(os.path.basename(__file__))
    report_manager = multiprocessing.Manager()
    report_queue = report_manager.Queue()

    logging.log.info("Starting .csv preparation for LSTM")

    # find .csv files: cell_logext_P012_3_S14_C11.csv
    cell_logext_csv = []
    slave_cell_found = [[" "] * cfg.NUM_CELLS_PER_SLAVE for _ in range(cfg.NUM_SLAVES_MAX)]
    with os.scandir(cfg.CSV_RESULT_DIR) as iterator:
        re_str_logext_csv = cfg.CSV_FILENAME_05_RESULT_BASE_CELL_RE.replace("(\w)", cfg.CSV_FILENAME_05_TYPE_LOG_EXT)
        re_pat_logext_csv = re.compile(re_str_logext_csv)
        for entry in iterator:
            re_match_logext_csv = re_pat_logext_csv.fullmatch(entry.name)
            if re_match_logext_csv:
                param_id = int(re_match_logext_csv.group(1))
                param_nr = int(re_match_logext_csv.group(2))
                slave_id = int(re_match_logext_csv.group(3))
                cell_id = int(re_match_logext_csv.group(4))
                cell_csv = {"param_id": param_id, "param_nr": param_nr, "slave_id": slave_id, "cell_id": cell_id,
                            "filename": entry.name}
                cell_logext_csv.append(cell_csv)
                logext_csv_task_queue.put(cell_csv)
                if (slave_id < 0) or (slave_id >= cfg.NUM_SLAVES_MAX):
                    logging.warning("Found unusual slave_id: %u" % slave_id)
                else:
                    if (cell_id < 0) or (cell_id >= cfg.NUM_CELLS_PER_SLAVE):
                        logging.warning("Found unusual cell_id: %u" % cell_id)
                    else:
                        if slave_cell_found[slave_id][cell_id] == "x":
                            logging.warning("Found more than one entry for S%02u:C%02u" % (slave_id, cell_id))
                        else:
                            slave_cell_found[slave_id][cell_id] = "x"

    print("Found the following files:\n"
          "' ' = no file found, 'X' = file found -> added")
    print("   Cells:   ", end="")
    for cell_id in range(0, cfg.NUM_CELLS_PER_SLAVE):
        print("x", end="")
    print("")
    for slave_id in range(0, cfg.NUM_SLAVES_MAX):
        print("Slave %2u:   " % slave_id, end="")
        for cell_id in range(0, cfg.NUM_CELLS_PER_SLAVE):
            print(slave_cell_found[slave_id][cell_id], end="")
        print("")

    # Create processes
    processes = []
    logging.log.info("Starting processes to prepare LSTM input data...")
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        logging.log.debug("  Starting process %u" % processorNumber)
        processes.append(multiprocessing.Process(target=prepare_thread,
                                                 args=(processorNumber, logext_csv_task_queue, report_queue,)))
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].start()
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].join()
        logging.log.debug("Joined process %u" % processorNumber)

    logging.log.info("\nMerge csv files...\n")

    # TODO: merge all csv files together - only copy the header once
    # - in cfg.LSTM_INPUT_DIR: find all "test_result_%03u.csv"
    # - merge into "test_result.csv"

    logging.log.info("\n\n========== All tasks ended - summary ========== \n")

    while True:
        if (report_queue is None) or report_queue.empty():
            break  # no more reports

        try:
            slave_report = report_queue.get_nowait()
        except multiprocessing.queues.Empty:
            break  # no more reports

        if slave_report is None:
            break  # no more reports

        report_msg = slave_report["msg"]
        report_level = slave_report["level"]

        if report_level == logging.ERROR:
            logging.log.error(report_msg)
        elif report_level == logging.WARNING:
            logging.log.warning(report_msg)
        elif report_level == logging.INFO:
            logging.log.info(report_msg)
        elif report_level == logging.DEBUG:
            logging.log.debug(report_msg)
        elif report_level == logging.CRITICAL:
            logging.log.critical(report_msg)

    stop_timestamp = datetime.now()

    logging.log.info("\nScript runtime: %s h:mm:ss.ms" % str(stop_timestamp - start_timestamp))


def prepare_thread(processor_number, slave_queue, thread_report_queue):
    while True:
        if (slave_queue is None) or slave_queue.empty():
            break  # no more files

        try:
            queue_entry = slave_queue.get_nowait()
        except multiprocessing.queues.Empty:
            break  # no more files

        if queue_entry is None:
            break  # no more files

        slave_id = queue_entry["slave_id"]
        cell_id = queue_entry["cell_id"]
        param_id = queue_entry["param_id"]
        param_nr = queue_entry["param_nr"]

        if not ((slave_id == 4) and (cell_id == 7)):
            continue  # for debugging individual cells

        filename_csv = queue_entry["filename"]
        logging.log.info("Thread %u S%02u:C%02u - preparing LSTM input data" % (processor_number, slave_id, cell_id))

        num_infos = 0
        num_warnings = 0
        num_errors = 0

        # I. LOG file
        logging.log.debug("Thread %u S%02u:C%02u - reading LOG file" % (processor_number, slave_id, cell_id))
        log_df = pd.read_csv(cfg.CSV_RESULT_DIR + filename_csv, header=0, sep=cfg.CSV_SEP)

        # get rid of completely unnecessary data:
        log_df.drop(CSV_LABELS_DELETE_IMMEDIATELY, axis=1, inplace=True)

        # insert record_id --> better to do this at the beginning since the LSTM might also need it in the check-up CSV
        log_df[CSV_LABEL_RECORD_ID] = log_df.index + 1

        # filter: cycling, charging/discharging, running
        log_df_use = log_df[(log_df[CSV_LABEL_SCH_STATE_PH] == 1)  # <- SCH_CELL_PH_AUTO_CYCLING; charging/discharging:
                            & ((log_df[CSV_LABEL_SCH_STATE_CHG] == 1) | (log_df[CSV_LABEL_SCH_STATE_CHG] == 2))
                            & (log_df[CSV_LABEL_SCH_STATE_SUB] == 4)]  # <- SCH_CELL_SUB_RUNNING

        # insert cycle count column:
        log_df_use[CSV_LABEL_CYCLE_COUNT] = 0
        # fill cycle count column:

        # replace charging/discharging column values
        log_df_use[CSV_LABEL_SCH_STATE_CHG][log_df_use[CSV_LABEL_SCH_STATE_CHG] == 1] = 37  # charging, main cycle
        log_df_use[CSV_LABEL_SCH_STATE_CHG][log_df_use[CSV_LABEL_SCH_STATE_CHG] == 2] = 40  # discharging, main cycle

        # TODO: fix gaps (1x at start, 1x at end: copy values, but set current to 0)
        # - detect where increment of CSV_LABEL_TIMESTAMP is > threshold (e.g. 10 seconds)
        # - for each gap, iterate through and insert one row at the beginning (+2s) and the end (-2s) of the gap:
        #   - copy all values, but make current = 0

        # TODO: adjust delta_q/e as needed for LSTM
        # - detect positive and negative "edges" of CSV_LABEL_SCH_STATE_CHG
        # - for each edge positive/negative edge tuple, iterate:
        # - increment cycle counter
        # - fill CSV_LABEL_CYCLE_COUNT of complete cycle with current cycle counter
        # - fill CSV_LABEL_CHARGING_CAPACITY / CSV_LABEL_WH_CHARGING segments with last before edge value (hold)

        # TODO: reset cycle counter at every checkup
        # - detect edges in log_df (not log_df_use!)
        #   from CSV_LABEL_SCH_STATE_PH = 2 (SCH_CELL_PH_AUTO_CHECKUP) to 1 (SCH_CELL_PH_AUTO_CYCLING)
        # - the first cycle after the checkup shall be 1:
        #   --> decrement everything log_df_use[CSV_LABEL_CYCLE_COUNT] with index > last check-up index

        # free memory
        log_df = ""
        del log_df
        gc.collect()

        # drop other unnecessary columns
        log_df_use.drop(CSV_LABELS_DELETE_LATER, axis=1, inplace=True)

        # convert time
        log_df_use[CSV_LABEL_TIMESTAMP] = \
            (log_df_use[CSV_LABEL_TIMESTAMP] - log_df_use[CSV_LABEL_TIMESTAMP].iloc[0]) / 3600.0

        # insert step time
        log_df_use[CSV_LABEL_STEP_TIME] = log_df_use[CSV_LABEL_TIMESTAMP] - log_df_use[CSV_LABEL_TIMESTAMP].shift(1)
        log_df_use[CSV_LABEL_STEP_TIME].iloc[0] = 0

        # rename columns
        log_df_use.rename(columns={CSV_LABEL_TIMESTAMP: CSV_LABEL_TIME,  # time
                                   CSV_LABEL_V_CELL: CSV_LABEL_VOLTAGE,  # voltage
                                   CSV_LABEL_I_CELL: CSV_LABEL_CURRENT,  # current
                                   CSV_LABEL_T_CELL: CSV_LABEL_TEMPERATURE,  # temperature
                                   CSV_LABEL_DELTA_Q_CHG: CSV_LABEL_CHARGING_CAPACITY,  # delta_q_chg
                                   CSV_LABEL_DELTA_Q_DISCHG: CSV_LABEL_DISCHARGING_CAPACITY,  # delta_q_dischg
                                   CSV_LABEL_DELTA_E_CHG: CSV_LABEL_WH_CHARGING,  # delta_e_chg
                                   CSV_LABEL_DELTA_E_DISCHG: CSV_LABEL_WH_DISCHARGE,  # delta_e_dischg
                                   #
                                   }, inplace=True)

        # insert test_name
        lstm_cell_id = ((param_id - 1) * CELLS_PER_PARAMETER + (param_nr - 1))
        log_df_use[CSV_LABEL_TEST_NAME] = TEST_NAME % lstm_cell_id

        # TODO: change order of columns

        # save csv
        log_df_use.to_csv(cfg.LSTM_INPUT_DIR + ("test_result_%03u.csv" % lstm_cell_id),
                                            index=False, sep=cfg.CSV_SEP, float_format="%.9f")

        # free memory
        log_df_use = ""
        del log_df_use
        gc.collect()

        # reporting to main thread
        report_msg = f"%s - S%02u:C%02u - prepared LSTM input data: %u infos, %u warnings, %u errors"\
                     % (filename_csv, slave_id, cell_id, num_infos, num_warnings, num_errors)
        report_level = logging.INFO
        if num_errors > 0:
            report_level = logging.ERROR
        elif num_warnings > 0:
            report_level = logging.WARNING

        cell_report = {"msg": report_msg, "level": report_level}
        thread_report_queue.put(cell_report)

    slave_queue.close()
    # thread_report_queue.close()
    logging.log.info("Thread %u - no more slaves - exiting" % processor_number)


if __name__ == "__main__":
    run()
