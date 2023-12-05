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


# config -> for a complete run, set FIX_PULSE, FIX_DQ_DE and FIX_CHECKUP_NUM = True
FIX_PULSE = False  # FIXME: set True if you have new data
FIX_DQ_DE = True
FIX_CHECKUP_NUM = True
SKIP_ALREADY_EDITED_LOG_FILES = True
TIME_MAX_LOOKAROUND_S = (15 * 60)  # 15 * 60 = 15 minutes
NUM_MATCHING_DATA_ROWS_PER_PULSE_PATTERN = 27
CSV_LABEL_TIMESTAMP = "timestamp_s"
CSV_LABEL_PULSE_V = "v_raw_V"
CSV_LABEL_PULSE_I = "i_raw_A"
CSV_LABEL_AGE_CHG_RATE = "age_chg_rate"
CSV_LABEL_AGE_DISCHG_RATE = "age_dischg_rate"
CSV_LABEL_T_AVG = "t_avg_degC"
CSV_LABEL_SD_BLOCK_ID = "sd_block_id"
CSV_LABEL_EOC_CYC_CONDITION = "cyc_condition"
CSV_LABEL_EOC_CYC_CHARGED = "cyc_charged"
CSV_LABEL_EOC_NUM_CYCLES_CU = "num_cycles_checkup"
CSV_LABEL_T_START = "t_start_degC"
CSV_LABEL_T_END = "t_end_degC"
CSV_LABEL_CYC_DURATION = "cyc_duration_s"
CSV_LABEL_CAP_CHARGED_EST = "cap_aged_est_Ah"
CSV_LABEL_SOH_CAP = "soh_cap"
CSV_LABEL_DELTA_Q = "delta_q_Ah"
CSV_LABEL_DELTA_Q_CHG = "delta_q_chg_Ah"
CSV_LABEL_DELTA_Q_DISCHG = "delta_q_dischg_Ah"
CSV_LABEL_DELTA_E = "delta_e_Wh"
CSV_LABEL_DELTA_E_CHG = "delta_e_chg_Wh"
CSV_LABEL_DELTA_E_DISCHG = "delta_e_dischg_Wh"
CSV_LABEL_COULOMB_EFFICIENCY = "coulomb_efficiency"
CSV_LABEL_ENERGY_EFFICIENCY = "energy_efficiency"
CSV_LABEL_TOTAL_Q_CONDITION = "total_q_condition_Ah"
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
CSV_LABEL_TOTAL_E_CONDITION = "total_e_condition_Wh"
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
CSV_LABEL_TOTAL_Q_CHG = "total_q_chg_Ah"
CSV_LABEL_TOTAL_Q_DISCHG = "total_q_dischg_Ah"
CSV_LABEL_TOTAL_E_CHG = "total_e_chg_Wh"
CSV_LABEL_TOTAL_E_DISCHG = "total_e_dischg_Wh"
CSV_LABEL_DELTA_DQ_CHG = "delta_dQ_chg"
CSV_LABEL_DELTA_DQ_DISCHG = "delta_dQ_dischg"
CSV_LABEL_DELTA_DE_CHG = "delta_dE_chg"
CSV_LABEL_DELTA_DE_DISCHG = "delta_dE_dischg"
CSV_LABEL_SCH_STATE_DEC = "scheduler_state_dec"
CSV_LABEL_SCH_STATE_PH = "scheduler_state_phase"
CSV_LABEL_SCH_STATE_CU = "scheduler_state_checkup"
CSV_LABEL_SCH_STATE_SUB = "scheduler_state_sub"
CSV_LABEL_SCH_STATE_CHG = "scheduler_state_charge"
CSV_LABEL_V_CELL = "v_raw_V"
CSV_LABEL_I_CELL = "i_raw_A"
CSV_LABEL_P_CELL = "p_raw_W"
CSV_LABEL_T_CELL = "t_cell_degC"
CSV_LABEL_SOC_EST = "soc_est"
CSV_LABEL_OCV_EST = "ocv_est_V"
CSV_LABEL_UPTIME_TICKS = "uptime_ticks"
# CSV_LABEL_OLD = "_old"
CU_MAX_LENGTH_S = (2 * 24 * 60 * 60)  # (2 * 24 * 60 * 60) = 2 days
DQ_START_MAX = 0.005  # if abs() of initial delta_Q is greater than this, inform
DQ_START_MAX_WARN = 0.1  # if abs() of initial delta_Q is greater than this, warn
DQ_NEW_REL = 0.1  # if next delta_q deviates by this percentage (0.1 = 10%), assume it was reset/this is a new run
SD_DELTA_BLOCK_IDS_SEARCH = 200 * cfg.SD_BLOCK_SIZE_BYTES  # 200 * ... = search up to 200 SD_block_ID blocks ...
#    ... in advance of EOC location to find LOG that can be matched with the EOC data set
MAX_RETRY_LOG_MATCH = 200  # when looking for matching LOG row based on the EOC message, go up a maximum number of this
# rows do find matching delta_q (for statistics - the total_xxx values should be the same even after delta_q was reset)
DQ_CMP_REL_FINE = 0.03  # 0.03 = 3%, compare delta_q and total_q_xxx for matching EOC/LOG ...
DE_CMP_REL_FINE = 0.05  # 0.05 = 5%, compare delta_e and total_e_xxx for matching EOC/LOG ...
# ... - if it deviates by more than this percentage (0.01 = 1%), warn - alternatively, ...
TOTAL_DQ_CMP_ABS_FINE = 0.5  # ... compare if absolute dQ deviation is smaller than this (useful for near zero values)
TOTAL_DE_CMP_ABS_FINE = 2.0  # ... compare if absolute dE deviation is smaller than this (useful for near zero values)
DQ_CMP_ABS_FINE = 0.01  # for delta_q (not total_q) comparisons
DE_CMP_ABS_FINE = 0.04  # for delta_e (not total_e) comparisons
DQ_EIS_ABS_FINE = 0.05  # for delta_q (not total_q) comparisons --> tolerated deviation during EIS
DE_EIS_ABS_FINE = 0.2  # for delta_e (not total_e) comparisons --> tolerated deviation during EIS
DQ_CYC_ABS_FINE = 0.01  # for delta_q (not total_q) comparisons --> tolerated deviation during cycling (discharging)
DE_CYC_ABS_FINE = 0.04  # for delta_e (not total_e) comparisons --> tolerated deviation during cycling (discharging)
MAX_DELTA_DQ_IDLE = 0.000002  # maximum plausible abs(delta_dQ) when delta_q shouldn't be changing --> inform
MAX_DELTA_DQ_IDLE_WARN = 0.1  # maximum plausible abs(delta_dQ) when delta_q shouldn't be changing --> warn
MAX_DELTA_DE_IDLE = 0.000008  # maximum plausible abs(delta_dQ) when delta_q shouldn't be changing --> inform
MAX_DELTA_DE_IDLE_WARN = 0.4  # maximum plausible abs(delta_dQ) when delta_q shouldn't be changing --> warn
MAX_DELTA_DQ_START = 0.002  # maximum plausible abs(delta_dQ) at the start of the experiment (before first RUN state)
# 2 sec charging with 1 A = 0.00055 Ah
# 2 sec charging with 4 mA = ca. 0.000002 Ah
LOSS_RATIO_MIN_EXPECT = 0.80
LOSS_RATIO_MAX_EXPECT = 1.02  # loss ratio > 1 sometimes occurs for overcharged cells
LOSS_RATIO_DEFAULT = 0.95
KNOWN_GAPS_TIMESTAMP_START = [1665648000, 1666195200, 1666256400, 1666821600, 1667113200, 1667826000, 1668193200,
                              1668610800, 1668668400, 1669035600, 1669114800, 1669536000, 1669964400, 1670396400,
                              1672059600, 1672794000, 1672909200, 1674748800]
KNOWN_GAPS_TIMESTAMP_END = [1665658800, 1666209600, 1666260000, 1666825200, 1667390400, 1667840400, 1668196800,
                            1668628800, 1668672000, 1669042800, 1669118400, 1669539600, 1669968000, 1670400000,
                            1672063200, 1672797600, 1672912800, 1674756000]


# constants
# NUMBER_OF_PROCESSORS_TO_USE = multiprocessing.cpu_count()  -> heavy lagging, memory overflow...
NUMBER_OF_PROCESSORS_TO_USE_LOG = 1  # multithreading will cause RAM overflow here, at least with my computer/laptop
NUMBER_OF_PROCESSORS_TO_USE_EOC = math.ceil(multiprocessing.cpu_count() / 4)  # computer lags heavily without "/ 2" or 4
if cfg.computer == cfg.ComputerID.IPE_AVT_SIM:
    # FIXME: for NUMBER_OF_PROCESSORS_TO_USE_LOG, use / 4 for Apr'23 backup and probably / 8 for final backup
    NUMBER_OF_PROCESSORS_TO_USE_LOG = math.ceil(multiprocessing.cpu_count() / 4)  # ca. 100 GB RAM for 8 cores (Feb'23)
    NUMBER_OF_PROCESSORS_TO_USE_EOC = math.ceil(multiprocessing.cpu_count() / 4)

fix_log_csv_task_queue = multiprocessing.Queue()
fix_eoc_csv_task_queue = multiprocessing.Queue()
# report_manager = multiprocessing.Manager()
# report_queue = report_manager.Queue()
# report_queue = multiprocessing.Queue()


def run():
    start_timestamp = datetime.now()
    logging.log.info(os.path.basename(__file__))
    # report_queue = multiprocessing.Queue()
    report_manager = multiprocessing.Manager()
    report_queue = report_manager.Queue()

    if FIX_PULSE:
        # bug in Logger.c of cycler: PULSE data [27:60] was not stored on the SD card -> merge with Influx
        fix_pulse_csv(report_queue)
    else:
        logging.log.info("\n\n========== Skipping FIX_PULSE ========== \n")

    if FIX_DQ_DE:
        # multiple bugs in cycler with dQ/dE.
        # - dQ/dE in LOG is always correct, but it is not separated into dQ/dE_chg/dischg
        # - dQ/dE_chg/dischg in EOC is always correct, but it doesn't include rows for EIS/PULSE charges
        # - total_dQ_chg/dischg in EOC is always correct (no but)
        # - total_dQ_cond in EOC is mostly wrong -> better to ignore it
        # - total_dE_cond and total_dE_dischg is almost always wrong -> better to ignore it
        # - total_dE_chg is always correct ? (to be verified!), but ..._dischg not available
        # solution:
        # I. LOG file
        # - insert new columns dQ/dE_chg/dischg (4x)
        # - insert new columns total_dQ/dE_(CU_RT/CYC_OT/others_RT/others_OT/sum)_chg/dischg (20x)
        # - walk along all rows and increment dQ/dE_chg/dischg accordingly (for rising/falling dQ/dE)
        # - also calculate total_... (at the same time or later, vectorized?)
        # II. EOC file:
        # - delete total_dQ/dE_cond
        # - add total_dQ/dE_(CU_RT/CYC_OT/others_RT/others_OT/sum)_chg/dischg with the ones from LOG*
        # - compare old total_dQ/dE_chg/dischg with total_dQ/dE_sum_chg/dischg (warn if necessary)
        # - delete old total_dQ/dE_chg/dischg columns
        # * use SD_block_ID from EOC to look up row index of LOG, then go up until LOG message is found - probably go
        #   up a bot more and/or check if dQ is plausible/matches with old (and correct) dQ/dE from EOC
        fix_log_dq_de(report_queue)
    else:
        logging.log.info("\n\n========== Skipping FIX_DQ_DE ========== \n")

    if FIX_CHECKUP_NUM or FIX_DQ_DE:
        # The cycler increments num_cycles_checkup in EOC data just before the end of the CU cap check discharge EOC
        # message is sent. That means that the rest of the checkup has a num_cycles_checkup of +1, which is counter-
        # intuitive -> mark the rest of CU EOC messages with (num_cycles_checkup) instead of (num_cycles_checkup + 1)
        fix_eoc_csv(report_queue)
    else:
        logging.log.info("\n\n========== Skipping FIX_CHECKUP_NUM / FIX_DQ_DE ========== \n")

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


def fix_pulse_csv(report_queue):
    logging.log.info("========== 1) Fix missing PULSE data ==========")

    # find .csv files: cell_pls_P012_3_S14_C11.csv
    cell_pulse_csv = []
    slave_cell_found = [[" "] * cfg.NUM_CELLS_PER_SLAVE for _ in range(cfg.NUM_SLAVES_MAX)]
    with os.scandir(cfg.CSV_RESULT_DIR) as iterator:
        re_str_pulse_csv = cfg.CSV_FILENAME_05_RESULT_BASE_CELL_RE.replace("(\w)", cfg.CSV_FILENAME_05_TYPE_PULSE)
        re_pat_pulse_csv = re.compile(re_str_pulse_csv)
        for entry in iterator:
            re_match_pulse_csv = re_pat_pulse_csv.fullmatch(entry.name)
            if re_match_pulse_csv:
                param_id = int(re_match_pulse_csv.group(1))
                param_nr = int(re_match_pulse_csv.group(2))
                slave_id = int(re_match_pulse_csv.group(3))
                cell_id = int(re_match_pulse_csv.group(4))
                cell_csv = {"param_id": param_id, "param_nr": param_nr, "slave_id": slave_id, "cell_id": cell_id,
                             "filename": entry.name}
                cell_pulse_csv.append(cell_csv)
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

    # List found slaves/cells for user
    print("   Cells:   ", end="")
    for cell_id in range(0, cfg.NUM_CELLS_PER_SLAVE):
        print("x", end="")
    print("")
    for slave_id in range(0, cfg.NUM_SLAVES_MAX):
        print("Slave %2u:   " % slave_id, end="")
        for cell_id in range(0, cfg.NUM_CELLS_PER_SLAVE):
            print(slave_cell_found[slave_id][cell_id], end="")
        print("")

    logging.log.info("Start downloading Influx data...")

    # open Influx Client
    warnings.simplefilter("ignore", MissingPivotFunction)
    client = InfluxDBClient(url=cfg.influx_url, token=cfg.influx_token, org=cfg.influx_org)
    query_api = client.query_api()

    # for each found pulse file
    for cell_pulse_file in cell_pulse_csv:
        num_not_converted_error = 0
        num_converted_ok = 0
        num_converted_warnings = 0

        slave_id = cell_pulse_file["slave_id"]
        cell_id = cell_pulse_file["cell_id"]

        # if not (((slave_id == 4) and (cell_id == 4)) or ((slave_id == 17) and (cell_id == 6))):
        # if not ((slave_id == 12) and (cell_id == 7)):
        #     continue  # for debugging

        pulse_df = pd.read_csv(cfg.CSV_RESULT_DIR + cell_pulse_file["filename"], header=0, sep=cfg.CSV_SEP)
        i_start = pulse_df.index[0]
        i_max = pulse_df.index[-1]

        # for each pulse pattern
        while i_start < i_max:
            # group by sd_block_id
            timestamp_range = pulse_df[CSV_LABEL_TIMESTAMP][pulse_df[CSV_LABEL_SD_BLOCK_ID] ==
                                                            pulse_df[CSV_LABEL_SD_BLOCK_ID][i_start]]
            i_end = timestamp_range.index[-1]
            timestamp_min_search = int(timestamp_range.iat[0] - TIME_MAX_LOOKAROUND_S)
            timestamp_max_search = int(timestamp_range.iat[-1] + TIME_MAX_LOOKAROUND_S)
            logging.log.debug("Influx - S%02u:C%02u - csv row %5u - %5u" % (slave_id, cell_id, i_start, i_end))

            query = '''
            from(bucket: "{influxBucket}")
              |> range(start: {t_min}, stop: {t_max})
              |> filter(fn: (r) => r["_measurement"] == "pulse_data")
              |> filter(fn: (r) => r["_field"] == "pulse_v" or r["_field"] == "pulse_i")
              |> filter(fn: (r) => r["slave_id"] == "{slave_id}")
              |> filter(fn: (r) => r["slave_cell"] == "{cell_id}")
              |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
              |> map(fn: (r) => ({{
                {pulse_v}: r.pulse_v,
                {pulse_i}: r.pulse_i,
                unixtimestamp: uint(v: r._time)
            }}))
            '''.format(influxBucket=cfg.influx_bucket,
                       t_min=timestamp_min_search,
                       t_max=timestamp_max_search,
                       slave_id=slave_id,
                       cell_id=cell_id,
                       pulse_v=CSV_LABEL_PULSE_V,
                       pulse_i=CSV_LABEL_PULSE_I
                       )
            logging.log.debug("   Waiting for database...")
            influx_df = query_api.query_data_frame(query)
            logging.log.debug("   Processing and export to CSV...")
            influx_df = influx_df.drop("result", axis=1)
            influx_df = influx_df.drop("table", axis=1)
            influx_df["unixtimestamp"] = (influx_df["unixtimestamp"] / 1000000000)

            num_rows_csv = timestamp_range.shape[0]
            num_rows_influx = influx_df.shape[0]
            issue_solved = True
            has_warning = False
            if num_rows_csv != num_rows_influx:
                issue_solved = False
                has_warning = True
                # if (slave_id == 4) and (cell_id == 7) and (i_start == 1037):
                #     print("debug here")
                if (num_rows_influx >= num_rows_csv) and (num_rows_influx <= 5 * num_rows_csv):
                    # e.g. Slave 4, Cell 9,  2022-10-21 5:45 ... 5:50: 2x pulse data sent from slave to raspi script
                    # same message, different timestamp offset
                    # http://ipepdvmssqldb2.ipe.kit.edu:3000/d/pqJqKzu7k/cell-pulses?orgId=1&var-xx=4&var-yy=9&from=1666323918776&to=1666324185370
                    dt_ms = influx_df["unixtimestamp"] - round(influx_df["unixtimestamp"])
                    influx_df["unique_code"] = influx_df[CSV_LABEL_PULSE_V].map("{:.4f}".format)\
                                               + influx_df[CSV_LABEL_PULSE_I].map("{:.4f}".format)\
                                               + dt_ms.map("{:.3f}".format)
                    influx_df_new = influx_df.drop_duplicates(subset=['unique_code'])
                    influx_df_new = influx_df_new.reset_index()  # reindex(range(0, influx_df_new.shape[0]))
                    num_rows_influx = influx_df_new.shape[0]
                    if num_rows_csv == num_rows_influx:
                        issue_solved = True
                    else:
                        # manually walk through elements one by one until the next duplicate is found
                        influx_df["matched"] = 0
                        # influx_df_new = pd.DataFrame(columns=[CSV_LABEL_TIMESTAMP, CSV_LABEL_PULSE_V,
                        #                                       CSV_LABEL_PULSE_I, "unique_code", "matched"])
                        influx_df_new = pd.DataFrame()
                        k = 0
                        k_max = influx_df.shape[0]
                        while k < k_max:
                            if influx_df.at[k, "matched"] == 0:  # was influx_df.loc[k, "matched"]
                                for m in range(k + 1, k_max):
                                    if influx_df["unique_code"][k] == influx_df["unique_code"][m]:
                                        influx_df.at[m, "matched"] = 1
                                        break  # -> don't break, we need to find all duplicates??
                                influx_df.at[k, "matched"] = 1
                                influx_df_new = influx_df_new.append(influx_df.iloc[k])  # deprecated, but ...
                                # influx_df_new = pd.concat([influx_df_new, influx_df.iloc[k]])  # this doesn't work
                            k = k + 1

                        influx_df_new.drop("matched", inplace=True, axis=1)
                        influx_df_new = influx_df_new.reset_index()
                        num_rows_influx = influx_df_new.shape[0]
                        if num_rows_csv == num_rows_influx:
                            issue_solved = True
                        else:
                            print("debug here")
                    influx_df_new.drop("unique_code", inplace=True, axis=1)
                    influx_df = influx_df_new
                if issue_solved:
                    logging.log.info("Influx - S%02u:C%02u - csv row %5u - %5u: size of csv and influx data "
                                     "doesn't match, but I was able to identify and delete duplicates"
                                     % (slave_id, cell_id, i_start, i_end))
                else:
                    logging.log.error("Influx - S%02u:C%02u - csv row %5u - %5u: size of csv and influx data doesn't "
                                      "match -> skip" % (slave_id, cell_id, i_start, i_end))
            if issue_solved:
                # influx_df.sort_values(axis=1, by="unixtimestamp", inplace=True)  # sorting not necessary?
                t_offset = influx_df["unixtimestamp"][0] - timestamp_range.iloc[0]
                influx_df["unixtimestamp"] = round(influx_df["unixtimestamp"] - t_offset,
                                                   cfg.PULSE_TIME_OFFSET_S_MAX_DECIMALS + 1)
                # noinspection PyTypeChecker
                if not all(influx_df["unixtimestamp"].values == timestamp_range.values):
                    if not has_warning:
                        logging.log.info("Influx - S%02u:C%02u - csv row %5u - %5u: timestamps of csv and influx data "
                                         "don't match -> copy anyway" % (slave_id, cell_id, i_start, i_end))
                        has_warning = True
                i_end_m = i_start + NUM_MATCHING_DATA_ROWS_PER_PULSE_PATTERN
                if i_end_m > i_max:
                    i_end_m = i_max
                # noinspection PyTypeChecker
                if not all(influx_df[CSV_LABEL_PULSE_V].iloc[0:NUM_MATCHING_DATA_ROWS_PER_PULSE_PATTERN].values ==
                           pulse_df[CSV_LABEL_PULSE_V].iloc[i_start:i_end_m].values) or \
                    not all(influx_df[CSV_LABEL_PULSE_I].iloc[0:NUM_MATCHING_DATA_ROWS_PER_PULSE_PATTERN].values ==
                                   pulse_df[CSV_LABEL_PULSE_I].iloc[i_start:i_end_m].values):
                    if not has_warning:
                        logging.log.warning("Influx - S%02u:C%02u - csv row %5u - %5u: first V/I rows of csv and influx"
                                            " data don't match -> copy anyway" % (slave_id, cell_id, i_start, i_end))
                        has_warning = True

                influx_df.set_index(pulse_df.loc[i_start:i_end, CSV_LABEL_PULSE_V].index, inplace=True)
                pulse_df.loc[i_start:i_end, CSV_LABEL_PULSE_V] = influx_df[CSV_LABEL_PULSE_V]
                pulse_df.loc[i_start:i_end, CSV_LABEL_PULSE_I] = influx_df[CSV_LABEL_PULSE_I]
                if has_warning:
                    num_converted_warnings = num_converted_warnings + 1
                else:
                    num_converted_ok = num_converted_ok + 1
            else:
                num_not_converted_error = num_not_converted_error + 1

            i_start = i_end + 1

        # custom float formats by string conversion:
        pulse_df[CSV_LABEL_AGE_CHG_RATE] = pulse_df[CSV_LABEL_AGE_CHG_RATE].map("{:.2f}".format)
        pulse_df[CSV_LABEL_AGE_DISCHG_RATE] = pulse_df[CSV_LABEL_AGE_DISCHG_RATE].map("{:.2f}".format)
        pulse_df[CSV_LABEL_T_AVG] = pulse_df[CSV_LABEL_T_AVG].map("{:.2f}".format)

        pulse_df[CSV_LABEL_PULSE_V] = pulse_df[CSV_LABEL_PULSE_V].map("{:.4f}".format)
        pulse_df[CSV_LABEL_PULSE_I] = pulse_df[CSV_LABEL_PULSE_I].map("{:.4f}".format)

        # write to csv, with remaining floats using %.3f float format
        pulse_df.to_csv(cfg.CSV_RESULT_DIR + (cfg.CSV_FILENAME_05_RESULT_BASE_CELL
                                              % (cfg.CSV_FILENAME_05_TYPE_PULSE, cell_pulse_file["param_id"],
                                                 cell_pulse_file["param_nr"], slave_id, cell_id)),
                            index=False, sep=cfg.CSV_SEP, float_format="%.3f")

        report_msg = f"%s - S%02u:C%02u - %u converted without and %u converted with warnings, %u couldn't be " \
                     f"converted (error)" % (cell_pulse_file["filename"], slave_id, cell_id, num_converted_ok,
                                             num_converted_warnings, num_not_converted_error)
        report_level = logging.INFO
        if num_not_converted_error > 0:
            report_level = logging.ERROR
        elif num_converted_warnings > 0:
            report_level = logging.WARNING

        cell_report = {"msg": report_msg, "level": report_level}
        report_queue.put(cell_report)

    client.close()


def fix_log_dq_de(report_queue):
    logging.log.info("========== 2) Extend LOG data: dQ/dE ==========")

    # find .csv files: cell_log_P012_3_S14_C11.csv
    cell_log_csv = []
    slave_cell_found = [[" "] * cfg.NUM_CELLS_PER_SLAVE for _ in range(cfg.NUM_SLAVES_MAX)]
    with os.scandir(cfg.CSV_RESULT_DIR) as iterator:
        re_str_log_csv = cfg.CSV_FILENAME_05_RESULT_BASE_CELL_RE.replace("(\w)", cfg.CSV_FILENAME_05_TYPE_LOG)
        re_pat_log_csv = re.compile(re_str_log_csv)
        for entry in iterator:
            re_match_log_csv = re_pat_log_csv.fullmatch(entry.name)
            if re_match_log_csv:
                param_id = int(re_match_log_csv.group(1))
                param_nr = int(re_match_log_csv.group(2))
                slave_id = int(re_match_log_csv.group(3))
                cell_id = int(re_match_log_csv.group(4))
                cell_csv = {"param_id": param_id, "param_nr": param_nr, "slave_id": slave_id, "cell_id": cell_id,
                             "filename": entry.name}
                cell_log_csv.append(cell_csv)
                fix_log_csv_task_queue.put(cell_csv)
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

    # List found slaves/cells for user
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
    logging.log.info("Starting processes to extend LOG data...")
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE_LOG):
        logging.log.debug("  Starting process %u" % processorNumber)
        processes.append(multiprocessing.Process(target=fix_log_csv_thread,
                                                 args=(processorNumber, fix_log_csv_task_queue, report_queue, )))
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE_LOG):
        processes[processorNumber].start()
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE_LOG):
        processes[processorNumber].join()
        logging.log.debug("Joined process %u" % processorNumber)


def fix_log_csv_thread(processor_number, slave_queue, thread_report_queue):
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

        # if not ((slave_id == 8) and (cell_id == 1)):
        #     continue  # for debugging individual cells

        filename_csv = queue_entry["filename"]
        logging.log.info("Thread %u S%02u:C%02u - extending LOG data" % (processor_number, slave_id, cell_id))
        # num_runs = 0
        num_points_edited = 0
        num_infos = 0
        num_warnings = 0
        num_errors = 0

        # I. LOG file
        logging.log.debug("Thread %u S%02u:C%02u - reading LOG file" % (processor_number, slave_id, cell_id))
        log_df = pd.read_csv(cfg.CSV_RESULT_DIR + filename_csv, header=0, sep=cfg.CSV_SEP)

        # check if the file was already modified - if so, warn (and skip depending on configuration)
        if CSV_LABEL_DELTA_Q_CHG in log_df:
            # the file was probably already modified!
            if SKIP_ALREADY_EDITED_LOG_FILES:
                report_msg = f"%s - S%02u:C%02u - Warning: extending LOG - data skipped because file was already " \
                             f"edited before" % (filename_csv, slave_id, cell_id)
                report_level = logging.WARNING
                cell_report = {"msg": report_msg, "level": report_level}
                logging.log.warning(report_msg)
                thread_report_queue.put(cell_report)
                continue  # go to next queue entry
            else:
                logging.log.warning("%s - S%02u:C%02u - Warning: extending LOG - data was already edited before "
                                    "-> continue anyway" % (filename_csv, slave_id, cell_id))

        # - insert new columns dQ/dE_chg/dischg (4x)
        default_value = 0  # np.nan
        log_df[CSV_LABEL_DELTA_Q_CHG] = default_value
        log_df[CSV_LABEL_DELTA_Q_DISCHG] = default_value
        log_df[CSV_LABEL_DELTA_E_CHG] = default_value
        log_df[CSV_LABEL_DELTA_E_DISCHG] = default_value

        # - insert new columns total_dQ/dE_(CU_RT/CYC_OT/others_RT/others_OT/sum)_chg/dischg (20x)
        log_df[CSV_LABEL_TOTAL_Q_CHG_CU_RT] = default_value
        log_df[CSV_LABEL_TOTAL_Q_DISCHG_CU_RT] = default_value
        log_df[CSV_LABEL_TOTAL_Q_CHG_CYC_OT] = default_value
        log_df[CSV_LABEL_TOTAL_Q_DISCHG_CYC_OT] = default_value
        log_df[CSV_LABEL_TOTAL_Q_CHG_OTHER_RT] = default_value
        log_df[CSV_LABEL_TOTAL_Q_DISCHG_OTHER_RT] = default_value
        log_df[CSV_LABEL_TOTAL_Q_CHG_OTHER_OT] = default_value
        log_df[CSV_LABEL_TOTAL_Q_DISCHG_OTHER_OT] = default_value
        log_df[CSV_LABEL_TOTAL_Q_CHG_SUM] = default_value
        log_df[CSV_LABEL_TOTAL_Q_DISCHG_SUM] = default_value
        log_df[CSV_LABEL_TOTAL_E_CHG_CU_RT] = default_value
        log_df[CSV_LABEL_TOTAL_E_DISCHG_CU_RT] = default_value
        log_df[CSV_LABEL_TOTAL_E_CHG_CYC_OT] = default_value
        log_df[CSV_LABEL_TOTAL_E_DISCHG_CYC_OT] = default_value
        log_df[CSV_LABEL_TOTAL_E_CHG_OTHER_RT] = default_value
        log_df[CSV_LABEL_TOTAL_E_DISCHG_OTHER_RT] = default_value
        log_df[CSV_LABEL_TOTAL_E_CHG_OTHER_OT] = default_value
        log_df[CSV_LABEL_TOTAL_E_DISCHG_OTHER_OT] = default_value
        log_df[CSV_LABEL_TOTAL_E_CHG_SUM] = default_value
        log_df[CSV_LABEL_TOTAL_E_DISCHG_SUM] = default_value

        log_df[CSV_LABEL_DELTA_DQ_CHG] = default_value
        log_df[CSV_LABEL_DELTA_DQ_DISCHG] = default_value
        log_df[CSV_LABEL_DELTA_DE_CHG] = default_value
        log_df[CSV_LABEL_DELTA_DE_DISCHG] = default_value

        # - walk along all rows and increment dQ/dE_chg/dischg accordingly (for rising/falling dQ/dE)
        # cut into sections from IDLE_PREPARE/WAIT_FOR_START->RUNNING to ...
        #    ... RUNNING->FINISH/FINISHED/IDLE_PREPARE/WAIT_FOR_START (+ 1 LOG after) = 1 "run"

        # list of run start indexes = first with RUNNING sub state (in CHARGE/DISCHARGE/DYNAMIC charge state) ...
        #    ... following a FINISH/FINISHED/IDLE_PREPARE/WAIT_FOR_START/START sub state (or a CALENDAR_AGING chg state)
        run_start_indexes = log_df.index[(log_df[CSV_LABEL_SCH_STATE_SUB] == state_sub.RUNNING)
                                          & ((log_df[CSV_LABEL_SCH_STATE_CHG] == state_chg.CHARGE)
                                             | (log_df[CSV_LABEL_SCH_STATE_CHG] == state_chg.DISCHARGE)
                                             | (log_df[CSV_LABEL_SCH_STATE_CHG] == state_chg.DYNAMIC))
                                          & ((log_df[CSV_LABEL_SCH_STATE_SUB].shift(1) == state_sub.FINISH)
                                             | (log_df[CSV_LABEL_SCH_STATE_SUB].shift(1) == state_sub.FINISHED)
                                             | (log_df[CSV_LABEL_SCH_STATE_SUB].shift(1) == state_sub.IDLE_PREPARE)
                                             | (log_df[CSV_LABEL_SCH_STATE_SUB].shift(1) == state_sub.WAIT_FOR_START)
                                             | (log_df[CSV_LABEL_SCH_STATE_SUB].shift(1) == state_sub.START)
                                             | (log_df[CSV_LABEL_SCH_STATE_CHG].shift(1) == state_chg.CALENDAR_AGING)
                                             | (log_df[CSV_LABEL_SCH_STATE_CHG].shift(1) == state_chg.IDLE))]

        # list of run end indexes = first with FINISH/FINISHED/IDLE_PREPARE/WAIT_FOR_START/START sub state following ...
        #    ... a RUNNING sub state (in CHARGE/DISCHARGE/DYNAMIC charge state)
        run_end_indexes = log_df.index[((log_df[CSV_LABEL_SCH_STATE_SUB] == state_sub.FINISH)
                                        | (log_df[CSV_LABEL_SCH_STATE_SUB] == state_sub.FINISHED)
                                        | ((log_df[CSV_LABEL_SCH_STATE_SUB] == state_sub.IDLE_PREPARE)
                                           & (log_df[CSV_LABEL_SCH_STATE_DEC] != 0x00000010))  # = reboot
                                        | (log_df[CSV_LABEL_SCH_STATE_SUB] == state_sub.WAIT_FOR_START)
                                        | (log_df[CSV_LABEL_SCH_STATE_SUB] == state_sub.START))
                                       & ((log_df[CSV_LABEL_SCH_STATE_CHG].shift(1) == state_chg.CHARGE)
                                          | (log_df[CSV_LABEL_SCH_STATE_CHG].shift(1) == state_chg.DISCHARGE)
                                          | (log_df[CSV_LABEL_SCH_STATE_CHG].shift(1) == state_chg.DYNAMIC))
                                       & (log_df[CSV_LABEL_SCH_STATE_SUB].shift(1) == state_sub.RUNNING)]

        i_run = 0
        # for k in range(0, 1):  # for some cells, we need to do this twice
        # while i_run < (min(run_start_indexes.shape[0], run_end_indexes.shape[0]) - 1):
        while i_run < min(run_start_indexes.shape[0], run_end_indexes.shape[0]):
            if i_run < (min(run_start_indexes.shape[0], run_end_indexes.shape[0]) - 1):
                if run_end_indexes[i_run] > run_start_indexes[i_run + 1]:
                    run_end_indexes = run_end_indexes.insert(i_run, run_start_indexes[i_run + 1] - 1)
                    continue
            if run_end_indexes[i_run] < run_start_indexes[i_run]:
                # go up to start of "RUN" state
                seek_start = 0
                seek_end = run_end_indexes[i_run]
                if i_run > 0:
                    seek_start = run_end_indexes[i_run - 1] + 1
                log_state_sub_previous = log_df[CSV_LABEL_SCH_STATE_SUB][seek_start:seek_end]
                all_previous_run_indexes = log_state_sub_previous.index[log_state_sub_previous == state_sub.RUNNING]
                if all_previous_run_indexes.shape[0] > 0:
                    new_start_index = all_previous_run_indexes[0]
                    run_start_indexes = run_start_indexes.insert(i_run, new_start_index)
                else:
                    # no previous run -> delete this end index
                    run_end_indexes = run_end_indexes.drop(run_end_indexes[i_run])
                continue
            i_run = i_run + 1

        # free memory
        log_state_sub_previous = ""
        all_previous_run_indexes = ""
        del log_state_sub_previous
        del all_previous_run_indexes
        gc.collect()

        num_run_starts = run_start_indexes.shape[0]
        num_run_ends = run_end_indexes.shape[0]
        if num_run_ends > num_run_starts:
            print("debug here")
            run_end_indexes = run_end_indexes.drop(run_end_indexes[-1])  # drop last index
            num_run_ends = run_end_indexes.shape[0]
        if run_start_indexes[0] > run_end_indexes[0]:
            # start begins with RUNNING? (unusual) -> add index 0 to the beginning
            print("debug here")
            num_warnings = num_warnings + 1
            run_start_indexes = run_start_indexes.insert(0, 0)
            num_run_starts = run_start_indexes.shape[0]
        if num_run_starts > num_run_ends:
            # end ends with RUNNING? (likely) -> add last index to the end => not working for permanently paused cells
            # run_end_indexes = run_end_indexes.insert(run_end_indexes.shape[0], log_df.index[-1])
            # instead, go down to end of "RUN" state
            seek_start = run_start_indexes[-1] + 1
            seek_end = log_df.index[-1] + 1
            log_state_sub = log_df[CSV_LABEL_SCH_STATE_SUB][seek_start:seek_end]
            all_run_indexes = log_state_sub.index[log_state_sub == state_sub.RUNNING]
            if all_run_indexes.shape[0] > 0:
                new_end_index = all_run_indexes[-1]
                run_end_indexes = run_end_indexes.append(pd.Index([new_end_index]))
                num_run_ends = run_end_indexes.shape[0]
            else:
                # no run following -> delete this end index
                run_start_indexes = run_start_indexes.drop(run_start_indexes[-1])
                num_run_starts = run_start_indexes.shape[0]

        if num_run_starts > num_run_ends:
            # number of run_start_indexes elements is still larger than run_end_indexes ==> error
            logging.log.error("Thread %u S%02u:C%02u - Error: more run start (%u) that run endings (%u)"
                              % (processor_number, slave_id, cell_id, num_run_starts, num_run_ends))
            num_errors = num_errors + 1
        elif num_run_starts < num_run_ends:
            # number of run_start_indexes elements is still larger than run_end_indexes ==> error
            logging.log.error("Thread %u S%02u:C%02u - Error: less run start (%u) that run endings (%u)"
                              % (processor_number, slave_id, cell_id, num_run_starts, num_run_ends))
            num_errors = num_errors + 1
        else:
            if not all(run_end_indexes - run_start_indexes > 0):
                # some run_start_indexes come after the run_end_indexes ==> error
                faulty_start_indexes = run_start_indexes[run_end_indexes - run_start_indexes <= 0]
                faulty_end_indexes = run_end_indexes[run_end_indexes - run_start_indexes <= 0]
                logging.log.error("Thread %u S%02u:C%02u - Error: Some run starts lie before the run endings:\n"
                                  "Faulty starts:\n%s\nFaulty ends:\n%s"
                                  % (processor_number, slave_id, cell_id,
                                     log_df[CSV_LABEL_TIMESTAMP][faulty_start_indexes],
                                     log_df[CSV_LABEL_TIMESTAMP][faulty_end_indexes]))
                num_errors = num_errors + 1

        logging.log.debug("Thread %u S%02u:C%02u - walking through runs" % (processor_number, slave_id, cell_id))
        # last_index = log_df.index[-1] + 1

        # detect unusually high delta_q :
        log_dq_pre = log_df[CSV_LABEL_DELTA_Q][0:run_start_indexes[0]]
        log_ddq_pre = log_dq_pre - log_dq_pre.shift(1)
        if log_ddq_pre.shape[0] > 0:
            log_ddq_pre.iloc[0] = 0
            log_ddq_pre_max = max(abs(log_ddq_pre))
            if log_ddq_pre_max > MAX_DELTA_DQ_START:
                logging.log.warning("Thread %u S%02u:C%02u - Warning: Unusually high delta_dQ (%.6f) before start"
                                    % (processor_number, slave_id, cell_id, log_ddq_pre_max))
                num_warnings = num_warnings + 1

        # free memory
        log_dq_pre = ""
        log_ddq_pre = ""
        del log_dq_pre
        del log_ddq_pre
        gc.collect()

        num_runs = num_run_starts
        for i_run in range(0, min(num_run_starts, num_run_ends)):
            start_index = run_start_indexes[i_run]
            end_index = run_end_indexes[i_run]
            log_df_run = log_df[start_index:(end_index + 1)]
            delta_dQ_start = log_df_run[CSV_LABEL_DELTA_Q][start_index]
            delta_dE_start = log_df_run[CSV_LABEL_DELTA_E][start_index]
            if abs(delta_dQ_start) > DQ_START_MAX:
                timestamp_int = log_df[CSV_LABEL_TIMESTAMP][start_index]
                timestamp_string = pd.to_datetime(timestamp_int, unit="s")
                tmp_str = f"Initial delta_q unusually high (%.4f) at log index %u " \
                          f"(%s) -> set to 0" % (log_df[CSV_LABEL_DELTA_Q][start_index], start_index, timestamp_string)
                is_known_gap = is_data_gap_known(timestamp_int)
                if (not is_known_gap) and (abs(delta_dQ_start) > DQ_START_MAX_WARN):
                    logging.log.warning("Thread %u S%02u:C%02u - Warning: %s"
                                        % (processor_number, slave_id, cell_id, tmp_str))
                    num_warnings = num_warnings + 1
                else:
                    is_known_str = ""
                    if is_known_gap:
                        is_known_str = " (known data gap)"
                    logging.log.info("Thread %u S%02u:C%02u - Info: %s%s"
                                     % (processor_number, slave_id, cell_id, tmp_str, is_known_str))
                    num_infos = num_infos + 1
                delta_dQ_start = 0
                delta_dE_start = 0
                # S04:C07 - Warning: Initial delta_q unusually high (-0.0119) at log index 301247   -> OK
                #   -> because of a reboot, state of re-entry not properly chosen, but OK here because error very small
                # S04:C07 - Warning: Initial delta_q unusually high (1.1927) at log index 887750    ->OK
                #   -> halloween internet failure, state of re-entry not properly chosen. Sucks, but no good
                #      (universally applicable) fix

            # calculate difference between adjacent dQ's / dE's
            delta_dQ = log_df_run[CSV_LABEL_DELTA_Q] - log_df_run[CSV_LABEL_DELTA_Q].shift(1)
            delta_dQ[start_index] = delta_dQ_start  # fill first with start value
            delta_dQ[end_index] = 0  # set last to 0

            delta_dE = log_df_run[CSV_LABEL_DELTA_E] - log_df_run[CSV_LABEL_DELTA_E].shift(1)
            delta_dE[start_index] = delta_dE_start
            delta_dE[end_index] = 0  # set last to 0

            # copy to chg/dischg series and eliminate all that are not charging/discharging
            delta_dQ_chg = delta_dQ.copy()
            delta_dQ_dischg = -delta_dQ
            delta_dQ_chg[delta_dQ_chg <= 0] = 0
            delta_dQ_dischg[delta_dQ_dischg <= 0] = 0

            delta_dE_chg = delta_dE.copy()
            delta_dE_dischg = -delta_dE
            delta_dE_chg[delta_dE_chg <= 0] = 0
            delta_dE_dischg[delta_dE_dischg <= 0] = 0

            # copy delta_dQ/dE_chg/dischg to main data frame
            log_df.loc[start_index:end_index, CSV_LABEL_DELTA_DQ_CHG] = delta_dQ_chg
            log_df.loc[start_index:end_index, CSV_LABEL_DELTA_DQ_DISCHG] = delta_dQ_dischg
            log_df.loc[start_index:end_index, CSV_LABEL_DELTA_DE_CHG] = delta_dE_chg
            log_df.loc[start_index:end_index, CSV_LABEL_DELTA_DE_DISCHG] = delta_dE_dischg

            # sum up to current value
            delta_dQ_chg = delta_dQ_chg.cumsum()
            delta_dQ_dischg = delta_dQ_dischg.cumsum()

            delta_dE_chg = delta_dE_chg.cumsum()
            delta_dE_dischg = delta_dE_dischg.cumsum()

            # copy rest to main data frame (delta_dQ_chg is now delta_Q_chg, ...)
            log_df.loc[start_index:end_index, CSV_LABEL_DELTA_Q_CHG] = delta_dQ_chg
            log_df.loc[start_index:end_index, CSV_LABEL_DELTA_Q_DISCHG] = delta_dQ_dischg

            log_df.loc[start_index:end_index, CSV_LABEL_DELTA_E_CHG] = delta_dE_chg
            log_df.loc[start_index:end_index, CSV_LABEL_DELTA_E_DISCHG] = delta_dE_dischg

            num_points_edited = num_points_edited + end_index - start_index + 1
            # last_index = end_index + 1
            # find first value where dQ was reset
            if i_run < (num_run_starts - 1):
                max_index = run_start_indexes[i_run + 1]
            else:
                max_index = log_df.index[-1] + 1
            if max_index > (end_index + 1):
                last_dQ = log_df[CSV_LABEL_DELTA_Q][end_index]
                forward_dQ = log_df[CSV_LABEL_DELTA_Q][(end_index + 1):max_index]
                if last_dQ > 0:  # 0 <= min_dQ <= max_dQ
                    min_dQ = ((1 - DQ_NEW_REL) * last_dQ)
                    max_dQ = ((1 + DQ_NEW_REL) * last_dQ)
                else:  # min_dQ <= max_dQ <= 0
                    min_dQ = ((1 + DQ_NEW_REL) * last_dQ)
                    max_dQ = ((1 - DQ_NEW_REL) * last_dQ)
                deviating_indexes = forward_dQ[(forward_dQ < min_dQ) | (forward_dQ > max_dQ)]

                forward_start_index = end_index + 1
                forward_end_index = max_index - 1
                if deviating_indexes.shape[0] > 0:
                    forward_end_index = deviating_indexes.index[0] - 1
                # else: all are the same -> extend to (max_index - 1)

                # extend last values to the index just before the reset
                if forward_end_index >= forward_start_index:
                    log_df.loc[forward_start_index:forward_end_index, CSV_LABEL_DELTA_Q_CHG] = \
                        log_df[CSV_LABEL_DELTA_Q_CHG][end_index]
                    log_df.loc[forward_start_index:forward_end_index, CSV_LABEL_DELTA_Q_DISCHG] = \
                        log_df[CSV_LABEL_DELTA_Q_DISCHG][end_index]

                    log_df.loc[forward_start_index:forward_end_index, CSV_LABEL_DELTA_E_CHG] = \
                        log_df[CSV_LABEL_DELTA_E_CHG][end_index]
                    log_df.loc[forward_start_index:forward_end_index, CSV_LABEL_DELTA_E_DISCHG] = \
                        log_df[CSV_LABEL_DELTA_E_DISCHG][end_index]

                    # last_index = forward_end_index + 1

                # detect unusually high delta_q :
                log_dq_inter = log_df[CSV_LABEL_DELTA_Q][forward_start_index:forward_end_index]
                log_ddq_inter = log_dq_inter - log_dq_inter.shift(1)
                if log_ddq_inter.shape[0] > 0:
                    log_ddq_inter.iloc[0] = 0
                    log_ddq_inter_max = max(abs(log_ddq_inter))
                    if log_ddq_inter_max > MAX_DELTA_DQ_IDLE:
                        timestamp_int = log_df[CSV_LABEL_TIMESTAMP][forward_start_index]
                        timestamp_string = pd.to_datetime(timestamp_int, unit="s")
                        tmp_str = f"Thread %u S%02u:C%02u - Warning: Unusually high delta_dQ (%.6f) after run %u (%s)" \
                                  % (processor_number, slave_id, cell_id, log_ddq_inter_max, i_run, timestamp_string)
                        is_known_gap = is_data_gap_known(timestamp_int)
                        if (not is_known_gap) and (abs(log_ddq_inter_max) > MAX_DELTA_DQ_IDLE_WARN):
                            logging.log.warning("Thread %u S%02u:C%02u - Warning: %s"
                                                % (processor_number, slave_id, cell_id, tmp_str))
                            num_warnings = num_warnings + 1
                        else:
                            is_known_str = ""
                            if is_known_gap:
                                is_known_str = " (known data gap)"
                            logging.log.info("Thread %u S%02u:C%02u - Info: %s%s"
                                             % (processor_number, slave_id, cell_id, tmp_str, is_known_str))
                            num_infos = num_infos + 1

        # free memory
        delta_dQ = ""
        delta_dQ_chg = ""
        delta_dQ_dischg = ""
        delta_dE = ""
        delta_dE_chg = ""
        delta_dE_dischg = ""
        deviating_indexes = ""
        forward_dQ = ""
        log_ddq_inter = ""
        log_dq_inter = ""
        log_df_run = ""
        run_end_indexes = ""
        run_start_indexes = ""
        del delta_dQ
        del delta_dQ_chg
        del delta_dQ_dischg
        del delta_dE
        del delta_dE_chg
        del delta_dE_dischg
        del deviating_indexes
        del forward_dQ
        del log_ddq_inter
        del log_dq_inter
        del log_df_run
        del run_end_indexes
        del run_start_indexes
        gc.collect()

        logging.log.debug("Thread %u S%02u:C%02u - calculating total_dQ/dE's" % (processor_number, slave_id, cell_id))

        # Check-Up
        # Total delta_Q/E during check-up (room temperature) - charging => CAP_MEAS_CHG*
        log_df.loc[(log_df[CSV_LABEL_SCH_STATE_CU] == state_cu.CAP_MEAS_CHG)
                   | (log_df[CSV_LABEL_SCH_STATE_CU] == state_cu.CAP_MEAS_DISCHG), CSV_LABEL_TOTAL_Q_CHG_CU_RT] = \
            log_df[CSV_LABEL_DELTA_DQ_CHG]
        log_df.loc[(log_df[CSV_LABEL_SCH_STATE_CU] == state_cu.CAP_MEAS_CHG)
                   | (log_df[CSV_LABEL_SCH_STATE_CU] == state_cu.CAP_MEAS_DISCHG), CSV_LABEL_TOTAL_E_CHG_CU_RT] = \
            log_df[CSV_LABEL_DELTA_DE_CHG]

        # Total delta_Q/E during check-up (room temperature) - discharging => CAP_MEAS_DISCHG*
        log_df.loc[(log_df[CSV_LABEL_SCH_STATE_CU] == state_cu.CAP_MEAS_CHG)
                   | (log_df[CSV_LABEL_SCH_STATE_CU] == state_cu.CAP_MEAS_DISCHG), CSV_LABEL_TOTAL_Q_DISCHG_CU_RT] = \
            log_df[CSV_LABEL_DELTA_DQ_DISCHG]
        log_df.loc[(log_df[CSV_LABEL_SCH_STATE_CU] == state_cu.CAP_MEAS_CHG)
                   | (log_df[CSV_LABEL_SCH_STATE_CU] == state_cu.CAP_MEAS_DISCHG), CSV_LABEL_TOTAL_E_DISCHG_CU_RT] = \
            log_df[CSV_LABEL_DELTA_DE_DISCHG]
        # *also include CAP_MEAS_DISCHG / CAP_MEAS_CHG because one LOG after this state can also include some dQ/dE

        # Cycling
        # Total delta_Q/E during cycling or calendar aging (operational temperature) - charging => CYCLING
        log_df.loc[log_df[CSV_LABEL_SCH_STATE_PH] == state_ph.CYCLING, CSV_LABEL_TOTAL_Q_CHG_CYC_OT] = \
            log_df[CSV_LABEL_DELTA_DQ_CHG]
        log_df.loc[log_df[CSV_LABEL_SCH_STATE_PH] == state_ph.CYCLING, CSV_LABEL_TOTAL_E_CHG_CYC_OT] = \
            log_df[CSV_LABEL_DELTA_DE_CHG]

        # Total delta_Q/E during cycling or calendar aging (operational temperature) - discharging => CYCLING
        log_df.loc[log_df[CSV_LABEL_SCH_STATE_PH] == state_ph.CYCLING, CSV_LABEL_TOTAL_Q_DISCHG_CYC_OT] = \
            log_df[CSV_LABEL_DELTA_DQ_DISCHG]
        log_df.loc[log_df[CSV_LABEL_SCH_STATE_PH] == state_ph.CYCLING, CSV_LABEL_TOTAL_E_DISCHG_CYC_OT] = \
            log_df[CSV_LABEL_DELTA_DE_DISCHG]

        # Other RT -> only at checkup
        # Total delta_Q/E others (room temperature) - charging => PREPARE_SET, PREPARE_DISCHARGE, RT_EIS_PULSE
        log_df.loc[(log_df[CSV_LABEL_SCH_STATE_CU] == state_cu.PREPARE_SET)
                   | (log_df[CSV_LABEL_SCH_STATE_CU] == state_cu.PREPARE_DISCHG)
                   | (log_df[CSV_LABEL_SCH_STATE_CU] == state_cu.RT_EIS_PULSE),
                   CSV_LABEL_TOTAL_Q_CHG_OTHER_RT] \
            = log_df[CSV_LABEL_DELTA_DQ_CHG]
        log_df.loc[(log_df[CSV_LABEL_SCH_STATE_CU] == state_cu.PREPARE_SET)
                   | (log_df[CSV_LABEL_SCH_STATE_CU] == state_cu.PREPARE_DISCHG)
                   | (log_df[CSV_LABEL_SCH_STATE_CU] == state_cu.RT_EIS_PULSE),
                   CSV_LABEL_TOTAL_E_CHG_OTHER_RT] \
            = log_df[CSV_LABEL_DELTA_DE_CHG]

        # Total delta_Q/E others (room temperature) - discharging => PREPARE_SET, PREPARE_DISCHARGE, RT_EIS_PULSE
        log_df.loc[(log_df[CSV_LABEL_SCH_STATE_CU] == state_cu.PREPARE_SET)
                   | (log_df[CSV_LABEL_SCH_STATE_CU] == state_cu.PREPARE_DISCHG)
                   | (log_df[CSV_LABEL_SCH_STATE_CU] == state_cu.RT_EIS_PULSE),
                   CSV_LABEL_TOTAL_Q_DISCHG_OTHER_RT] \
            = log_df[CSV_LABEL_DELTA_DQ_DISCHG]
        log_df.loc[(log_df[CSV_LABEL_SCH_STATE_CU] == state_cu.PREPARE_SET)
                   | (log_df[CSV_LABEL_SCH_STATE_CU] == state_cu.PREPARE_DISCHG)
                   | (log_df[CSV_LABEL_SCH_STATE_CU] == state_cu.RT_EIS_PULSE),
                   CSV_LABEL_TOTAL_E_DISCHG_OTHER_RT] \
            = log_df[CSV_LABEL_DELTA_DE_DISCHG]

        # Other OT -> only at checkup
        # Total delta_Q/E others (operation temperature) - charging => OT_EIS_PULSE, FOLLOW_UP
        log_df.loc[(log_df[CSV_LABEL_SCH_STATE_CU] == state_cu.OT_EIS_PULSE)
                   | (log_df[CSV_LABEL_SCH_STATE_CU] == state_cu.FOLLOW_UP), CSV_LABEL_TOTAL_Q_CHG_OTHER_OT] \
            = log_df[CSV_LABEL_DELTA_DQ_CHG]
        log_df.loc[(log_df[CSV_LABEL_SCH_STATE_CU] == state_cu.OT_EIS_PULSE)
                   | (log_df[CSV_LABEL_SCH_STATE_CU] == state_cu.FOLLOW_UP), CSV_LABEL_TOTAL_E_CHG_OTHER_OT] \
            = log_df[CSV_LABEL_DELTA_DE_CHG]

        # Total delta_Q/E others (operation temperature) - discharging => OT_EIS_PULSE, FOLLOW_UP
        log_df.loc[(log_df[CSV_LABEL_SCH_STATE_CU] == state_cu.OT_EIS_PULSE)
                   | (log_df[CSV_LABEL_SCH_STATE_CU] == state_cu.FOLLOW_UP), CSV_LABEL_TOTAL_Q_DISCHG_OTHER_OT] \
            = log_df[CSV_LABEL_DELTA_DQ_DISCHG]
        log_df.loc[(log_df[CSV_LABEL_SCH_STATE_CU] == state_cu.OT_EIS_PULSE)
                   | (log_df[CSV_LABEL_SCH_STATE_CU] == state_cu.FOLLOW_UP), CSV_LABEL_TOTAL_E_DISCHG_OTHER_OT] \
            = log_df[CSV_LABEL_DELTA_DE_DISCHG]

        # sum of total sums (the dQ/dE's are mutually exclusive
        log_df[CSV_LABEL_TOTAL_Q_CHG_SUM] = log_df[CSV_LABEL_TOTAL_Q_CHG_CU_RT] \
                                                  + log_df[CSV_LABEL_TOTAL_Q_CHG_CYC_OT] \
                                                  + log_df[CSV_LABEL_TOTAL_Q_CHG_OTHER_RT] \
                                                  + log_df[CSV_LABEL_TOTAL_Q_CHG_OTHER_OT]
        log_df[CSV_LABEL_TOTAL_E_CHG_SUM] = log_df[CSV_LABEL_TOTAL_E_CHG_CU_RT] \
                                                  + log_df[CSV_LABEL_TOTAL_E_CHG_CYC_OT] \
                                                  + log_df[CSV_LABEL_TOTAL_E_CHG_OTHER_RT] \
                                                  + log_df[CSV_LABEL_TOTAL_E_CHG_OTHER_OT]

        log_df[CSV_LABEL_TOTAL_Q_DISCHG_SUM] = log_df[CSV_LABEL_TOTAL_Q_DISCHG_CU_RT] \
                                                  + log_df[CSV_LABEL_TOTAL_Q_DISCHG_CYC_OT] \
                                                  + log_df[CSV_LABEL_TOTAL_Q_DISCHG_OTHER_RT] \
                                                  + log_df[CSV_LABEL_TOTAL_Q_DISCHG_OTHER_OT]
        log_df[CSV_LABEL_TOTAL_E_DISCHG_SUM] = log_df[CSV_LABEL_TOTAL_E_DISCHG_CU_RT] \
                                                  + log_df[CSV_LABEL_TOTAL_E_DISCHG_CYC_OT] \
                                                  + log_df[CSV_LABEL_TOTAL_E_DISCHG_OTHER_RT] \
                                                  + log_df[CSV_LABEL_TOTAL_E_DISCHG_OTHER_OT]

        # check if one of CSV_LABEL_DELTA_DQ_CHG is <> 0 where CSV_LABEL_TOTAL_Q_CHG_SUM is 0
        dq_chg_check = log_df[CSV_LABEL_DELTA_DQ_CHG][log_df[CSV_LABEL_TOTAL_Q_CHG_SUM] == 0]
        dq_chg_check = dq_chg_check[abs(dq_chg_check) > MAX_DELTA_DQ_IDLE]
        dq_dischg_check = log_df[CSV_LABEL_DELTA_DQ_DISCHG][log_df[CSV_LABEL_TOTAL_Q_DISCHG_SUM] == 0]
        dq_dischg_check = dq_dischg_check[abs(dq_dischg_check) > MAX_DELTA_DQ_IDLE]
        de_chg_check = log_df[CSV_LABEL_DELTA_DE_CHG][log_df[CSV_LABEL_TOTAL_E_CHG_SUM] == 0]
        de_chg_check = de_chg_check[abs(de_chg_check) > MAX_DELTA_DE_IDLE]
        de_dischg_check = log_df[CSV_LABEL_DELTA_DE_DISCHG][log_df[CSV_LABEL_TOTAL_E_DISCHG_SUM] == 0]
        de_dischg_check = de_dischg_check[abs(de_dischg_check) > MAX_DELTA_DE_IDLE]
        if ((dq_chg_check.shape[0] > 0) or (dq_dischg_check.shape[0] > 0)
                or (de_chg_check.shape[0] > 0) or (de_dischg_check.shape[0] > 0)):
            need_warning = False
            msg = "Unusually high delta_dQ/dE where total_Q/E is not changing"
            if dq_chg_check.shape[0] > 0:
                msg = msg + f"\n=== dQ_chg_check ===\n%s" % dq_chg_check
                if max(abs(dq_chg_check)) > MAX_DELTA_DQ_IDLE_WARN:
                    need_warning = True
            if dq_dischg_check.shape[0] > 0:
                msg = msg + f"\n=== dQ_dischg_check ===\n%s" % dq_dischg_check
                if max(abs(dq_dischg_check)) > MAX_DELTA_DQ_IDLE_WARN:
                    need_warning = True
            if de_chg_check.shape[0] > 0:
                msg = msg + f"\n=== dE_chg_check ===\n%s" % de_chg_check
                if max(abs(de_chg_check) > MAX_DELTA_DE_IDLE_WARN):
                    need_warning = True
            if de_dischg_check.shape[0] > 0:
                msg = msg + f"\n=== dE_dischg_check ===\n%s" % de_dischg_check
                if max(abs(de_dischg_check)) > MAX_DELTA_DE_IDLE_WARN:
                    need_warning = True
            if need_warning:
                logging.log.warning("Thread %u S%02u:C%02u - Warning: %s" % (processor_number, slave_id, cell_id, msg))
                num_warnings = num_warnings + 1
            else:
                logging.log.info("Thread %u S%02u:C%02u - Info: %s" % (processor_number, slave_id, cell_id, msg))
                num_infos = num_infos + 1
            # S01:C10 - Unusually high delta_dQ/dE where total_Q/E is not changing      304111      --> OK
            #   -> because of a reboot (small deviation because last dQ was not stored on flash directly before reboot)

        # free memory
        dq_chg_check = ""
        dq_dischg_check = ""
        de_chg_check = ""
        de_dischg_check = ""
        del dq_chg_check
        del dq_dischg_check
        del de_dischg_check
        gc.collect()

        # up to now, we only have delta dQ/dE's, now, sum up the totals:
        log_df[CSV_LABEL_TOTAL_Q_CHG_CU_RT] = log_df[CSV_LABEL_TOTAL_Q_CHG_CU_RT].cumsum()
        log_df[CSV_LABEL_TOTAL_Q_DISCHG_CU_RT] = log_df[CSV_LABEL_TOTAL_Q_DISCHG_CU_RT].cumsum()
        log_df[CSV_LABEL_TOTAL_Q_CHG_CYC_OT] = log_df[CSV_LABEL_TOTAL_Q_CHG_CYC_OT].cumsum()
        log_df[CSV_LABEL_TOTAL_Q_DISCHG_CYC_OT] = log_df[CSV_LABEL_TOTAL_Q_DISCHG_CYC_OT].cumsum()
        log_df[CSV_LABEL_TOTAL_Q_CHG_OTHER_RT] = log_df[CSV_LABEL_TOTAL_Q_CHG_OTHER_RT].cumsum()
        log_df[CSV_LABEL_TOTAL_Q_DISCHG_OTHER_RT] = log_df[CSV_LABEL_TOTAL_Q_DISCHG_OTHER_RT].cumsum()
        log_df[CSV_LABEL_TOTAL_Q_CHG_OTHER_OT] = log_df[CSV_LABEL_TOTAL_Q_CHG_OTHER_OT].cumsum()
        log_df[CSV_LABEL_TOTAL_Q_DISCHG_OTHER_OT] = log_df[CSV_LABEL_TOTAL_Q_DISCHG_OTHER_OT].cumsum()
        log_df[CSV_LABEL_TOTAL_Q_CHG_SUM] = log_df[CSV_LABEL_TOTAL_Q_CHG_SUM].cumsum()
        log_df[CSV_LABEL_TOTAL_Q_DISCHG_SUM] = log_df[CSV_LABEL_TOTAL_Q_DISCHG_SUM].cumsum()

        log_df[CSV_LABEL_TOTAL_E_CHG_CU_RT] = log_df[CSV_LABEL_TOTAL_E_CHG_CU_RT].cumsum()
        log_df[CSV_LABEL_TOTAL_E_DISCHG_CU_RT] = log_df[CSV_LABEL_TOTAL_E_DISCHG_CU_RT].cumsum()
        log_df[CSV_LABEL_TOTAL_E_CHG_CYC_OT] = log_df[CSV_LABEL_TOTAL_E_CHG_CYC_OT].cumsum()
        log_df[CSV_LABEL_TOTAL_E_DISCHG_CYC_OT] = log_df[CSV_LABEL_TOTAL_E_DISCHG_CYC_OT].cumsum()
        log_df[CSV_LABEL_TOTAL_E_CHG_OTHER_RT] = log_df[CSV_LABEL_TOTAL_E_CHG_OTHER_RT].cumsum()
        log_df[CSV_LABEL_TOTAL_E_DISCHG_OTHER_RT] = log_df[CSV_LABEL_TOTAL_E_DISCHG_OTHER_RT].cumsum()
        log_df[CSV_LABEL_TOTAL_E_CHG_OTHER_OT] = log_df[CSV_LABEL_TOTAL_E_CHG_OTHER_OT].cumsum()
        log_df[CSV_LABEL_TOTAL_E_DISCHG_OTHER_OT] = log_df[CSV_LABEL_TOTAL_E_DISCHG_OTHER_OT].cumsum()
        log_df[CSV_LABEL_TOTAL_E_CHG_SUM] = log_df[CSV_LABEL_TOTAL_E_CHG_SUM].cumsum()
        log_df[CSV_LABEL_TOTAL_E_DISCHG_SUM] = log_df[CSV_LABEL_TOTAL_E_DISCHG_SUM].cumsum()

        logging.log.debug("Thread %u S%02u:C%02u - applying custom formats" % (processor_number, slave_id, cell_id))
        # custom float formats by string conversion:
        log_df[CSV_LABEL_T_CELL] = log_df[CSV_LABEL_T_CELL].map("{:.2f}".format)

        log_df[CSV_LABEL_V_CELL] = log_df[CSV_LABEL_V_CELL].map("{:.4f}".format)
        log_df[CSV_LABEL_I_CELL] = log_df[CSV_LABEL_I_CELL].map("{:.4f}".format)
        log_df[CSV_LABEL_P_CELL] = log_df[CSV_LABEL_P_CELL].map("{:.4f}".format)
        log_df[CSV_LABEL_SOC_EST] = log_df[CSV_LABEL_SOC_EST].map("{:.4f}".format)
        log_df[CSV_LABEL_OCV_EST] = log_df[CSV_LABEL_OCV_EST].map("{:.4f}".format)

        log_df[CSV_LABEL_DELTA_Q] = log_df[CSV_LABEL_DELTA_Q].map("{:.6f}".format)
        log_df[CSV_LABEL_DELTA_E] = log_df[CSV_LABEL_DELTA_E].map("{:.6f}".format)
        log_df[CSV_LABEL_DELTA_Q_CHG] = log_df[CSV_LABEL_DELTA_Q_CHG].map("{:.6f}".format)
        log_df[CSV_LABEL_DELTA_Q_DISCHG] = log_df[CSV_LABEL_DELTA_Q_DISCHG].map("{:.6f}".format)
        log_df[CSV_LABEL_DELTA_E_CHG] = log_df[CSV_LABEL_DELTA_E_CHG].map("{:.6f}".format)
        log_df[CSV_LABEL_DELTA_E_DISCHG] = log_df[CSV_LABEL_DELTA_E_DISCHG].map("{:.6f}".format)
        log_df[CSV_LABEL_DELTA_DQ_CHG] = log_df[CSV_LABEL_DELTA_DQ_CHG].map("{:.6f}".format)
        log_df[CSV_LABEL_DELTA_DQ_DISCHG] = log_df[CSV_LABEL_DELTA_DQ_DISCHG].map("{:.6f}".format)
        log_df[CSV_LABEL_DELTA_DE_CHG] = log_df[CSV_LABEL_DELTA_DE_CHG].map("{:.6f}".format)
        log_df[CSV_LABEL_DELTA_DE_DISCHG] = log_df[CSV_LABEL_DELTA_DE_DISCHG].map("{:.6f}".format)

        flt_format = "{:.4f}"  # use the same as the total_d/q in the original EOC
        log_df[CSV_LABEL_TOTAL_Q_CHG_CU_RT] \
            = log_df[CSV_LABEL_TOTAL_Q_CHG_CU_RT].map(flt_format.format)
        log_df[CSV_LABEL_TOTAL_Q_DISCHG_CU_RT] \
            = log_df[CSV_LABEL_TOTAL_Q_DISCHG_CU_RT].map(flt_format.format)
        log_df[CSV_LABEL_TOTAL_Q_CHG_CYC_OT] \
            = log_df[CSV_LABEL_TOTAL_Q_CHG_CYC_OT].map(flt_format.format)
        log_df[CSV_LABEL_TOTAL_Q_DISCHG_CYC_OT] \
            = log_df[CSV_LABEL_TOTAL_Q_DISCHG_CYC_OT].map(flt_format.format)
        log_df[CSV_LABEL_TOTAL_Q_CHG_OTHER_RT] \
            = log_df[CSV_LABEL_TOTAL_Q_CHG_OTHER_RT].map(flt_format.format)
        log_df[CSV_LABEL_TOTAL_Q_DISCHG_OTHER_RT] \
            = log_df[CSV_LABEL_TOTAL_Q_DISCHG_OTHER_RT].map(flt_format.format)
        log_df[CSV_LABEL_TOTAL_Q_CHG_OTHER_OT] \
            = log_df[CSV_LABEL_TOTAL_Q_CHG_OTHER_OT].map(flt_format.format)
        log_df[CSV_LABEL_TOTAL_Q_DISCHG_OTHER_OT] \
            = log_df[CSV_LABEL_TOTAL_Q_DISCHG_OTHER_OT].map(flt_format.format)
        log_df[CSV_LABEL_TOTAL_Q_CHG_SUM] \
            = log_df[CSV_LABEL_TOTAL_Q_CHG_SUM].map(flt_format.format)
        log_df[CSV_LABEL_TOTAL_Q_DISCHG_SUM] \
            = log_df[CSV_LABEL_TOTAL_Q_DISCHG_SUM].map(flt_format.format)

        log_df[CSV_LABEL_TOTAL_E_CHG_CU_RT] \
            = log_df[CSV_LABEL_TOTAL_E_CHG_CU_RT].map(flt_format.format)
        log_df[CSV_LABEL_TOTAL_E_DISCHG_CU_RT] \
            = log_df[CSV_LABEL_TOTAL_E_DISCHG_CU_RT].map(flt_format.format)
        log_df[CSV_LABEL_TOTAL_E_CHG_CYC_OT] \
            = log_df[CSV_LABEL_TOTAL_E_CHG_CYC_OT].map(flt_format.format)
        log_df[CSV_LABEL_TOTAL_E_DISCHG_CYC_OT] \
            = log_df[CSV_LABEL_TOTAL_E_DISCHG_CYC_OT].map(flt_format.format)
        log_df[CSV_LABEL_TOTAL_E_CHG_OTHER_RT] \
            = log_df[CSV_LABEL_TOTAL_E_CHG_OTHER_RT].map(flt_format.format)
        log_df[CSV_LABEL_TOTAL_E_DISCHG_OTHER_RT] \
            = log_df[CSV_LABEL_TOTAL_E_DISCHG_OTHER_RT].map(flt_format.format)
        log_df[CSV_LABEL_TOTAL_E_CHG_OTHER_OT] \
            = log_df[CSV_LABEL_TOTAL_E_CHG_OTHER_OT].map(flt_format.format)
        log_df[CSV_LABEL_TOTAL_E_DISCHG_OTHER_OT] \
            = log_df[CSV_LABEL_TOTAL_E_DISCHG_OTHER_OT].map(flt_format.format)
        log_df[CSV_LABEL_TOTAL_E_CHG_SUM] \
            = log_df[CSV_LABEL_TOTAL_E_CHG_SUM].map(flt_format.format)
        log_df[CSV_LABEL_TOTAL_E_DISCHG_SUM] \
            = log_df[CSV_LABEL_TOTAL_E_DISCHG_SUM].map(flt_format.format)

        logging.log.debug("Thread %u S%02u:C%02u - writing new LOG file - this may take a while..."
                          % (processor_number, slave_id, cell_id))
        # write to csv, with remaining floats using %.4f float format
        log_df.to_csv(cfg.CSV_RESULT_DIR + (cfg.CSV_FILENAME_05_RESULT_BASE_CELL
                                            % (cfg.CSV_FILENAME_05_TYPE_LOG_EXT, queue_entry["param_id"],
                                               queue_entry["param_nr"], slave_id, cell_id)),
                      index=False, sep=cfg.CSV_SEP, float_format="%.3f")

        # free memory
        log_df = ""
        del log_df
        gc.collect()

        # reporting to main thread
        report_msg = f"%s - S%02u:C%02u - extended LOG data: %u runs with a total of %u points edited, %u infos, " \
                     f"%u warnings, %u errors" % (filename_csv, slave_id, cell_id, num_runs, num_points_edited,
                                                  num_infos, num_warnings, num_errors)
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


def is_data_gap_known(unixtimestamp_s):
    for i in range(0, len(KNOWN_GAPS_TIMESTAMP_START)):
        if unixtimestamp_s >= KNOWN_GAPS_TIMESTAMP_START[i]:
            if unixtimestamp_s <= KNOWN_GAPS_TIMESTAMP_END[i]:
                return True
        else:
            return False


def fix_eoc_csv(report_queue):
    logging.log.info("========== 3) Fix EOC data: num_cycles_checkup total_dQ/dE ==========")

    # find .csv files: cell_eoc_P012_3_S14_C11.csv
    cell_eoc_csv = []
    cell_log_csv = []
    slave_cell_found = [[" "] * cfg.NUM_CELLS_PER_SLAVE for _ in range(cfg.NUM_SLAVES_MAX)]
    with os.scandir(cfg.CSV_RESULT_DIR) as iterator:
        re_str_eoc_csv = cfg.CSV_FILENAME_05_RESULT_BASE_CELL_RE.replace("(\w)", cfg.CSV_FILENAME_05_TYPE_EOC)
        re_pat_eoc_csv = re.compile(re_str_eoc_csv)
        re_str_log_csv = cfg.CSV_FILENAME_05_RESULT_BASE_CELL_RE.replace("(\w)", cfg.CSV_FILENAME_05_TYPE_LOG_EXT)
        re_pat_log_csv = re.compile(re_str_log_csv)
        for entry in iterator:
            re_match_eoc_csv = re_pat_eoc_csv.fullmatch(entry.name)
            re_match_log_csv = re_pat_log_csv.fullmatch(entry.name)
            if re_match_eoc_csv:
                param_id = int(re_match_eoc_csv.group(1))
                param_nr = int(re_match_eoc_csv.group(2))
                slave_id = int(re_match_eoc_csv.group(3))
                cell_id = int(re_match_eoc_csv.group(4))
                cell_csv = {"param_id": param_id, "param_nr": param_nr, "slave_id": slave_id, "cell_id": cell_id,
                             "eoc_filename": entry.name}
                cell_eoc_csv.append(cell_csv)
                if (slave_id < 0) or (slave_id >= cfg.NUM_SLAVES_MAX):
                    logging.warning("Found unusual slave_id: %u" % slave_id)
                else:
                    if (cell_id < 0) or (cell_id >= cfg.NUM_CELLS_PER_SLAVE):
                        logging.warning("Found unusual cell_id: %u" % cell_id)
                    else:
                        if (slave_cell_found[slave_id][cell_id] == "b") or (slave_cell_found[slave_id][cell_id] == "e"):
                            logging.warning("Found more than one entry for S%02u:C%02u" % (slave_id, cell_id))
                        elif slave_cell_found[slave_id][cell_id] == "l":
                            slave_cell_found[slave_id][cell_id] = "b"
                        else:
                            slave_cell_found[slave_id][cell_id] = "e"
            elif re_match_log_csv:
                param_id = int(re_match_log_csv.group(1))
                param_nr = int(re_match_log_csv.group(2))
                slave_id = int(re_match_log_csv.group(3))
                cell_id = int(re_match_log_csv.group(4))
                cell_csv = {"param_id": param_id, "param_nr": param_nr, "slave_id": slave_id, "cell_id": cell_id,
                             "log_filename": entry.name}
                cell_log_csv.append(cell_csv)
                if (slave_id < 0) or (slave_id >= cfg.NUM_SLAVES_MAX):
                    logging.warning("Found unusual slave_id: %u" % slave_id)
                else:
                    if (cell_id < 0) or (cell_id >= cfg.NUM_CELLS_PER_SLAVE):
                        logging.warning("Found unusual cell_id: %u" % cell_id)
                    else:
                        if (slave_cell_found[slave_id][cell_id] == "b") or (slave_cell_found[slave_id][cell_id] == "l"):
                            logging.warning("Found more than one entry for S%02u:C%02u" % (slave_id, cell_id))
                        elif slave_cell_found[slave_id][cell_id] == "e":
                            slave_cell_found[slave_id][cell_id] = "b"
                        else:
                            slave_cell_found[slave_id][cell_id] = "l"

    for cell_eoc in cell_eoc_csv:
        for cell_log in cell_log_csv:
            if ((cell_eoc["param_id"] == cell_log["param_id"])
                    and (cell_eoc["param_nr"] == cell_log["param_nr"])
                    and (cell_eoc["slave_id"] == cell_log["slave_id"])
                    and (cell_eoc["cell_id"] == cell_log["cell_id"])):
                cell_eoc["log_filename"] = cell_log["log_filename"]
                slave_id = cell_eoc["slave_id"]
                cell_id = cell_eoc["cell_id"]
                if ((slave_id >= 0) and (slave_id < cfg.NUM_SLAVES_MAX)
                        and (cell_id >= 0) and (cell_id < cfg.NUM_CELLS_PER_SLAVE)):
                    slave_cell_found[slave_id][cell_id] = "X"
                fix_eoc_csv_task_queue.put(cell_eoc)
                break

    # List found slaves/cells for user
    print("Found the following files:\n"
          "' ' = no file found, 'e' = only EOC, 'l' = only LOG, 'b' = both found, 'X' = both found & matching -> added")
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
    logging.log.info("Starting processes to fix EOC data...")
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE_EOC):
        logging.log.debug("  Starting process %u" % processorNumber)
        processes.append(multiprocessing.Process(target=fix_eoc_csv_thread,
                                                 args=(processorNumber, fix_eoc_csv_task_queue, report_queue, )))
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE_EOC):
        processes[processorNumber].start()
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE_EOC):
        processes[processorNumber].join()
        logging.log.debug("Joined process %u" % processorNumber)


def fix_eoc_csv_thread(processor_number, slave_queue, thread_report_queue):
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

        # if not ((slave_id == 8) and (cell_id == 1)):
        #     continue  # for debugging individual cells

        filename_eoc_csv = queue_entry["eoc_filename"]
        filename_log_csv = queue_entry["log_filename"]
        logging.log.info("Thread %u S%02u:C%02u - fixing EOC data" % (processor_number, slave_id, cell_id))
        num_converted = 0
        num_dQdE_total_updates = 0
        num_warnings = 0
        num_errors = 0

        eoc_df = pd.read_csv(cfg.CSV_RESULT_DIR + filename_eoc_csv, header=0, sep=cfg.CSV_SEP)

        if FIX_CHECKUP_NUM:
            # update CSV_LABEL_EOC_NUM_CYCLES_CU
            cu_start_indexes = eoc_df.index[(eoc_df[CSV_LABEL_EOC_CYC_CONDITION] == 2)
                                            & (eoc_df[CSV_LABEL_EOC_CYC_CHARGED] == 1)]
            for i_start in cu_start_indexes:
                cu_nr = eoc_df.at[i_start, CSV_LABEL_EOC_NUM_CYCLES_CU]
                timestamp_cu_max_start = eoc_df.at[i_start, CSV_LABEL_TIMESTAMP] + 1
                timestamp_cu_max_end = timestamp_cu_max_start + CU_MAX_LENGTH_S
                cu_update_indexes = eoc_df.index[(eoc_df[CSV_LABEL_TIMESTAMP] >= timestamp_cu_max_start)
                                                 & (eoc_df[CSV_LABEL_TIMESTAMP] <= timestamp_cu_max_end)
                                                 & (eoc_df[CSV_LABEL_EOC_NUM_CYCLES_CU] == (cu_nr + 1))
                                                 & ((eoc_df[CSV_LABEL_EOC_CYC_CONDITION] == 2)
                                                    | (eoc_df[CSV_LABEL_EOC_CYC_CONDITION] == 0))]
                if cu_update_indexes.shape[0] == 0:
                    num_warnings = num_warnings + 1
                else:
                    eoc_df.loc[cu_update_indexes, CSV_LABEL_EOC_NUM_CYCLES_CU] = \
                        eoc_df.loc[cu_update_indexes, CSV_LABEL_EOC_NUM_CYCLES_CU] - 1
                    num_converted = num_converted + 1
        else:
            logging.log.info("\n\n========== Skipping FIX_CHECKUP_NUM ========== \n")

        if FIX_DQ_DE:
            # II. EOC file:
            logging.log.debug("Thread %u S%02u:C%02u - extending EOC data" % (processor_number, slave_id, cell_id))

            # - delete total_dQ/dE_cond -> they are mostly wrong anyway
            if CSV_LABEL_TOTAL_Q_CONDITION in eoc_df:
                eoc_df.drop(CSV_LABEL_TOTAL_Q_CONDITION, axis=1, inplace=True)
            if CSV_LABEL_TOTAL_E_CONDITION in eoc_df:
                eoc_df.drop(CSV_LABEL_TOTAL_E_CONDITION, axis=1, inplace=True)

            # - add total_dQ/dE_(CU_RT/CYC_OT/others_RT/others_OT/sum)_chg/dischg with the ones from LOG*
            # * use SD_block_ID from EOC to look up row index of LOG, then go up until LOG message is found - probably
            #   go up a bot more and/or check if dQ is plausible/matches with old (and correct) dQ/dE from EOC
            # -> add empty new columns, fill later
            eoc_df[CSV_LABEL_TOTAL_Q_CHG_CU_RT] = 0
            eoc_df[CSV_LABEL_TOTAL_Q_DISCHG_CU_RT] = 0
            eoc_df[CSV_LABEL_TOTAL_Q_CHG_CYC_OT] = 0
            eoc_df[CSV_LABEL_TOTAL_Q_DISCHG_CYC_OT] = 0
            eoc_df[CSV_LABEL_TOTAL_Q_CHG_OTHER_RT] = 0
            eoc_df[CSV_LABEL_TOTAL_Q_DISCHG_OTHER_RT] = 0
            eoc_df[CSV_LABEL_TOTAL_Q_CHG_OTHER_OT] = 0
            eoc_df[CSV_LABEL_TOTAL_Q_DISCHG_OTHER_OT] = 0
            eoc_df[CSV_LABEL_TOTAL_Q_CHG_SUM] = 0
            eoc_df[CSV_LABEL_TOTAL_Q_DISCHG_SUM] = 0

            eoc_df[CSV_LABEL_TOTAL_E_CHG_CU_RT] = 0
            eoc_df[CSV_LABEL_TOTAL_E_DISCHG_CU_RT] = 0
            eoc_df[CSV_LABEL_TOTAL_E_CHG_CYC_OT] = 0
            eoc_df[CSV_LABEL_TOTAL_E_DISCHG_CYC_OT] = 0
            eoc_df[CSV_LABEL_TOTAL_E_CHG_OTHER_RT] = 0
            eoc_df[CSV_LABEL_TOTAL_E_DISCHG_OTHER_RT] = 0
            eoc_df[CSV_LABEL_TOTAL_E_CHG_OTHER_OT] = 0
            eoc_df[CSV_LABEL_TOTAL_E_DISCHG_OTHER_OT] = 0
            eoc_df[CSV_LABEL_TOTAL_E_CHG_SUM] = 0
            eoc_df[CSV_LABEL_TOTAL_E_DISCHG_SUM] = 0

            dTQC_CU_RT = 0
            dTQC_cyc_OT = 0
            dTQC_other_RT = 0
            dTQC_other_OT = 0
            dTQCS = 0

            dTQD_CU_RT = 0
            dTQD_cyc_OT = 0
            dTQD_other_RT = 0
            dTQD_other_OT = 0
            dTQDS = 0

            dTEC_CU_RT = 0
            dTEC_cyc_OT = 0
            dTEC_other_RT = 0
            dTEC_other_OT = 0
            dTECS = 0

            dTED_CU_RT = 0
            dTED_cyc_OT = 0
            dTED_other_RT = 0
            dTED_other_OT = 0
            dTEDS = 0

            # open LOG file
            log_df = pd.read_csv(cfg.CSV_RESULT_DIR + filename_log_csv, header=0, sep=cfg.CSV_SEP)

            loss_ratio = log_df[CSV_LABEL_TOTAL_E_DISCHG_SUM].iloc[-1] / log_df[CSV_LABEL_TOTAL_E_CHG_SUM].iloc[-1]
            if pd.isna(loss_ratio) or (loss_ratio < LOSS_RATIO_MIN_EXPECT) or (loss_ratio > LOSS_RATIO_MAX_EXPECT):
                # this is typical for cells that were overcharged before their end of life
                # --> try to use loss ratio at last check up?
                found_valid_loss_ratio = False
                eoc_cu_df = eoc_df[(eoc_df[CSV_LABEL_EOC_CYC_CONDITION] == 2)
                                   & (eoc_df[CSV_LABEL_EOC_CYC_CHARGED] == 1)]
                if eoc_cu_df.shape[0] > 0:
                    # if eoc_cu_df.shape[0] > 1:
                    #     sd_block_id = eoc_cu_df[CSV_LABEL_SD_BLOCK_ID].iloc[-2]  # sd_block_id of 2 CU dischg before
                    # else:
                    sd_block_id = eoc_cu_df[CSV_LABEL_SD_BLOCK_ID].iloc[-1]  # sd_block_id of last check-up dischg
                    log_index = log_df.index[
                        (log_df[CSV_LABEL_SD_BLOCK_ID] >= (sd_block_id - SD_DELTA_BLOCK_IDS_SEARCH))
                        & (log_df[CSV_LABEL_SD_BLOCK_ID] < sd_block_id)]
                    num_log_indexes = log_index.shape[0]
                    if num_log_indexes > 0:
                        log_index_use = log_index[-1]
                        e_dischg_sum = log_df.loc[log_index_use, CSV_LABEL_TOTAL_E_DISCHG_SUM]
                        e_chg_sum = log_df.loc[log_index_use, CSV_LABEL_TOTAL_E_CHG_SUM]
                        loss_ratio = e_dischg_sum / e_chg_sum
                        if not (pd.isna(loss_ratio) or (loss_ratio < LOSS_RATIO_MIN_EXPECT)
                                or (loss_ratio > LOSS_RATIO_MAX_EXPECT)):
                            found_valid_loss_ratio = True
                            logging.log.debug("Thread %u S%02u:C%02u - using loss ratio = total_E_dischg / total_E_chg "
                                                "= %.4f of last check-up discharge"
                                                % (processor_number, slave_id, cell_id, loss_ratio))
                if not found_valid_loss_ratio:
                    logging.log.warning("Thread %u S%02u:C%02u - Warning: unexpected loss ratio = total_E_dischg / "
                                        "total_E_chg = %.4f, using default = %.4f"
                                        % (processor_number, slave_id, cell_id, loss_ratio, LOSS_RATIO_DEFAULT))
                    loss_ratio = LOSS_RATIO_DEFAULT
                    num_warnings = num_warnings + 1

            # - compare old total_dQ/dE_chg/dischg with total_dQ/dE_sum_chg/dischg (warn if necessary)
            # for each row in EOC, find matching SD block ID in log (and go up a bit, see tablet notes)
            for eoc_index, eoc_item in eoc_df.iterrows():
                sd_block_id = eoc_item[CSV_LABEL_SD_BLOCK_ID]
                log_index = log_df.index[(log_df[CSV_LABEL_SD_BLOCK_ID] >= (sd_block_id - SD_DELTA_BLOCK_IDS_SEARCH))
                                         & (log_df[CSV_LABEL_SD_BLOCK_ID] < sd_block_id)]
                num_log_indexes = log_index.shape[0]
                if num_log_indexes <= 0:
                    logging.log.error("Thread %u S%02u:C%02u - no matching LOG for EOC row index %u found "
                                      "-> skipping row" % (processor_number, slave_id, cell_id, eoc_index))
                    num_errors = num_errors + 1
                else:
                    log_index_use = log_index[-1]

                    # check if result is similar
                    delta_q_eoc = eoc_df.loc[eoc_index, CSV_LABEL_DELTA_Q]
                    if delta_q_eoc > 0:  # 0 <= min_dQ <= max_dQ
                        min_dQ = ((1 - DQ_CMP_REL_FINE) * delta_q_eoc) - DQ_CMP_ABS_FINE
                        max_dQ = ((1 + DQ_CMP_REL_FINE) * delta_q_eoc) + DQ_CMP_ABS_FINE
                    else:  # min_dQ <= max_dQ <= 0
                        min_dQ = ((1 + DQ_CMP_REL_FINE) * delta_q_eoc) - DQ_CMP_ABS_FINE
                        max_dQ = ((1 - DQ_CMP_REL_FINE) * delta_q_eoc) + DQ_CMP_ABS_FINE
                    num_retry = MAX_RETRY_LOG_MATCH
                    delta_q_log = log_df.loc[log_index_use, CSV_LABEL_DELTA_Q]
                    dq_initial = delta_q_eoc - delta_q_log
                    match = False
                    while (log_index_use > 0) and (num_retry > 0):
                        if (delta_q_log < min_dQ) or (delta_q_log > max_dQ):
                            # no match (yet) -> continue
                            num_retry = num_retry - 1
                            log_index_use = log_index_use - 1
                            delta_q_log = log_df.loc[log_index_use, CSV_LABEL_DELTA_Q]
                            continue
                        # match
                        match = True
                        break
                    if not match:
                        if num_retry == MAX_RETRY_LOG_MATCH:  # special case: log_index_use was 0 before entering loop
                            if (delta_q_log >= min_dQ) or (delta_q_log <= max_dQ):
                                match = True
                    if not match:
                        # warn -> but copy anyway - what can we do?
                        timestamp_log_int = log_df[CSV_LABEL_TIMESTAMP][log_index_use]
                        timestamp_log_string = pd.to_datetime(timestamp_log_int, unit="s")
                        timestamp_eoc_int = eoc_item[CSV_LABEL_TIMESTAMP]
                        timestamp_eoc_string = pd.to_datetime(timestamp_eoc_int, unit="s")
                        logging.log.warning("Thread %u S%02u:C%02u - Warning: delta_q of supposedly matching LOG index "
                                            "%u (%s) and EOC row index %u (%s) vary unusually much (%.6f) -> use anyway"
                                            % (processor_number, slave_id, cell_id, log_index_use, timestamp_log_string,
                                               eoc_index, timestamp_eoc_string, dq_initial))
                        num_warnings = num_warnings + 1
                    else:
                        # check other stats
                        total_dQ_chg_eoc = eoc_df.loc[eoc_index, CSV_LABEL_TOTAL_Q_CHG]
                        total_dQ_chg_log = log_df.loc[log_index_use, CSV_LABEL_TOTAL_Q_CHG_SUM] + dTQCS
                        if abs(total_dQ_chg_eoc - total_dQ_chg_log) <= (DQ_CMP_REL_FINE * total_dQ_chg_log):
                            # only matches if total_dQ_chg_log is >= 0 & relative deviation smaller than DQ_CMP_REL_FINE
                            total_dQ_chg_check = True
                        elif abs(total_dQ_chg_eoc - total_dQ_chg_log) <= TOTAL_DQ_CMP_ABS_FINE:
                            # matches if the absolute deviation is small (useful if the values are both very small)
                            total_dQ_chg_check = True
                        else:
                            total_dQ_chg_check = False  # check failed -> deviation unusually high

                        total_dQ_dischg_eoc = eoc_df.loc[eoc_index, CSV_LABEL_TOTAL_Q_DISCHG]
                        total_dQ_dischg_log = log_df.loc[log_index_use, CSV_LABEL_TOTAL_Q_DISCHG_SUM] + dTQDS
                        if abs(total_dQ_dischg_eoc - total_dQ_dischg_log) <= (DQ_CMP_REL_FINE * total_dQ_dischg_log):
                            total_dQ_dischg_check = True  # see comment above
                        elif abs(total_dQ_dischg_eoc - total_dQ_dischg_log) <= TOTAL_DQ_CMP_ABS_FINE:
                            total_dQ_dischg_check = True  # see comment above
                        else:
                            total_dQ_dischg_check = False  # see comment above

                        total_dE_chg_eoc = eoc_df.loc[eoc_index, CSV_LABEL_TOTAL_E_CHG]
                        total_dE_chg_log = log_df.loc[log_index_use, CSV_LABEL_TOTAL_E_CHG_SUM] + dTECS
                        if abs(total_dE_chg_eoc - total_dE_chg_log) <= (DE_CMP_REL_FINE * total_dE_chg_log):
                            total_dE_chg_check = True  # see comment above
                        elif abs(total_dE_chg_eoc - total_dE_chg_log) <= TOTAL_DE_CMP_ABS_FINE:
                            total_dE_chg_check = True  # see comment above
                        else:
                            total_dE_chg_check = False  # see comment above

                        # don't check CSV_LABEL_TOTAL_E_DISCHG, it is almost always wrong
                        # -> partly the reason why we're doing this

                        if total_dQ_chg_check and total_dQ_dischg_check:
                            if not total_dE_chg_check:
                                # total_dE_chg_check not passed -> warning (we don't trust it a lot)
                                timestamp_log_int = log_df[CSV_LABEL_TIMESTAMP][log_index_use]
                                timestamp_log_string = pd.to_datetime(timestamp_log_int, unit="s")
                                timestamp_eoc_int = eoc_df[CSV_LABEL_TIMESTAMP][eoc_index]
                                timestamp_eoc_string = pd.to_datetime(timestamp_eoc_int, unit="s")
                                logging.log.warning("Thread %u S%02u:C%02u - Warning: total_dE_chg of supposedly "
                                                    "matching LOG index %u (%s) and EOC row index %u (%s) vary "
                                                    "unusually much -> use anyway"
                                                    % (processor_number, slave_id, cell_id, log_index_use,
                                                       timestamp_log_string, eoc_index, timestamp_eoc_string))
                                num_warnings = num_warnings + 1
                        else:
                            # total_dQ_chg_check or total_dQ_dischg_check not passed -> error
                            timestamp_log_int = log_df[CSV_LABEL_TIMESTAMP][log_index_use]
                            timestamp_log_string = pd.to_datetime(timestamp_log_int, unit="s")
                            timestamp_eoc_int = eoc_df[CSV_LABEL_TIMESTAMP][eoc_index]
                            timestamp_eoc_string = pd.to_datetime(timestamp_eoc_int, unit="s")
                            logging.log.warning("Thread %u S%02u:C%02u - Warning: total_dQ_chg/dischg of supposedly "
                                                "matching LOG index %u (%s) and EOC row index %u (%s) vary unusually "
                                                "much -> use anyway"
                                                % (processor_number, slave_id, cell_id, log_index_use,
                                                   timestamp_log_string, eoc_index, timestamp_eoc_string))
                            num_warnings = num_warnings + 1
                    # There is a difference between the "manually cumsum'd" total_q/e and the ones from the EOC. This
                    # happens mainly when the current is not constant/slowly changing, e.g. during EIS or WLTP cycles.
                    # Since we trust the total_q coming from the original EOC files, adjust our manually collected
                    # total_q_sum from the extended LOG.
                    # If there is a new deviation, and this EOC is for...
                    #    ... cyc_cond = 0, cyc_charged = 0 (OT EIS/pulses)     -> add to others_OT_chg/dischg...
                    #    ... cyc_cond = 0, cyc_charged = 1 (RT EIS/pulses)     -> add to others_RT_chg/dischg...
                    #    ... cyc_cond = 1, cyc_charged = 0 (WLTP/profile)      -> add to cyc_OT_chg/dischg...
                    #    ... cyc_cond = 1, cyc_charged = 1 (WLTP/profile)      -> add to cyc_OT_chg/dischg... [warn*]
                    #    ... others                                            -> add to CU_RT_chg/dischg...  [warn*]
                    #                                                             ... to match _sum
                    #    If dTQC/dTQD/dTEC/dTED is > threshold, warn. For [warn*], the threshold is very small.
                    # Also correct total_e_chg_sum (here TECS) as above. Since total_e_dischg wasn't properly calculated
                    # in the cycler (bug), we don't trust the one from the EOC. Adjust total_e_dischg_sum (TEDS) using:
                    # dTEDS = dTECS * loss_ratio      , where
                    # loss_ratio = log[TEDS].iloc[-1] / log[TECS].iloc[-1]      (last entry from the *LOG* file)
                    #     i.e. try not to change the power loss ratio from the LOG
                    #
                    # Note:
                    # All these errors typically are < 2.5 % even for calendar aging cells (EIS has a relatively large
                    # contribution to overall charge/discharge), but we still try to eliminate them as good as possible.
                    cyc_cond = eoc_item[CSV_LABEL_EOC_CYC_CONDITION]
                    cyc_charged = eoc_item[CSV_LABEL_EOC_CYC_CHARGED]
                    dTQC_new = eoc_df.loc[eoc_index, CSV_LABEL_TOTAL_Q_CHG]\
                            - log_df.loc[log_index_use, CSV_LABEL_TOTAL_Q_CHG_SUM] - dTQCS
                    dTQD_new = eoc_df.loc[eoc_index, CSV_LABEL_TOTAL_Q_DISCHG]\
                            - log_df.loc[log_index_use, CSV_LABEL_TOTAL_Q_DISCHG_SUM] - dTQDS
                    dTEC_new = eoc_df.loc[eoc_index, CSV_LABEL_TOTAL_E_CHG]\
                            - log_df.loc[log_index_use, CSV_LABEL_TOTAL_E_CHG_SUM] - dTECS
                    dTED_new = loss_ratio * dTEC_new
                    # dTEDS = eoc_df.loc[eoc_index, CSV_LABEL_TOTAL_E_DISCHG]\
                    #         - log_df.loc[log_index_use, CSV_LABEL_TOTAL_E_DISCHG_SUM]  -> can't trust this
                    if cyc_cond == 0:
                        if ((abs(dTQC_new) > DQ_EIS_ABS_FINE) or (abs(dTQD_new) > DQ_EIS_ABS_FINE)
                                or (abs(dTEC_new) > DE_EIS_ABS_FINE) or (abs(dTED_new) > DE_EIS_ABS_FINE)):
                            timestamp_eoc_int = eoc_df[CSV_LABEL_TIMESTAMP][eoc_index]
                            timestamp_eoc_string = pd.to_datetime(timestamp_eoc_int, unit="s")
                            timestamp_log_int = log_df[CSV_LABEL_TIMESTAMP][log_index_use]
                            timestamp_log_string = pd.to_datetime(timestamp_log_int, unit="s")
                            logging.log.error("Thread %u S%02u:C%02u - Error: at EOC row index %u (%s) and matching "
                                              "LOG index %u (%s), unusually high delta of total_D/E/chg/dischg (EIS):\n"
                                              "   dTQC_new:   %.6f\n   dTQD_new:   %.6f\n   dTEC_new:   %.6f\n"
                                              "   dTED_new:   %.6f"
                                              % (processor_number, slave_id, cell_id, eoc_index, timestamp_eoc_string,
                                                 log_index_use, timestamp_log_string,
                                                 dTQC_new, dTQD_new, dTEC_new, dTED_new))
                            num_errors = num_errors + 1
                            # SLave 12, Cell 0: 2022-10-13, 10:15...10:17 --> cycler bug on re-entry in CU_RT_EIS_PULSE
                            # state after reboot
                        if cyc_charged == 0:
                            dTQC_other_OT = dTQC_other_OT + dTQC_new
                            dTQD_other_OT = dTQD_other_OT + dTQD_new
                            dTEC_other_OT = dTEC_other_OT + dTEC_new
                            dTED_other_OT = dTED_other_OT + dTED_new
                        elif cyc_charged == 1:
                            dTQC_other_RT = dTQC_other_RT + dTQC_new
                            dTQD_other_RT = dTQD_other_RT + dTQD_new
                            dTEC_other_RT = dTEC_other_RT + dTEC_new
                            dTED_other_RT = dTED_other_RT + dTED_new
                        else:
                            timestamp_eoc_int = eoc_df[CSV_LABEL_TIMESTAMP][eoc_index]
                            timestamp_eoc_string = pd.to_datetime(timestamp_eoc_int, unit="s")
                            logging.log.error("Thread %u S%02u:C%02u - Warning: invalid cyc_charged %.0f at EOC row "
                                              "index %u (%s)" % (processor_number, slave_id, cell_id, cyc_charged,
                                                                 eoc_index, timestamp_eoc_string))
                            num_warnings = num_warnings + 1
                    elif cyc_cond == 1:
                        if cyc_charged == 0:
                            if ((abs(dTQC_new) > DQ_CYC_ABS_FINE) or (abs(dTQD_new) > DQ_CYC_ABS_FINE)
                                    or (abs(dTEC_new) > DE_CYC_ABS_FINE) or (abs(dTED_new) > DE_CYC_ABS_FINE)):
                                timestamp_eoc_int = eoc_df[CSV_LABEL_TIMESTAMP][eoc_index]
                                timestamp_eoc_string = pd.to_datetime(timestamp_eoc_int, unit="s")
                                logging.log.error("Thread %u S%02u:C%02u - Error: at EOC row index %u (%s), unusually "
                                                  "high delta of total_D/E/chg/dischg (CYC dischg):\n   dTQC_new:   "
                                                  "%.6f\n   dTQD_new:   %.6f\n   dTEC_new:   %.6f\n   dTED_new:   %.6f"
                                                  % (processor_number, slave_id, cell_id, eoc_index,
                                                     timestamp_eoc_string, dTQC_new, dTQD_new, dTEC_new, dTED_new))
                                num_errors = num_errors + 1
                            dTQC_cyc_OT = dTQC_cyc_OT + dTQC_new
                            dTQD_cyc_OT = dTQD_cyc_OT + dTQD_new
                            dTEC_cyc_OT = dTEC_cyc_OT + dTEC_new
                            dTED_cyc_OT = dTED_cyc_OT + dTED_new
                        elif cyc_charged == 1:
                            if ((abs(dTQC_new) > DQ_CMP_ABS_FINE) or (abs(dTQD_new) > DQ_CMP_ABS_FINE)
                                    or (abs(dTEC_new) > DE_CMP_ABS_FINE) or (abs(dTED_new) > DE_CMP_ABS_FINE)):
                                timestamp_eoc_int = eoc_df[CSV_LABEL_TIMESTAMP][eoc_index]
                                timestamp_eoc_string = pd.to_datetime(timestamp_eoc_int, unit="s")
                                logging.log.error("Thread %u S%02u:C%02u - Error: at EOC row index %u (%s), unusually "
                                                  "high delta of total_D/E/chg/dischg (CYC chg):\n   dTQC_new:   %.6f\n"
                                                  "   dTQD_new:   %.6f\n   dTEC_new:   %.6f\n   dTED_new:   %.6f"
                                                  % (processor_number, slave_id, cell_id, eoc_index,
                                                     timestamp_eoc_string, dTQC_new, dTQD_new, dTEC_new, dTED_new))
                                num_errors = num_errors + 1
                            dTQC_cyc_OT = dTQC_cyc_OT + dTQC_new
                            dTQD_cyc_OT = dTQD_cyc_OT + dTQD_new
                            dTEC_cyc_OT = dTEC_cyc_OT + dTEC_new
                            dTED_cyc_OT = dTED_cyc_OT + dTED_new
                        else:
                            timestamp_eoc_int = eoc_df[CSV_LABEL_TIMESTAMP][eoc_index]
                            timestamp_eoc_string = pd.to_datetime(timestamp_eoc_int, unit="s")
                            logging.log.error("Thread %u S%02u:C%02u - Error: invalid cyc_charged %.0f at EOC row index"
                                              " %u (%s)" % (processor_number, slave_id, cell_id, cyc_charged, eoc_index,
                                                            timestamp_eoc_string))
                            num_errors = num_errors + 1
                    elif cyc_cond == 2:
                        if ((abs(dTQC_new) > DQ_CMP_ABS_FINE) or (abs(dTQD_new) > DQ_CMP_ABS_FINE)
                                or (abs(dTEC_new) > DE_CMP_ABS_FINE) or (abs(dTED_new) > DE_CMP_ABS_FINE)):
                            timestamp_eoc_int = eoc_df[CSV_LABEL_TIMESTAMP][eoc_index]
                            timestamp_eoc_string = pd.to_datetime(timestamp_eoc_int, unit="s")
                            logging.log.error("Thread %u S%02u:C%02u - Error: at EOC row index %u (%s), unusually high "
                                              "delta of total_D/E/chg/dischg (CU):\n   dTQC_new:   %.6f\n"
                                              "   dTQD_new:   %.6f\n   dTEC_new:   %.6f\n   dTED_new:   %.6f"
                                              % (processor_number, slave_id, cell_id, eoc_index, timestamp_eoc_string,
                                                 dTQC_new, dTQD_new, dTEC_new, dTED_new))
                            num_errors = num_errors + 1
                        if (cyc_charged == 0) or (cyc_charged == 1):
                            dTQC_CU_RT = dTQC_CU_RT + dTQC_new
                            dTQD_CU_RT = dTQD_CU_RT + dTQD_new
                            dTEC_CU_RT = dTEC_CU_RT + dTEC_new
                            dTED_CU_RT = dTED_CU_RT + dTED_new
                        else:
                            timestamp_eoc_int = eoc_df[CSV_LABEL_TIMESTAMP][eoc_index]
                            timestamp_eoc_string = pd.to_datetime(timestamp_eoc_int, unit="s")
                            logging.log.error("Thread %u S%02u:C%02u - Error: invalid cyc_charged %.0f at EOC row index"
                                              " %u (%s)" % (processor_number, slave_id, cell_id, cyc_charged, eoc_index,
                                                            timestamp_eoc_string))
                            num_errors = num_errors + 1
                    else:
                        timestamp_eoc_int = eoc_df[CSV_LABEL_TIMESTAMP][eoc_index]
                        timestamp_eoc_string = pd.to_datetime(timestamp_eoc_int, unit="s")
                        logging.log.error("Thread %u S%02u:C%02u - Error: invalid cyc_cond %.0f at EOC row index %u "
                                          "(%s)" % (processor_number, slave_id, cell_id, cyc_cond, eoc_index,
                                                    timestamp_eoc_string))
                        num_errors = num_errors + 1

                    dTQCS = dTQCS + dTQC_new
                    dTQDS = dTQDS + dTQD_new
                    dTECS = dTECS + dTEC_new
                    dTEDS = dTEDS + dTED_new

                    eoc_df.loc[eoc_index, CSV_LABEL_TOTAL_Q_CHG_CU_RT] = \
                        log_df.loc[log_index_use, CSV_LABEL_TOTAL_Q_CHG_CU_RT] + dTQC_CU_RT
                    eoc_df.loc[eoc_index, CSV_LABEL_TOTAL_Q_DISCHG_CU_RT] = \
                        log_df.loc[log_index_use, CSV_LABEL_TOTAL_Q_DISCHG_CU_RT] + dTQD_CU_RT
                    eoc_df.loc[eoc_index, CSV_LABEL_TOTAL_Q_CHG_CYC_OT] = \
                        log_df.loc[log_index_use, CSV_LABEL_TOTAL_Q_CHG_CYC_OT] + dTQC_cyc_OT
                    eoc_df.loc[eoc_index, CSV_LABEL_TOTAL_Q_DISCHG_CYC_OT] = \
                        log_df.loc[log_index_use, CSV_LABEL_TOTAL_Q_DISCHG_CYC_OT] + dTQD_cyc_OT
                    eoc_df.loc[eoc_index, CSV_LABEL_TOTAL_Q_CHG_OTHER_RT] = \
                        log_df.loc[log_index_use, CSV_LABEL_TOTAL_Q_CHG_OTHER_RT] + dTQC_other_RT
                    eoc_df.loc[eoc_index, CSV_LABEL_TOTAL_Q_DISCHG_OTHER_RT] = \
                        log_df.loc[log_index_use, CSV_LABEL_TOTAL_Q_DISCHG_OTHER_RT] + dTQD_other_RT
                    eoc_df.loc[eoc_index, CSV_LABEL_TOTAL_Q_CHG_OTHER_OT] = \
                        log_df.loc[log_index_use, CSV_LABEL_TOTAL_Q_CHG_OTHER_OT] + dTQC_other_OT
                    eoc_df.loc[eoc_index, CSV_LABEL_TOTAL_Q_DISCHG_OTHER_OT] = \
                        log_df.loc[log_index_use, CSV_LABEL_TOTAL_Q_DISCHG_OTHER_OT] + dTQD_other_OT
                    eoc_df.loc[eoc_index, CSV_LABEL_TOTAL_Q_CHG_SUM] = \
                        log_df.loc[log_index_use, CSV_LABEL_TOTAL_Q_CHG_SUM] + dTQCS
                    eoc_df.loc[eoc_index, CSV_LABEL_TOTAL_Q_DISCHG_SUM] = \
                        log_df.loc[log_index_use, CSV_LABEL_TOTAL_Q_DISCHG_SUM] + dTQDS

                    eoc_df.loc[eoc_index, CSV_LABEL_TOTAL_E_CHG_CU_RT] = \
                        log_df.loc[log_index_use, CSV_LABEL_TOTAL_E_CHG_CU_RT] + dTEC_CU_RT
                    eoc_df.loc[eoc_index, CSV_LABEL_TOTAL_E_DISCHG_CU_RT] = \
                        log_df.loc[log_index_use, CSV_LABEL_TOTAL_E_DISCHG_CU_RT] + dTED_CU_RT
                    eoc_df.loc[eoc_index, CSV_LABEL_TOTAL_E_CHG_CYC_OT] = \
                        log_df.loc[log_index_use, CSV_LABEL_TOTAL_E_CHG_CYC_OT] + dTEC_cyc_OT
                    eoc_df.loc[eoc_index, CSV_LABEL_TOTAL_E_DISCHG_CYC_OT] = \
                        log_df.loc[log_index_use, CSV_LABEL_TOTAL_E_DISCHG_CYC_OT] + dTED_cyc_OT
                    eoc_df.loc[eoc_index, CSV_LABEL_TOTAL_E_CHG_OTHER_RT] = \
                        log_df.loc[log_index_use, CSV_LABEL_TOTAL_E_CHG_OTHER_RT] + dTEC_other_RT
                    eoc_df.loc[eoc_index, CSV_LABEL_TOTAL_E_DISCHG_OTHER_RT] = \
                        log_df.loc[log_index_use, CSV_LABEL_TOTAL_E_DISCHG_OTHER_RT] + dTED_other_RT
                    eoc_df.loc[eoc_index, CSV_LABEL_TOTAL_E_CHG_OTHER_OT] = \
                        log_df.loc[log_index_use, CSV_LABEL_TOTAL_E_CHG_OTHER_OT] + dTEC_other_OT
                    eoc_df.loc[eoc_index, CSV_LABEL_TOTAL_E_DISCHG_OTHER_OT] = \
                        log_df.loc[log_index_use, CSV_LABEL_TOTAL_E_DISCHG_OTHER_OT] + dTED_other_OT
                    eoc_df.loc[eoc_index, CSV_LABEL_TOTAL_E_CHG_SUM] = \
                        log_df.loc[log_index_use, CSV_LABEL_TOTAL_E_CHG_SUM] + dTECS
                    eoc_df.loc[eoc_index, CSV_LABEL_TOTAL_E_DISCHG_SUM] = \
                        log_df.loc[log_index_use, CSV_LABEL_TOTAL_E_DISCHG_SUM] + dTEDS

                    num_dQdE_total_updates = num_dQdE_total_updates + 1

            logging.log.debug("Thread %u S%02u:C%02u - preparing to write EOC data"
                              % (processor_number, slave_id, cell_id))

            # - delete old total_dQ/dE_chg/dischg columns -> they were replaced by total_dQ/dE_chg/dischg_sum
            if CSV_LABEL_TOTAL_Q_CHG in eoc_df:
                eoc_df.drop(CSV_LABEL_TOTAL_Q_CHG, axis=1, inplace=True)
            if CSV_LABEL_TOTAL_Q_DISCHG in eoc_df:
                eoc_df.drop(CSV_LABEL_TOTAL_Q_DISCHG, axis=1, inplace=True)
            if CSV_LABEL_TOTAL_E_CHG in eoc_df:
                eoc_df.drop(CSV_LABEL_TOTAL_E_CHG, axis=1, inplace=True)
            if CSV_LABEL_TOTAL_E_DISCHG in eoc_df:
                eoc_df.drop(CSV_LABEL_TOTAL_E_DISCHG, axis=1, inplace=True)
        else:
            logging.log.info("\n\n========== Skipping FIX_DQ_DE ========== \n")

        # custom float formats by string conversion:
        eoc_df[CSV_LABEL_AGE_CHG_RATE] = eoc_df[CSV_LABEL_AGE_CHG_RATE].map("{:.2f}".format)
        eoc_df[CSV_LABEL_AGE_DISCHG_RATE] = eoc_df[CSV_LABEL_AGE_DISCHG_RATE].map("{:.2f}".format)
        eoc_df[CSV_LABEL_T_START] = eoc_df[CSV_LABEL_T_START].map("{:.2f}".format)
        eoc_df[CSV_LABEL_T_END] = eoc_df[CSV_LABEL_T_END].map("{:.2f}".format)
        eoc_df[CSV_LABEL_CYC_DURATION] = eoc_df[CSV_LABEL_CYC_DURATION].map("{:.2f}".format)

        eoc_df[CSV_LABEL_TIMESTAMP] = eoc_df[CSV_LABEL_TIMESTAMP].map("{:.3f}".format)

        eoc_df[CSV_LABEL_CAP_CHARGED_EST] = eoc_df[CSV_LABEL_CAP_CHARGED_EST].map("{:.6f}".format)
        eoc_df[CSV_LABEL_SOH_CAP] = eoc_df[CSV_LABEL_SOH_CAP].map("{:.6f}".format)
        eoc_df[CSV_LABEL_DELTA_Q] = eoc_df[CSV_LABEL_DELTA_Q].map("{:.6f}".format)
        eoc_df[CSV_LABEL_DELTA_Q_CHG] = eoc_df[CSV_LABEL_DELTA_Q_CHG].map("{:.6f}".format)
        eoc_df[CSV_LABEL_DELTA_Q_DISCHG] = eoc_df[CSV_LABEL_DELTA_Q_DISCHG].map("{:.6f}".format)
        eoc_df[CSV_LABEL_DELTA_E] = eoc_df[CSV_LABEL_DELTA_E].map("{:.6f}".format)
        eoc_df[CSV_LABEL_DELTA_E_CHG] = eoc_df[CSV_LABEL_DELTA_E_CHG].map("{:.6f}".format)
        eoc_df[CSV_LABEL_DELTA_E_DISCHG] = eoc_df[CSV_LABEL_DELTA_E_DISCHG].map("{:.6f}".format)
        eoc_df[CSV_LABEL_COULOMB_EFFICIENCY] = eoc_df[CSV_LABEL_COULOMB_EFFICIENCY].map("{:.6f}".format)
        eoc_df[CSV_LABEL_ENERGY_EFFICIENCY] = eoc_df[CSV_LABEL_ENERGY_EFFICIENCY].map("{:.6f}".format)

        if CSV_LABEL_TOTAL_Q_CHG_CU_RT in eoc_df:  # depends on FIX_DQ_DE and if file was edited before
            flt_format = "{:.2f}"  # for debugging, use "{:.4f}" (no accuracy loss through rounding)
            eoc_df[CSV_LABEL_TOTAL_Q_CHG_CU_RT] = eoc_df[CSV_LABEL_TOTAL_Q_CHG_CU_RT].map(flt_format.format)
            eoc_df[CSV_LABEL_TOTAL_Q_DISCHG_CU_RT] = eoc_df[CSV_LABEL_TOTAL_Q_DISCHG_CU_RT].map(flt_format.format)
            eoc_df[CSV_LABEL_TOTAL_Q_CHG_CYC_OT] = eoc_df[CSV_LABEL_TOTAL_Q_CHG_CYC_OT].map(flt_format.format)
            eoc_df[CSV_LABEL_TOTAL_Q_DISCHG_CYC_OT] = eoc_df[CSV_LABEL_TOTAL_Q_DISCHG_CYC_OT].map(flt_format.format)
            eoc_df[CSV_LABEL_TOTAL_Q_CHG_OTHER_RT] = eoc_df[CSV_LABEL_TOTAL_Q_CHG_OTHER_RT].map(flt_format.format)
            eoc_df[CSV_LABEL_TOTAL_Q_DISCHG_OTHER_RT] = eoc_df[CSV_LABEL_TOTAL_Q_DISCHG_OTHER_RT].map(flt_format.format)
            eoc_df[CSV_LABEL_TOTAL_Q_CHG_OTHER_OT] = eoc_df[CSV_LABEL_TOTAL_Q_CHG_OTHER_OT].map(flt_format.format)
            eoc_df[CSV_LABEL_TOTAL_Q_DISCHG_OTHER_OT] = eoc_df[CSV_LABEL_TOTAL_Q_DISCHG_OTHER_OT].map(flt_format.format)
            eoc_df[CSV_LABEL_TOTAL_Q_CHG_SUM] = eoc_df[CSV_LABEL_TOTAL_Q_CHG_SUM].map(flt_format.format)
            eoc_df[CSV_LABEL_TOTAL_Q_DISCHG_SUM] = eoc_df[CSV_LABEL_TOTAL_Q_DISCHG_SUM].map(flt_format.format)

            eoc_df[CSV_LABEL_TOTAL_E_CHG_CU_RT] = eoc_df[CSV_LABEL_TOTAL_E_CHG_CU_RT].map(flt_format.format)
            eoc_df[CSV_LABEL_TOTAL_E_DISCHG_CU_RT] = eoc_df[CSV_LABEL_TOTAL_E_DISCHG_CU_RT].map(flt_format.format)
            eoc_df[CSV_LABEL_TOTAL_E_CHG_CYC_OT] = eoc_df[CSV_LABEL_TOTAL_E_CHG_CYC_OT].map(flt_format.format)
            eoc_df[CSV_LABEL_TOTAL_E_DISCHG_CYC_OT] = eoc_df[CSV_LABEL_TOTAL_E_DISCHG_CYC_OT].map(flt_format.format)
            eoc_df[CSV_LABEL_TOTAL_E_CHG_OTHER_RT] = eoc_df[CSV_LABEL_TOTAL_E_CHG_OTHER_RT].map(flt_format.format)
            eoc_df[CSV_LABEL_TOTAL_E_DISCHG_OTHER_RT] = eoc_df[CSV_LABEL_TOTAL_E_DISCHG_OTHER_RT].map(flt_format.format)
            eoc_df[CSV_LABEL_TOTAL_E_CHG_OTHER_OT] = eoc_df[CSV_LABEL_TOTAL_E_CHG_OTHER_OT].map(flt_format.format)
            eoc_df[CSV_LABEL_TOTAL_E_DISCHG_OTHER_OT] = eoc_df[CSV_LABEL_TOTAL_E_DISCHG_OTHER_OT].map(flt_format.format)
            eoc_df[CSV_LABEL_TOTAL_E_CHG_SUM] = eoc_df[CSV_LABEL_TOTAL_E_CHG_SUM].map(flt_format.format)
            eoc_df[CSV_LABEL_TOTAL_E_DISCHG_SUM] = eoc_df[CSV_LABEL_TOTAL_E_DISCHG_SUM].map(flt_format.format)

        # put CSV_LABEL_UPTIME_TICKS to the end again
        eoc_df[CSV_LABEL_UPTIME_TICKS + "2"] = eoc_df[CSV_LABEL_UPTIME_TICKS]
        eoc_df.drop(CSV_LABEL_UPTIME_TICKS, axis=1, inplace=True)
        eoc_df.rename(columns={CSV_LABEL_UPTIME_TICKS + "2": CSV_LABEL_UPTIME_TICKS}, inplace=True)

        # write to csv, with remaining floats using %.4f float format
        logging.log.debug("Thread %u S%02u:C%02u - writing EOC data" % (processor_number, slave_id, cell_id))
        eoc_df.to_csv(cfg.CSV_RESULT_DIR + (cfg.CSV_FILENAME_05_RESULT_BASE_CELL
                                            % (cfg.CSV_FILENAME_05_TYPE_EOC_FIXED, queue_entry["param_id"],
                                               queue_entry["param_nr"], slave_id, cell_id)),
                      index=False, sep=cfg.CSV_SEP, float_format="%.4f")

        # reporting to main thread
        report_msg = f"%s - S%02u:C%02u - fixed EOC data: %u check-up numbers edited, %u total_d/e rows edited - " \
                     f"number of warnings: %u, number of errors: %u"\
                     % (filename_eoc_csv, slave_id, cell_id, num_converted, num_dQdE_total_updates,
                        num_warnings, num_errors)
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
