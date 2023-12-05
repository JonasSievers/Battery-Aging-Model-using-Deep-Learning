import gc
import struct
# import time
import pandas as pd
import config_preprocessing as cfg
import multiprocessing
from datetime import datetime
import os
import re
import config_logging as logging
import numpy as np
from cytoolz import interleave


# config
TRY_TO_USE_SPACER_DATA = True  # if True, try using data based on magic word although FS box was corrupted & recovered
TRY_TO_USE_CONFLICTING_DATA = False  # if True, try using data based on magic word although FS box differs
PROGRESS_NOTIFICATION_DELTA_T = (7 * 24 * 60 * 60)  # 7 weeks. Note that it is only checked in EIS message branch, so...
# it doesn't make sense to make this smaller than the smallest check-up interval

header_slave_log = "timestamp" + cfg.CSV_SEP + \
                   "timestamp_origin" + cfg.CSV_SEP + \
                   "sd_block_id" + cfg.CSV_SEP + \
                   "high_level_state" + cfg.CSV_SEP + \
                   "scheduler_condition" + cfg.CSV_SEP + \
                   "controller_condition" + cfg.CSV_SEP + \
                   "v_main_V" + cfg.CSV_SEP + \
                   "i_main_A" + cfg.CSV_SEP + \
                   "p_main_W" + cfg.CSV_SEP + \
                   "v_aux_V" + cfg.CSV_SEP + \
                   "v_main_state" + cfg.CSV_SEP + \
                   "i_main_state" + cfg.CSV_SEP + \
                   "v_aux_state" + cfg.CSV_SEP + \
                   "monitor_enable" + cfg.CSV_SEP + \
                   "dc_state" + cfg.CSV_SEP + \
                   "log_state" + cfg.CSV_SEP + \
                   "eth_status" + cfg.CSV_SEP + \
                   "eth_fault" + cfg.CSV_SEP + \
                   "sd_fault" + cfg.CSV_SEP + \
                   "sd_fill" + cfg.CSV_SEP + \
                   "uptime_ticks" + \
                   cfg.CSV_NEWLINE

header_cell_log = "timestamp_s" + cfg.CSV_SEP + \
                  "timestamp_origin" + cfg.CSV_SEP + \
                  "sd_block_id" + cfg.CSV_SEP + \
                  "v_raw_V" + cfg.CSV_SEP + \
                  "i_raw_A" + cfg.CSV_SEP + \
                  "p_raw_W" + cfg.CSV_SEP + \
                  "t_cell_degC" + cfg.CSV_SEP + \
                  "delta_q_Ah" + cfg.CSV_SEP + \
                  "delta_e_Wh" + cfg.CSV_SEP + \
                  "soc_est" + cfg.CSV_SEP + \
                  "ocv_est_V" + cfg.CSV_SEP + \
                  "meas_condition" + cfg.CSV_SEP + \
                  "bms_condition" + cfg.CSV_SEP + \
                  "v_state" + cfg.CSV_SEP + \
                  "i_state" + cfg.CSV_SEP + \
                  "t_state" + cfg.CSV_SEP + \
                  "over_q" + cfg.CSV_SEP + \
                  "under_q" + cfg.CSV_SEP + \
                  "over_soc" + cfg.CSV_SEP + \
                  "under_soc" + cfg.CSV_SEP + \
                  "eol_cap" + cfg.CSV_SEP + \
                  "eol_imp" + cfg.CSV_SEP + \
                  "scheduler_state_dec" + cfg.CSV_SEP + \
                  "scheduler_state_phase" + cfg.CSV_SEP + \
                  "scheduler_state_checkup" + cfg.CSV_SEP + \
                  "scheduler_state_charge" + cfg.CSV_SEP + \
                  "scheduler_timeout" + cfg.CSV_SEP + \
                  "scheduler_cu_pending" + cfg.CSV_SEP + \
                  "scheduler_pause_pending" + cfg.CSV_SEP + \
                  "scheduler_state_sub" + \
                  cfg.CSV_NEWLINE
# Note: There are 12x cell_log per slave_log. The cell_log doesn't contain data from the slave_log anymore. However, the
# sd_block_id in combination with the slave ID (in the filename) makes it possible to look up slave data from cell_log.
# Note: I left away '"scheduler_state_hex" + cfg.CSV_SEP + \' between scheduler_state_dec and scheduler_state_phase
# because it was very similar / redundant with scheduler_state_dec and took away quite some space.

header_cell_eoc = "timestamp_s" + cfg.CSV_SEP + \
                  "timestamp_origin" + cfg.CSV_SEP + \
                  "sd_block_id" + cfg.CSV_SEP + \
                  "cyc_condition" + cfg.CSV_SEP + \
                  "cyc_charged" + cfg.CSV_SEP + \
                  "age_type" + cfg.CSV_SEP + \
                  "age_temp" + cfg.CSV_SEP + \
                  "age_soc" + cfg.CSV_SEP + \
                  "age_chg_rate" + cfg.CSV_SEP + \
                  "age_dischg_rate" + cfg.CSV_SEP + \
                  "age_profile" + cfg.CSV_SEP + \
                  "v_max_target_V" + cfg.CSV_SEP + \
                  "v_min_target_V" + cfg.CSV_SEP + \
                  "i_chg_max_A" + cfg.CSV_SEP + \
                  "i_dischg_min_A" + cfg.CSV_SEP + \
                  "i_chg_cutoff_A" + cfg.CSV_SEP + \
                  "i_dischg_cutoff_A" + cfg.CSV_SEP + \
                  "cap_aged_est_Ah" + cfg.CSV_SEP + \
                  "soh_cap" + cfg.CSV_SEP + \
                  "delta_q_Ah" + cfg.CSV_SEP + \
                  "delta_q_chg_Ah" + cfg.CSV_SEP + \
                  "delta_q_dischg_Ah" + cfg.CSV_SEP + \
                  "delta_e_Wh" + cfg.CSV_SEP + \
                  "delta_e_chg_Wh" + cfg.CSV_SEP + \
                  "delta_e_dischg_Wh" + cfg.CSV_SEP + \
                  "coulomb_efficiency" + cfg.CSV_SEP + \
                  "energy_efficiency" + cfg.CSV_SEP + \
                  "ocv_est_start_V" + cfg.CSV_SEP + \
                  "ocv_est_end_V" + cfg.CSV_SEP + \
                  "soc_est_start" + cfg.CSV_SEP + \
                  "soc_est_end" + cfg.CSV_SEP + \
                  "t_start_degC" + cfg.CSV_SEP + \
                  "t_end_degC" + cfg.CSV_SEP + \
                  "cyc_duration_s" + cfg.CSV_SEP + \
                  "num_cycles_op" + cfg.CSV_SEP + \
                  "num_cycles_checkup" + cfg.CSV_SEP + \
                  "total_q_condition_Ah" + cfg.CSV_SEP + \
                  "total_q_chg_Ah" + cfg.CSV_SEP + \
                  "total_q_dischg_Ah" + cfg.CSV_SEP + \
                  "total_e_condition_Wh" + cfg.CSV_SEP + \
                  "total_e_chg_Wh" + cfg.CSV_SEP + \
                  "total_e_dischg_Wh" + cfg.CSV_SEP + \
                  "uptime_ticks" + \
                  cfg.CSV_NEWLINE

header_cell_eis = "timestamp_s" + cfg.CSV_SEP + \
                  "timestamp_origin" + cfg.CSV_SEP + \
                  "sd_block_id" + cfg.CSV_SEP + \
                  "cyc_charged" + cfg.CSV_SEP + \
                  "is_rt" + cfg.CSV_SEP + \
                  "soc_nom" + cfg.CSV_SEP + \
                  "valid" + cfg.CSV_SEP + \
                  "z_ref_init_mOhm" + cfg.CSV_SEP + \
                  "z_ref_now_mOhm" + cfg.CSV_SEP + \
                  "soh_imp" + cfg.CSV_SEP + \
                  "ocv_est_avg_V" + cfg.CSV_SEP + \
                  "t_avg_degC" + cfg.CSV_SEP + \
                  "eis_duration_s" + cfg.CSV_SEP + \
                  "uptime_ticks" + cfg.CSV_SEP + \
                  "seq_nr" + cfg.CSV_SEP + \
                  "freq_Hz" + cfg.CSV_SEP + \
                  "z_amp_mOhm" + cfg.CSV_SEP + \
                  "z_ph_deg" + \
                  cfg.CSV_NEWLINE
# Note: seq_nr is introduced to identify lines that belong to the same electrochemical impedance spectroscopy (EIS) run

header_cell_pulse = "timestamp_s" + cfg.CSV_SEP + \
                    "timestamp_origin" + cfg.CSV_SEP + \
                    "sd_block_id" + cfg.CSV_SEP + \
                    "cyc_charged" + cfg.CSV_SEP + \
                    "is_rt" + cfg.CSV_SEP + \
                    "soc_nom" + cfg.CSV_SEP + \
                    "age_type" + cfg.CSV_SEP + \
                    "age_temp" + cfg.CSV_SEP + \
                    "age_soc" + cfg.CSV_SEP + \
                    "age_chg_rate" + cfg.CSV_SEP + \
                    "age_dischg_rate" + cfg.CSV_SEP + \
                    "age_profile" + cfg.CSV_SEP + \
                    "t_avg_degC" + cfg.CSV_SEP + \
                    "r_ref_10ms_mOhm" + cfg.CSV_SEP + \
                    "r_ref_1s_mOhm" + cfg.CSV_SEP + \
                    "uptime_ticks" + cfg.CSV_SEP + \
                    "seq_nr" + cfg.CSV_SEP + \
                    "v_raw_V" + cfg.CSV_SEP + \
                    "i_raw_A" + \
                    cfg.CSV_NEWLINE
# Note: seq_nr is introduced to identify lines that belong to the same pulse pattern run

header_tmgmt_log = "timestamp" + cfg.CSV_SEP + \
                   "timestamp_origin" + cfg.CSV_SEP + \
                   "sd_block_id" + cfg.CSV_SEP + \
                   "high_level_state" + cfg.CSV_SEP + \
                   "scheduler_condition" + cfg.CSV_SEP + \
                   "controller_condition" + cfg.CSV_SEP + \
                   "v_main_V" + cfg.CSV_SEP + \
                   "i_main_A" + cfg.CSV_SEP + \
                   "p_main_W" + cfg.CSV_SEP + \
                   "v_aux_V" + cfg.CSV_SEP + \
                   "v_main_state" + cfg.CSV_SEP + \
                   "i_main_state" + cfg.CSV_SEP + \
                   "v_aux_state" + cfg.CSV_SEP + \
                   "monitor_enable" + cfg.CSV_SEP + \
                   "dc_state" + cfg.CSV_SEP + \
                   "log_state" + cfg.CSV_SEP + \
                   "sd_fill" + cfg.CSV_SEP + \
                   "valve_air_en" + cfg.CSV_SEP + \
                   "fan_pcb_en" + cfg.CSV_SEP + \
                   "fan_radiator_en" + cfg.CSV_SEP + \
                   "pump_hot_set" + cfg.CSV_SEP + \
                   "pump_hot_meas" + cfg.CSV_SEP + \
                   "chiller_t_set_degC" + cfg.CSV_SEP + \
                   "chiller_t_meas_degC" + cfg.CSV_SEP + \
                   "chiller_n_set_rpm" + cfg.CSV_SEP + \
                   "chiller_n_meas_rpm" + cfg.CSV_SEP + \
                   "chiller_state" + cfg.CSV_SEP + \
                   "r_iso_state" + cfg.CSV_SEP + \
                   "r_iso_Ohm" + cfg.CSV_SEP + \
                   "t_pipe_1_degC" + cfg.CSV_SEP + \
                   "t_pipe_2_degC" + cfg.CSV_SEP + \
                   "t_pipe_3_degC" + cfg.CSV_SEP + \
                   "t_pipe_1_state" + cfg.CSV_SEP + \
                   "t_pipe_2_state" + cfg.CSV_SEP + \
                   "t_pipe_3_state" + cfg.CSV_SEP + \
                   "t_controller_degC" + cfg.CSV_SEP + \
                   "t_box_cold_degC" + cfg.CSV_SEP + \
                   "hum_box_cold_percent" + cfg.CSV_SEP + \
                   "press_box_cold_hPa" + cfg.CSV_SEP + \
                   "dew_point_box_cold_degC" + cfg.CSV_SEP + \
                   "abs_hum_box_cold_g_m3" + cfg.CSV_SEP + \
                   "t_ambient_degC" + cfg.CSV_SEP + \
                   "hum_ambient_percent" + cfg.CSV_SEP + \
                   "press_ambient_hPa" + cfg.CSV_SEP + \
                   "dew_point_ambient_degC" + cfg.CSV_SEP + \
                   "abs_hum_ambient_g_m3" + cfg.CSV_SEP + \
                   "uptime_ticks" + \
                   cfg.CSV_NEWLINE

header_pool_log = "timestamp" + cfg.CSV_SEP + \
                  "timestamp_origin" + cfg.CSV_SEP + \
                  "sd_block_id" + cfg.CSV_SEP + \
                  "scheduler_state_dec" + cfg.CSV_SEP + \
                  "scheduler_state_phase" + cfg.CSV_SEP + \
                  "scheduler_state_checkup" + cfg.CSV_SEP + \
                  "scheduler_cu_pending" + cfg.CSV_SEP + \
                  "scheduler_pause_pending" + cfg.CSV_SEP + \
                  "scheduler_state_sub" + cfg.CSV_SEP + \
                  "t_pool_set_degC" + cfg.CSV_SEP + \
                  "t_pool_set_ramped_degC" + cfg.CSV_SEP + \
                  "t_pool_meas_degC" + cfg.CSV_SEP + \
                  "t_plate_degC" + cfg.CSV_SEP + \
                  "has_ot" + cfg.CSV_SEP + \
                  "is_stable" + cfg.CSV_SEP + \
                  "has_timeout" + cfg.CSV_SEP + \
                  "condition" + cfg.CSV_SEP + \
                  "t_pool_state" + cfg.CSV_SEP + \
                  "t_plate_state" + cfg.CSV_SEP + \
                  "peltier_i_set_sum_A" + cfg.CSV_SEP + \
                  "peltier_i_meas_sum_A" + cfg.CSV_SEP + \
                  "peltier_num_en" + cfg.CSV_SEP + \
                  "peltier_i_set_1_A" + cfg.CSV_SEP + \
                  "peltier_i_set_2_A" + cfg.CSV_SEP + \
                  "peltier_i_set_3_A" + cfg.CSV_SEP + \
                  "peltier_i_set_4_A" + cfg.CSV_SEP + \
                  "peltier_i_meas_1_A" + cfg.CSV_SEP + \
                  "peltier_i_meas_2_A" + cfg.CSV_SEP + \
                  "peltier_i_meas_3_A" + cfg.CSV_SEP + \
                  "peltier_i_meas_4_A" + cfg.CSV_SEP + \
                  "peltier_en_1" + cfg.CSV_SEP + \
                  "peltier_en_2" + cfg.CSV_SEP + \
                  "peltier_en_3" + cfg.CSV_SEP + \
                  "peltier_en_4" + cfg.CSV_SEP + \
                  "peltier_i_state_1" + cfg.CSV_SEP + \
                  "peltier_i_state_2" + cfg.CSV_SEP + \
                  "peltier_i_state_3" + cfg.CSV_SEP + \
                  "peltier_i_state_4" + cfg.CSV_SEP + \
                  "peltier_ctrl_state_1" + cfg.CSV_SEP + \
                  "peltier_ctrl_state_2" + cfg.CSV_SEP + \
                  "peltier_ctrl_state_3" + cfg.CSV_SEP + \
                  "peltier_ctrl_state_4" + cfg.CSV_SEP + \
                  "peltier_condition_1" + cfg.CSV_SEP + \
                  "peltier_condition_2" + cfg.CSV_SEP + \
                  "peltier_condition_3" + cfg.CSV_SEP + \
                  "peltier_condition_4" + \
                   cfg.CSV_NEWLINE
# Note: There are 4x pool_log per tmgmt_log. The pool_log doesn't contain data from the tmgmt_log anymore. However, the
# sd_block_id in combination with the tmgmt ID (in the filename) makes it possible to look up tmgmt data from pool_log.
# Note: I left away '"scheduler_state_hex" + cfg.CSV_SEP + \' between scheduler_state_dec and scheduler_state_phase
# because it was very similar / redundant with scheduler_state_dec and took away quite some space.

header_slave_config = "sd_block_id" + cfg.CSV_SEP + \
                      "slave_id" + cfg.CSV_SEP + \
                      "board_type" + cfg.CSV_SEP + \
                      "my_ip_adr" + cfg.CSV_SEP + \
                      "server_ip_adr" + cfg.CSV_SEP + \
                      "my_mac_adr" + cfg.CSV_SEP + \
                      "num_EIS" + cfg.CSV_SEP + \
                      "EIS_SoC_0" + cfg.CSV_SEP + \
                      "EIS_SoC_1" + cfg.CSV_SEP + \
                      "EIS_SoC_2" + cfg.CSV_SEP + \
                      "EIS_SoC_3" + cfg.CSV_SEP + \
                      "EIS_SoC_4" + cfg.CSV_SEP + \
                      "EIS_SoC_5" + cfg.CSV_SEP + \
                      "EIS_SoC_6" + cfg.CSV_SEP + \
                      "EIS_SoC_7" + cfg.CSV_SEP + \
                      "EIS_SoC_8" + cfg.CSV_SEP + \
                      "EIS_SoC_9" + \
                      cfg.CSV_NEWLINE

header_cell_config = "sd_block_id" + cfg.CSV_SEP + \
                     "slave_id" + cfg.CSV_SEP + \
                     "cell_id" + cfg.CSV_SEP + \
                     "parameter_id" + cfg.CSV_SEP + \
                     "parameter_nr" + cfg.CSV_SEP + \
                     "cell_used" + cfg.CSV_SEP + \
                     "cell_type" + cfg.CSV_SEP + \
                     "t_sns_type" + cfg.CSV_SEP + \
                     "age_type" + cfg.CSV_SEP + \
                     "age_temp" + cfg.CSV_SEP + \
                     "age_soc" + cfg.CSV_SEP + \
                     "age_chg_rate" + cfg.CSV_SEP + \
                     "age_dischg_rate" + cfg.CSV_SEP + \
                     "age_profile" + cfg.CSV_SEP + \
                     "V_max_cyc_V" + cfg.CSV_SEP + \
                     "V_min_cyc_V" + cfg.CSV_SEP + \
                     "V_max_cu_V" + cfg.CSV_SEP + \
                     "V_min_cu_V" + cfg.CSV_SEP + \
                     "I_chg_max_cyc_A" + cfg.CSV_SEP + \
                     "I_dischg_max_cyc_A" + cfg.CSV_SEP + \
                     "I_chg_max_cu_A" + cfg.CSV_SEP + \
                     "I_dischg_max_cu_A" + cfg.CSV_SEP + \
                     "I_chg_cutoff_cyc_A" + cfg.CSV_SEP + \
                     "I_dischg_cutoff_cyc_A" + cfg.CSV_SEP + \
                     "I_chg_cutoff_cu_A" + cfg.CSV_SEP + \
                     "I_dischg_cutoff_cu_A" + cfg.CSV_SEP + \
                     "I_chg_pulse_cu_A" + cfg.CSV_SEP + \
                     "I_dischg_pulse_cu_A" + \
                     cfg.CSV_NEWLINE

header_tmgmt_config = "sd_block_id" + cfg.CSV_SEP + \
                      "slave_id" + cfg.CSV_SEP + \
                      "board_type" + cfg.CSV_SEP + \
                      "my_ip_adr" + cfg.CSV_SEP + \
                      "server_ip_adr" + cfg.CSV_SEP + \
                      "my_mac_adr" + \
                      cfg.CSV_NEWLINE

header_pool_config = "sd_block_id" + cfg.CSV_SEP + \
                     "slave_id" + cfg.CSV_SEP + \
                     "pool_id" + cfg.CSV_SEP + \
                     "pool_used" + cfg.CSV_SEP + \
                     "is_in_cold_circuit" + cfg.CSV_SEP + \
                     "t_operation_degC" + cfg.CSV_SEP + \
                     "t_checkup_degC" + \
                     cfg.CSV_NEWLINE


# constants
NUMBER_OF_PROCESSORS_TO_USE = multiprocessing.cpu_count()
NUM_CELLS_PER_SLAVE = cfg.NUM_CELLS_PER_SLAVE
NUM_POOLS_PER_TMGMT = cfg.NUM_POOLS_PER_TMGMT
NUM_PELTIERS_PER_POOL = cfg.NUM_PELTIERS_PER_POOL
task_queue = multiprocessing.Queue()
report_queue = multiprocessing.Queue()


def run():
    start_timestamp = datetime.now()
    logging.log.info(os.path.basename(__file__))

    # find .csv files
    slaves_csv = []
    with os.scandir(cfg.CSV_WORKING_DIR) as iterator:
        re_str_csv = cfg.CSV_FILENAME_04_SD_BLOCK_ID_UPTIME_TIMESTAMP.replace("%s%02u", "([ST])(\d+)")
        re_pat_csv = re.compile(re_str_csv)
        for entry in iterator:
            re_match_csv = re_pat_csv.fullmatch(entry.name)
            if re_match_csv:
                slave_type = re_match_csv.group(1)
                slave_id = int(re_match_csv.group(2))
                slave_csv = {"id": slave_id, "type": slave_type}
                slaves_csv.append(slave_csv)

    # find .img files
    slaves_img = []
    with os.scandir(cfg.SD_IMAGE_PATH) as iterator:
        re_pat_img_slave = re.compile(cfg.SD_IMAGE_FILE_REGEX_SLAVE)
        re_pat_img_tmgmt = re.compile(cfg.SD_IMAGE_FILE_REGEX_TMGMT)
        for entry in iterator:
            re_match_img_slave = re_pat_img_slave.fullmatch(entry.name)
            if re_match_img_slave:
                slave_id = int(re_match_img_slave.group(2))
                img_size = os.path.getsize(cfg.SD_IMAGE_PATH + entry.name)
                if img_size == 0:
                    logging.log.error("Slave S%02u - Image file is empty -> ignore image" % slave_id)
                elif (img_size % cfg.SD_BLOCK_SIZE_BYTES) != 0:
                    logging.log.error("Slave S%02u - Image file size is not a multiple of %u bytes -> ignore image"
                                      % (slave_id, cfg.SD_BLOCK_SIZE_BYTES))
                else:
                    slave_img = {"filename": entry.name, "id": slave_id, "type": "S"}
                    slaves_img.append(slave_img)
            else:
                re_match_img_tmgmt = re_pat_img_tmgmt.fullmatch(entry.name)
                if re_match_img_tmgmt:
                    slave_id = int(re_match_img_tmgmt.group(2))
                    img_size = os.path.getsize(cfg.SD_IMAGE_PATH + entry.name)
                    if img_size == 0:
                        logging.log.error("Slave T%02u - Image file is empty -> ignore image" % slave_id)
                    elif (img_size % cfg.SD_BLOCK_SIZE_BYTES) != 0:
                        logging.log.error("Slave T%02u - Image file size is not a multiple of %u bytes -> ignore image"
                                          % (slave_id, cfg.SD_BLOCK_SIZE_BYTES))
                    else:
                        slave_img = {"filename": entry.name, "id": slave_id, "type": "T"}
                        slaves_img.append(slave_img)

    # only used slaves of which a .csv and .img exists
    slaves_string = []
    for slave_csv in slaves_csv:
        for slave_img in slaves_img:
            if (slave_csv["id"] == slave_img["id"]) and (slave_csv["type"] == slave_img["type"]):
                slaves_string.append("%s%02u" % (slave_csv["type"], slave_csv["id"]))
                task_queue.put(slave_img)

    logging.log.info("Found .csv/.img files for slaves: %s" % slaves_string)

    # Create processes
    processes = []
    logging.log.info("Starting processes")
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        logging.log.debug("  Starting process %u" % processorNumber)
        processes.append(multiprocessing.Process(target=run_thread, args=(processorNumber, task_queue, report_queue, )))
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].start()
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].join()
        logging.log.debug("Joined process %u" % processorNumber)

    logging.log.info("\n\n========== All processes ended - summary ========== \n")

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
        num_valid_data_blocks = slave_report["num_valid_data_blocks"]
        num_invalid_data_blocks = slave_report["num_invalid_data_blocks"]
        num_dubious_blocks = slave_report["num_dubious_blocks"]
        num_conflicting_blocks = slave_report["num_conflicting_blocks"]
        num_gaps = slave_report["num_gaps"]
        report_msg_ext = report_msg + "\n" + \
                         f"   %u valid blocks converted\n" \
                         f"   %u invalid blocks detected (but couldn't be converted)\n" \
                         f"   %u dubious blocks found (FS box = expected data type in info/backup sector differs)\n" \
                         f"   %u conflicting blocks found (FS box = expected data type and found data type differs)\n" \
                         f"   %u gaps in FS info/backup found (data loss or corrupted data likely!)\n" \
                         % (num_valid_data_blocks, num_invalid_data_blocks, num_dubious_blocks, num_conflicting_blocks,
                            num_gaps)

        if report_level == logging.ERROR:
            logging.log.error(report_msg_ext)
        elif report_level == logging.WARNING:
            logging.log.warning(report_msg_ext)
        elif report_level == logging.INFO:
            logging.log.info(report_msg_ext)
        elif report_level == logging.DEBUG:
            logging.log.debug(report_msg_ext)
        elif report_level == logging.CRITICAL:
            logging.log.critical(report_msg_ext)

    stop_timestamp = datetime.now()

    logging.log.info("\nScript runtime: %s h:mm:ss.ms" % str(stop_timestamp - start_timestamp))


def run_thread(processor_number, slave_queue, thread_report_queue):
    while True:
        if (slave_queue is None) or slave_queue.empty():
            break  # no more files

        try:
            queue_entry = slave_queue.get_nowait()
        except multiprocessing.queues.Empty:
            break  # no more files

        if queue_entry is None:
            break  # no more files

        slave_id = queue_entry["id"]
        # if slave_id != 0:
        #     continue  # for debugging individual slaves

        type_id = queue_entry["type"]
        filename_img = queue_entry["filename"]
        logging.log.info("Thread %u Slave %s%02u - start decoding SD data"
                         % (processor_number, type_id, slave_id))

        num_dubious_blocks = 0  # number of blocks where FS block matches Magic Word, but doesn't match backup block
        num_conflicting_blocks = 0  # number of blocks where FS block doesn't match Magic Word
        num_gaps = 0  # number of blocks that are unused/invalid and lie in between valid data
        num_valid_data_blocks = 0  # number of blocks that were converted
        num_invalid_data_blocks = 0  # number of blocks that were detected but couldn't be converted

        # create empty file lists for slave...
        file_slave_log = None
        file_cell_log = np.empty(NUM_CELLS_PER_SLAVE, dtype=object)
        file_cell_eoc = np.empty(NUM_CELLS_PER_SLAVE, dtype=object)
        file_cell_eis = np.empty(NUM_CELLS_PER_SLAVE, dtype=object)
        file_cell_pulse = np.empty(NUM_CELLS_PER_SLAVE, dtype=object)
        # ...and thermal management
        file_tmgmt_log = None
        file_pool_log = np.empty(NUM_POOLS_PER_TMGMT, dtype=object)

        if type_id == "S":

            # open/create files, write header
            tmp_filename = cfg.CSV_FILENAME_05_RESULT_BASE_SLAVE % (cfg.CSV_FILENAME_05_TYPE_LOG, type_id, slave_id)
            file_slave_log = open(cfg.CSV_RESULT_DIR + tmp_filename, "w")
            file_slave_log.write(header_slave_log)

            for i in range(0, NUM_CELLS_PER_SLAVE):
                # identifies the "parameter set", e.g. 17 = cyclic aging, 0°C, 0-100%, 1/3 C charging, 1/3 C discharging
                parameter_set_id = cfg.PARAMETER_SET_ID_FROM_SXX_CXX[slave_id][i]

                # identifies the index of the cell in the parameter set, e.g. S01 C08 has index 1, S05 C05 index 3
                # if three cells age with the same parameter set, the index goes from 1 to 3
                parameter_set_cell_nr = cfg.PARAMETER_SET_CELL_NR_FROM_SXX_CXX[slave_id][i]

                tmp_filename = cfg.CSV_FILENAME_05_RESULT_BASE_CELL % (cfg.CSV_FILENAME_05_TYPE_LOG, parameter_set_id,
                                                                       parameter_set_cell_nr, slave_id, i)
                file_cell_log[i] = open(cfg.CSV_RESULT_DIR + tmp_filename, "w")
                file_cell_log[i].write(header_cell_log)

                tmp_filename = cfg.CSV_FILENAME_05_RESULT_BASE_CELL % (cfg.CSV_FILENAME_05_TYPE_EIS, parameter_set_id,
                                                                       parameter_set_cell_nr, slave_id, i)
                file_cell_eis[i] = open(cfg.CSV_RESULT_DIR + tmp_filename, "w")
                file_cell_eis[i].write(header_cell_eis)

                tmp_filename = cfg.CSV_FILENAME_05_RESULT_BASE_CELL % (cfg.CSV_FILENAME_05_TYPE_EOC, parameter_set_id,
                                                                       parameter_set_cell_nr, slave_id, i)
                file_cell_eoc[i] = open(cfg.CSV_RESULT_DIR + tmp_filename, "w")
                file_cell_eoc[i].write(header_cell_eoc)

                tmp_filename = cfg.CSV_FILENAME_05_RESULT_BASE_CELL % (cfg.CSV_FILENAME_05_TYPE_PULSE, parameter_set_id,
                                                                       parameter_set_cell_nr, slave_id, i)
                file_cell_pulse[i] = open(cfg.CSV_RESULT_DIR + tmp_filename, "w")
                file_cell_pulse[i].write(header_cell_pulse)
        elif type_id == "T":

            # open/create files
            tmp_filename = cfg.CSV_FILENAME_05_RESULT_BASE_SLAVE % (cfg.CSV_FILENAME_05_TYPE_LOG, type_id, slave_id)
            file_tmgmt_log = open(cfg.CSV_RESULT_DIR + tmp_filename, "w")
            file_tmgmt_log.write(header_tmgmt_log)

            for i in range(0, NUM_POOLS_PER_TMGMT):
                tmp_filename = cfg.CSV_FILENAME_05_RESULT_BASE_POOL % (cfg.CSV_FILENAME_05_TYPE_LOG, slave_id, i)
                file_pool_log[i] = open(cfg.CSV_RESULT_DIR + tmp_filename, "w")
                file_pool_log[i].write(header_pool_log)
        else:
            # generate report, then go to next slave
            report_msg = f"Thread %u Slave %s%02u - unknown slave type -> skipping" \
                         % (processor_number, type_id, slave_id)
            report_level = logging.ERROR
            logging.log.error(report_msg)  # also warn, so that the user can fix the issue while the rest is running
            slave_report = {"msg": report_msg, "level": report_level, "num_dubious_blocks": num_dubious_blocks,
                            "num_conflicting_blocks": num_conflicting_blocks, "num_gaps": num_gaps,
                            "num_valid_data_blocks": num_valid_data_blocks,
                            "num_invalid_data_blocks": num_invalid_data_blocks}
            thread_report_queue.put(slave_report)
            continue

        # read timestamp file
        filename_04 = cfg.CSV_WORKING_DIR + (cfg.CSV_FILENAME_04_SD_BLOCK_ID_UPTIME_TIMESTAMP % (type_id, slave_id))
        sd_timestamp_df = pd.read_csv(filename_04, header=0, sep=cfg.CSV_SEP)
        sd_timestamp_df.drop("stm_tim_uptime", inplace=True, axis=1)

        # open image file
        file_img = open(cfg.SD_IMAGE_PATH + filename_img, "rb")

        # read FS info start block
        data_block = file_img.read(cfg.SD_BLOCK_SIZE_BYTES)
        re_pat = re.compile(cfg.BATCYC_FS_INFO_STR)
        re_match = re_pat.match(data_block.decode(cfg.IMAGE_ENCODING))
        if not re_match:
            # generate report, close opened files, then go to next slave
            report_msg = f"Thread %u Slave %s%02u - unknown/invalid FS info start block" \
                         % (processor_number, type_id, slave_id)
            report_level = logging.ERROR
            logging.log.error(report_msg)  # also warn, so that the user can fix the issue while the rest is running
            slave_report = {"msg": report_msg, "level": report_level, "num_dubious_blocks": num_dubious_blocks,
                            "num_conflicting_blocks": num_conflicting_blocks, "num_gaps": num_gaps,
                            "num_valid_data_blocks": num_valid_data_blocks,
                            "num_invalid_data_blocks": num_invalid_data_blocks}
            thread_report_queue.put(slave_report)

            if type_id == "S":
                close_slv_files(file_slave_log, file_cell_log, file_cell_eis, file_cell_eoc, file_cell_pulse, file_img)
            else:  # elif type_id == "T":
                close_tmgmt_files(file_tmgmt_log, file_pool_log, file_img)
            continue
        num_fs_blocks = int(re_match.group(1))
        num_data_blocks = int(re_match.group(2))
        if num_data_blocks > ((num_fs_blocks - 1) * cfg.NUM_BOXES_PER_FS_BLOCK):
            # generate report, close opened files, then go to next slave
            report_msg = f"Thread %u Slave %s%02u - implausible FS info start block" \
                         % (processor_number, type_id, slave_id)
            report_level = logging.ERROR
            logging.log.error(report_msg)  # also warn, so that the user can fix the issue while the rest is running
            slave_report = {"msg": report_msg, "level": report_level, "num_dubious_blocks": num_dubious_blocks,
                            "num_conflicting_blocks": num_conflicting_blocks, "num_gaps": num_gaps,
                            "num_valid_data_blocks": num_valid_data_blocks,
                            "num_invalid_data_blocks": num_invalid_data_blocks}
            thread_report_queue.put(slave_report)

            if type_id == "S":
                close_slv_files(file_slave_log, file_cell_log, file_cell_eis, file_cell_eoc, file_cell_pulse, file_img)
            else:  # elif type_id == "T":
                close_tmgmt_files(file_tmgmt_log, file_pool_log, file_img)
            continue

        # build up FS info
        logging.log.debug("Thread %u Slave %s%02u - Building up FS info..." % (processor_number, type_id, slave_id))
        fs_info = pd.Series(cfg.SdBox.SD_BOX_UNUSED, dtype=int, index=range(num_data_blocks))
        fs_info = fill_fs_boxes(file_img, fs_info, num_fs_blocks, processor_number, type_id, slave_id)

        # read FS backup start block
        file_img.seek(num_fs_blocks * cfg.SD_BLOCK_SIZE_BYTES, os.SEEK_SET)
        data_block = file_img.read(cfg.SD_BLOCK_SIZE_BYTES)
        re_pat = re.compile(cfg.BATCYC_FS_BACKUP_STR)
        re_match = re_pat.match(data_block.decode(cfg.IMAGE_ENCODING))
        if not re_match:
            # generate report, close opened files, then go to next slave
            report_msg = f"Thread %u Slave %s%02u - unknown/invalid FS backup start block" \
                         % (processor_number, type_id, slave_id)
            report_level = logging.ERROR
            logging.log.error(report_msg)  # also warn, so that the user can fix the issue while the rest is running
            slave_report = {"msg": report_msg, "level": report_level, "num_dubious_blocks": num_dubious_blocks,
                            "num_conflicting_blocks": num_conflicting_blocks, "num_gaps": num_gaps,
                            "num_valid_data_blocks": num_valid_data_blocks,
                            "num_invalid_data_blocks": num_invalid_data_blocks}
            thread_report_queue.put(slave_report)

            if type_id == "S":
                close_slv_files(file_slave_log, file_cell_log, file_cell_eis, file_cell_eoc, file_cell_pulse, file_img)
            else:  # elif type_id == "T":
                close_tmgmt_files(file_tmgmt_log, file_pool_log, file_img)
            continue

        # build up FS backup
        logging.log.debug("Thread %u Slave %s%02u - Building up FS backup..." % (processor_number, type_id, slave_id))
        fs_backup = pd.Series(cfg.SdBox.SD_BOX_UNUSED, dtype=int, index=range(num_data_blocks))
        fs_backup = fill_fs_boxes(file_img, fs_backup, num_fs_blocks, processor_number, type_id, slave_id)

        # analyze differences
        fs_diff = fs_info[fs_backup != fs_backup]
        logging.log.debug("Thread %u Slave %s%02u - %u differences between FS info and backup: fs_diff = \n%s"
                          % (processor_number, type_id, slave_id, fs_diff.shape[0], fs_diff))
        num_dubious_blocks = fs_diff.shape[0]
        if num_dubious_blocks > 0:
            logging.log.warning("Thread %u Slave %s%02u - Found %u differences between FS info/backup at indexes:\n%s"
                                % (processor_number, type_id, slave_id, num_dubious_blocks, fs_diff.index))
            # If FS info is 0, use FS backup:
            for i in fs_diff.index:
                if (fs_info[i] == 0) or (fs_info[i] > cfg.SD_BOX_MAX_VALID):
                    fs_info[i] = fs_backup[i]

        # analyze gaps
        fs_is_used = (fs_info > 0)
        fs_is_used_diff = fs_is_used[fs_is_used.shift(1) - fs_is_used < 0]
        num_gaps = fs_is_used_diff.shape[0]
        num_gaps_remaining = num_gaps
        if num_gaps > 0:
            if (num_gaps == 1) and (fs_is_used_diff.index[0] == 1000):
                # this is very likely caused by the 0xFF SD card bug --> see comment in fill_fs_boxes()
                logging.log.info("Thread %u Slave %s%02u - Found %u gaps in FS info/backup blocks, ending at:\n%s\n"
                                 "    This is most likely caused by a bug in the slave software and not data loss."
                                 % (processor_number, type_id, slave_id, num_gaps, fs_is_used_diff))
                num_gaps = num_gaps - 1  # suppresses the warning in case this is the only gap
            else:
                logging.log.warning("Thread %u Slave %s%02u - Found %u gaps in FS info/backup blocks, ending at:\n%s"
                                    % (processor_number, type_id, slave_id, num_gaps, fs_is_used_diff))

        # free memory
        fs_backup = ""
        fs_diff = ""
        fs_is_used = ""
        fs_is_used_diff = ""
        del fs_backup
        del fs_diff
        del fs_is_used
        del fs_is_used_diff
        gc.collect()

        # go through data (compare with expected data block from FS)
        sd_block_id = num_fs_blocks * cfg.SD_BLOCK_SIZE_BYTES * 2
        file_img.seek(sd_block_id, os.SEEK_SET)
        sd_block_id = sd_block_id - cfg.SD_BLOCK_SIZE_BYTES
        gap_begun = False
        found_config = False
        timestamp = 0
        timestamp_origin = cfg.TimestampOrigin.ERROR
        i_next_sd_df = 0
        timestamp_next_progress = PROGRESS_NOTIFICATION_DELTA_T
        eis_soc_list = [None] * cfg.NUM_EIS_SOCS_MAX
        for i in range(0, num_data_blocks):
            sd_block_id = sd_block_id + cfg.SD_BLOCK_SIZE_BYTES
            data_block = file_img.read(cfg.SD_BLOCK_SIZE_BYTES)
            magic_word = data_block[0:4]
            data_type = cfg.SdBox.SD_BOX_UNUSED
            if magic_word == b'MLB1':
                data_type = cfg.SdBox.SD_BOX_LOG_DATA
            elif magic_word == b'MLT1':
                data_type = cfg.SdBox.SD_BOX_TLOG_DATA
            elif magic_word == b'MLE2':
                data_type = cfg.SdBox.SD_BOX_EOC_DATA
            elif magic_word == b'MLI2':
                data_type = cfg.SdBox.SD_BOX_EIS_DATA
            elif magic_word == b'MLP1':
                data_type = cfg.SdBox.SD_BOX_PULSE_DATA
            elif (magic_word == b'\x00\x00\x00\x00') or (magic_word == b'\xFF\xFF\xFF\xFF'):
                pass  # empty block
            else:
                cfg_byte = data_block[0:1]
                if cfg_byte == b'C':
                    if found_config:
                        logging.log.warning("Thread %u Slave %s%02u - Duplicate configuration found at data index %u "
                                            "-> skip" % (processor_number, type_id, slave_id, i))
                        continue  # skip
                    elif type_id != "S":
                        logging.log.warning("Thread %u Slave %s%02u - Invalid configuration found at data index %u "
                                            "-> skip" % (processor_number, type_id, slave_id, i))
                        continue  # skip
                    else:
                        data_type = cfg.SdBox.SD_BOX_CFG_DATA
                        found_config = True
                elif cfg_byte == b'T':
                    if found_config:
                        logging.log.warning("Thread %u Slave %s%02u - Duplicate configuration found at data index %u "
                                            "-> skip" % (processor_number, type_id, slave_id, i))
                        continue  # skip
                    elif type_id != "T":
                        logging.log.warning("Thread %u Slave %s%02u - Invalid configuration found at data index %u "
                                            "-> skip" % (processor_number, type_id, slave_id, i))
                        continue  # skip
                    else:
                        data_type = cfg.SdBox.SD_BOX_TCFG_DATA
                        found_config = True
                else:
                    logging.log.warning("Thread %u Slave %s%02u - Unknown magic word '%s' at data index %u -> skip"
                                        % (processor_number, type_id, slave_id, magic_word.decode(cfg.IMAGE_ENCODING),
                                           i))
                    continue  # skip, the data is corrupted / invalid / unusable

            expected_type = fs_info[i]
            if expected_type == cfg.SdBox.SD_BOX_SPACER:
                if gap_begun:
                    gap_begun = False
                    num_gaps_remaining = num_gaps_remaining - 1
                if TRY_TO_USE_SPACER_DATA:
                    logging.log.info("Thread %u Slave %s%02u - Found spacer FS box at data index %u -> try to use"
                                     % (processor_number, type_id, slave_id, i))

                else:
                    logging.log.info("Thread %u Slave %s%02u - Found spacer FS box at data index %u -> skip"
                                     % (processor_number, type_id, slave_id, i))
                    continue
            elif expected_type != data_type:
                num_conflicting_blocks = num_conflicting_blocks + 1
                if TRY_TO_USE_CONFLICTING_DATA:
                    if expected_type == cfg.SdBox.SD_BOX_UNUSED:
                        if not gap_begun:
                            gap_begun = True  # even if no more gaps are expected -> try to use more data
                    elif gap_begun:
                        gap_begun = False
                        num_gaps_remaining = num_gaps_remaining - 1
                    logging.log.warning("Thread %u Slave %s%02u - Found magic word '%s' at data index %u, but expected "
                                        "another -> try to use based on magic word anyway"
                                        % (processor_number, type_id, slave_id, magic_word.decode(cfg.IMAGE_ENCODING),
                                           i))
                else:
                    if expected_type == cfg.SdBox.SD_BOX_UNUSED:
                        if gap_begun:
                            if num_gaps_remaining <= 0:
                                break  # no more gaps expected and likely no mora data available -> this is the end
                        else:
                            if num_gaps_remaining > 0:
                                gap_begun = True
                            else:
                                break  # no more gaps expected -> this is the end
                    elif gap_begun:
                        gap_begun = False
                        num_gaps_remaining = num_gaps_remaining - 1
                    logging.log.warning("Thread %u Slave %s%02u - Found magic word '%s' at data index %u, but expected "
                                        "another -> skip"
                                        % (processor_number, type_id, slave_id, magic_word.decode(cfg.IMAGE_ENCODING),
                                           i))
                    continue
            elif expected_type == cfg.SdBox.SD_BOX_UNUSED:  # == data_type
                if not gap_begun:
                    if num_gaps_remaining > 0:
                        gap_begun = True
                    else:
                        break  # no more gaps expected -> this is the end
            else:  # expected_type == data_type and expected_type is not SD_BOX_UNUSED
                if gap_begun:
                    gap_begun = False
                    num_gaps_remaining = num_gaps_remaining - 1

            # at this point, expected_type == data_type, or we shall try to use data_type anyway
            # success = False
            if data_type == cfg.SdBox.SD_BOX_LOG_DATA:
                # try to update timestamp
                [i_next_sd_df, timestamp, timestamp_origin] =\
                    find_timestamp(sd_timestamp_df, i_next_sd_df, sd_block_id, timestamp, timestamp_origin,
                                   type_id, slave_id)
                success = decode_log(data_block, sd_block_id, slave_id, timestamp, timestamp_origin,
                                     file_slave_log, file_cell_log, i)
            elif data_type == cfg.SdBox.SD_BOX_TLOG_DATA:
                [i_next_sd_df, timestamp, timestamp_origin] =\
                    find_timestamp(sd_timestamp_df, i_next_sd_df, sd_block_id, timestamp, timestamp_origin,
                                   type_id, slave_id)
                success = decode_tlog(data_block, sd_block_id, slave_id, timestamp, timestamp_origin,
                                      file_tmgmt_log, file_pool_log, i)
                if timestamp > timestamp_next_progress:
                    timestamp_next_progress = timestamp + PROGRESS_NOTIFICATION_DELTA_T
                    # show progress
                    date_text = pd.to_datetime(timestamp, unit="s")
                    logging.log.info("Thread %u Slave %s%02u - progress: currently at %s"
                                     % (processor_number, type_id, slave_id, date_text))
                    # flush tmgmt files
                    if type_id == "T":
                        flush_tmgmt_files(file_tmgmt_log, file_pool_log, file_img)
            elif data_type == cfg.SdBox.SD_BOX_EOC_DATA:
                success = decode_eoc(data_block, sd_block_id, slave_id, timestamp, timestamp_origin, file_cell_eoc, i)
            elif data_type == cfg.SdBox.SD_BOX_EIS_DATA:
                success = decode_eis(data_block, sd_block_id, slave_id, timestamp, timestamp_origin, file_cell_eis, i,
                                     eis_soc_list)
                if timestamp > timestamp_next_progress:
                    timestamp_next_progress = timestamp + PROGRESS_NOTIFICATION_DELTA_T
                    # show progress
                    date_text = pd.to_datetime(timestamp, unit="s")
                    logging.log.info("Thread %u Slave %s%02u - progress: currently at %s"
                                     % (processor_number, type_id, slave_id, date_text))
                    # flush slave files
                    if type_id == "S":
                        flush_slv_files(file_slave_log, file_cell_log, file_cell_eis, file_cell_eoc, file_cell_pulse,
                                        file_img)
            elif data_type == cfg.SdBox.SD_BOX_PULSE_DATA:
                success = decode_pulse(data_block, sd_block_id, slave_id, timestamp, timestamp_origin, file_cell_pulse,
                                       i)
            elif data_type == cfg.SdBox.SD_BOX_CFG_DATA:
                [success, eis_soc_list] = decode_slave_config(data_block, sd_block_id, slave_id, i)
            elif data_type == cfg.SdBox.SD_BOX_TCFG_DATA:
                success = decode_tmgmt_config(data_block, sd_block_id, slave_id, i)
            elif data_type == cfg.SdBox.SD_BOX_UNUSED:
                continue  # just skip, without increasing counters
            else:
                logging.log.error("Thread %u Slave %s%02u - expected valid data_type, but it is invalid, at data index "
                                  "%u" % (processor_number, type_id, slave_id, i))
                num_invalid_data_blocks = num_invalid_data_blocks + 1
                continue
            if success:
                num_valid_data_blocks = num_valid_data_blocks + 1
            else:
                num_invalid_data_blocks = num_invalid_data_blocks + 1
        
        if type_id == "S":
            close_slv_files(file_slave_log, file_cell_log, file_cell_eis, file_cell_eoc, file_cell_pulse, file_img)
        else:  # elif type_id == "T":
            close_tmgmt_files(file_tmgmt_log, file_pool_log, file_img)

        # reporting to main thread
        report_msg = f"Thread %u Slave %s%02u - done." % (processor_number, type_id, slave_id)
        if (num_valid_data_blocks == 0) or (num_conflicting_blocks > 0):
            report_level = logging.ERROR
        elif (num_invalid_data_blocks > 0) or (num_dubious_blocks > 0) or (num_gaps > 0):
            report_level = logging.WARNING
        else:
            report_level = logging.INFO
        slave_report = {"msg": report_msg, "level": report_level, "num_dubious_blocks": num_dubious_blocks,
                        "num_conflicting_blocks": num_conflicting_blocks, "num_gaps": num_gaps,
                        "num_valid_data_blocks": num_valid_data_blocks,
                        "num_invalid_data_blocks": num_invalid_data_blocks}
        thread_report_queue.put(slave_report)

        # also inform, so that the user knows early what's going on
        report_msg_ext = report_msg + "\n" + \
                         f"   %u valid blocks converted\n" \
                         f"   %u invalid blocks detected (but couldn't be converted)\n" \
                         f"   %u dubious blocks found (FS box = expected data type in info/backup sector differs)\n" \
                         f"   %u conflicting blocks found (FS box = expected data type and found data type differs)\n" \
                         f"   %u gaps in FS info/backup found (data loss or corrupted data likely!)\n" \
                         % (num_valid_data_blocks, num_invalid_data_blocks, num_dubious_blocks, num_conflicting_blocks,
                            num_gaps)
        if report_level == logging.ERROR:
            logging.log.error(report_msg_ext)
        elif report_level == logging.WARNING:
            logging.log.warning(report_msg_ext)
        elif report_level == logging.INFO:
            logging.log.info(report_msg_ext)
        elif report_level == logging.DEBUG:
            logging.log.debug(report_msg_ext)
        elif report_level == logging.CRITICAL:
            logging.log.critical(report_msg_ext)

    logging.log.info("Thread %u - no more slaves - exiting" % processor_number)
    slave_queue.close()
    thread_report_queue.close()


if __name__ == "__main__":
    run()


def close_slv_files(file_slave_log, file_cell_log, file_cell_eis, file_cell_eoc, file_cell_pulse, file_img):
    file_slave_log.close()
    for i in range(0, NUM_CELLS_PER_SLAVE):
        file_cell_log[i].close()
        file_cell_eis[i].close()
        file_cell_eoc[i].close()
        file_cell_pulse[i].close()
    file_img.close()


def close_tmgmt_files(file_tmgmt_log, file_pool_log, file_img):
    file_tmgmt_log.close()
    for i in range(0, NUM_POOLS_PER_TMGMT):
        file_pool_log[i].close()
    file_img.close()


def flush_slv_files(file_slave_log, file_cell_log, file_cell_eis, file_cell_eoc, file_cell_pulse, file_img):
    file_slave_log.flush()
    for i in range(0, NUM_CELLS_PER_SLAVE):
        file_cell_log[i].flush()
        file_cell_eis[i].flush()
        file_cell_eoc[i].flush()
        file_cell_pulse[i].flush()
    file_img.flush()


def flush_tmgmt_files(file_tmgmt_log, file_pool_log, file_img):
    file_tmgmt_log.flush()
    for i in range(0, NUM_POOLS_PER_TMGMT):
        file_pool_log[i].flush()
    file_img.flush()


def uint_from_bytes(blocks):
    return int.from_bytes(blocks, byteorder="big", signed=False)


def sint_from_bytes(blocks):
    return int.from_bytes(blocks, byteorder="big", signed=True)


# improvement from 94.4 to 2.4 second = 97.5 % faster!!!
def fill_fs_boxes(file_img, fs_info, num_fs_blocks, processor_number, type_id, slave_id):
    # define data type of raw file block:
    dt = np.dtype([('num_boxes', '>i2'), ('_1', 'u1', 10), ('boxes', 'u1', 500)])

    # read compete FS section (either info or backup) depending on when fill_fs_boxes(...) is called
    data_np = np.fromfile(file_img, dtype=dt, count=(num_fs_blocks-1), sep='')

    # convert number of boxes field to pandas data series
    num_boxes_ser = pd.Series(data_np['num_boxes'])

    # fix early bug of cycler board which couldn't deal with SD cards whose empty storage was all "0xFF" --> 0xFFFF = -1
    # --> the config (1 entry) was written by a script on the computer, but the cycler skipped the FS block
    if num_boxes_ser[0] == -1:
        num_boxes_ser[0] = 1

    # filter valid boxes (> 0 and <= 1000)
    num_boxes_valid_ser = num_boxes_ser[(num_boxes_ser > 0) & (num_boxes_ser <= cfg.NUM_BOXES_PER_FS_BLOCK)]

    # find gaps and warn if gaps exist
    nb_valid_index_ser = pd.Series(num_boxes_valid_ser.index)
    nb_valid_index_gaps_ser = nb_valid_index_ser[(nb_valid_index_ser.shift(1) - nb_valid_index_ser) < -1.0]
    if nb_valid_index_gaps_ser.shape[0] > 0:
        logging.log.warning("Thread %u Slave %s%02u - FS num_blocks contains gaps!"
                            % (processor_number, type_id, slave_id))

    data_np = data_np[num_boxes_valid_ser.index[0]:(num_boxes_valid_ser.index[-1] + 1)]
    boxes_df = pd.DataFrame(data_np['boxes'])
    boxes_df = boxes_df[boxes_df.index.isin(nb_valid_index_ser)]
    boxes_df_h = boxes_df / 16
    boxes_df_h = boxes_df_h.astype(int)
    boxes_df_l = boxes_df % 16

    boxes_df_combined =\
        pd.DataFrame(list(interleave([boxes_df_h.values.transpose(), boxes_df_l.values.transpose()]))).transpose()
    boxes_df_combined.index = nb_valid_index_ser.values

    overall_size = boxes_df_combined.shape[0] + 1
    boxes_df_combined = boxes_df_combined.reindex(range(0, overall_size))
    boxes_df_combined.fillna(0, inplace=True)
    boxes_df_combined = boxes_df_combined.astype(int)

    num_boxes_ser = num_boxes_valid_ser.astype(int).reindex(range(0, overall_size))
    num_boxes_ser.fillna(0, inplace=True)
    num_boxes_ser = num_boxes_ser.astype(int)

    # get all rows that are not full
    num_boxes_ser_not_full = num_boxes_ser[num_boxes_ser < cfg.NUM_BOXES_PER_FS_BLOCK]
    for index, item in num_boxes_ser_not_full.items():
        # set boxes to 0 where nothing should be according to num_boxes
        boxes_df_combined.loc[index, item:cfg.NUM_BOXES_PER_FS_BLOCK] = cfg.SdBox.SD_BOX_UNUSED

    boxes_list = boxes_df_combined.stack().values
    fs_info[0:boxes_list.shape[0]] = boxes_list
    return fs_info
    # # free memory
    # data_np = ""
    # num_boxes_ser = ""
    # nb_valid_index_ser = ""
    # nb_valid_index_gaps_ser = ""
    # del data_np
    # del num_boxes_ser
    # del nb_valid_index_ser
    # del nb_valid_index_gap<s_ser
    # gc.collect()


def find_timestamp(sd_timestamp_df, i_next_sd_df, sd_block_id, timestamp, timestamp_origin, type_id, slave_id):
    match = False
    while not match:
        sd_block_id_from_df = sd_timestamp_df.SD_block_ID[i_next_sd_df]
        if sd_block_id_from_df == sd_block_id:
            # match (as expected)
            timestamp = sd_timestamp_df.unixtimestamp[i_next_sd_df]
            timestamp_origin = int(sd_timestamp_df.timestamp_origin[i_next_sd_df])
            i_next_sd_df = i_next_sd_df + 1
            match = True
        elif sd_block_id_from_df < sd_block_id:
            # block id in data frame too low, increment index and retry
            i_next_sd_df = i_next_sd_df + 1
        else:  # (sd_block_id_from_df > sd_block_id) or invalid
            # couldn't find timestamp! -> at this point, this should raise an Error, however fill timestamp anyway
            timestamp = timestamp + cfg.DELTA_T_LOG
            timestamp_origin = cfg.TimestampOrigin.ERROR
            logging.log.error("Thread ? Slave %s%02u - couldn't find timestamp for SD block ID %u"
                              % (type_id, slave_id, sd_block_id))
            break
    return [i_next_sd_df, timestamp, timestamp_origin]


def decode_log(data_block, sd_block_id, slave_id, timestamp, timestamp_origin, file_slave_log, file_cell_log,
               data_index):
    # data_block[4] contains slave_id
    if data_block[4] != slave_id:
        text = f"Thread ? Slave S%02u - found different slave_id %u in LOG at data index %u" \
               % (slave_id, data_block[4], data_index)
        if data_block[4] == 0:
            logging.log.debug(text)  # this occasionally happens, especially at the beginning of the experiment
        else:
            logging.log.warning(text)  # this usually shouldn't happen, so warn

    # I played around quite some time and I think this is the fastest:
    scheduler_condition = data_block[5] >> 4  # (data_block[5] >> 4) & 0x0F
    controller_condition = data_block[5] & 0x0F
    high_level_state = data_block[6]
    log_state = data_block[7] >> 4  # (data_block[7] >> 4) & 0x0F
    dc_state = data_block[7] & 0x0F
    uptime_ticks = uint_from_bytes(data_block[8:12])
    p_main_W = struct.unpack(">f", data_block[12:16])[0]
    v_main_V = struct.unpack(">f", data_block[16:20])[0]
    v_aux_V = struct.unpack(">f", data_block[20:24])[0]
    i_main_A = struct.unpack(">f", data_block[24:28])[0]
    eth_status = data_block[28] >> 6  # (data_block[28] >> 6) & 0x03
    v_main_state = data_block[28] & 0x3F
    eth_fault = data_block[29] >> 7  # data_block[29] >> 7 & 0x01
    sd_fault = (data_block[29] >> 6) & 0x01
    v_aux_state = data_block[29] & 0x3F
    monitor_enable = data_block[30] >> 7  # (data_block[30] >> 7) & 0x01
    i_main_state = data_block[30] & 0x3F
    sd_fill = sint_from_bytes(data_block[31:32])

    string_slave_log = format(timestamp, ".3f") + cfg.CSV_SEP + \
                       str(timestamp_origin) + cfg.CSV_SEP + \
                       str(sd_block_id) + cfg.CSV_SEP + \
                       str(high_level_state) + cfg.CSV_SEP + \
                       str(scheduler_condition) + cfg.CSV_SEP + \
                       str(controller_condition) + cfg.CSV_SEP + \
                       format(v_main_V, ".2f") + cfg.CSV_SEP + \
                       format(i_main_A, ".2f") + cfg.CSV_SEP + \
                       format(p_main_W, ".2f") + cfg.CSV_SEP + \
                       format(v_aux_V, ".2f") + cfg.CSV_SEP + \
                       str(v_main_state) + cfg.CSV_SEP + \
                       str(i_main_state) + cfg.CSV_SEP + \
                       str(v_aux_state) + cfg.CSV_SEP + \
                       str(monitor_enable) + cfg.CSV_SEP + \
                       str(dc_state) + cfg.CSV_SEP + \
                       str(log_state) + cfg.CSV_SEP + \
                       str(eth_status) + cfg.CSV_SEP + \
                       str(eth_fault) + cfg.CSV_SEP + \
                       str(sd_fault) + cfg.CSV_SEP + \
                       str(sd_fill) + cfg.CSV_SEP + \
                       str(uptime_ticks) + \
                       cfg.CSV_NEWLINE

    file_slave_log.write(string_slave_log)

    for i_cell in range(0, cfg.NUM_CELLS_PER_SLAVE):
        i_block_start = 32 + i_cell * 40
        i_block_end = i_block_start + 40
        cell_data_block = data_block[i_block_start:i_block_end]

        scheduler_state = uint_from_bytes(cell_data_block[0:4])
        scheduler_state_phase = (scheduler_state >> 16) & 0xF
        scheduler_state_checkup = (scheduler_state >> 12) & 0xF
        scheduler_state_charge = (scheduler_state >> 8) & 0xF
        scheduler_state_sub = scheduler_state & 0xF
        scheduler_timeout = (scheduler_state >> 6) & 0x1
        scheduler_cu_pending = (scheduler_state >> 5) & 0x1
        scheduler_pause_pending = (scheduler_state >> 4) & 0x1

        v_raw_V = struct.unpack(">f", cell_data_block[4:8])[0]
        i_raw_A = struct.unpack(">f", cell_data_block[8:12])[0]
        t_cell_degC = struct.unpack(">f", cell_data_block[12:16])[0]
        delta_q_Ah = struct.unpack(">f", cell_data_block[16:20])[0]
        soc_est = struct.unpack(">f", cell_data_block[20:24])[0] * 100
        ocv_est_V = struct.unpack(">f", cell_data_block[24:28])[0]
        over_q = cell_data_block[28] >> 7
        under_q = (cell_data_block[28] >> 6) & 0x01
        v_state = cell_data_block[28] & 0x3F
        over_soc = cell_data_block[29] >> 7
        under_soc = (cell_data_block[29] >> 6) & 0x01
        i_state = cell_data_block[29] & 0x3F
        eol_cap = cell_data_block[30] >> 7
        eol_imp = (cell_data_block[30] >> 6) & 0x01
        t_state = cell_data_block[30] & 0x3F
        meas_condition = cell_data_block[31] >> 4
        bms_condition = cell_data_block[31] & 0x0F
        p_raw_W = struct.unpack(">f", cell_data_block[32:36])[0]
        delta_e_Wh = struct.unpack(">f", cell_data_block[36:40])[0]

        string_cell_log = format(timestamp, ".3f") + cfg.CSV_SEP + \
                          str(timestamp_origin) + cfg.CSV_SEP + \
                          str(sd_block_id) + cfg.CSV_SEP + \
                          format(v_raw_V, ".4f") + cfg.CSV_SEP + \
                          format(i_raw_A, ".4f") + cfg.CSV_SEP + \
                          format(p_raw_W, ".2f") + cfg.CSV_SEP + \
                          format(t_cell_degC, ".2f") + cfg.CSV_SEP + \
                          format(delta_q_Ah, ".6f") + cfg.CSV_SEP + \
                          format(delta_e_Wh, ".6f") + cfg.CSV_SEP + \
                          format(soc_est, ".2f") + cfg.CSV_SEP + \
                          format(ocv_est_V, ".4f") + cfg.CSV_SEP + \
                          str(meas_condition) + cfg.CSV_SEP + \
                          str(bms_condition) + cfg.CSV_SEP + \
                          str(v_state) + cfg.CSV_SEP + \
                          str(i_state) + cfg.CSV_SEP + \
                          str(t_state) + cfg.CSV_SEP + \
                          str(over_q) + cfg.CSV_SEP + \
                          str(under_q) + cfg.CSV_SEP + \
                          str(over_soc) + cfg.CSV_SEP + \
                          str(under_soc) + cfg.CSV_SEP + \
                          str(eol_cap) + cfg.CSV_SEP + \
                          str(eol_imp) + cfg.CSV_SEP + \
                          format(scheduler_state, "d") + cfg.CSV_SEP + \
                          str(scheduler_state_phase) + cfg.CSV_SEP + \
                          str(scheduler_state_checkup) + cfg.CSV_SEP + \
                          str(scheduler_state_charge) + cfg.CSV_SEP + \
                          str(scheduler_timeout) + cfg.CSV_SEP + \
                          str(scheduler_cu_pending) + cfg.CSV_SEP + \
                          str(scheduler_pause_pending) + cfg.CSV_SEP + \
                          str(scheduler_state_sub) + \
                          cfg.CSV_NEWLINE
        # Note: To reduce file size, I left away 'format(scheduler_state, "#010X") + cfg.CSV_SEP + \' between
        # 'format(scheduler_state, "d") + cfg.CSV_SEP + \' and 'str(scheduler_state_phase) + cfg.CSV_SEP + \'
        # Note: To reduce file size, I changed soc_est accuracy from .4f to .2f - it wasn't that exact anyway.
        # Note: To reduce file size, I changed p_raw_W accuracy from .4f to .2f.
        file_cell_log[i_cell].write(string_cell_log)
    return True


def decode_tlog(data_block, sd_block_id, slave_id, timestamp, timestamp_origin, file_tmgmt_log, file_pool_log,
                data_index):
    # data_block[4] contains slave_id
    if data_block[4] != slave_id:
        text = f"Thread ? Slave T%02u - found different slave_id %u in TLOG at data index %u" \
               % (slave_id, data_block[4], data_index)
        if data_block[4] == 0:
            logging.log.debug(text)  # this occasionally happens, especially at the beginning of the experiment
        else:
            logging.log.warning(text)  # this usually shouldn't happen, so warn

    scheduler_condition = data_block[5] >> 4  # (data_block[5] >> 4) & 0x0F
    controller_condition = data_block[5] & 0x0F
    high_level_state = data_block[6]
    log_state = data_block[7] >> 4
    dc_state = data_block[7] & 0x0F
    uptime_ticks = uint_from_bytes(data_block[8:12])
    p_main_W = struct.unpack(">f", data_block[12:16])[0]
    v_main_V = struct.unpack(">f", data_block[16:20])[0]
    v_aux_V = struct.unpack(">f", data_block[20:24])[0]
    i_main_A = struct.unpack(">f", data_block[24:28])[0]
    # eth_status = data_block[28] >> 6
    v_main_state = data_block[28] & 0x3F
    # eth_fault = data_block[29] >> 7
    # sd_fault = (data_block[29] >> 6) & 0x01
    v_aux_state = data_block[29] & 0x3F
    monitor_enable = data_block[30] >> 7
    i_main_state = data_block[30] & 0x3F
    sd_fill = sint_from_bytes(data_block[31:32])
    valve_air_en = data_block[32]
    fan_pcb_en = data_block[33]
    fan_radiator_en = data_block[34]
    pump_hot_set = data_block[35]
    chiller_t_set_degC = struct.unpack(">f", data_block[36:40])[0]
    chiller_t_meas_degC = struct.unpack(">f", data_block[40:44])[0]
    chiller_n_set_rpm = struct.unpack(">f", data_block[44:48])[0]
    chiller_n_meas_rpm = struct.unpack(">f", data_block[48:52])[0]
    chiller_state = data_block[52]
    pump_hot_meas = data_block[53]
    r_iso_state = data_block[54]
    # unused = data_block[55]
    r_iso_Ohm = struct.unpack(">f", data_block[56:60])[0]
    t_pipe_1_state = data_block[60]
    t_pipe_2_state = data_block[61]
    t_pipe_3_state = data_block[62]
    # t_pipe_4_state = data_block[63] -> not in use
    # t_other_1_state = data_block[64] -> not in use
    # t_other_2_state = data_block[65] -> not in use
    # t_other_3_state = data_block[66] -> not in use
    # t_other_4_state = data_block[67] -> not in use
    t_pipe_1_degC = struct.unpack(">f", data_block[68:72])[0]
    t_pipe_2_degC = struct.unpack(">f", data_block[72:76])[0]
    t_pipe_3_degC = struct.unpack(">f", data_block[76:80])[0]
    # t_pipe_4_degC = struct.unpack(">f", data_block[80:84])[0] -> not in use
    # t_other_1_degC = struct.unpack(">f", data_block[84:88])[0] -> not in use
    # t_other_2_degC = struct.unpack(">f", data_block[88:92])[0] -> not in use
    # t_other_3_degC = struct.unpack(">f", data_block[92:96])[0] -> not in use
    # t_other_4_degC = struct.unpack(">f", data_block[96:100])[0] -> not in use
    # scheduler_state = struct.unpack(">f", data_block[100:104])[0] -> not in use
    t_box_cold_degC = struct.unpack(">f", data_block[104:108])[0]
    hum_box_cold_percent = struct.unpack(">f", data_block[108:112])[0]
    press_box_cold_hPa = struct.unpack(">f", data_block[112:116])[0]
    dew_point_box_cold_degC = struct.unpack(">f", data_block[116:120])[0]
    abs_hum_box_cold_g_m3 = struct.unpack(">f", data_block[120:124])[0]
    t_ambient_degC = struct.unpack(">f", data_block[124:128])[0]
    hum_ambient_percent = struct.unpack(">f", data_block[128:132])[0]
    press_ambient_hPa = struct.unpack(">f", data_block[132:136])[0]
    dew_point_ambient_degC = struct.unpack(">f", data_block[136:140])[0]
    abs_hum_ambient_g_m3 = struct.unpack(">f", data_block[140:144])[0]
    t_controller_degC = struct.unpack(">f", data_block[144:148])[0]

    string_tmgmt_log = format(timestamp, ".3f") + cfg.CSV_SEP + \
                       str(timestamp_origin) + cfg.CSV_SEP + \
                       str(sd_block_id) + cfg.CSV_SEP + \
                       str(high_level_state) + cfg.CSV_SEP + \
                       str(scheduler_condition) + cfg.CSV_SEP + \
                       str(controller_condition) + cfg.CSV_SEP + \
                       format(v_main_V, ".2f") + cfg.CSV_SEP + \
                       format(i_main_A, ".2f") + cfg.CSV_SEP + \
                       format(p_main_W, ".2f") + cfg.CSV_SEP + \
                       format(v_aux_V, ".2f") + cfg.CSV_SEP + \
                       str(v_main_state) + cfg.CSV_SEP + \
                       str(i_main_state) + cfg.CSV_SEP + \
                       str(v_aux_state) + cfg.CSV_SEP + \
                       str(monitor_enable) + cfg.CSV_SEP + \
                       str(dc_state) + cfg.CSV_SEP + \
                       str(log_state) + cfg.CSV_SEP + \
                       str(sd_fill) + cfg.CSV_SEP + \
                       str(valve_air_en) + cfg.CSV_SEP + \
                       str(fan_pcb_en) + cfg.CSV_SEP + \
                       str(fan_radiator_en) + cfg.CSV_SEP + \
                       str(pump_hot_set) + cfg.CSV_SEP + \
                       str(pump_hot_meas) + cfg.CSV_SEP + \
                       format(chiller_t_set_degC, ".1f") + cfg.CSV_SEP + \
                       format(chiller_t_meas_degC, ".2f") + cfg.CSV_SEP + \
                       str(int(chiller_n_set_rpm)) + cfg.CSV_SEP + \
                       str(int(chiller_n_meas_rpm)) + cfg.CSV_SEP + \
                       str(chiller_state) + cfg.CSV_SEP + \
                       str(r_iso_state) + cfg.CSV_SEP + \
                       str(r_iso_Ohm) + cfg.CSV_SEP + \
                       format(t_pipe_1_degC, ".2f") + cfg.CSV_SEP + \
                       format(t_pipe_2_degC, ".2f") + cfg.CSV_SEP + \
                       format(t_pipe_3_degC, ".2f") + cfg.CSV_SEP + \
                       str(t_pipe_1_state) + cfg.CSV_SEP + \
                       str(t_pipe_2_state) + cfg.CSV_SEP + \
                       str(t_pipe_3_state) + cfg.CSV_SEP + \
                       format(t_controller_degC, ".2f") + cfg.CSV_SEP + \
                       format(t_box_cold_degC, ".2f") + cfg.CSV_SEP + \
                       format(hum_box_cold_percent, ".2f") + cfg.CSV_SEP + \
                       format(press_box_cold_hPa, ".2f") + cfg.CSV_SEP + \
                       format(dew_point_box_cold_degC, ".2f") + cfg.CSV_SEP + \
                       format(abs_hum_box_cold_g_m3, ".2f") + cfg.CSV_SEP + \
                       format(t_ambient_degC, ".2f") + cfg.CSV_SEP + \
                       format(hum_ambient_percent, ".2f") + cfg.CSV_SEP + \
                       format(press_ambient_hPa, ".2f") + cfg.CSV_SEP + \
                       format(dew_point_ambient_degC, ".2f") + cfg.CSV_SEP + \
                       format(abs_hum_ambient_g_m3, ".2f") + cfg.CSV_SEP + \
                       str(uptime_ticks) + \
                       cfg.CSV_NEWLINE
    file_tmgmt_log.write(string_tmgmt_log)

    for i_pool in range(0, cfg.NUM_POOLS_PER_TMGMT):
        # pool data
        i_block_start = 160 + i_pool * 24
        i_block_end = i_block_start + 24
        pool_data_block = data_block[i_block_start:i_block_end]

        scheduler_state = uint_from_bytes(pool_data_block[0:4])
        scheduler_state_phase = (scheduler_state >> 16) & 0xF
        scheduler_state_checkup = (scheduler_state >> 12) & 0xF
        scheduler_state_sub = scheduler_state & 0xF
        scheduler_cu_pending = (scheduler_state >> 5) & 0x1
        scheduler_pause_pending = (scheduler_state >> 4) & 0x1

        t_pool_meas_degC = struct.unpack(">f", pool_data_block[4:8])[0]
        t_plate_degC = struct.unpack(">f", pool_data_block[8:12])[0]
        t_pool_state = pool_data_block[12]
        t_plate_state = pool_data_block[13]
        has_ot = pool_data_block[14] >> 7
        is_stable = (pool_data_block[14] >> 6) & 0x01
        has_timeout = (pool_data_block[14] >> 5) & 0x01
        condition = pool_data_block[15]
        t_pool_set_degC = struct.unpack(">f", pool_data_block[16:20])[0]
        t_pool_set_ramped_degC = struct.unpack(">f", pool_data_block[20:24])[0]

        # peltier data (4 per pool)
        i_block_start = 256 + i_pool * 64
        i_block_end = i_block_start + 64
        peltier_data_block = data_block[i_block_start:i_block_end]

        peltier_i_meas_1_A = struct.unpack(">f", peltier_data_block[0:4])[0]
        peltier_i_meas_2_A = struct.unpack(">f", peltier_data_block[16:20])[0]
        peltier_i_meas_3_A = struct.unpack(">f", peltier_data_block[32:36])[0]
        peltier_i_meas_4_A = struct.unpack(">f", peltier_data_block[48:52])[0]
        peltier_i_state_1 = peltier_data_block[4]
        peltier_i_state_2 = peltier_data_block[20]
        peltier_i_state_3 = peltier_data_block[36]
        peltier_i_state_4 = peltier_data_block[52]
        peltier_en_1 = peltier_data_block[5]
        peltier_en_2 = peltier_data_block[21]
        peltier_en_3 = peltier_data_block[37]
        peltier_en_4 = peltier_data_block[53]
        peltier_ctrl_state_1 = peltier_data_block[6]
        peltier_ctrl_state_2 = peltier_data_block[22]
        peltier_ctrl_state_3 = peltier_data_block[38]
        peltier_ctrl_state_4 = peltier_data_block[54]
        peltier_condition_1 = peltier_data_block[7]
        peltier_condition_2 = peltier_data_block[23]
        peltier_condition_3 = peltier_data_block[39]
        peltier_condition_4 = peltier_data_block[55]
        peltier_i_set_1_A = struct.unpack(">f", peltier_data_block[8:12])[0]
        peltier_i_set_2_A = struct.unpack(">f", peltier_data_block[24:28])[0]
        peltier_i_set_3_A = struct.unpack(">f", peltier_data_block[40:44])[0]
        peltier_i_set_4_A = struct.unpack(">f", peltier_data_block[56:60])[0]

        peltier_i_meas_sum_A = peltier_i_meas_1_A + peltier_i_meas_2_A + peltier_i_meas_3_A + peltier_i_meas_4_A
        peltier_num_en = peltier_en_1 + peltier_en_2 + peltier_en_3 + peltier_en_4
        peltier_i_set_sum_A = peltier_i_set_1_A + peltier_i_set_2_A + peltier_i_set_3_A + peltier_i_set_4_A

        string_pool_log = format(timestamp, ".3f") + cfg.CSV_SEP + \
                          str(timestamp_origin) + cfg.CSV_SEP + \
                          str(sd_block_id) + cfg.CSV_SEP + \
                          format(scheduler_state, "d") + cfg.CSV_SEP + \
                          str(scheduler_state_phase) + cfg.CSV_SEP + \
                          str(scheduler_state_checkup) + cfg.CSV_SEP + \
                          str(scheduler_cu_pending) + cfg.CSV_SEP + \
                          str(scheduler_pause_pending) + cfg.CSV_SEP + \
                          str(scheduler_state_sub) + cfg.CSV_SEP + \
                          format(t_pool_set_degC, ".2f") + cfg.CSV_SEP + \
                          format(t_pool_set_ramped_degC, ".2f") + cfg.CSV_SEP + \
                          format(t_pool_meas_degC, ".2f") + cfg.CSV_SEP + \
                          format(t_plate_degC, ".2f") + cfg.CSV_SEP + \
                          str(has_ot) + cfg.CSV_SEP + \
                          str(is_stable) + cfg.CSV_SEP + \
                          str(has_timeout) + cfg.CSV_SEP + \
                          str(condition) + cfg.CSV_SEP + \
                          str(t_pool_state) + cfg.CSV_SEP + \
                          str(t_plate_state) + cfg.CSV_SEP + \
                          format(peltier_i_set_sum_A, ".2f") + cfg.CSV_SEP + \
                          format(peltier_i_meas_sum_A, ".2f") + cfg.CSV_SEP + \
                          str(peltier_num_en) + cfg.CSV_SEP + \
                          format(peltier_i_set_1_A, ".2f") + cfg.CSV_SEP + \
                          format(peltier_i_set_2_A, ".2f") + cfg.CSV_SEP + \
                          format(peltier_i_set_3_A, ".2f") + cfg.CSV_SEP + \
                          format(peltier_i_set_4_A, ".2f") + cfg.CSV_SEP + \
                          format(peltier_i_meas_1_A, ".2f") + cfg.CSV_SEP + \
                          format(peltier_i_meas_2_A, ".2f") + cfg.CSV_SEP + \
                          format(peltier_i_meas_3_A, ".2f") + cfg.CSV_SEP + \
                          format(peltier_i_meas_4_A, ".2f") + cfg.CSV_SEP + \
                          str(peltier_en_1) + cfg.CSV_SEP + \
                          str(peltier_en_2) + cfg.CSV_SEP + \
                          str(peltier_en_3) + cfg.CSV_SEP + \
                          str(peltier_en_4) + cfg.CSV_SEP + \
                          str(peltier_i_state_1) + cfg.CSV_SEP + \
                          str(peltier_i_state_2) + cfg.CSV_SEP + \
                          str(peltier_i_state_3) + cfg.CSV_SEP + \
                          str(peltier_i_state_4) + cfg.CSV_SEP + \
                          str(peltier_ctrl_state_1) + cfg.CSV_SEP + \
                          str(peltier_ctrl_state_2) + cfg.CSV_SEP + \
                          str(peltier_ctrl_state_3) + cfg.CSV_SEP + \
                          str(peltier_ctrl_state_4) + cfg.CSV_SEP + \
                          str(peltier_condition_1) + cfg.CSV_SEP + \
                          str(peltier_condition_2) + cfg.CSV_SEP + \
                          str(peltier_condition_3) + cfg.CSV_SEP + \
                          str(peltier_condition_4) + \
                          cfg.CSV_NEWLINE
        # Note: To reduce file size, I left away 'format(scheduler_state, "#010X") + cfg.CSV_SEP + \' between
        # 'format(scheduler_state, "d") + cfg.CSV_SEP + \' and 'str(scheduler_state_phase) + cfg.CSV_SEP + \'
        file_pool_log[i_pool].write(string_pool_log)
    return True


def decode_eoc(data_block, sd_block_id, slave_id, timestamp, timestamp_origin, file_cell_eoc, data_index):
    # data_block[4] contains slave_id
    if data_block[4] != slave_id:
        logging.log.warning(f"Thread ? Slave S%02u - found different slave_id %u in EOC at data index %u"
                            % (slave_id, data_block[4], data_index))  # this usually shouldn't happen, so warn

    cell_id = data_block[5]
    if cell_id >= cfg.NUM_CELLS_PER_SLAVE:
        logging.log.error(f"Thread ? Slave S%02u - found invalid cell_id %u in EOC at data index %u -> skip"
                          % (slave_id, cell_id, data_index))
        return False  # invalid cell ID -> can't write to file -> skip

    cyc_charged = data_block[6] >> 7
    cyc_condition = data_block[6] & 0x7F
    age_type = data_block[7]
    uptime_ticks = uint_from_bytes(data_block[8:12])
    age_temp = sint_from_bytes(data_block[12:13])
    age_soc = data_block[13]
    age_chg_rate = data_block[14] / 100
    age_dischg_rate = data_block[15] / 100
    ocv_est_start_V = struct.unpack(">f", data_block[16:20])[0]
    ocv_est_end_V = struct.unpack(">f", data_block[20:24])[0]
    soc_est_start = struct.unpack(">f", data_block[24:28])[0] * 100
    soc_est_end = struct.unpack(">f", data_block[28:32])[0] * 100
    t_start_degC = struct.unpack(">f", data_block[32:36])[0]
    t_end_degC = struct.unpack(">f", data_block[36:40])[0]
    cyc_duration_s = struct.unpack(">f", data_block[40:44])[0]
    delta_q_Ah = struct.unpack(">f", data_block[44:48])[0]
    total_q_condition_Ah = struct.unpack(">f", data_block[48:52])[0]
    total_q_chg_Ah = struct.unpack(">f", data_block[52:56])[0]
    total_q_dischg_Ah = struct.unpack(">f", data_block[56:60])[0]
    num_cycles_op = uint_from_bytes(data_block[60:64])
    delta_e_Wh = struct.unpack(">f", data_block[64:68])[0]
    total_e_condition_Wh = struct.unpack(">f", data_block[68:72])[0]
    total_e_chg_Wh = struct.unpack(">f", data_block[72:76])[0]
    total_e_dischg_Wh = struct.unpack(">f", data_block[76:80])[0]
    coulomb_efficiency = struct.unpack(">f", data_block[80:84])[0] * 100
    energy_efficiency = struct.unpack(">f", data_block[84:88])[0] * 100
    v_max_target_V = struct.unpack(">f", data_block[88:92])[0]
    v_min_target_V = struct.unpack(">f", data_block[92:96])[0]
    i_chg_max_A = struct.unpack(">f", data_block[96:100])[0]
    i_dischg_min_A = struct.unpack(">f", data_block[100:104])[0]
    i_chg_cutoff_A = struct.unpack(">f", data_block[104:108])[0]
    i_dischg_cutoff_A = struct.unpack(">f", data_block[108:112])[0]
    delta_q_chg_Ah = struct.unpack(">f", data_block[112:116])[0]
    delta_q_dischg_Ah = struct.unpack(">f", data_block[116:120])[0]
    delta_e_chg_Wh = struct.unpack(">f", data_block[120:124])[0]
    delta_e_dischg_Wh = struct.unpack(">f", data_block[124:128])[0]
    age_profile = data_block[128]
    num_cycles_checkup = data_block[129]
    cap_aged_est_Ah = struct.unpack(">f", data_block[132:136])[0]
    soh_cap = struct.unpack(">f", data_block[136:140])[0] * 100

    string_cell_eoc = format(timestamp, ".3f") + cfg.CSV_SEP + \
                      str(timestamp_origin) + cfg.CSV_SEP + \
                      str(sd_block_id) + cfg.CSV_SEP + \
                      str(cyc_condition) + cfg.CSV_SEP + \
                      str(cyc_charged) + cfg.CSV_SEP + \
                      str(age_type) + cfg.CSV_SEP + \
                      str(age_temp) + cfg.CSV_SEP + \
                      str(age_soc) + cfg.CSV_SEP + \
                      format(age_chg_rate, ".2f") + cfg.CSV_SEP + \
                      format(age_dischg_rate, ".2f") + cfg.CSV_SEP + \
                      str(age_profile) + cfg.CSV_SEP + \
                      format(v_max_target_V, ".4f") + cfg.CSV_SEP + \
                      format(v_min_target_V, ".4f") + cfg.CSV_SEP + \
                      format(i_chg_max_A, ".4f") + cfg.CSV_SEP + \
                      format(i_dischg_min_A, ".4f") + cfg.CSV_SEP + \
                      format(i_chg_cutoff_A, ".4f") + cfg.CSV_SEP + \
                      format(i_dischg_cutoff_A, ".4f") + cfg.CSV_SEP + \
                      format(cap_aged_est_Ah, ".6f") + cfg.CSV_SEP + \
                      format(soh_cap, ".6f") + cfg.CSV_SEP + \
                      format(delta_q_Ah, ".6f") + cfg.CSV_SEP + \
                      format(delta_q_chg_Ah, ".6f") + cfg.CSV_SEP + \
                      format(delta_q_dischg_Ah, ".6f") + cfg.CSV_SEP + \
                      format(delta_e_Wh, ".6f") + cfg.CSV_SEP + \
                      format(delta_e_chg_Wh, ".6f") + cfg.CSV_SEP + \
                      format(delta_e_dischg_Wh, ".6f") + cfg.CSV_SEP + \
                      format(coulomb_efficiency, ".6f") + cfg.CSV_SEP + \
                      format(energy_efficiency, ".6f") + cfg.CSV_SEP + \
                      format(ocv_est_start_V, ".4f") + cfg.CSV_SEP + \
                      format(ocv_est_end_V, ".4f") + cfg.CSV_SEP + \
                      format(soc_est_start, ".4f") + cfg.CSV_SEP + \
                      format(soc_est_end, ".4f") + cfg.CSV_SEP + \
                      format(t_start_degC, ".2f") + cfg.CSV_SEP + \
                      format(t_end_degC, ".2f") + cfg.CSV_SEP + \
                      format(cyc_duration_s, ".2f") + cfg.CSV_SEP + \
                      str(num_cycles_op) + cfg.CSV_SEP + \
                      str(num_cycles_checkup) + cfg.CSV_SEP + \
                      format(total_q_condition_Ah, ".4f") + cfg.CSV_SEP + \
                      format(total_q_chg_Ah, ".4f") + cfg.CSV_SEP + \
                      format(total_q_dischg_Ah, ".4f") + cfg.CSV_SEP + \
                      format(total_e_condition_Wh, ".4f") + cfg.CSV_SEP + \
                      format(total_e_chg_Wh, ".4f") + cfg.CSV_SEP + \
                      format(total_e_dischg_Wh, ".4f") + cfg.CSV_SEP + \
                      str(uptime_ticks) + \
                      cfg.CSV_NEWLINE

    file_cell_eoc[cell_id].write(string_cell_eoc)
    return True


def decode_eis(data_block, sd_block_id, slave_id, timestamp, timestamp_origin, file_cell_eis, data_index, eis_soc_list):
    # data_block[4] contains slave_id
    if data_block[4] != slave_id:
        logging.log.warning(f"Thread ? Slave S%02u - found different slave_id %u in EIS at data index %u"
                            % (slave_id, data_block[4], data_index))  # this usually shouldn't happen, so warn

    cell_id = data_block[5]
    if cell_id >= cfg.NUM_CELLS_PER_SLAVE:
        logging.log.error(f"Thread ? Slave S%02u - found invalid cell_id %u in EIS at data index %u -> skip"
                          % (slave_id, cell_id, data_index))
        return False  # invalid cell ID -> can't write to file -> skip

    valid = data_block[6] >> 7
    num_eis_points = data_block[6] & 0x7F
    if num_eis_points > cfg.NUM_EIS_POINTS_MAX_EXPECTED_ON_SD:
        if num_eis_points > cfg.NUM_EIS_POINTS_MAX_FITTING_ON_SD:
            logging.log.error(f"Thread ? Slave S%02u - num_eis_points (%u) > MAX (%u) in EIS at data index %u -> crop"
                              % (slave_id, num_eis_points, cfg.NUM_EIS_POINTS_MAX_FITTING_ON_SD, data_index))
            num_eis_points = cfg.NUM_EIS_POINTS_MAX_FITTING_ON_SD
        else:
            logging.log.warning(f"Thread ? Slave S%02u - num_eis_points (%u) > TYP (%u) in EIS at data index %u"
                                % (slave_id, num_eis_points, cfg.NUM_EIS_POINTS_MAX_EXPECTED_ON_SD, data_index))
            # leave num_eis_points unchanged

    is_rt = data_block[7] >> 7
    op_cond = data_block[7] & 0x7F
    soc_nom = int(0)
    if (op_cond > 0) and (op_cond <= cfg.NUM_EIS_SOCS_MAX):
        i_list = op_cond - 1
        soc_nom_table = eis_soc_list[i_list]
        if soc_nom_table is None:
            logging.log.warning(f"Thread ? Slave S%02u - EIS at SoC with index %u: SoC not initialized - at data index "
                                f"%u" % (slave_id, i_list, data_index))  # this usually shouldn't happen, so warn
        else:
            soc_nom = int(soc_nom_table)
    elif op_cond == 0:
        logging.log.warning(f"Thread ? Slave S%02u - manual EIS found, using SoC_nom = 0%% - at data index %u"
                            % (slave_id, data_index))  # this usually shouldn't happen, so warn
    else:
        logging.log.warning(f"Thread ? Slave S%02u - EIS SoC index %u too high, using SoC_nom = 0%% - at data index "
                            f"%u" % (slave_id, op_cond - 1, data_index))  # this usually shouldn't happen, so warn

    cyc_charged = is_rt  # in our experiment, the cell is charged between EIS at RT and discharged between EIS at OT

    uptime_ticks = uint_from_bytes(data_block[8:12])
    ocv_est_avg_V = struct.unpack(">f", data_block[12:16])[0]
    t_avg_degC = struct.unpack(">f", data_block[16:20])[0]
    eis_duration_s = struct.unpack(">f", data_block[20:24])[0]
    z_ref_init_mOhm = struct.unpack(">f", data_block[24:28])[0] * 1000
    z_ref_now_mOhm = struct.unpack(">f", data_block[28:32])[0] * 1000
    soh_imp = struct.unpack(">f", data_block[32:36])[0] * 100

    string_cell_eis_base = format(timestamp, ".3f") + cfg.CSV_SEP + \
                           str(timestamp_origin) + cfg.CSV_SEP + \
                           str(sd_block_id) + cfg.CSV_SEP + \
                           str(cyc_charged) + cfg.CSV_SEP + \
                           str(is_rt) + cfg.CSV_SEP + \
                           str(soc_nom) + cfg.CSV_SEP + \
                           str(valid) + cfg.CSV_SEP + \
                           format(z_ref_init_mOhm, ".3f") + cfg.CSV_SEP + \
                           format(z_ref_now_mOhm, ".3f") + cfg.CSV_SEP + \
                           format(soh_imp, ".6f") + cfg.CSV_SEP +\
                           format(ocv_est_avg_V, ".4f") + cfg.CSV_SEP + \
                           format(t_avg_degC, ".2f") + cfg.CSV_SEP + \
                           format(eis_duration_s, ".2f") + cfg.CSV_SEP + \
                           str(uptime_ticks) + cfg.CSV_SEP

    i_offset = 44
    for i_seq in range(0, num_eis_points):
        freq_Hz = struct.unpack(">f", data_block[i_offset:(i_offset + 4)])[0]
        z_amp_mOhm = struct.unpack(">f", data_block[(i_offset + 4):(i_offset + 8)])[0]
        z_ph_deg = struct.unpack(">f", data_block[(i_offset + 8):(i_offset + 12)])[0]

        string_cell_eis = string_cell_eis_base + \
                          str(i_seq) + cfg.CSV_SEP + \
                          format(freq_Hz, ".4f") + cfg.CSV_SEP + \
                          format(z_amp_mOhm, ".3f") + cfg.CSV_SEP + \
                          format(z_ph_deg, ".3f") + \
                          cfg.CSV_NEWLINE
        file_cell_eis[cell_id].write(string_cell_eis)
        i_offset = i_offset + 12
    return True


def decode_pulse(data_block, sd_block_id, slave_id, timestamp, timestamp_origin, file_cell_pulse, data_index):
    # data_block[4] contains slave_id
    if data_block[4] != slave_id:
        logging.log.warning(f"Thread ? Slave S%02u - found different slave_id %u in PULSE at data index %u"
                            % (slave_id, data_block[4], data_index))  # this usually shouldn't happen, so warn

    cell_id = data_block[5]
    if cell_id >= cfg.NUM_CELLS_PER_SLAVE:
        logging.log.error(f"Thread ? Slave S%02u - found invalid cell_id %u in PULSE at data index %u -> skip"
                          % (slave_id, cell_id, data_index))
        return False  # invalid cell ID -> can't write to file -> skip

    soc_nom = data_block[6]
    # unused = data_block[7]
    uptime_ticks = uint_from_bytes(data_block[8:12])
    cyc_charged = data_block[12] >> 7
    cyc_condition = data_block[12] & 0x7F
    if cyc_condition == 2:
        is_rt = 1
    elif cyc_condition == 3:
        is_rt = 0
    else:
        is_rt = 1
        logging.log.warning(f"Thread ? Slave S%02u - found invalid cyc_condition %u in PULSE at data index %u"
                            % (slave_id, cyc_condition, data_index))  # this usually shouldn't happen, so warn
    age_type = data_block[13]
    age_temp = sint_from_bytes(data_block[14:15])
    age_soc = data_block[15]
    age_chg_rate = data_block[16] / 100
    age_dischg_rate = data_block[17] / 100
    t_avg_degC = struct.unpack(">f", data_block[18:22])[0]
    r_ref_10ms_mOhm = struct.unpack(">f", data_block[22:26])[0] * 1000
    r_ref_1s_mOhm = struct.unpack(">f", data_block[26:30])[0] * 1000
    age_profile = data_block[30]

    string_cell_pulse_base = cfg.CSV_SEP + \
                             str(timestamp_origin) + cfg.CSV_SEP + \
                             str(sd_block_id) + cfg.CSV_SEP + \
                             str(cyc_charged) + cfg.CSV_SEP + \
                             str(is_rt) + cfg.CSV_SEP + \
                             str(soc_nom) + cfg.CSV_SEP + \
                             str(age_type) + cfg.CSV_SEP + \
                             str(age_temp) + cfg.CSV_SEP + \
                             str(age_soc) + cfg.CSV_SEP + \
                             format(age_chg_rate, ".2f") + cfg.CSV_SEP + \
                             format(age_dischg_rate, ".2f") + cfg.CSV_SEP + \
                             str(age_profile) + cfg.CSV_SEP + \
                             format(t_avg_degC, ".2f") + cfg.CSV_SEP + \
                             format(r_ref_10ms_mOhm, ".3f") + cfg.CSV_SEP + \
                             format(r_ref_1s_mOhm, ".3f") + cfg.CSV_SEP + \
                             str(uptime_ticks) + cfg.CSV_SEP
    i_offset = 32
    for i_seq in range(0, cfg.NUM_PULSE_LOG_POINTS):
        v_raw_V = uint_from_bytes(data_block[i_offset:(i_offset + 2)]) / 10000  # 100 µV steps
        i_raw_A = sint_from_bytes(data_block[(i_offset + 2):(i_offset + 4)]) / 1000  # 1 mA steps
        timestamp_use = timestamp + cfg.PULSE_TIME_OFFSET_S[i_seq]
        string_cell_pulse = format(timestamp_use, ".3f") +\
                            string_cell_pulse_base + \
                            str(i_seq) + cfg.CSV_SEP + \
                            format(v_raw_V, ".4f") + cfg.CSV_SEP + \
                            format(i_raw_A, ".4f") + \
                            cfg.CSV_NEWLINE
        file_cell_pulse[cell_id].write(string_cell_pulse)
        i_offset = i_offset + 4
    return True


def decode_slave_config(data_block, sd_block_id, slave_id, data_index):
    if data_block[1] != slave_id:
        logging.log.error(f"Thread ? Slave S%02u - found different slave_id %u in CFG at data index %u"
                          % (slave_id, data_block[4], data_index))

    # open slave config file and write header
    tmp_filename = cfg.CSV_FILENAME_05_RESULT_BASE_SLAVE % (cfg.CSV_FILENAME_05_TYPE_CONFIG, "S", slave_id)
    file_slave_cfg = open(cfg.CSV_RESULT_DIR + tmp_filename, "w")
    file_slave_cfg.write(header_slave_config)

    # write slave config
    board_type_str = f"%#04X = '%s'" % (data_block[0], chr(data_block[0]))
    my_ip_adr_str = f"%u.%u.%u.%u" % (data_block[2], data_block[3], data_block[4], data_block[5])
    server_ip_adr_str = f"%u.%u.%u.%u" % (data_block[6], data_block[7], data_block[8], data_block[9])
    my_mac_adr_str = f"%02X:%02X:%02X:%02X:%02X:%02X"\
                 % (data_block[10], data_block[11], data_block[12], data_block[13], data_block[14], data_block[15])
    num_eis = data_block[16]
    eis_soc = [data_block[17], data_block[18], data_block[19], data_block[20], data_block[21],
               data_block[22], data_block[23], data_block[24], data_block[25], data_block[26]]

    string_slave_config = str(sd_block_id) + cfg.CSV_SEP + \
                          str(slave_id) + cfg.CSV_SEP + \
                          board_type_str + cfg.CSV_SEP + \
                          my_ip_adr_str + cfg.CSV_SEP + \
                          server_ip_adr_str + cfg.CSV_SEP + \
                          my_mac_adr_str + cfg.CSV_SEP + \
                          str(num_eis) + cfg.CSV_SEP + \
                          str(eis_soc[0]) + cfg.CSV_SEP + \
                          str(eis_soc[1]) + cfg.CSV_SEP + \
                          str(eis_soc[2]) + cfg.CSV_SEP + \
                          str(eis_soc[3]) + cfg.CSV_SEP + \
                          str(eis_soc[4]) + cfg.CSV_SEP + \
                          str(eis_soc[5]) + cfg.CSV_SEP + \
                          str(eis_soc[6]) + cfg.CSV_SEP + \
                          str(eis_soc[7]) + cfg.CSV_SEP + \
                          str(eis_soc[8]) + cfg.CSV_SEP + \
                          str(eis_soc[9]) + \
                          cfg.CSV_NEWLINE
    file_slave_cfg.write(string_slave_config)

    # close slave config file
    file_slave_cfg.close()

    for i_cell in range(0, NUM_CELLS_PER_SLAVE):
        i_block_start = 28 + i_cell * 40
        i_block_end = i_block_start + 40
        cell_data_block = data_block[i_block_start:i_block_end]

        # open cell config file and write header
        parameter_set_id = cfg.PARAMETER_SET_ID_FROM_SXX_CXX[slave_id][i_cell]
        parameter_set_cell_nr = cfg.PARAMETER_SET_CELL_NR_FROM_SXX_CXX[slave_id][i_cell]
        tmp_filename = cfg.CSV_FILENAME_05_RESULT_BASE_CELL % (cfg.CSV_FILENAME_05_TYPE_CONFIG, parameter_set_id,
                                                               parameter_set_cell_nr, slave_id, i_cell)
        file_cell_cfg = open(cfg.CSV_RESULT_DIR + tmp_filename, "w")
        file_cell_cfg.write(header_cell_config)

        # write cell config
        cell_used = cell_data_block[0]
        cell_type = cell_data_block[1]
        age_type = cell_data_block[2]
        age_temp = sint_from_bytes(cell_data_block[3:4])
        age_soc = cell_data_block[4]
        age_chg_rate = cell_data_block[5] / 100
        age_dischg_rate = cell_data_block[6] / 100
        t_sns_type = cell_data_block[7]
        V_max_cyc_V = uint_from_bytes(cell_data_block[8:10]) / 1000
        V_min_cyc_V = uint_from_bytes(cell_data_block[10:12]) / 1000
        V_max_cu_V = uint_from_bytes(cell_data_block[12:14]) / 1000
        V_min_cu_V = uint_from_bytes(cell_data_block[14:16]) / 1000
        I_chg_max_cyc_A = uint_from_bytes(cell_data_block[16:18]) / 1000
        I_dischg_max_cyc_A = -uint_from_bytes(cell_data_block[18:20]) / 1000
        I_chg_max_cu_A = uint_from_bytes(cell_data_block[20:22]) / 1000
        I_dischg_max_cu_A = -uint_from_bytes(cell_data_block[22:24]) / 1000
        I_chg_cutoff_cyc_A = uint_from_bytes(cell_data_block[24:26]) / 1000
        I_dischg_cutoff_cyc_A = -uint_from_bytes(cell_data_block[26:28]) / 1000
        I_chg_cutoff_cu_A = uint_from_bytes(cell_data_block[28:30]) / 1000
        I_dischg_cutoff_cu_A = -uint_from_bytes(cell_data_block[30:32]) / 1000
        I_chg_pulse_cu_A = uint_from_bytes(cell_data_block[32:34]) / 1000
        I_dischg_pulse_cu_A = -uint_from_bytes(cell_data_block[34:36]) / 1000
        age_profile = cell_data_block[36]

        string_cell_config = str(sd_block_id) + cfg.CSV_SEP + \
                             str(slave_id) + cfg.CSV_SEP + \
                             str(i_cell) + cfg.CSV_SEP + \
                             str(parameter_set_id) + cfg.CSV_SEP + \
                             str(parameter_set_cell_nr) + cfg.CSV_SEP + \
                             str(cell_used) + cfg.CSV_SEP + \
                             str(cell_type) + cfg.CSV_SEP + \
                             str(t_sns_type) + cfg.CSV_SEP + \
                             str(age_type) + cfg.CSV_SEP + \
                             str(age_temp) + cfg.CSV_SEP + \
                             str(age_soc) + cfg.CSV_SEP + \
                             str(age_chg_rate) + cfg.CSV_SEP + \
                             str(age_dischg_rate) + cfg.CSV_SEP + \
                             str(age_profile) + cfg.CSV_SEP + \
                             str(V_max_cyc_V) + cfg.CSV_SEP + \
                             str(V_min_cyc_V) + cfg.CSV_SEP + \
                             str(V_max_cu_V) + cfg.CSV_SEP + \
                             str(V_min_cu_V) + cfg.CSV_SEP + \
                             str(I_chg_max_cyc_A) + cfg.CSV_SEP + \
                             str(I_dischg_max_cyc_A) + cfg.CSV_SEP + \
                             str(I_chg_max_cu_A) + cfg.CSV_SEP + \
                             str(I_dischg_max_cu_A) + cfg.CSV_SEP + \
                             str(I_chg_cutoff_cyc_A) + cfg.CSV_SEP + \
                             str(I_dischg_cutoff_cyc_A) + cfg.CSV_SEP + \
                             str(I_chg_cutoff_cu_A) + cfg.CSV_SEP + \
                             str(I_dischg_cutoff_cu_A) + cfg.CSV_SEP + \
                             str(I_chg_pulse_cu_A) + cfg.CSV_SEP + \
                             str(I_dischg_pulse_cu_A) + \
                             cfg.CSV_NEWLINE
        file_cell_cfg.write(string_cell_config)

        # close cell config file
        file_cell_cfg.close()

    eis_soc_list = [None] * cfg.NUM_EIS_SOCS_MAX
    for i_eis in range(0, num_eis):
        eis_soc_list[i_eis] = eis_soc[i_eis]

    return [True, eis_soc_list]


def decode_tmgmt_config(data_block, sd_block_id, slave_id, data_index):
    if data_block[1] != slave_id:
        logging.log.error(f"Thread ? Slave T%02u - found different slave_id %u in CFG at data index %u"
                          % (slave_id, data_block[4], data_index))

    # open tmgmt config file and write header
    tmp_filename = cfg.CSV_FILENAME_05_RESULT_BASE_SLAVE % (cfg.CSV_FILENAME_05_TYPE_CONFIG, "T", slave_id)
    file_tmgmt_cfg = open(cfg.CSV_RESULT_DIR + tmp_filename, "w")
    file_tmgmt_cfg.write(header_tmgmt_config)

    # write tmgmt config
    board_type_str = f"%#04X = '%s'" % (data_block[0], chr(data_block[0]))
    my_ip_adr_str = f"%u.%u.%u.%u" % (data_block[2], data_block[3], data_block[4], data_block[5])
    server_ip_adr_str = f"%u.%u.%u.%u" % (data_block[6], data_block[7], data_block[8], data_block[9])
    my_mac_adr_str = f"%02X:%02X:%02X:%02X:%02X:%02X"\
                 % (data_block[10], data_block[11], data_block[12], data_block[13], data_block[14], data_block[15])

    string_tmgmt_config = str(sd_block_id) + cfg.CSV_SEP + \
                          str(slave_id) + cfg.CSV_SEP + \
                          board_type_str + cfg.CSV_SEP + \
                          my_ip_adr_str + cfg.CSV_SEP + \
                          server_ip_adr_str + cfg.CSV_SEP + \
                          my_mac_adr_str + \
                          cfg.CSV_NEWLINE
    file_tmgmt_cfg.write(string_tmgmt_config)

    # close tmgmt config file
    file_tmgmt_cfg.close()

    for i_pool in range(0, NUM_POOLS_PER_TMGMT):
        i_block_start = 28 + i_pool * 40
        i_block_end = i_block_start + 40
        pool_data_block = data_block[i_block_start:i_block_end]

        # open pool config file and write header
        tmp_filename = cfg.CSV_FILENAME_05_RESULT_BASE_POOL % (cfg.CSV_FILENAME_05_TYPE_CONFIG, slave_id, i_pool)
        file_pool_cfg = open(cfg.CSV_RESULT_DIR + tmp_filename, "w")
        file_pool_cfg.write(header_pool_config)

        # write pool config
        pool_used = pool_data_block[0] >> 4
        is_in_cold_circuit = pool_data_block[1]
        t_operation_degC = sint_from_bytes(pool_data_block[2:4]) / 100
        t_checkup_degC = sint_from_bytes(pool_data_block[4:6]) / 100

        string_pool_config = str(sd_block_id) + cfg.CSV_SEP + \
                             str(slave_id) + cfg.CSV_SEP + \
                             str(i_pool) + cfg.CSV_SEP + \
                             str(pool_used) + cfg.CSV_SEP + \
                             str(is_in_cold_circuit) + cfg.CSV_SEP + \
                             str(t_operation_degC) + cfg.CSV_SEP + \
                             str(t_checkup_degC) + \
                             cfg.CSV_NEWLINE
        file_pool_cfg.write(string_pool_config)

        # close pool config file
        file_pool_cfg.close()
    return True
