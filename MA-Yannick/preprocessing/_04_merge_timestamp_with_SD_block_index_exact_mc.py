import pandas
import pandas as pd
import matplotlib.pyplot as plt
import config_preprocessing as cfg
import multiprocessing
from datetime import datetime, timedelta
import os
import re
import config_logging as logging
import gc


# config
INFLUX_SEARCH_LEFT_AT_BEGINNING_SECONDS = (20 * 60)  # 20*60 -> search up to 15 minutes before first matched timestamp
# if we find a previously unmatched but now matching uptime value, e.g. slave 4:
# http://ipepdvmssqldb2.ipe.kit.edu:3000/d/YxIcA31nk/slave-log?orgId=1&var-slave_id=4&var-my_interval=2s&viewPanel=6&from=1665597541502&to=1665598542578
MAX_NUM_NAN_SIMPLE_INTERPOLATION = 200  # if 200 or fewer values are missing, interpolate them without using Influx
MAX_STM_TICK_SLOPE_DEVIATION = 0.2  # 0.2 -> 20%, maximum deviation from stm tick slope ...
# ... (if higher, fall back to DELTA_T_STM_TICK )
MIN_TIME_DEVIATION_FILL_WITH_INFLUX = 0.50  # 0.50 -> -50%, maximum negative time deviation ...
MAX_TIME_DEVIATION_FILL_WITH_INFLUX = 1.10  # 1.10 -> +10%, maximum positive time deviation ...
# ... if previously unmatched SD and Influx data is available and their expected delta_timestamps vary less than that
# MIN_TIME_DEVIATION_FILL_SD = 0.95  # 0.95 -> -5%, maximum negative time deviation ...
MAX_TIME_DEVIATION_FILL_SD = 1.05  # 1.05 -> +5%, maximum positive time deviation ...
# ... if unmatchable SD (but no Influx) data is available and their expected delta_timestamps vary less than that
NUM_COMPARE_AND_COPY_BLOCKS = 500  # when interpolating & influx data is available, compare & copy with this block size
NUM_COMPARE_AND_COPY_BLOCKS_DECR = 50  # ...if unequal, decrement block size by this. Fallback: copy individually
DELTA_STM_TICKS_SD_MAX_BEFORE_RESYNC = 1000  # if the SD uptime rises but deviates by more than this, try to resync ...
# ... with Influx because there might be a reboot in between (e.g. Slave 4, 19.10.22, around 21:22)
MIN_DELTA_T_WARNING = cfg.DELTA_T_LOG / 4  # output a warning if the final minimum delta_timestamp is smaller than this
INFLUX_UPTIME_MIN_USE = cfg.UPTIME_MIN_REBOOT_THRESHOLD  # don't use Influx values below this threshold
NUM_MULTIPLE_MATCHES_INFLUX = 5  # if multiple SD/Influx matches are needed for reliability, use this number of matches


# constants
DELTA_UPTIME_REBOOT = cfg.UPTIME_MIN_REBOOT_THRESHOLD
DELTA_UPTIME_REBOOT_SMALL = cfg.UPTIME_MIN_REBOOT_SMALL_THRESHOLD
pd.options.mode.chained_assignment = None  # default="warn"
# timestamp_row = cfg.CSV_FILE_HEADER_UNIX_TIMESTAMP
# sd_block_id_row = cfg.CSV_FILE_HEADER_SD_BLOCK_ID
# uptime_row = cfg.CSV_FILE_HEADER_STM_TIM_UPTIME
NUMBER_OF_PROCESSORS_TO_USE = multiprocessing.cpu_count()
task_queue = multiprocessing.Queue()
report_queue = multiprocessing.Queue()


def run():
    start_timestamp = datetime.now()
    logging.log.info(os.path.basename(__file__))

    # find files
    slaves_sd_complete = []
    slaves_sd_with_timestamp = []
    slaves_influx = []
    with os.scandir(cfg.CSV_WORKING_DIR) as iterator:
        re_str_sd_complete = cfg.CSV_FILENAME_01_SD_UPTIME.replace("%s%02u", "([ST])(\d+)")
        re_pat_sd_complete = re.compile(re_str_sd_complete)
        re_str_sd_with_timestamp = cfg.CSV_FILENAME_03_TIMESTAMP_AND_SD_BLOCK_ID_PEAKS.replace("%s%02u", "([ST])(\d+)")
        re_pat_sd_with_timestamp = re.compile(re_str_sd_with_timestamp)
        re_str_influx = cfg.CSV_FILENAME_02_INFLUX_UPTIME.replace("%s%02u", "([ST])(\d+)")
        re_pat_influx = re.compile(re_str_influx)
        for entry in iterator:
            re_match_sd_complete = re_pat_sd_complete.fullmatch(entry.name)
            if re_match_sd_complete:
                slave_type = re_match_sd_complete.group(1)
                slave_id = int(re_match_sd_complete.group(2))
                slave = {"id": slave_id, "type": slave_type}
                slaves_sd_complete.append(slave)
            else:
                re_match_sd_with_timestamp = re_pat_sd_with_timestamp.fullmatch(entry.name)
                if re_match_sd_with_timestamp:
                    slave_type = re_match_sd_with_timestamp.group(1)
                    slave_id = int(re_match_sd_with_timestamp.group(2))
                    slave = {"id": slave_id, "type": slave_type}
                    slaves_sd_with_timestamp.append(slave)
                else:
                    re_match_influx = re_pat_influx.fullmatch(entry.name)
                    if re_match_influx:
                        slave_type = re_match_influx.group(1)
                        slave_id = int(re_match_influx.group(2))
                        slave = {"id": slave_id, "type": slave_type}
                        slaves_influx.append(slave)

    # only used slaves of which a .csv from the SD and influx exists
    slaves_string = []
    for slave_sd_complete in slaves_sd_complete:
        for slave_sd_with_timestamp in slaves_sd_with_timestamp:
            for slave_influx in slaves_influx:
                if (slave_sd_complete["id"] == slave_sd_with_timestamp["id"] == slave_influx["id"])\
                        and (slave_sd_complete["type"] == slave_sd_with_timestamp["type"] == slave_influx["type"]):
                    task_queue.put(slave_sd_complete)
                    slaves_string.append("%s%02u" % (slave_sd_complete["type"], slave_sd_complete["id"]))

    logging.log.info("Found .csv files for slaves: %s" % slaves_string)

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

        if slave_report["msg_nans"] is not None:
            logging.log.error(slave_report["msg_nans"])

        if slave_report["level"] == logging.ERROR:
            logging.log.error(slave_report["msg"])
        elif slave_report["level"] == logging.WARNING:
            logging.log.warning(slave_report["msg"])
        elif slave_report["level"] == logging.INFO:
            logging.log.info(slave_report["msg"])

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
        type_id = queue_entry["type"]
        logging.log.info("Thread %u Slave %s%02u - start assigning exact timestamps to SD data"
                         % (processor_number, type_id, slave_id))

        # read SD data of slave n# in chunks
        filename_01 = cfg.CSV_WORKING_DIR + (cfg.CSV_FILENAME_01_SD_UPTIME % (type_id, slave_id))
        sd_df = pd.read_csv(filename_01, header=0, sep=cfg.CSV_SEP)

        # read the fixed timestamps from the 3rd script:
        # Peak values (highest uptime counter) are the known timestamp references
        filename_03 = cfg.CSV_WORKING_DIR + (cfg.CSV_FILENAME_03_TIMESTAMP_AND_SD_BLOCK_ID_PEAKS % (type_id, slave_id))
        timestamp_df = pd.read_csv(filename_03, sep=cfg.CSV_SEP, header=0, index_col=0)
        timestamp_df["unixtimestamp_ms"] = (timestamp_df.unixtimestamp * 1000).astype("int64")
        # timestamp_df.unixtimestamp = timestamp_df.unixtimestamp.astype(int)

        logging.log.debug("Thread %u Slave %s%02u - timestamp_df (1):\n%s"
                          % (processor_number, type_id, slave_id, timestamp_df))

        # Merge the timestamp df with the datadf, so that the few peak values of the sd card get a timestamp
        merged_df = pd.merge(left=sd_df, right=timestamp_df, how="left", on="SD_block_ID")
        # merged_df.to_csv("temp.csv")

        # Dataframe with only sd card rows with a fixed timestamp (peak values)
        # The index of this df is the row number of the sdcard dataframe
        values_with_time_stamp_df = merged_df[merged_df.unixtimestamp_ms.notnull()]

        values_with_time_stamp_df["unixtimestamp_str"] =\
            pandas.to_datetime(values_with_time_stamp_df.unixtimestamp_ms, unit="ms")

        logging.log.debug("Thread %u Slave %s%02u - values_with_time_stamp_df:\n%s"
                          % (processor_number, type_id, slave_id, values_with_time_stamp_df.to_string()))

        filename_02 = cfg.CSV_WORKING_DIR + (cfg.CSV_FILENAME_02_INFLUX_UPTIME % (type_id, slave_id))
        influx_df = pd.read_csv(filename_02, sep=cfg.CSV_SEP)

        # assign timestamps to interim influx values
        i_session_start_min_ifx = 0
        i_session_start_min_sd = 0
        sd_df_merged = sd_df
        # itertuples seems to be much, much faster (>10x) than iterrows (see links below)
        # if the index is needed, use iterrows instead of for i in range ... with iloc/loc (>10x faster)
        # so itertuples is > 100x faster than for i in range ... with iloc/loc
        # https://stackoverflow.com/a/24871316/2738240
        # https://github.com/dimgold/Datathon-TAU/blob/master/7.6%20Pandas%20iterations%20test/Pandas_Iterations.ipynb
        # for index, row in values_with_time_stamp_df.iterrows():
        for row in values_with_time_stamp_df.itertuples():
            # Here, a 'session' is defined as the runtime of a slave board.
            # If the slave board reboots, a new session starts.

            # 1. in the complete influx data set, find first item of current session
            #    a. find last unixtimestamp of session (as in values_with_time_stamp_df) in complete influx data set
            #    b. go to left until the first value arrives that is significantly larger
            # 2. in the complete SD data set, find the first item of the current session
            #    a. find last unixtimestamp of session (as in values_with_time_stamp_df) in complete SD data set
            #    b. go to left until the first value arrives that is significantly larger

            # influx ---------------------------------------------------------------------------------------------------
            i_session_end_ifx = influx_df.unixtimestamp[influx_df.unixtimestamp == row.unixtimestamp].index.values[0]
            logging.log.debug("Thread %u Slave %s%02u - i_session_end_ifx:\n%s"
                              % (processor_number, type_id, slave_id, i_session_end_ifx))

            # select the window from the influx data set in which the current session starts ()
            ifx_sess_win_df = influx_df.iloc[i_session_start_min_ifx:(i_session_end_ifx + 1), :]

            # delete rows with small uptime -> they cause problems if cycler was rebooted multiple times close in time
            ifx_sess_win_df = ifx_sess_win_df[ifx_sess_win_df.stm_tim_uptime > INFLUX_UPTIME_MIN_USE]

            logging.log.debug("Thread %u Slave %s%02u - ifx_sess_win_df (1):\n%s"
                              % (processor_number, type_id, slave_id, ifx_sess_win_df))
            ifx_sess_win_df["min"] = ifx_sess_win_df.stm_tim_uptime[
                (ifx_sess_win_df.stm_tim_uptime.shift(-1) > ifx_sess_win_df.stm_tim_uptime)
                & (ifx_sess_win_df.stm_tim_uptime.shift(1) > (ifx_sess_win_df.stm_tim_uptime + DELTA_UPTIME_REBOOT))
                ]
            ifx_sess_win_df_min = ifx_sess_win_df.loc[~pd.isna(ifx_sess_win_df["min"]), "min"]
            logging.log.debug("Thread %u Slave %s%02u - ifx_sess_win_df_min:\n%s"
                              % (processor_number, type_id, slave_id, ifx_sess_win_df_min))
            i_session_start_ifx = i_session_start_min_ifx
            ifx_sess_win_df_min_length = len(ifx_sess_win_df_min.index)
            if ifx_sess_win_df_min_length > 0:
                # one or more minima found in between - select the last minimum as the start of the relevant session
                # (this occurs if the data frame starts with small sessions that were ignored in the peak detection)
                i_session_start_ifx = ifx_sess_win_df_min.index[-1]
            # else: no minima found - the first index is the start of the current session (ifx_sess_win_df is unchanged)

            ifx_sess_win_df = influx_df.iloc[i_session_start_ifx:(i_session_end_ifx + 1), :]
            ifx_sess_win_df["timestamp_origin"] = row.timestamp_origin

            # delete rows again
            ifx_sess_win_df = ifx_sess_win_df[ifx_sess_win_df.stm_tim_uptime > INFLUX_UPTIME_MIN_USE]

            # ifx_sess_win_df.drop("min", inplace=True, axis=1)  # drop "min" column
            # ifx_sess_win_df["unixtimestamp_str"] =\
            #     pandas.to_datetime(ifx_sess_win_df.unixtimestamp, unit="s")
            logging.log.debug("Thread %u Slave %s%02u - ifx_sess_win_df (2):\n%s"
                              % (processor_number, type_id, slave_id, ifx_sess_win_df))

            # SD -------------------------------------------------------------------------------------------------------
            i_session_end_sd = sd_df.SD_block_ID[sd_df.SD_block_ID == row.SD_block_ID].index.values[0]
            logging.log.debug("Thread %u Slave %s%02u - i_session_end_sd: %u"
                              % (processor_number, type_id, slave_id, i_session_end_sd))

            # select the window from the SD data set in which the current session starts ()
            sd_sess_win_df = sd_df.iloc[i_session_start_min_sd:(i_session_end_sd + 1), :]
            logging.log.debug("Thread %u Slave %s%02u - sd_sess_win_df (1):\n%s"
                              % (processor_number, type_id, slave_id, sd_sess_win_df))
            # sd_sess_win_df["min"] = sd_sess_win_df.stm_tim_uptime[
            #     (sd_sess_win_df.stm_tim_uptime.shift(-1) > sd_sess_win_df.stm_tim_uptime)
            #     & (sd_sess_win_df.stm_tim_uptime.shift(1) >
            #        (sd_sess_win_df.stm_tim_uptime + DELTA_UPTIME_REBOOT_SMALL))
            #     ]
            # use DELTA_UPTIME_REBOOT_SMALL instead of DELTA_UPTIME_REBOOT for SD card
            sd_sess_win_df["min"] = sd_sess_win_df.stm_tim_uptime[
                (sd_sess_win_df.stm_tim_uptime.shift(-1) > sd_sess_win_df.stm_tim_uptime)
                & (sd_sess_win_df.stm_tim_uptime.shift(1) > sd_sess_win_df.stm_tim_uptime)
                ]
            sd_sess_win_df_min = sd_sess_win_df.loc[~pd.isna(sd_sess_win_df["min"]), "min"]
            logging.log.debug("Thread %u Slave %s%02u - sd_sess_win_df_min:\n%s"
                              % (processor_number, type_id, slave_id, sd_sess_win_df_min))
            i_session_start_sd = i_session_start_min_sd
            sd_sess_win_df_min_length = len(sd_sess_win_df_min.index)
            if sd_sess_win_df_min_length > 0:
                # one or more minima found in between - select the last minimum as the start of the relevant session
                # (this occurs if the data frame starts with small sessions that were ignored in the peak detection)
                i_session_start_sd = sd_sess_win_df_min.index[-1]
            # else: no minima found - the first index is the start of the current session (sd_sess_win_df is unchanged)
            sd_sess_win_df = sd_df.iloc[i_session_start_sd:(i_session_end_sd + 1), :]
            # sd_sess_win_df.drop("min", inplace=True, axis=1)  # drop "min" column
            logging.log.debug("Thread %u Slave %s%02u - sd_sess_win_df (2):\n%s"
                              % (processor_number, type_id, slave_id, sd_sess_win_df))

            merged_df = pd.merge(
                left=sd_sess_win_df,
                right=ifx_sess_win_df,
                on="stm_tim_uptime",
                left_index=False,
                right_index=False,
                how="outer",
            )
            # TODO: what happens if uptime values occur twice in InfluxDB because they were resent by the Raspi/router?
            # e.g. Slave 14, 07.12.2022, 04:42:08 and 04:42:22

            logging.log.debug("Thread %u Slave %s%02u - merged_df:\n%s"
                              % (processor_number, type_id, slave_id, merged_df))

            ifx_sess_length = len(ifx_sess_win_df.index)
            if ifx_sess_length > 0:
                # noinspection PyArgumentList
                nan_count = merged_df.isna().sum()
                # use iat, because first index is not 0 and last index can't be accessed with -1
                start_date = pandas.to_datetime(ifx_sess_win_df.unixtimestamp.iat[0], unit="s")
                end_date = pandas.to_datetime(ifx_sess_win_df.unixtimestamp.iat[-1], unit="s")
                logging.log.info("Thread %u Slave %s%02u - Session from:   %s   to   %s\n"
                                 "   Missing influx rows: %u\n"
                                 "   Missing SD rows    : %u"
                                 % (processor_number, type_id, slave_id, start_date, end_date,
                                    nan_count.unixtimestamp, nan_count.SD_block_ID))
            else:
                logging.log.info("Thread %u Slave %s%02u - Empty session" % (processor_number, type_id, slave_id))

            sd_df_merged = pd.merge(
                left=sd_df_merged,
                right=merged_df,
                on="SD_block_ID",
                left_index=False,
                right_index=False,
                how="left",
                # how="outer", if we would merge influx and SD data logs, we could also use "outer".
                # Assume nothing special happens if the SD card data is missing, so we just use the SD card data
            )
            sd_df_merged.drop("stm_tim_uptime_y", inplace=True, axis=1)  # drop "stm_tim_uptime_y" column
            sd_df_merged.rename(columns={"stm_tim_uptime_x": "stm_tim_uptime"}, inplace=True)
            if "unixtimestamp_x" in sd_df_merged.columns:
                sd_df_merged["unixtimestamp"] =\
                    sd_df_merged.unixtimestamp_x.combine_first(sd_df_merged.unixtimestamp_y)
                sd_df_merged.drop("unixtimestamp_x", inplace=True, axis=1)
                sd_df_merged.drop("unixtimestamp_y", inplace=True, axis=1)
                # sd_df_merged.rename(columns={"unixtimestamp_x": "unixtimestamp"}, inplace=True)

                sd_df_merged["timestamp_origin"] =\
                    sd_df_merged.timestamp_origin_x.combine_first(sd_df_merged["timestamp_origin_y"])
                sd_df_merged.drop("timestamp_origin_x", inplace=True, axis=1)
                sd_df_merged.drop("timestamp_origin_y", inplace=True, axis=1)

            i_session_start_min_ifx = i_session_end_ifx + 1
            i_session_start_min_sd = i_session_end_sd + 1

        # sd_df_nan = sd_df_merged[pd.isna(sd_df_merged.unixtimestamp)]
        # logging.log.debug("Thread %u Slave %s%02u - sd_df_nan:\n%s"
        #                   % (processor_number, type_id, slave_id, sd_df_nan))

        sd_df_merged.index.name = "index"

        # find first entry where timestamp is not nan
        # first_valid_index = -1
        sd_df_merged_timestamp_not_nan = sd_df_merged.unixtimestamp[~pd.isna(sd_df_merged.unixtimestamp)]
        if sd_df_merged_timestamp_not_nan.shape[0] == 0:
            first_valid_index = -1
        else:
            first_valid_index = sd_df_merged_timestamp_not_nan.index[0]

        # for index_sd_df_merged, row_sd_df_merged in sd_df_merged.iterrows():  # we need the index, so no itertuples
        #     if not pd.isna(row_sd_df_merged.unixtimestamp):
        #         first_valid_index = index_sd_df_merged
        #         break

        # collect garbage to (hopefully?) free memory
        # https://stackoverflow.com/questions/39100971/how-do-i-release-memory-used-by-a-pandas-dataframe
        sd_df = ""
        merged_df = ""
        ifx_sess_win_df = ""
        sd_sess_win_df = ""
        timestamp_df = ""
        values_with_time_stamp_df = ""
        sd_df_merged_timestamp_not_nan = ""
        del sd_df
        del merged_df
        del ifx_sess_win_df
        del sd_sess_win_df
        del timestamp_df
        del values_with_time_stamp_df
        del sd_df_merged_timestamp_not_nan
        gc.collect()

        report_msg_nans = None
        if first_valid_index < 0:
            # no valid timestamp found?
            report_msg = f"Thread %u Slave %s%02u - No valid timestamp found!" % (processor_number, type_id, slave_id)
            report_level = logging.ERROR
            # logging.log.error(report_msg)
        else:
            logging.log.info("Thread %u Slave %s%02u - Extrapolating beginning..."
                                % (processor_number, type_id, slave_id))
            # fix beginning:
            # e.g. slave 4:
            #   http://ipepdvmssqldb2.ipe.kit.edu:3000/d/YxIcA31nk/slave-log?orgId=1&var-slave_id=4&var-my_interval=2s&viewPanel=6&from=1665597541502&to=1665598542578
            # - go down to minima using delta_uptime * cfg.DELTA_T_STM_TICK [cfg.TimestampOrigin.EXTRAPOLATED]
            # - if there is another (small) peak, search in influx raw data if similar values can be found within a
            #   20-minute timeframe left --> match/merge them [cfg.TimestampOrigin.INFLUX_ESTIMATED]
            # - else go down with 2 second steps (extrapolate with cfg.DELTA_T_LOG) [cfg.TimestampOrigin.EXTRAPOLATED]
            first_matched_timestamp = sd_df_merged.unixtimestamp.iat[first_valid_index]
            min_search_timestamp = first_matched_timestamp - INFLUX_SEARCH_LEFT_AT_BEGINNING_SECONDS
            last_timestamp = first_matched_timestamp
            last_uptime = sd_df_merged.stm_tim_uptime.iat[first_valid_index]
            for i in range(first_valid_index-1, -1, -1):
                # count down from (first_valid_index-1) to 0 --> iterate through all nan's at beginning of sd_df_merged
                new_uptime = sd_df_merged.stm_tim_uptime.iat[i]
                uptime_diff = new_uptime - last_uptime
                if uptime_diff < 0:  # we go 'down the hill' backwards in time
                    # --> previously we found no influx data, so go down to minimum using uptime
                    last_timestamp = last_timestamp + uptime_diff * cfg.DELTA_T_STM_TICK
                    sd_df_merged.unixtimestamp.iat[i] = last_timestamp
                    sd_df_merged.timestamp_origin.iat[i] = cfg.TimestampOrigin.EXTRAPOLATED
                elif uptime_diff > DELTA_UPTIME_REBOOT:  # there was a reboot in between
                    # new maximum --> find it in influx raw data (limited to a certain timeframe of interest)
                    influx_df_start = influx_df[(influx_df.unixtimestamp > min_search_timestamp)
                                                & (influx_df.unixtimestamp < last_timestamp)]
                    k_max = influx_df_start.shape[0] - 1
                    k_match = -1
                    # influx_df_start.stm_tim_uptime[influx_df_start.stm_tim_uptime == new_uptime]
                    tmp_df = influx_df_start.stm_tim_uptime.reset_index(drop=True)
                    match_df = tmp_df[tmp_df == new_uptime].index
                    if match_df.shape[0] > 0:
                        k_match = match_df[0]
                    # for k in range(k_max, -1, -1):
                    #     if new_uptime == influx_df_start.stm_tim_uptime.iat[k]:
                    #         k_match = k  # we found the uptime of the SD in Influx
                    #         break
                    if k_match >= 0:
                        # found a match in the influx data right before the first matched timestamp --> try to join
                        i_x = i
                        find_influx_match = False
                        while k_match >= 0:
                            # there is more data from Influx
                            uptime_sd = sd_df_merged.stm_tim_uptime.iat[i_x]
                            uptime_influx = influx_df_start.stm_tim_uptime.iat[k_match]
                            if uptime_sd == uptime_influx:  # exact match --> copy and decrement both counters
                                last_timestamp = influx_df_start.unixtimestamp.iat[k_match]
                                sd_df_merged.unixtimestamp.iat[i_x] = last_timestamp
                                sd_df_merged.timestamp_origin.iat[i_x] = cfg.TimestampOrigin.INFLUX_ESTIMATED
                                i_x = i_x - 1
                                k_match = k_match - 1
                                find_influx_match = False
                            elif (((not find_influx_match)
                                   and (uptime_influx > (uptime_sd + DELTA_UPTIME_REBOOT)))
                                  or (uptime_influx < uptime_sd)):
                                # *new* peak in influx  -or-  SD uptime to high --> decrement SD index
                                # we also need to assign a time to the SD card data:
                                uptime_diff = (uptime_sd - last_uptime)
                                if uptime_diff < 0:
                                    # go down to minimum using uptime
                                    last_timestamp = last_timestamp + uptime_diff * cfg.DELTA_T_STM_TICK
                                    sd_df_merged.unixtimestamp.iat[i_x] = last_timestamp
                                    sd_df_merged.timestamp_origin.iat[i_x] = cfg.TimestampOrigin.EXTRAPOLATED
                                elif uptime_influx > (uptime_sd + DELTA_UPTIME_REBOOT):
                                    k_match = k_match - 1  # influx uptime too high (new peak) -> decrement influx index
                                    find_influx_match = True  # Next: decrement k_match until we find a match
                                    continue  # do not decrement i_x, do not assign last_uptime
                                else:
                                    # --> interpolate with fixed timestamp
                                    if uptime_diff > DELTA_UPTIME_REBOOT:  # there was a reboot in between
                                        last_timestamp = (last_timestamp - cfg.DELTA_T_REBOOT
                                                          - last_uptime * cfg.DELTA_T_STM_TICK)  # optional
                                    else:  # most likely, there was no reboot in between
                                        last_timestamp = last_timestamp - cfg.DELTA_T_LOG
                                    sd_df_merged.unixtimestamp.iat[i_x] = last_timestamp
                                    sd_df_merged.timestamp_origin.iat[i_x] = cfg.TimestampOrigin.EXTRAPOLATED
                                i_x = i_x - 1
                            else:  # here: equal to "elif uptime_sd < uptime_influx" --> influx uptime too high
                                k_match = k_match - 1  # --> decrement influx index until we find a match
                                continue  # do not assign last_uptime
                            last_uptime = uptime_sd
                        # done extrapolating the beginning from Influx data

                        # there is probably more data on the SD though! --> extrapolate with methods from above
                        while i_x >= 0:
                            uptime_sd = sd_df_merged.stm_tim_uptime.iat[i_x]
                            uptime_diff = (uptime_sd - last_uptime)
                            if uptime_diff < 0:
                                # go down to minimum using uptime
                                last_timestamp = last_timestamp + uptime_diff * cfg.DELTA_T_STM_TICK
                                sd_df_merged.unixtimestamp.iat[i_x] = last_timestamp
                                sd_df_merged.timestamp_origin.iat[i_x] = cfg.TimestampOrigin.EXTRAPOLATED
                            else:
                                # --> interpolate with fixed timestamp
                                if uptime_diff > DELTA_UPTIME_REBOOT:  # there was a reboot in between
                                    last_timestamp = (last_timestamp - cfg.DELTA_T_REBOOT
                                                      - last_uptime * cfg.DELTA_T_STM_TICK)  # optional
                                else:  # most likely, there was no reboot in between
                                    last_timestamp = last_timestamp - cfg.DELTA_T_LOG
                                sd_df_merged.unixtimestamp.iat[i_x] = last_timestamp
                                sd_df_merged.timestamp_origin.iat[i_x] = cfg.TimestampOrigin.EXTRAPOLATED
                            i_x = i_x - 1
                            last_uptime = uptime_sd
                        break
                    else:
                        # there was a reboot, but we found no Influx match --> interpolate with fixed timestamp
                        last_timestamp = (last_timestamp - cfg.DELTA_T_REBOOT
                                          - last_uptime * cfg.DELTA_T_STM_TICK)  # optional
                        sd_df_merged.unixtimestamp.iat[i] = last_timestamp
                        sd_df_merged.timestamp_origin.iat[i] = cfg.TimestampOrigin.EXTRAPOLATED
                else:
                    # not really a new maximum (wrong Ethernet message order / very short session?)
                    # --> interpolate with fixed timestamp
                    last_timestamp = last_timestamp - cfg.DELTA_T_LOG
                    sd_df_merged.unixtimestamp.iat[i] = last_timestamp
                    sd_df_merged.timestamp_origin.iat[i] = cfg.TimestampOrigin.EXTRAPOLATED

                last_uptime = new_uptime

            # interpolate gaps
            df_nan = sd_df_merged[sd_df_merged.unixtimestamp.isna()]
            df_nan["i"] = df_nan.index
            df_nan_from = df_nan[((df_nan.i - df_nan.i.shift(1)) != 1) | pd.isna(df_nan.i - df_nan.i.shift(1))]
            df_nan_from.reset_index(drop=True, inplace=True)
            df_nan_to = df_nan.i[((df_nan.i.shift(-1) - df_nan.i) != 1) | pd.isna(df_nan.i.shift(-1) - df_nan.i)]
            df_nan_to.reset_index(drop=True, inplace=True)
            df_nan_range = df_nan_from.join(df_nan_to, lsuffix="_from", rsuffix="_to")
            df_nan_range["num_nans"] = df_nan_range["i_to"] - df_nan_range["i_from"] + 1
            # df_nan_range.drop("SD_block_ID_to", inplace=True, axis=1)
            # df_nan_range.rename(columns={"SD_block_ID_from": "SD_block_ID"}, inplace=True)

            # collect garbage to (hopefully?) free memory
            df_nan = ""
            df_nan_from = ""
            df_nan_to = ""
            influx_df_start = ""
            del df_nan
            del df_nan_from
            del df_nan_to
            del influx_df_start
            gc.collect()

            logging.log.info("Thread %u Slave %s%02u - Interpolating gaps (found %u)..."
                                % (processor_number, type_id, slave_id, df_nan_range.shape[0]))

            for row in df_nan_range.itertuples():
                num_nans = row.num_nans
                i_1 = row.i_from - 1  # this is the last valid index in sd_df_merged before a series of nans
                i_2 = row.i_to + 1  # this is the first valid index in sd_df_merged after a series of nans
                if num_nans < MAX_NUM_NAN_SIMPLE_INTERPOLATION:
                    t_1 = sd_df_merged.unixtimestamp[i_1]
                    u_1 = sd_df_merged.stm_tim_uptime[i_1]
                    t_2 = sd_df_merged.unixtimestamp[i_2]
                    u_2 = sd_df_merged.stm_tim_uptime[i_2]
                    du = u_2 - u_1
                    if du > 0:  # case A: short gap, rising uptime --> interpolate using uptime (tested, works)
                        logging.log.debug("Thread %u Slave %s%02u -  %s - %s  - case A (short gap, rising uptime)"
                                          % (processor_number, type_id, slave_id,
                                             pd.to_datetime(t_1, unit="s"), pd.to_datetime(t_2, unit="s")))
                        time_per_uptime = (t_2 - t_1) / du
                        if time_per_uptime > cfg.DELTA_T_STM_TICK * (1.0 + MAX_STM_TICK_SLOPE_DEVIATION):
                            time_per_uptime = cfg.DELTA_T_STM_TICK  # fall back
                        sd_df_merged.unixtimestamp[row.i_from:i_2] =\
                            t_1 + (sd_df_merged.stm_tim_uptime[row.i_from:i_2] - u_1) * time_per_uptime
                        sd_df_merged.timestamp_origin[row.i_from:i_2] = cfg.TimestampOrigin.INTERPOLATED
                    # elif du < -DELTA_UPTIME_REBOOT:  # case B: short gap, reboot in between
                    else:  # case B: short gap, reboot in between (SD card data -> don't check for DELTA_UPTIME_REBOOT)
                        logging.log.debug("Thread %u Slave %s%02u -  %s - %s  - case B (short gap, reboot in between)"
                                          % (processor_number, type_id, slave_id,
                                             pd.to_datetime(t_1, unit="s"), pd.to_datetime(t_2, unit="s")))
                        # i) go up forward from beginning until top is reached (tested, works)
                        i_low = row.i_from
                        while i_low < i_2:
                            new_uptime = sd_df_merged.stm_tim_uptime[i_low]
                            uptime_diff = (new_uptime - u_1)
                            if uptime_diff > 0:  # we go forward and uptime is rising
                                # go up to top using uptime
                                t_1 = t_1 + uptime_diff * cfg.DELTA_T_STM_TICK
                                sd_df_merged.unixtimestamp[i_low] = t_1
                                sd_df_merged.timestamp_origin.iat[i_low] = cfg.TimestampOrigin.INTERPOLATED
                                u_1 = new_uptime
                                i_low = i_low + 1
                            else:  # there was a reboot in between
                                # note that we don't need DELTA_UPTIME_REBOOT, because we are using SD card
                                # data. Assume there are no double entries in SD card.
                                break

                        # ii) go down backwards from bottom is reached (tested, works)
                        i_high = row.i_to
                        while i_high >= i_low:
                            new_uptime = sd_df_merged.stm_tim_uptime[i_high]
                            uptime_diff = (new_uptime - u_2)
                            if uptime_diff < 0:  # we go backward and uptime is falling
                                # go down to bottom using uptime
                                t_2 = t_2 + uptime_diff * cfg.DELTA_T_STM_TICK
                                sd_df_merged.unixtimestamp[i_high] = t_2
                                sd_df_merged.timestamp_origin.iat[i_high] = cfg.TimestampOrigin.INTERPOLATED
                                u_2 = new_uptime
                                i_high = i_high - 1
                            else:  # there was a reboot in between
                                # note that we don't need DELTA_UPTIME_REBOOT, because we are using SD card
                                # data. Assume there are no double entries in SD card.
                                last_uptime = u_2
                                last_timestamp = t_2
                                break

                        # iii) fill values in between (tested, works))
                        if i_high >= i_low:
                            # there is still data in between!

                            # logging.log.warning("Thread %u Slave %s%02u -  %s - %s  - unimplemented case: B->iii)"
                            #                     % (processor_number, type_id, slave_id,
                            #                        pd.to_datetime(t_1, unit="s"), pd.to_datetime(t_2, unit="s")))
                            while i_high >= i_low:
                                uptime_sd = sd_df_merged.stm_tim_uptime[i_high]
                                uptime_diff = (uptime_sd - last_uptime)
                                if uptime_diff < 0:  # go down to bottom using uptime
                                    last_timestamp = last_timestamp + uptime_diff * cfg.DELTA_T_STM_TICK
                                else:  # since we use (reliable) SD card data, assume there was a reboot
                                    last_timestamp = (last_timestamp - cfg.DELTA_T_REBOOT
                                                      - last_uptime * cfg.DELTA_T_STM_TICK)  # optional
                                sd_df_merged.unixtimestamp[i_high] = last_timestamp
                                sd_df_merged.timestamp_origin[i_high] = cfg.TimestampOrigin.INTERPOLATED
                                last_uptime = uptime_sd
                                i_high = i_high - 1
                    # else:
                    #     # case X
                    #     logging.log.warning("Thread %u Slave %s%02u -  %s - %s  - unimplemented case: X"
                    #                         % (processor_number, type_id, slave_id,
                    #                            pd.to_datetime(t_1, unit="s"), pd.to_datetime(t_2, unit="s")))
                else:  # long gap --> consider Influx values
                    t_1 = sd_df_merged.unixtimestamp[i_1]
                    t_2 = 0
                    influx_window_df = ""
                    try:
                        t_2 = sd_df_merged.unixtimestamp[i_2]
                        influx_window_df = influx_df[(influx_df.unixtimestamp > t_1) & (influx_df.unixtimestamp < t_2)]
                        influx_window_df.reset_index(drop=True, inplace=True)
                        num_influx_points = influx_window_df.shape[0]
                    except KeyError:
                        # logging.log.critical("Thread %u Slave %s%02u -  i_1= %u, i_2 = %u  - KeyError"
                        #                      % (processor_number, type_id, slave_id, i_1, i_2))
                        num_influx_points = -2  # maybe this happens at the end of the Series? --> there is no more data

                    if num_influx_points > 0:  # Influx data available
                        # t_1_influx = influx_window_df.unixtimestamp.iat[0]
                        # t_2_influx = influx_window_df.unixtimestamp.iat[-1]
                        # dt_influx = t_2_influx - t_1_influx
                        # dt_sd_est = (num_nans - 1) * cfg.DELTA_T_LOG
                        # if ((dt_influx >= dt_sd_est * MIN_TIME_DEVIATION_FILL_WITH_INFLUX)
                        #         & (dt_influx <= dt_sd_est * MAX_TIME_DEVIATION_FILL_WITH_INFLUX)):
                        # use number of data points instead of time difference because sometimes the SD time range is
                        # significantly larger just because of 1-2 values at the very end. Here, we don't care if the
                        # cycler was turned off during a significant amount of the time, we just care if we find a match
                        if ((num_influx_points >= num_nans * MIN_TIME_DEVIATION_FILL_WITH_INFLUX)
                                & (num_influx_points <= num_nans * MAX_TIME_DEVIATION_FILL_WITH_INFLUX)):
                            logging.log.debug("Thread %u Slave %s%02u -  %s - %s  - case C (long gap, Influx data with "
                                              "comparable number of data points available)"
                                              % (processor_number, type_id, slave_id,
                                                 pd.to_datetime(t_1, unit="s"), pd.to_datetime(t_2, unit="s")))
                            # case C: long gap, but we have Influx & SD values, their delta_timestamps match very well
                            # (tested, works)
                            i_low = row.i_from  # index of first nan in sd_df_merged
                            i_influx = 0
                            i_influx_max = influx_window_df.shape[0]
                            no_more_influx_data = False
                            last_timestamp = sd_df_merged.unixtimestamp[i_1]
                            last_uptime = sd_df_merged.stm_tim_uptime[i_1]

                            while i_low < i_2:  # can we vectorize stuff in here? --> at least compare & copy in blocks
                                uptime_sd = sd_df_merged.stm_tim_uptime[i_low]
                                if i_influx < i_influx_max:
                                    uptime_influx = influx_window_df.stm_tim_uptime[i_influx]
                                else:
                                    uptime_influx = -1
                                    no_more_influx_data = True
                                if uptime_sd == uptime_influx:  # exact match --> copy time, increment both counters
                                    # try to compare and copy in blocks
                                    max_copy_size = min(i_2 - i_low, i_influx_max - i_influx)  # - 1
                                    copy_size = min(NUM_COMPARE_AND_COPY_BLOCKS, max_copy_size)
                                    found_match = False
                                    i_max_influx = (i_influx+copy_size)
                                    i_max_sd = (i_low+copy_size)
                                    compare_block_influx = influx_window_df.stm_tim_uptime[i_influx:i_max_influx]
                                    compare_block_sd = sd_df_merged.stm_tim_uptime[i_low:i_max_sd]
                                    compare_block_influx.reset_index(drop=True, inplace=True)
                                    compare_block_sd.reset_index(drop=True, inplace=True)
                                    while not found_match:
                                        if compare_block_influx.equals(compare_block_sd):
                                            sd_df_merged.unixtimestamp[i_low:i_max_sd] =\
                                                influx_window_df.unixtimestamp[i_influx:i_max_influx]
                                            sd_df_merged.timestamp_origin[i_low:i_max_sd] =\
                                                cfg.TimestampOrigin.INFLUX_ESTIMATED
                                            last_timestamp = influx_window_df.unixtimestamp[i_max_influx - 1]
                                            uptime_sd = compare_block_sd.iat[-1]
                                            i_low = i_max_sd
                                            i_influx = i_max_influx
                                            found_match = True
                                        elif copy_size > NUM_COMPARE_AND_COPY_BLOCKS_DECR:  # reduce copy size
                                            copy_size = copy_size - NUM_COMPARE_AND_COPY_BLOCKS_DECR
                                            i_max_influx = (i_influx+copy_size)
                                            i_max_sd = (i_low+copy_size)
                                            compare_block_influx =\
                                                compare_block_influx.iloc[:-NUM_COMPARE_AND_COPY_BLOCKS_DECR]
                                            compare_block_sd = compare_block_sd.iloc[:-NUM_COMPARE_AND_COPY_BLOCKS_DECR]
                                        else:
                                            break  # copy individually
                                    compare_block_influx = ""
                                    compare_block_sd = ""
                                    if not found_match:
                                        last_timestamp = influx_window_df.unixtimestamp[i_influx]
                                        sd_df_merged.unixtimestamp[i_low] = last_timestamp
                                        sd_df_merged.timestamp_origin[i_low] = cfg.TimestampOrigin.INFLUX_ESTIMATED
                                        i_low = i_low + 1
                                        i_influx = i_influx + 1
                                elif (uptime_influx < (uptime_sd - DELTA_UPTIME_REBOOT_SMALL)
                                      or (uptime_influx > uptime_sd) or no_more_influx_data):
                                    # reboot in influx
                                    #    --> increment using SD uptime or fixed timestep until SD reaches reboot
                                    # OR influx uptime is too high
                                    #    --> increment using SD uptime or fixed timestep until SD reaches influx
                                    # we also need to increment the SD counter
                                    uptime_diff = (uptime_sd - last_uptime)
                                    resync = False
                                    if uptime_diff > DELTA_STM_TICKS_SD_MAX_BEFORE_RESYNC:
                                        resync = True
                                    # if (uptime_diff > 0) and (no_more_influx_data or (uptime_influx < uptime_sd)):
                                    if (uptime_diff > 0) and not resync:
                                        # go up to peak (or until Influx data continues) using uptime
                                        last_timestamp = last_timestamp + uptime_diff * cfg.DELTA_T_STM_TICK
                                        sd_df_merged.unixtimestamp[i_low] = last_timestamp
                                        sd_df_merged.timestamp_origin[i_low] = cfg.TimestampOrigin.INTERPOLATED
                                    else:  # uptime_diff <= 0 or too high deviation
                                        # Since we use SD card data, assume there was a reboot.
                                        # Reboot in SD as well --> find next Influx match and go backwards from there on
                                        # at this point, uptime_influx > uptime_sd
                                        # if (uptime_influx > uptime_sd) and (no_more_influx_data is False):
                                        # --> the code below is ok even without that check

                                        i_match = -1
                                        if i_influx < i_influx_max:
                                            i_low_x = i_low  # i_low_x = i_low + 1

                                            retry_matching = False
                                            while (i_match == -1) and (i_low_x < i_2):
                                                uptime_sd = sd_df_merged.stm_tim_uptime[i_low_x]
                                                if uptime_sd < uptime_influx:
                                                    # sd data too low --> need to increment for match
                                                    i_low_x = i_low_x + 1
                                                elif uptime_sd == uptime_influx:
                                                    # found match --> check if there are multiple matches!
                                                    match_m = True
                                                    i_low_max = sd_df_merged.stm_tim_uptime.shape[0]
                                                    for m in range(1, NUM_MULTIPLE_MATCHES_INFLUX):
                                                        i_influx_m = i_influx + m
                                                        i_low_m = i_low_x + m
                                                        if (i_influx_m >= i_influx_max) or (i_low_m >= i_low_max):
                                                            match_m = False  # end of index --> assume there's no match
                                                            break
                                                        uptime_influx_m = influx_window_df.stm_tim_uptime[i_influx_m]
                                                        uptime_sd_m = sd_df_merged.stm_tim_uptime[i_low_m]
                                                        if uptime_influx_m != uptime_sd_m:
                                                            match_m = False  # values differ --> no match
                                                            retry_matching = True
                                                            break
                                                    if match_m:
                                                        i_match = i_low_x
                                                        break
                                                    else:
                                                        # no match, increment sd since this is probably caused by
                                                        # frequent reboots which are not in influx
                                                        i_low_x = i_low_x + 1
                                                elif resync:  # sd data too high and resync needed --> increment influx
                                                    i_influx = i_influx + 1
                                                    if i_influx < i_influx_max:
                                                        uptime_influx = influx_window_df.stm_tim_uptime[i_influx]
                                                else:  # uptime_sd < uptime_influx
                                                    if retry_matching:
                                                        i_low_x = i_low_x + 1
                                                        continue  # chance of finding a match after reboot --> retry
                                                    else:
                                                        break  # sd data too high --> no match
                                        if i_match >= 0:
                                            # found a matching influx data --> go backwards from there on
                                            i = i_match
                                            last_timestamp = influx_window_df.unixtimestamp[i_influx]
                                            sd_df_merged.unixtimestamp[i] = last_timestamp
                                            sd_df_merged.timestamp_origin[i] = cfg.TimestampOrigin.INFLUX_ESTIMATED
                                            initial_last_uptime = uptime_sd
                                            while i > i_low:
                                                i = i - 1
                                                last_uptime = uptime_sd
                                                uptime_sd = sd_df_merged.stm_tim_uptime[i]
                                                uptime_diff = (uptime_sd - last_uptime)
                                                if uptime_diff < 0:  # go down to bottom using uptime
                                                    last_timestamp = last_timestamp + uptime_diff * cfg.DELTA_T_STM_TICK
                                                else:  # since we use (reliable) SD card data, assume there was a reboot
                                                    last_timestamp = (last_timestamp - cfg.DELTA_T_REBOOT
                                                                      - last_uptime * cfg.DELTA_T_STM_TICK)  # optional
                                                sd_df_merged.unixtimestamp[i] = last_timestamp
                                                sd_df_merged.timestamp_origin[i] = cfg.TimestampOrigin.INTERPOLATED
                                            i_influx = i_influx + 1
                                            uptime_sd = initial_last_uptime
                                            i_low = i_match  # is incremented +1 below (continue with next unmatched)

                                        else:  # didn't find match --> interpolate from next SD data using uptime
                                            last_timestamp = sd_df_merged.unixtimestamp[i_2]
                                            last_uptime = sd_df_merged.stm_tim_uptime[i_2]
                                            i_high = row.i_to
                                            while i_high >= i_low:
                                                uptime_sd = sd_df_merged.stm_tim_uptime[i_high]
                                                uptime_diff = (uptime_sd - last_uptime)
                                                if uptime_diff < 0:  # go down to bottom using uptime
                                                    last_timestamp = last_timestamp + uptime_diff * cfg.DELTA_T_STM_TICK
                                                else:  # since we use (reliable) SD card data, assume there was a reboot
                                                    last_timestamp = (last_timestamp - cfg.DELTA_T_REBOOT
                                                                      - last_uptime * cfg.DELTA_T_STM_TICK)  # optional
                                                sd_df_merged.unixtimestamp[i_high] = last_timestamp
                                                sd_df_merged.timestamp_origin[i_high] = cfg.TimestampOrigin.INTERPOLATED
                                                last_uptime = uptime_sd
                                                i_high = i_high - 1
                                            i_low = i_2
                                    i_low = i_low + 1
                                else:  # here: equal to "elif uptime_sd > uptime_influx" --> influx uptime too low
                                    i_influx = i_influx + 1  # --> increment influx index until we find a match
                                    continue  # do not assign last_uptime   and last_timestamp???
                                last_uptime = uptime_sd
                        elif num_influx_points < num_nans * MIN_TIME_DEVIATION_FILL_WITH_INFLUX:
                            # case F3/F4 not enough influx data -> e.g. slave 2, 21.11.22 14:13 - 15:21
                            # -> very bad data quality --> treat as if there was no influx data
                            num_influx_points = -1
                        else:
                            # Influx data time difference doesn't match with SD card time difference

                            pass  # untested

                            logging.log.warning("Thread %u Slave %s%02u - unimplemented case: D/E/G"
                                                % (processor_number, type_id, slave_id))
                    if num_influx_points <= 0:
                        # no Influx data available (tested, works)
                        dt_sd_est = (num_nans - 1) * cfg.DELTA_T_LOG
                        if num_influx_points == -2:  # end of Series
                            t_2 = t_1 + dt_sd_est  # assume data fits perfectly
                        dt = t_2 - t_1
                        # if ((dt >= dt_sd_est * MIN_TIME_DEVIATION_FILL_SD)
                        #      & (dt <= dt_sd_est * MAX_TIME_DEVIATION_FILL_SD)):
                        if dt <= dt_sd_est * MAX_TIME_DEVIATION_FILL_SD:
                            if num_influx_points < 0:  # almost no data available --> ignore it --> case F3
                                case_text = "F3 (long gap, no Influx data"
                            else:  # no data available --> case F1
                                case_text = "F1 (long gap, almost no Influx data (ignore)"
                            case_text = case_text + ", SD card data with comparable time difference available)"
                            # go up from start of gap until peak is reached
                            # then go down from end of gap until min is reached
                            # if data is left, spread individual reboots evenly
                            stick_sessions_to_end = False
                        else:
                            if num_influx_points < 0:  # almost no data available --> ignore it --> case F4
                                case_text = "F4 (long gap, no Influx data"
                            else:  # no data available --> case F1
                                case_text = "F2 (long gap, almost no Influx data (ignore)"
                            case_text = case_text + ", SD card data too short for time difference)"
                            # go up from start of gap until peak is reached
                            # then go down from end of gap until min is reached
                            # if data is left, stick all data to the end
                            stick_sessions_to_end = True
                        logging.log.debug("Thread %u Slave %s%02u -  %s - %s  - case %s"
                                          % (processor_number, type_id, slave_id, pd.to_datetime(t_1, unit="s"),
                                             pd.to_datetime(t_2, unit="s"), case_text))
                        # find sessions (using vectors)
                        u_1 = sd_df_merged.stm_tim_uptime[i_1]
                        u_2 = 0
                        sd_roi_df = sd_df_merged[(i_1+1):i_2]  # ROI (region of interest)
                        sd_roi_min = \
                            sd_roi_df.stm_tim_uptime[(sd_roi_df.stm_tim_uptime.shift(1) - sd_roi_df.stm_tim_uptime)
                                                     > DELTA_UPTIME_REBOOT_SMALL].index
                        sd_roi_max =\
                            sd_roi_df.stm_tim_uptime[(sd_roi_df.stm_tim_uptime - sd_roi_df.stm_tim_uptime.shift(-1))
                                                     > DELTA_UPTIME_REBOOT_SMALL].index
                        if num_influx_points != -2:
                            u_2 = sd_df_merged.stm_tim_uptime[i_2]

                        if (num_influx_points == -2) or (sd_roi_min is None) or (sd_roi_min.shape[0] == 0)\
                                or (sd_roi_max is None) or (sd_roi_max.shape[0] == 0):
                            # there is no reboot, just interpolate linearly like short gaps

                            if num_influx_points == -2:
                                du = 1
                            else:
                                du = u_2 - u_1

                            if du > 0:
                                if num_influx_points == -2:
                                    time_per_uptime = cfg.DELTA_T_STM_TICK
                                else:
                                    time_per_uptime = (t_2 - t_1) / du
                                    if time_per_uptime > cfg.DELTA_T_STM_TICK * (1.0 + MAX_STM_TICK_SLOPE_DEVIATION):
                                        time_per_uptime = cfg.DELTA_T_STM_TICK  # fall back
                                i_low = i_1 + 1
                                if ((sd_df_merged.stm_tim_uptime.loc[i_low] > sd_df_merged.stm_tim_uptime.loc[i_1])
                                        and (sd_roi_max is not None)):
                                    if sd_roi_max.shape[0] > 0:
                                        # SD card uptime still rising --> go up from start of gap until peak is reached
                                        i_high = sd_roi_max[0]
                                        sd_sess_win_uptime = sd_roi_df.stm_tim_uptime.loc[i_low:i_high]
                                        sd_df_merged.unixtimestamp[i_low:(i_high + 1)] =\
                                            t_1 + (cfg.DELTA_T_STM_TICK * (sd_sess_win_uptime - u_1))
                                        sd_df_merged.timestamp_origin[i_low:(i_high + 1)] =\
                                            cfg.TimestampOrigin.INTERPOLATED
                                        i_low = i_high + 1
                                        t_1 = sd_df_merged.unixtimestamp[i_high] + cfg.DELTA_T_REBOOT
                                        u_1 = sd_df_merged.stm_tim_uptime[i_low]
                                    else:
                                        i_low = row.i_from
                                else:
                                    i_low = row.i_from

                                sd_df_merged.unixtimestamp[i_low:i_2] =\
                                    t_1 + (sd_df_merged.stm_tim_uptime[i_low:i_2] - u_1) * time_per_uptime
                                sd_df_merged.timestamp_origin[row.i_from:i_2] = cfg.TimestampOrigin.INTERPOLATED
                            else:
                                logging.log.warning("Thread %u Slave %s%02u -  %s - %s  - case F2/F3: expected rising "
                                                    "uptime but it was falling instead - I don't know what to do :("
                                                    % (processor_number, type_id, slave_id,
                                                       pd.to_datetime(t_1, unit="s"),
                                                       pd.to_datetime(t_2, unit="s")))
                        else:

                            i_low = i_1 + 1
                            if sd_df_merged.stm_tim_uptime.loc[i_low] > sd_df_merged.stm_tim_uptime.loc[i_1]:
                                # SD card uptime is still rising --> go up from start of gap until peak is reached
                                i_high = sd_roi_max[0]
                                sd_sess_win_uptime = sd_roi_df.stm_tim_uptime.loc[:i_high]
                                sd_df_merged.unixtimestamp[i_low:(i_high + 1)] = t_1 + (cfg.DELTA_T_STM_TICK *
                                                                                        (sd_sess_win_uptime - u_1))
                                sd_df_merged.timestamp_origin[i_low:(i_high + 1)] = cfg.TimestampOrigin.INTERPOLATED
                                # last timestamp before remaining gap:
                                last_timestamp_low = sd_df_merged.unixtimestamp[i_high]
                            else:
                                last_timestamp_low = t_1  # FIXME: untested (didn't happen yet)
                                sd_roi_min = sd_roi_min.append(pd.Index([i_low]))

                            i_high = i_2 - 1
                            if sd_df_merged.stm_tim_uptime.loc[i_2] > sd_df_merged.stm_tim_uptime.loc[i_high]:
                                # SD card uptime already rose --> go down from end of gap until min is reached
                                i_low = sd_roi_min[-1]
                                sd_sess_win_uptime = sd_roi_df.stm_tim_uptime.loc[i_low:]
                                sd_df_merged.unixtimestamp[i_low:(i_high + 1)] = t_2 + (cfg.DELTA_T_STM_TICK *
                                                                                        (sd_sess_win_uptime - u_2))
                                sd_df_merged.timestamp_origin[i_low:(i_high + 1)] = cfg.TimestampOrigin.INTERPOLATED
                                # first timestamp after remaining gap minus time difference since uptime = 0
                                last_timestamp_high = sd_df_merged.unixtimestamp[i_low]\
                                                      - sd_roi_df.stm_tim_uptime.loc[i_low] * cfg.DELTA_T_STM_TICK
                            else:
                                # first timestamp after remaining gap minus time difference since uptime = 0
                                last_timestamp_high = t_2 - sd_df_merged.stm_tim_uptime.loc[i_2] * cfg.DELTA_T_STM_TICK
                                sd_roi_max = sd_roi_max.append(pd.Index([i_high]))

                            num_sessions = sd_roi_max.size - 1
                            # num_sessions = sd_roi_min.size - 1
                            # if sd_roi_min.size != sd_roi_max.size:
                            #     print("check this out")

                            if num_sessions > 0:
                                # there are more sessions left to distribute
                                dt_total = last_timestamp_high - last_timestamp_low  # available time to fill in data
                                if dt_total <= 0:
                                    logging.log.warning("Thread %u Slave %s%02u -  %s - %s  - case F2/F3: no time left "
                                                        "to fill with remaining SD data --> using last timestamp"
                                                        % (processor_number, type_id, slave_id,
                                                           pd.to_datetime(t_1, unit="s"),
                                                           pd.to_datetime(t_2, unit="s")))
                                    i_low = sd_roi_max[0] + 1
                                    i_high = sd_roi_min[-1] - 1
                                    sd_df_merged.unixtimestamp[i_low:i_high] = last_timestamp_low
                                    sd_df_merged.timestamp_origin[i_low:i_high] = cfg.TimestampOrigin.ERROR
                                else:
                                    # sum of ticks from 0 to max:
                                    ticks_sd_total = sum(sd_roi_df.stm_tim_uptime.loc[sd_roi_max[1:]])
                                    # estimated ticks available for data:
                                    ticks_use = cfg.DELTA_T_STM_TICK
                                    dt_reboot_use = cfg.DELTA_T_REBOOT
                                    ticks_gap = (dt_total - (num_sessions + 1) * dt_reboot_use) / ticks_use
                                    if ticks_sd_total > ticks_gap:
                                        # not enough time in gap for default tick + pause time --> decrease tick timing
                                        ratio = ticks_gap / ticks_sd_total
                                        ticks_use = ratio * ticks_use
                                        if ticks_use < ((1 - MAX_STM_TICK_SLOPE_DEVIATION) * cfg.DELTA_T_STM_TICK):
                                            logging.log.warning("Thread %u Slave %s%02u -  %s - %s  - case F2/F3: time "
                                                                "very short to fill with remaining SD data\n   "
                                                                "using  unusually small DELTA_T_STM_TICK "
                                                                "(%f %% of nominal time)"
                                                                % (processor_number, type_id, slave_id,
                                                                   pd.to_datetime(t_1, unit="s"),
                                                                   pd.to_datetime(t_2, unit="s"), ratio * 100.0))
                                    elif ticks_sd_total < ticks_gap:
                                        # time in gap larger than default tick + pause time
                                        if not stick_sessions_to_end:  # deviation reasonable --> increase reboot time
                                            dt_reboot_use = (dt_total - ticks_sd_total * ticks_use) / (num_sessions + 1)
                                        # else: leave dt_reboot_use as is and stick to end

                                    last_timestamp = last_timestamp_high - dt_reboot_use
                                    for i_sess in range(num_sessions - 1, -1, -1):  # go backwards
                                        i_low = sd_roi_min[i_sess]
                                        i_high = sd_roi_max[i_sess + 1]
                                        max_ticks = sd_df_merged.stm_tim_uptime[i_high]
                                        i_high = i_high + 1
                                        sd_df_merged.unixtimestamp[i_low:i_high] =\
                                            (last_timestamp
                                             + (sd_df_merged.stm_tim_uptime[i_low:i_high] - max_ticks) * ticks_use)
                                        sd_df_merged.timestamp_origin[i_low:i_high] = cfg.TimestampOrigin.INTERPOLATED
                                        last_timestamp = sd_df_merged.unixtimestamp[i_low] - dt_reboot_use\
                                                         - sd_df_merged.stm_tim_uptime[i_low] * ticks_use
                            # collect garbage
                            sd_sess_win_uptime = ""
                            del sd_sess_win_uptime

                        # collect garbage
                        sd_roi_df = ""
                        del sd_roi_df

                    # collect garbage
                    influx_window_df = ""
                    del influx_window_df
                    gc.collect()

            # collect garbage
            influx_df = ""
            df_nan_range = ""
            del influx_df
            del df_nan_range
            gc.collect()

            # extrapolate end --> already done by gap algorithm?
            # logging.log.info("Thread %u Slave %s%02u - Extrapolating end..."
            #                     % (processor_number, type_id, slave_id))

            # check if nans still exist (shouldn't be the case)
            # sd_df_merged_nans = sd_df_merged.isnull().values.any()
            # if sd_df_merged_nans:
            # noinspection PyArgumentList
            sd_df_merged_num_nans = sd_df_merged.isnull().sum().sum()
            if sd_df_merged_num_nans > 0:
                report_msg_nans = f"Thread %u Slave %s%02u - %u NaN fields exist after gap filling!" \
                                  % (processor_number, type_id, slave_id, sd_df_merged_num_nans)
                # logging.log.error(report_msg_nans)

            # check if timestamp is strictly monotonically increasing (timestamp vector operation ... > X, e.g. X = 1 )

            sd_df_merged_timediff = sd_df_merged.unixtimestamp - sd_df_merged.unixtimestamp.shift(1)
            min_gap = sd_df_merged_timediff.min()
            max_gap = sd_df_merged_timediff.max()
            max_gap_str = str(timedelta(seconds=max_gap))
            min_unit_string = " h:mm:ss(.ms)"
            max_unit_string = " h:mm:ss(.ms)"
            if min_gap <= 0:
                if min_gap < 0:
                    min_gap_str = "-" + str(timedelta(seconds=-min_gap))
                    min_unit_string = min_unit_string + " - negative!"
                else:
                    min_gap_str = "0:00:00"
                    min_unit_string = min_unit_string + " - zero"
                report_msg = f"Thread %u Slave %s%02u - timestamp is not strictly monotonically increasing!\n" \
                              "   Min timestep: %25s %s\n" \
                              "   Max timestep: %25s %s\n" \
                              % (processor_number, type_id, slave_id,
                                 min_gap_str, min_unit_string, max_gap_str, max_unit_string)
                # logging.log.error(report_msg)
                report_level = logging.ERROR
            else:
                min_gap_str = str(timedelta(seconds=min_gap))
                if min_gap < MIN_DELTA_T_WARNING:
                    sd_df_merged_timediff_small = sd_df_merged_timediff[sd_df_merged_timediff < MIN_DELTA_T_WARNING]
                    report_msg = f"Thread %u Slave %s%02u - timestamp strictly monotonically increasing, but " \
                                  " unusually small min timestep.\n   Min timestep: %25s %s\n" \
                                  "   Max timestep: %25s %s\n%s\n" \
                                  % (processor_number, type_id, slave_id, min_gap_str, min_unit_string, max_gap_str,
                                     max_unit_string, sd_df_merged_timediff_small)
                    # logging.log.warning(report_msg)
                    report_level = logging.WARNING
                else:
                    report_msg = f"Thread %u Slave %s%02u - timestamp strictly monotonically increasing.\n" \
                                  "   Min timestep: %25s %s\n" \
                                  "   Max timestep: %25s %s\n" \
                                  % (processor_number, type_id, slave_id,
                                     min_gap_str, min_unit_string, max_gap_str, max_unit_string)
                    # logging.log.info(report_msg)
                    report_level = logging.INFO

            # sd_df_merged.unixtimestamp =\
            #     sd_df_merged.unixtimestamp.astype(int)  # -> doesn't work because of missing values
            sd_df_merged.to_csv(cfg.CSV_WORKING_DIR + (cfg.CSV_FILENAME_04_SD_BLOCK_ID_UPTIME_TIMESTAMP
                                                       % (type_id, slave_id)),
                                index_label="index", sep=cfg.CSV_SEP)
            if cfg.PLOT:
                fig = plt.figure()
                plt1 = fig.add_subplot(2, 1, 1)
                # plot difference of timestamps over SD card block ID
                plt1.plot(sd_df_merged.SD_block_ID, sd_df_merged_timediff)
                plt1.set_title("Slave %s%02u: timestamp difference over SD Block ID" % (type_id, slave_id))
                plt1.set_xlabel("SD Block ID")
                plt1.set_ylabel("Timestamp difference (log) [s]")
                if min_gap > 0:
                    plt1.set_yscale('log')
                plt1.grid(True)

                # plot timestamps over SD card block ID
                plt2 = fig.add_subplot(2, 1, 2, sharex=plt1)
                plt2.plot(sd_df_merged.SD_block_ID, sd_df_merged.unixtimestamp)
                plt2.set_title("Slave %s%02u: timestamp over SD Block ID" % (type_id, slave_id))
                plt2.set_xlabel("SD Block ID")
                plt2.set_ylabel("Timestamp")
                plt2.grid(True)

                plt.show()

            sd_df_merged = ""
            del sd_df_merged
            gc.collect()

        # logging_level = DEBUG  # DEBUG, INFO, WARNING, ERROR, CRITICAL
        slave_report = {"msg": report_msg, "level": report_level, "msg_nans": report_msg_nans}
        thread_report_queue.put(slave_report)

        logging.log.info("Thread %u Slave %s%02u - done" % (processor_number, type_id, slave_id))

    logging.log.info("Thread %u - no more slaves - exiting" % processor_number)
    slave_queue.close()
    thread_report_queue.close()


if __name__ == "__main__":
    run()
