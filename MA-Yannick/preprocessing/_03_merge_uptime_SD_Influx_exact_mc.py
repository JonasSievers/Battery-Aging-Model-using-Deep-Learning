import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import config_preprocessing as cfg
import multiprocessing
from datetime import datetime
import os
import re
import config_logging as logging


# config
MAX_LOOKAROUND = 1000  # search +/- this entries around the index of SD_block_ID that got the estimated timestamp
# to find the SD block ID with the exact TIM value
# 1000 --> ca. 2000 seconds = 33 minutes -> TIM value +/- 190735
# MAX_LOOKAROUND = 5000  # search +/- this entries around the index of SD_block_ID that got the estimated timestamp
# to find the SD block ID with the exact TIM value
# 5000 --> ca. 10000 seconds = 2h45m -> TIM value +/- 953675

# DELTA_TICKS_OFFSET_INFLUX = 47500  # 47500 = ca. 8,3 minutes
# the cycler usually restarted because of lost internet connection --> the SD logs always (?) go further than Influx
# for enhanced success of matching, add this value to the maximum uptime of the Influx table
MAX_SD_BLOCK_OFFSET_SECONDS = (24 * 60 * 60)  # 24 * 3600 = 1 day
MAX_SD_BLOCK_OFFSET_PERCENT = 0.05  # 0.02 = 2%
MAX_LOOKAROUND_MANUAL = 200000

R_SQUARED_MIN_WARNING = 0.9995  # if R-squared is smaller, output a warning that timestamps couldn't be reliably matched
ONLY_PLOT_SUSPICIOUS = False  # if True, only plot where timestamps couldn't be reliably matched (if cfg.PLOT == True)
ALWAYS_PLOT_SUSPICIOUS = True  # if True, always plot where timestamps couldn't be reliably matched (ignores cfg.PLOT)


# constants
# sd_block_id_row = cfg.CSV_FILE_HEADER_SD_BLOCK_ID
pd.options.mode.chained_assignment = None  # default="warn"
NUMBER_OF_PROCESSORS_TO_USE = multiprocessing.cpu_count()
task_queue = multiprocessing.Queue()
SD_BLOCKS_PER_SECOND = cfg.SD_BLOCK_SIZE_BYTES / cfg.DELTA_T_LOG
MAX_SD_BLOCK_OFFSET = SD_BLOCKS_PER_SECOND * MAX_SD_BLOCK_OFFSET_SECONDS


def run():
    start_timestamp = datetime.now()
    logging.log.info(os.path.basename(__file__))

    # find files
    slaves_sd = []
    slaves_influx = []
    with os.scandir(cfg.CSV_WORKING_DIR) as iterator:
        regex_string_sd = cfg.CSV_FILENAME_01_SD_UPTIME_PEAKS.replace("%s%02u", "([ST])(\d+)")
        regex_pattern_sd = re.compile(regex_string_sd)
        regex_string_influx = cfg.CSV_FILENAME_02_INFLUX_UPTIME_PEAKS.replace("%s%02u", "([ST])(\d+)")
        regex_pattern_influx = re.compile(regex_string_influx)
        for entry in iterator:
            re_match_sd = regex_pattern_sd.fullmatch(entry.name)
            if re_match_sd:
                slave_type = re_match_sd.group(1)
                slave_id = int(re_match_sd.group(2))
                slave = {"id": slave_id, "type": slave_type}
                slaves_sd.append(slave)
            else:
                re_match_influx = regex_pattern_influx.fullmatch(entry.name)
                if re_match_influx:
                    slave_type = re_match_influx.group(1)
                    slave_id = int(re_match_influx.group(2))
                    slave = {"id": slave_id, "type": slave_type}
                    slaves_influx.append(slave)

    # only used slaves of which a .csv from the SD and influx exists
    slaves_string = []
    for slave_sd in slaves_sd:
        for slave_influx in slaves_influx:
            if (slave_sd["id"] == slave_influx["id"]) and (slave_sd["type"] == slave_influx["type"]):
                task_queue.put(slave_sd)
                slaves_string.append("%s%02u" % (slave_sd["type"], slave_sd["id"]))

    logging.log.info("Found .csv files for slaves: %s" % slaves_string)

    # Create processes
    processes = []
    logging.log.info("Starting processes")
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        logging.log.debug("  Starting process %u" % processorNumber)
        processes.append(multiprocessing.Process(target=run_thread, args=(processorNumber, task_queue, )))
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].start()
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].join()
        logging.log.debug("Joined process %u" % processorNumber)

    stop_timestamp = datetime.now()
    logging.log.info("\nScript runtime: %s h:mm:ss.ms" % str(stop_timestamp - start_timestamp))


def run_thread(processor_number, slave_queue):
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
        logging.log.info("Thread %u Slave %s%02u - start merging of SD and InfluxDB uptime"
                         % (processor_number, type_id, slave_id))

        # timestamp_row = cfg.CSV_FILE_HEADER_UNIX_TIMESTAMP

        csv_file_path_sd_peaks = cfg.CSV_WORKING_DIR + (cfg.CSV_FILENAME_01_SD_UPTIME_PEAKS % (type_id, slave_id))
        csv_file_path_influx_peaks = cfg.CSV_WORKING_DIR + (cfg.CSV_FILENAME_02_INFLUX_UPTIME_PEAKS
                                                            % (type_id, slave_id))

        df1 = pd.read_csv(csv_file_path_sd_peaks, sep=cfg.CSV_SEP)
        df2 = pd.read_csv(csv_file_path_influx_peaks, sep=cfg.CSV_SEP)

        match_error = False
        if df1.shape[0] == df2.shape[0]:
            # same size --> assume matching index works

            df1.sort_values(["max"], inplace=True)
            logging.log.debug("Thread %u Slave %s%02u - df1:\n%s"
                              % (processor_number, type_id, slave_id, df1.to_string()))

            df2.sort_values(["max"], inplace=True)
            logging.log.debug("Thread %u Slave %s%02u - df2:\n%s"
                              % (processor_number, type_id, slave_id, df2.to_string()))
            # df2.max += DELTA_TICKS_OFFSET_INFLUX
            # logging.log.debug("Thread %u Slave %s%02u - df2 (2):\n%s"
            #                   % (processor_number, type_id, slave_id, df2.to_string()))

            logging.log.debug("Thread %u Slave %s%02u - df1 and df2 size equal, using simple index matching"
                              % (processor_number, type_id, slave_id))
            df3 = pd.merge(df1, df2, left_index=True, right_index=True)
            df3.sort_values("index_x", inplace=True)
        else:
            # try to match by "max" column --> even this worked well for all slaves (might also only use this)
            logging.log.info("Thread %u Slave %s%02u - df1 and df2 size unequal! Using advanced matching..."
                             % (processor_number, type_id, slave_id))
            # this doesn't work well:
            # df3 = pd.merge_asof(df2, df1, on="max", direction="forward", allow_exact_matches=True, )
            # merge manually instead:

            new_data_frame = []
            last_timestamp = 0
            last_sd_block_id = 0
            # use influx as baseline - if we don't have a peak in influx, we can't match it with the SD card
            for index_df2, row_df2 in df2.iterrows():
                match = False
                for index_df1, row_df1 in df1.iterrows():
                    if (row_df2["max"] > (row_df1["max"] - MAX_LOOKAROUND_MANUAL))\
                            and (row_df2["max"] < (row_df1["max"] + MAX_LOOKAROUND_MANUAL)):
                        if new_data_frame.__len__() == 0:
                            # first entry --> try to find match only based on max column
                            match = True
                        else:
                            delta_timestamp = row_df2.unixtimestamp - last_timestamp
                            delta_sd_block_id_ref = delta_timestamp * SD_BLOCKS_PER_SECOND
                            delta_sd_block_id_min = last_sd_block_id - MAX_SD_BLOCK_OFFSET\
                                                    + delta_sd_block_id_ref * (1.0 - MAX_SD_BLOCK_OFFSET_PERCENT)
                            delta_sd_block_id_max = last_sd_block_id + MAX_SD_BLOCK_OFFSET\
                                                    + delta_sd_block_id_ref * (1.0 + MAX_SD_BLOCK_OFFSET_PERCENT)
                            if (row_df1.SD_block_ID > delta_sd_block_id_min)\
                                    and (row_df1.SD_block_ID < delta_sd_block_id_max):
                                match = True
                    if match:
                        last_timestamp = row_df2.unixtimestamp
                        last_sd_block_id = row_df1.SD_block_ID
                        new_row = {"index_x": row_df2["index"],
                                   "unixtimestamp": row_df2.unixtimestamp,
                                   "stm_tim_uptime_x": row_df2.stm_tim_uptime,
                                   "max": row_df2["max"],
                                   "min_x": row_df2["min"],
                                   "index_y": row_df1["index"],
                                   "SD_block_ID": row_df1.SD_block_ID,
                                   "stm_tim_uptime_y": row_df1.stm_tim_uptime,
                                   "min_y": row_df1["min"],
                                   }
                        index_to_drop = int(row_df1["index"])
                        df1_index = df1[df1["index"] == index_to_drop].index
                        df1 = df1.drop(df1_index)
                        new_data_frame.append(new_row)
                        break
                if not match:
                    logging.log.warning("Thread %u Slave %s%02u - no match found for peak at unix timestamp %s"
                                        % (processor_number, type_id, slave_id,
                                           pd.to_datetime(row_df2.unixtimestamp, unit="s")))
                    match_error = True

            df3 = pd.DataFrame(new_data_frame)

        logging.log.debug("Thread %u Slave %s%02u - df3:\n%s" % (processor_number, type_id, slave_id, df3.to_string()))

        # print results
        slope, intercept, r_value, p_value, std_err = linregress(df3.unixtimestamp, df3.SD_block_ID)
        logging.log.debug("Thread %u Slave %s%02u:\n   Slope: %f\n   Intercept: %f\n   R-squared: %f"
                          % (processor_number, type_id, slave_id, slope, intercept, r_value**2))

        # try to find exact matches
        sd_csv_file_path = cfg.CSV_WORKING_DIR + (cfg.CSV_FILENAME_01_SD_UPTIME % (type_id, slave_id))
        sd_df = pd.read_csv(sd_csv_file_path, sep=cfg.CSV_SEP, header=0)
        # stm_tim_uptime_index = sd_df.columns.get_loc(cfg.CSV_FILE_HEADER_STM_TIM_UPTIME)
        # sd_block_id_index = sd_df.columns.get_loc(sd_block_id_row)
        for index, row in df3.iterrows():
            sd_block_id_start = row.SD_block_ID
            exact_uptime = row.stm_tim_uptime_x
            sd_df_index = sd_df.SD_block_ID[(sd_df.SD_block_ID == sd_block_id_start)].index.values[0]
            exact_sd_block_id = None  # 0
            timestamp_origin = cfg.TimestampOrigin.UNKNOWN
            for i in range(0, MAX_LOOKAROUND):
                i_1 = sd_df_index - i
                try:
                    uptime_1 = sd_df.stm_tim_uptime[i_1]
                    if uptime_1 == exact_uptime:
                        exact_sd_block_id = sd_df.SD_block_ID[i_1]
                        timestamp_origin = cfg.TimestampOrigin.INFLUX_EXACT
                        break
                    else:
                        i_2 = sd_df_index + i
                        uptime_2 = sd_df.stm_tim_uptime[i_2]
                        if uptime_2 == exact_uptime:
                            exact_sd_block_id = sd_df.SD_block_ID[i_2]
                            timestamp_origin = cfg.TimestampOrigin.INFLUX_EXACT
                            break
                except IndexError:
                    pass  # single positional indexer is out-of-bounds --> Influx data goes further than SD card data
            if exact_sd_block_id is None:
                # noinspection PyStringFormat
                logging.log.warning("Thread %u Slave %s%02u - No exact SD_block_id found for index %u with uptime %u"
                                    % (processor_number, type_id, slave_id, index, exact_uptime))
                exact_sd_block_id = sd_block_id_start  # at least use the start index
                timestamp_origin = cfg.TimestampOrigin.INFLUX_ESTIMATED
            df3.loc[index, "SD_block_ID_exact"] = exact_sd_block_id  # .astype("int64")
            df3.loc[index, "timestamp_origin"] = int(timestamp_origin)

        df3.SD_block_ID_exact = df3.SD_block_ID_exact.astype("int64")
        df3.timestamp_origin = df3.timestamp_origin.astype("int64")
        logging.log.debug("Thread %u Slave %s%02u - df3 (exact):\n%s"
                          % (processor_number, type_id, slave_id, df3.to_string()))

        # print results
        slope, intercept, r_value, p_value, std_err = linregress(df3.unixtimestamp, df3.SD_block_ID_exact)
        r_squared = r_value**2
        logging.log.debug("Thread %u Slave %s%02u:\n   Slope: %f\n   Intercept: %f\n   R-squared: %f"
                          % (processor_number, type_id, slave_id, slope, intercept, r_squared))

        plot_this = cfg.PLOT
        if r_squared < R_SQUARED_MIN_WARNING:
            logging.log.warning("Thread %u Slave %s%02u: R-squared suspiciously low: %f < %f\n"
                                "   Timestamps very likely couldn't be matched correctly."
                                % (processor_number, type_id, slave_id, r_squared, R_SQUARED_MIN_WARNING))
            if ALWAYS_PLOT_SUSPICIOUS:
                plot_this = True
        elif ONLY_PLOT_SUSPICIOUS or ALWAYS_PLOT_SUSPICIOUS:
            if match_error:
                if ALWAYS_PLOT_SUSPICIOUS:
                    plot_this = True
            elif not cfg.PLOT:
                plot_this = False

        export_df = df3[["unixtimestamp", "SD_block_ID_exact", "timestamp_origin"]]
        export_df.rename(columns={"SD_block_ID_exact": "SD_block_ID"}, inplace=True)
        csv_file_path_merged_table = cfg.CSV_WORKING_DIR + (cfg.CSV_FILENAME_03_TIMESTAMP_AND_SD_BLOCK_ID_PEAKS
                                                            % (type_id, slave_id))
        export_df.to_csv(csv_file_path_merged_table, index_label="index", header=True, mode="w", sep=cfg.CSV_SEP)

        if plot_this:  # cfg.PLOT:
            plt.scatter(df3.unixtimestamp, df3.SD_block_ID_exact, c="g")
            # plt.scatter(df3.unixtimestamp, df3["min"],c="b")
            plt.title("Slave %s%02u: Uptime over timestamp  from InfluxDB" % (type_id, slave_id))
            plt.xlabel("Timestamp")
            plt.ylabel("SCByte Count")
            plt.grid(True)
            plt.plot(df3.unixtimestamp, intercept + slope*df3.unixtimestamp, "r")
            plt.show()

        logging.log.info("Thread %u Slave %s%02u - done" % (processor_number, type_id, slave_id))

    logging.log.info("Thread %u - no more slaves - exiting" % processor_number)
    slave_queue.close()


if __name__ == "__main__":
    run()
