from influxdb_client import InfluxDBClient
from datetime import datetime
import time
import warnings
from influxdb_client.client.warnings import MissingPivotFunction
import pandas as pd
import matplotlib.pyplot as plt
import config_preprocessing as cfg
import multiprocessing
import os
import re
import config_logging as logging


# configuration
SD_CARD_DATA_END_TIME = datetime(2023, 4, 28, 0, 0, 0)  # Load influx up to this (UTC?) end date of the SD card data
LOAD_FROM_INFLUX = True
INFLUX_TIMESTAMP_MAX_VARIATION = 1.7  # 1.7 = 170 %, delete influx timestamp if 70 % higher than expected from uptime
INFLUX_TIMESTAMP_MIN_VARIATION = 0.3  # 0.3 =  30 %, delete influx timestamp if 70 % lower than expected from uptime
INFLUX_TIMESTAMP_MIN_REBOOT_TICKS = 2.0 * cfg.DELTA_T_LOG / cfg.DELTA_T_STM_TICK  # if uptime falls by more than ...
# ... INFLUX_TIMESTAMP_MIN_REBOOT_TICKS (here: 2x log period ticks), assume there was a reboot for sure
INFLUX_TIMESTAMP_DIFF_INDIVIDUAL = 3.0 * cfg.DELTA_T_LOG  # if timestamp deviates by more than ...
# ... INFLUX_TIMESTAMP_DIFF_INDIVIDUAL (here: 3x log period) left and right, delete this (individual) point

# constants
pd.options.mode.chained_assignment = None  # default="warn"
# uptime_row = cfg.CSV_FILE_HEADER_STM_TIM_UPTIME
NUMBER_OF_PROCESSORS_TO_USE = multiprocessing.cpu_count()
task_queue = multiprocessing.Queue()


def run():
    start_timestamp = datetime.now()
    logging.log.info(os.path.basename(__file__))

    # find files
    slaves = []
    slaves_string = []
    with os.scandir(cfg.CSV_WORKING_DIR) as iterator:
        regex_string = cfg.CSV_FILENAME_01_SD_UPTIME.replace("%s%02u", "([ST])(\d+)")
        regex_pattern = re.compile(regex_string)
        for entry in iterator:
            re_match = regex_pattern.fullmatch(entry.name)
            if re_match:
                slave_type = re_match.group(1)
                slave_id = int(re_match.group(2))
                slave = {"id": slave_id, "type": slave_type}
                slaves.append(slave)
                task_queue.put(slave)
                slaves_string.append("%s%02u" % (slave_type, slave_id))
    logging.log.info("Found .csv files for slaves: %s" % slaves_string)

    if LOAD_FROM_INFLUX:
        warnings.simplefilter("ignore", MissingPivotFunction)

        client = InfluxDBClient(url=cfg.influx_url, token=cfg.influx_token, org=cfg.influx_org)
        query_api = client.query_api()

        # =====================================================================================================
        # Query data
        # data base can send a month at maximum (ca.) = 30+24+3600 = 23887872 seconds
        # Iterate from start to now() in 23887872 second steps
        # What's the end date of the SD card data? -> 2023-02-01 (looked up manually)
        stop_date = SD_CARD_DATA_END_TIME
        stop_date_timestamp = int(time.mktime(stop_date.timetuple()))
        start_date_timestamp = 1665593100  # -> Mi, 12.10.2022  16:45:00 see schedule_2022-10-12_experiment_LG_HG2.txt

        # Split the timestamps into monthly chunks
        length = stop_date_timestamp - start_date_timestamp
        block_length = int(length / 52)  # for one year of data, this is ca. 1 week

        for slave in slaves:
            # Slave Number
            slave_name = "%s%02u" % (slave["type"], slave["id"])
            csv_file_path = cfg.CSV_WORKING_DIR + (cfg.CSV_FILENAME_02_INFLUX_UPTIME_RAW % (slave["type"], slave["id"]))

            header = True
            write_mode = "w"  # start with overwriting, change to append after first write

            for block_start in range(start_date_timestamp, (stop_date_timestamp - block_length + 1), block_length):
                block_stop = block_start + block_length  # - 1
                # https://docs.influxdata.com/flux/v0.x/stdlib/universe/range/
                # "Results include rows with _time values that match the specified start time."
                # --> block_stop[i] should be equal to block_start[i+1]
                percent_calc = int((block_start - start_date_timestamp) / length * 100)
                time_start = pd.to_datetime(block_start, unit="s")
                time_stop = pd.to_datetime(block_stop, unit="s")
                logging.log.info("Influx - Slave %s - %3u%% (now loading %s - %s)"
                                 % (slave_name, percent_calc, time_start, time_stop))

                query = '''
                from(bucket: "{influxBucket}")
                  |> range(start: {block_start}, stop: {block_stop})
                  |> filter(fn: (r) => r["_measurement"] == "slave_general")
                  |> filter(fn: (r) => r["_field"] == "uptime")
                  |> filter(fn: (r) => r["slave"] == "{slave}")
                  |> map(fn: (r) => ({{
                    uptime: r._value,
                    timestamp: uint(v: r._time)
                }}))
                '''.format(influxBucket=cfg.influx_bucket,
                           block_start=block_start,
                           block_stop=block_stop,
                           slave=slave_name
                           )
                logging.log.debug("   Waiting for database...")
                csv_result_pandas = query_api.query_data_frame(query)
                logging.log.debug("   Processing and export to CSV...")
                csv_result_pandas = csv_result_pandas.drop("result", axis=1)
                csv_result_pandas = csv_result_pandas.drop("table", axis=1)
                csv_result_pandas["timestamp"] = (csv_result_pandas["timestamp"] / 1000000000).astype(int)
                # print(csv_result_pandas.head())
                csv_result_pandas.rename(columns={"timestamp": "unixtimestamp"}, inplace=True)
                csv_result_pandas.rename(columns={"uptime": "stm_tim_uptime"}, inplace=True)
                csv_result_pandas.to_csv(csv_file_path, index=False, mode=write_mode, sep=cfg.CSV_SEP, header=header)
                write_mode = "a"
                header = False

        client.close()

    # Create processes
    processes = []
    logging.log.info("Starting processes")
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        logging.log.debug("  Starting process %u" % processorNumber)
        processes.append(multiprocessing.Process(target=run_thread, args=(processorNumber, task_queue,)))
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

        # Slave Number
        slave_id = queue_entry["id"]
        type_id = queue_entry["type"]

        logging.log.info("Thread %u Slave %s%02u - deleting implausible InfluxDB timestamps"
                         % (processor_number, type_id, slave_id))
        # for example: Slave 4, 31.10.2022 around 01:30:
        # http://ipepdvmssqldb2.ipe.kit.edu:3000/d/YxIcA31nk/slave-log?orgId=1&var-slave_id=4&var-my_interval=2s&viewPanel=6&from=1667175945188&to=1667176601553
        csv_file_path_raw = cfg.CSV_WORKING_DIR + (cfg.CSV_FILENAME_02_INFLUX_UPTIME_RAW % (type_id, slave_id))
        df_raw = pd.read_csv(csv_file_path_raw, sep=cfg.CSV_SEP, header=0)

        diff_df = ((df_raw.unixtimestamp - df_raw.unixtimestamp.shift(1))
                   / (df_raw.stm_tim_uptime - df_raw.stm_tim_uptime.shift(1)))

        # delete points where the slope deviates a lot from the expected slope
        # there are no nans in df_raw. nans in diff_df result from missing values at beginning/end due to shifting
        # --> include nans, values with a reasonable slope and values with a very high negative slope (reboot)
        df = df_raw[pd.isna(diff_df)
                    | ((diff_df > (cfg.DELTA_T_STM_TICK * INFLUX_TIMESTAMP_MIN_VARIATION))
                       & (diff_df < (cfg.DELTA_T_STM_TICK * INFLUX_TIMESTAMP_MAX_VARIATION)))
                    | (diff_df < -INFLUX_TIMESTAMP_MIN_REBOOT_TICKS)]

        # delete individual points without neighbors --> they occur very often during Ethernet failures and almost
        # always insert wrong timestamps. Individual points = timestamp (or uptime) left AND right deviates a lot.
        diff_df_1 = (df.unixtimestamp - df.unixtimestamp.shift(1))
        diff_df_2 = (df.unixtimestamp.shift(-1) - df.unixtimestamp)
        df = df[pd.isna(diff_df_1) | pd.isna(diff_df_2)
                | ((diff_df_1 > 0) & (diff_df_2 > 0)
                   & (diff_df_1 < INFLUX_TIMESTAMP_DIFF_INDIVIDUAL) & (diff_df_2 < INFLUX_TIMESTAMP_DIFF_INDIVIDUAL))]

        csv_file_path = cfg.CSV_WORKING_DIR + (cfg.CSV_FILENAME_02_INFLUX_UPTIME % (type_id, slave_id))
        df.to_csv(csv_file_path, index=False, mode="w", sep=cfg.CSV_SEP, header=True)

        logging.log.info("Thread %u Slave %s%02u - start finding InfluxDB uptime peaks"
                         % (processor_number, type_id, slave_id))

        df = pd.read_csv(csv_file_path, sep=cfg.CSV_SEP, header=0)
        df.stm_tim_uptime = pd.to_numeric(df.stm_tim_uptime, errors="coerce")
        df.unixtimestamp = pd.to_numeric(df.unixtimestamp, errors="coerce")
        # print(df.head())
        # timestamp uptime
        df.fillna(0, inplace=True)
        df["max"] = df.stm_tim_uptime[
            ((df.stm_tim_uptime.shift(1) < df.stm_tim_uptime)
             & (df.stm_tim_uptime.shift(-1) < (df.stm_tim_uptime - cfg.UPTIME_MIN_REBOOT_THRESHOLD)))
            | (df.stm_tim_uptime.shift(-1) < (df.stm_tim_uptime / 2))
            ]
        df["min"] = df.stm_tim_uptime[
            (df.stm_tim_uptime.shift(-1) > df.stm_tim_uptime)
            & (df.stm_tim_uptime.shift(+1) > (df.stm_tim_uptime + cfg.UPTIME_MIN_REBOOT_THRESHOLD))
            ]

        # print(df.loc[~pd.isna(df["max"]), "max"])
        # print(df["max"])
        peak_result = df[df["max"] > cfg.UPTIME_MIN_PEAK_THRESHOLD]

        csv_file_path_sd_peaks = cfg.CSV_WORKING_DIR + (cfg.CSV_FILENAME_01_SD_UPTIME_PEAKS
                                                        % (type_id, slave_id))
        df_sd = pd.read_csv(csv_file_path_sd_peaks, sep=cfg.CSV_SEP)
        last_sd_uptime = df_sd.tail(1).stm_tim_uptime.values[0]
        last_peak_index = peak_result.tail(1).index.values[0]
        # print(df[df.stm_tim_uptime == last_sd_uptime])
        last_result = df[(df.index > last_peak_index) & (df.stm_tim_uptime == last_sd_uptime)]
        # last_result = df[df.index > last_peak_index]
        # last_result = df[df.index == last_peak_index]
        # last_result = df[df.stm_tim_uptime == last_sd_uptime]
        last_result["max"] = last_result.stm_tim_uptime
        # print("\nlast_result\n", last_result)
        # print("\nlast_result["max"].index = ", last_result["max"].index, "\n")
        # print("\ndf = ", df, "\n")
        # print("\ndf.columns.get_loc("max") = ", df.columns.get_loc("max"), "\n")
        # print("\ndf.loc[last_result.index, df.columns.get_loc("max")] = ",
        #       df.iloc[last_result.index, df.columns.get_loc("max")], "\n")
        # print("x: ", df.loc[last_result.index, df.columns.get_loc(uptime_row)])
        df.iloc[last_result.index, df.columns.get_loc("max")] = last_result["max"]
        peak_result = pd.concat([peak_result, last_result])
        logging.log.debug(peak_result)

        csv_file_path_peaks = cfg.CSV_WORKING_DIR + (cfg.CSV_FILENAME_02_INFLUX_UPTIME_PEAKS
                                                     % (type_id, slave_id))
        peak_result.to_csv(csv_file_path_peaks, index_label="index", sep=cfg.CSV_SEP)

        # mask = df[cfg.CSV_FILE_HEADER_UNIX_TIMESTAMP].between(1667176600,1667382000)
        # print(df.iloc[787550:787591])
        # df.to_csv("temp1.csv")
        if cfg.PLOT:
            plt.plot(df.unixtimestamp, df.stm_tim_uptime)
            plt.scatter(df.unixtimestamp, df["max"], c="g")
            plt.scatter(df.unixtimestamp, df["min"], c="b")
            plt.title("Slave %s%02u: Uptime over timestamp  from InfluxDB" % (type_id, slave_id))
            plt.xlabel("Timestamp")
            plt.ylabel("Timer Count")
            plt.grid(True)
            plt.show()

        logging.log.info("Thread %u Slave %s%02u - done" % (processor_number, type_id, slave_id))

    logging.log.info("Thread %u - no more slaves - exiting" % processor_number)
    slave_queue.close()


if __name__ == "__main__":
    run()
