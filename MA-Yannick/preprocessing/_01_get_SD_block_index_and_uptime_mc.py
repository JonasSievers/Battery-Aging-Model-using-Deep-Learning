import config_preprocessing as cfg
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
from datetime import datetime
import os
import re
import config_logging as logging


# declaration of global variables and composite constants
CSV_FILE_HEADER = "SD_block_ID;stm_tim_uptime"
pd.options.mode.chained_assignment = None  # default="warn"
NUMBER_OF_PROCESSORS_TO_USE = multiprocessing.cpu_count()
task_queue = multiprocessing.Queue()


def run():
    start_timestamp = datetime.now()
    logging.log.info(os.path.basename(__file__))

    # find files
    with os.scandir(cfg.SD_IMAGE_PATH) as iterator:
        regex_pattern_slave = re.compile(cfg.SD_IMAGE_FILE_REGEX_SLAVE)
        regex_pattern_tmgmt = re.compile(cfg.SD_IMAGE_FILE_REGEX_TMGMT)
        for entry in iterator:
            re_match_slave = regex_pattern_slave.fullmatch(entry.name)
            if re_match_slave:
                slave_id = int(re_match_slave.group(2))
                logging.log.info("Found image of Slave %02u: '%s'" % (slave_id, entry.name))
                queue_entry = {"filename": entry.name, "slave_id": slave_id, "type": "S"}
                task_queue.put(queue_entry)
            else:
                re_match_tmgmt = regex_pattern_tmgmt.fullmatch(entry.name)
                if re_match_tmgmt:
                    slave_id = int(re_match_tmgmt.group(2))
                    logging.log.info("Found image of Thermal Management %02u: '%s'" % (slave_id, entry.name))
                    queue_entry = {"filename": entry.name, "slave_id": slave_id, "type": "T"}
                    task_queue.put(queue_entry)

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


def run_thread(processor_number, file_queue):
    while True:
        if (file_queue is None) or file_queue.empty():
            break  # no more files

        try:
            queue_entry = file_queue.get_nowait()
        except multiprocessing.queues.Empty:
            break  # no more files

        if queue_entry is None:
            break  # no more files

        sd_file_name = cfg.SD_IMAGE_PATH + queue_entry["filename"]
        slave_id = queue_entry["slave_id"]
        type_id = queue_entry["type"]
        logging.log.info("Thread %u Slave %s%02u - starting extraction of SD block index and uptime"
                         % (processor_number, type_id, slave_id))

        # check if file size is a multiple of the SD block size
        size = os.path.getsize(sd_file_name)
        logging.log.debug("Thread %u Slave %s%02u - Size of file: %u" % (processor_number, type_id, slave_id, size))
        if size == 0 or (size % cfg.SD_BLOCK_SIZE_BYTES) != 0:
            logging.log.warning("Thread %u Slave %s%02u - Block error, size of slave image '%s' not n*%u"
                                % (processor_number, type_id, slave_id, queue_entry["filename"],
                                   cfg.SD_BLOCK_SIZE_BYTES))
            raise Exception

        # open file, write header
        file = open(sd_file_name, "rb")
        csv_file_path = cfg.CSV_WORKING_DIR + (cfg.CSV_FILENAME_01_SD_UPTIME % (type_id, slave_id))
        csv_file_uptime = open(csv_file_path, "w")
        csv_file_uptime.write(CSV_FILE_HEADER + "\n")

        # write SD block ID and uptime to CSV file
        for i in range(0, size, cfg.SD_BLOCK_SIZE_BYTES):
            data = file.read(cfg.SD_BLOCK_SIZE_BYTES)

            if (data[0:4] == b'MLB1') or (data[0:4] == b'MLT1'):  # LOG or TLOG
                stm_tim_uptime = int.from_bytes(data[8:12], byteorder="big", signed=False)
                content_string = str(i) + ";" + str(stm_tim_uptime) + "\n"
                csv_file_uptime.write(content_string)

        csv_file_uptime.close()

        # find maxima/peaks of SD card uptime and store to csv
        df = pd.read_csv(csv_file_path, sep=cfg.CSV_SEP, header=0)
        # SD_block_ID;stm_tim_uptime
        # uptime_row = cfg.CSV_FILE_HEADER_STM_TIM_UPTIME
        df["max"] = df.stm_tim_uptime[
            (df.stm_tim_uptime.shift(1) < df.stm_tim_uptime) & (df.stm_tim_uptime.shift(-1) < df.stm_tim_uptime)
        ]
        df["min"] = df.stm_tim_uptime[
            (df.stm_tim_uptime.shift(1) > df.stm_tim_uptime) & (df.stm_tim_uptime.shift(-1) > df.stm_tim_uptime)
        ]

        logging.log.debug("\nThread %u Slave %s%02u:\n%s"
                          % (processor_number, type_id, slave_id, df.loc[~pd.isna(df["max"]), "max"].to_string()))
        csv_file_path_peaks = cfg.CSV_WORKING_DIR + (cfg.CSV_FILENAME_01_SD_UPTIME_PEAKS % (type_id, slave_id))
        df_peaks = df[df["max"] > cfg.UPTIME_MIN_PEAK_THRESHOLD]
        logging.log.debug("\nThread %u - df_peaks (1):\n%s" % (processor_number, df_peaks.to_string()))

        last_row = df.tail(1)
        last_row["max"] = last_row.stm_tim_uptime
        df.iloc[-1, df.columns.get_loc("max")] = last_row["max"]
        logging.log.debug("\nThread %u Slave %s%02u - last_row:\n%s" % (processor_number, type_id, slave_id, last_row))
        df_peaks = pd.concat([df_peaks, last_row])
        logging.log.debug("\nThread %u Slave %s%02u - df_peaks (2):\n%s"
                          % (processor_number, type_id, slave_id, df_peaks))

        csv_df = df_peaks.to_csv(csv_file_path_peaks, index_label="index", sep=cfg.CSV_SEP)
        logging.log.debug("\nThread %u Slave %s%02u - csv_df:\n%s" % (processor_number, type_id, slave_id, csv_df))

        if cfg.PLOT:
            plt.plot(df.index, df.stm_tim_uptime)
            plt.scatter(df.index, df["max"], c="g")
            plt.scatter(df.index, df["min"], c="r")

            plt.title("Slave %s%02u: stm_tim_uptime over index with local max detection" % (type_id, slave_id))
            plt.xlabel("Index Count")
            plt.ylabel("Timer Count")
            plt.grid(True)
            plt.show()

        logging.log.info("Thread %u Slave %s%02u - done" % (processor_number, type_id, slave_id))

    logging.log.info("Thread %u - no more slaves - exiting" % processor_number)
    file_queue.close()


if __name__ == "__main__":
    run()
