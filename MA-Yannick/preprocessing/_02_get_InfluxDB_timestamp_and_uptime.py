from influxdb_client import InfluxDBClient
from datetime import datetime
import time
import warnings
from influxdb_client.client.warnings import MissingPivotFunction
import pandas as pd
import matplotlib.pyplot as plt
import config_preprocessing as cfg
import os
import re
import config_logging as logging


# configuration
LOAD_FROM_INFLUX = False  # True
DELTA_UPTIME_REBOOT_THRESHOLD_MIN = 5000  # subsequent uptime needs to be at least this value lower in order to call
# it a new session

# constants
pd.options.mode.chained_assignment = None  # default='warn'
uptime_row = cfg.CSV_FILE_HEADER_STM_TIM_UPTIME
start_timestamp = datetime.now()


def run():
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
                slave = {'id': slave_id, 'type': slave_type}
                slaves.append(slave)
                slaves_string.append("%s%02u" % (slave_type, slave_id))
    logging.log.info("Found .csv files for slaves: %s" % slaves_string)

    if LOAD_FROM_INFLUX:
        warnings.simplefilter("ignore", MissingPivotFunction)

        client = InfluxDBClient(url=cfg.influx_url, token=cfg.influx_token, org=cfg.influx_org)
        query_api = client.query_api()

        # =====================================================================================================
        # Query data
        # Datenbank kann maximal nur einen Monat schicken = 30+24+3600 = 23887872 Sekunden
        # Iteration über Startpunkt bis now() in 23887872 Sekunden Schritten
        # Bis wann gehen die Daten der SD Karte? -> 2023-02-01 manuell nachgeschaut
        stop_date = datetime(2023, 2, 2, 0, 0, 0)
        stop_date_timestamp = int(time.mktime(stop_date.timetuple()))
        start_date_timestamp = 1665593100  # -> Mi, 12.10.2022  16:45:00 Siehe schedule_2022-10-12_experiment_LG_HG2.txt

        # Split the timestamps into monthly chunks
        length = stop_date_timestamp - start_date_timestamp
        block_length = int(length / 12)

        for slave in slaves:
            # Slave Number
            slave_name = "%s%02u" % (slave['type'], slave['id'])
            csv_file_path = cfg.CSV_WORKING_DIR + (cfg.CSV_FILENAME_02_INFLUX_UPTIME % (slave['type'], slave['id']))

            header = True
            write_mode = 'w'  # start with overwriting, change to append after first write

            for block_start in range(start_date_timestamp, stop_date_timestamp, block_length):
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
                logging.log.debug("   Warte auf Datenbank...")
                csv_result_pandas = query_api.query_data_frame(query)
                logging.log.debug("   Umrechnung und Export zu CSV...")
                csv_result_pandas = csv_result_pandas.drop('result', axis=1)
                csv_result_pandas = csv_result_pandas.drop('table', axis=1)
                csv_result_pandas['timestamp'] = (csv_result_pandas['timestamp'] / 1000000000).astype(int)
                # print(csv_result_pandas.head())
                csv_result_pandas.rename(columns={'timestamp': cfg.CSV_FILE_HEADER_UNIX_TIMESTAMP}, inplace=True)
                csv_result_pandas.rename(columns={'uptime': uptime_row}, inplace=True)
                csv_result_pandas.to_csv(csv_file_path, index=False, mode=write_mode, sep=cfg.CSV_SEP, header=header)
                write_mode = 'a'
                header = False

        client.close()

    for slave in slaves:
        # Slave Number
        slave_name = "%s%02u" % (slave['type'], slave['id'])
        csv_file_path = cfg.CSV_WORKING_DIR + (cfg.CSV_FILENAME_02_INFLUX_UPTIME % (slave['type'], slave['id']))

        logging.log.info("Analyzing uptime values - Slave %s" % slave_name)

        df = pd.read_csv(csv_file_path, sep=cfg.CSV_SEP, header=0)
        df[uptime_row] = pd.to_numeric(df[uptime_row], errors='coerce')
        df[cfg.CSV_FILE_HEADER_UNIX_TIMESTAMP] = pd.to_numeric(df[cfg.CSV_FILE_HEADER_UNIX_TIMESTAMP],
                                                               errors='coerce')
        # print(df.head())
        # timestamp uptime
        df.fillna(0, inplace=True)
        df['max'] = df[uptime_row][
            ((df[uptime_row].shift(1) < (df[uptime_row] - DELTA_UPTIME_REBOOT_THRESHOLD_MIN))
             & (df[uptime_row].shift(-1) < (df[uptime_row] - DELTA_UPTIME_REBOOT_THRESHOLD_MIN)))
            | (df[uptime_row].shift(-1) < (df[uptime_row] / 2))
        ]
        df['min'] = df[uptime_row][
            (df[uptime_row].shift(-1) > df[uptime_row]) & (df[uptime_row].shift(+1) > df[uptime_row])
        ]

        # print(df.loc[~pd.isna(df['max']), 'max'])
        # print(df['max'])
        peak_result = df[df['max'] > cfg.UPTIME_MIN_PEAK_THRESHOLD]

        csv_file_path_sd_peaks = cfg.CSV_WORKING_DIR + (cfg.CSV_FILENAME_01_SD_UPTIME_PEAKS
                                                        % (slave['type'], slave['id']))
        df_sd = pd.read_csv(csv_file_path_sd_peaks, sep=cfg.CSV_SEP)
        last_sd_uptime = df_sd.tail(1)[uptime_row].values[0]
        last_peak_index = peak_result.tail(1).index.values[0]
        # print(df[df[uptime_row] == last_sd_uptime])
        last_result = df[(df.index > last_peak_index) & (df[uptime_row] == last_sd_uptime)]
        # last_result = df[df.index > last_peak_index]
        # last_result = df[df.index == last_peak_index]
        # last_result = df[df[uptime_row] == last_sd_uptime]
        last_result['max'] = last_result[uptime_row]
        # print("\nlast_result\n", last_result)
        # print("\nlast_result['max'].index = ", last_result['max'].index, "\n")
        # print("\ndf = ", df, "\n")
        # print("\ndf.columns.get_loc('max') = ", df.columns.get_loc('max'), "\n")
        # print("\ndf.loc[last_result.index, df.columns.get_loc('max')] = ",
        #       df.iloc[last_result.index, df.columns.get_loc('max')], "\n")
        # print("x: ", df.loc[last_result.index, df.columns.get_loc(uptime_row)])
        df.iloc[last_result.index, df.columns.get_loc('max')] = last_result['max']
        peak_result = pd.concat([peak_result, last_result])
        logging.log.debug(peak_result)

        csv_file_path_peaks = cfg.CSV_WORKING_DIR + (cfg.CSV_FILENAME_02_INFLUX_UPTIME_PEAKS
                                                     % (slave['type'], slave['id']))
        peak_result.to_csv(csv_file_path_peaks, index_label="index", sep=cfg.CSV_SEP)

        # mask = df[cfg.CSV_FILE_HEADER_UNIX_TIMESTAMP].between(1667176600,1667382000)
        # print(df.iloc[787550:787591])
        # df.to_csv('temp1.csv')
        if cfg.PLOT:
            plt.plot(df[cfg.CSV_FILE_HEADER_UNIX_TIMESTAMP], df[uptime_row])
            plt.scatter(df[cfg.CSV_FILE_HEADER_UNIX_TIMESTAMP], df['max'], c="g")
            plt.scatter(df[cfg.CSV_FILE_HEADER_UNIX_TIMESTAMP], df['min'], c="b")
            plt.title("Uptime over timestamp  from InfluxDB")
            plt.xlabel("Timestamp")
            plt.ylabel("Timer Count")
            plt.grid(True)
            plt.show()

    stop_timestamp = datetime.now()
    logging.log.info("\nScript runtime: %s h:mm:ss.ms" % str(stop_timestamp - start_timestamp))


if __name__ == '__main__':
    run()
