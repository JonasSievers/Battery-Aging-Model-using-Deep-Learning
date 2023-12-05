import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import config_preprocessing as cfg


def run():
    slave = "S01"
    timestamp_row = cfg.CSV_FILE_HEADER_UNIX_TIMESTAMP

    csv_file_path_sd_peaks = cfg.CSV_WORKING_DIR + (cfg.CSV_FILENAME_01_SD_UPTIME_PEAKS % slave)
    csv_file_path_influx_peaks = cfg.CSV_WORKING_DIR + (cfg.CSV_FILENAME_02_INFLUX_UPTIME_PEAKS % slave)

    df1 = pd.read_csv(csv_file_path_sd_peaks, sep=";")
    df1.sort_values(["max"], inplace=True)

    df2 = pd.read_csv(csv_file_path_influx_peaks, sep=";")
    df2.sort_values(["max"], inplace=True)

    df3 = pd.merge_asof(df2, df1, on='max', direction='forward', allow_exact_matches=True, )
    df3.sort_values('index_x', inplace=True)
    print(df3)

    # print results
    slope, intercept, r_value, p_value, std_err = linregress(df3[timestamp_row],
                                                             df3[cfg.CSV_FILE_HEADER_SD_BLOCK_ID])
    print("Slope:", slope)
    print("Intercept:", intercept)
    print("R-squared:", r_value**2)

    export_df = df3[[timestamp_row, cfg.CSV_FILE_HEADER_SD_BLOCK_ID]]
    csv_file_path_merged_table = cfg.CSV_WORKING_DIR + (cfg.CSV_FILENAME_03_TIMESTAMP_AND_SD_BLOCK_ID_PEAKS % slave)
    export_df.to_csv(csv_file_path_merged_table, header=True, mode='w', sep=";")

    if cfg.PLOT:
        plt.scatter(df3[timestamp_row], df3[cfg.CSV_FILE_HEADER_SD_BLOCK_ID], c="g")
        # plt.scatter(df3['timestamp'], df3['min'],c="b")
        plt.title("Uptime over timestamp  from InfluxDB")
        plt.xlabel("Timestamp")
        plt.ylabel("SCByte Count")
        plt.grid(True)
        plt.plot(df3[timestamp_row], intercept + slope*df3[timestamp_row], 'r')
        plt.show()


if __name__ == '__main__':
    run()
