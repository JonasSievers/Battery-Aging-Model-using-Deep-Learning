import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import config_preprocessing as cfg


# constants
MAX_LOOKAROUND = 1000  # search +/- this entries around the index of SD_block_ID that got the estimated timestamp
# to find the SD block ID with the exact TIM value
# 1000 --> ca. 2000 seconds = 33 minutes -> TIM value +/- 190735
sd_block_id_row = cfg.CSV_FILE_HEADER_SD_BLOCK_ID
pd.options.mode.chained_assignment = None  # default='warn'


def run():
    slave = "S01"
    timestamp_row = cfg.CSV_FILE_HEADER_UNIX_TIMESTAMP

    csv_file_path_sd_peaks = cfg.CSV_WORKING_DIR + (cfg.CSV_FILENAME_01_SD_UPTIME_PEAKS % slave)
    csv_file_path_influx_peaks = cfg.CSV_WORKING_DIR + (cfg.CSV_FILENAME_02_INFLUX_UPTIME_PEAKS % slave)

    df1 = pd.read_csv(csv_file_path_sd_peaks, sep=cfg.CSV_SEP)
    df1.sort_values(["max"], inplace=True)

    df2 = pd.read_csv(csv_file_path_influx_peaks, sep=cfg.CSV_SEP)
    df2.sort_values(["max"], inplace=True)

    df3 = pd.merge_asof(df2, df1, on='max', direction='forward', allow_exact_matches=True, )
    df3.sort_values('index_x', inplace=True)
    print("df3:\n", df3.to_string())

    # print results
    slope, intercept, r_value, p_value, std_err = linregress(df3[timestamp_row],
                                                             df3[sd_block_id_row])
    print("   Slope:", slope)
    print("   Intercept:", intercept)
    print("   R-squared:", r_value**2)

    # try to find exact matches
    sd_csv_file_path = cfg.CSV_WORKING_DIR + (cfg.CSV_FILENAME_01_SD_UPTIME % slave)
    sd_df = pd.read_csv(sd_csv_file_path, sep=cfg.CSV_SEP, header=0)
    stm_tim_uptime_index = sd_df.columns.get_loc(cfg.CSV_FILE_HEADER_STM_TIM_UPTIME)
    sd_block_id_index = sd_df.columns.get_loc(sd_block_id_row)
    for index, row in df3.iterrows():
        sd_block_id_start = row[sd_block_id_row]
        exact_uptime = row[cfg.CSV_FILE_HEADER_STM_TIM_UPTIME + "_x"]
        sd_df_index = sd_df[sd_block_id_row][(sd_df[sd_block_id_row] == sd_block_id_start)].index.values[0]
        exact_sd_block_id = None  # 0
        for i in range(0, MAX_LOOKAROUND):
            i_1 = sd_df_index - i
            uptime_1 = sd_df.iloc[i_1, stm_tim_uptime_index]
            if uptime_1 == exact_uptime:
                exact_sd_block_id = sd_df.iloc[i_1, sd_block_id_index]
                break
            else:
                i_2 = sd_df_index + i
                uptime_2 = sd_df.iloc[i_2, stm_tim_uptime_index]
                if uptime_2 == exact_uptime:
                    exact_sd_block_id = sd_df.iloc[i_2, sd_block_id_index]
                    break
        if exact_sd_block_id is None:
            print("No exact SD_block_id found for index %u with uptime %u" % (index, exact_uptime))
            exact_sd_block_id = sd_block_id_start  # at least use the start index
        df3.loc[index, sd_block_id_row + '_exact'] = exact_sd_block_id  # .astype('int64')

    df3[sd_block_id_row + '_exact'] = df3[sd_block_id_row + '_exact'].astype('int64')
    print("df3 (exact):\n", df3.to_string())

    # print results
    slope, intercept, r_value, p_value, std_err = linregress(df3[timestamp_row],
                                                             df3[sd_block_id_row + '_exact'])
    print("   Slope:", slope)
    print("   Intercept:", intercept)
    print("   R-squared:", r_value**2)

    export_df = df3[[timestamp_row, sd_block_id_row + '_exact']]
    export_df.rename(columns={sd_block_id_row + '_exact': sd_block_id_row}, inplace=True)
    csv_file_path_merged_table = cfg.CSV_WORKING_DIR + (cfg.CSV_FILENAME_03_TIMESTAMP_AND_SD_BLOCK_ID_PEAKS % slave)
    export_df.to_csv(csv_file_path_merged_table, index_label="index", header=True, mode='w', sep=cfg.CSV_SEP)

    if cfg.PLOT:
        plt.scatter(df3[timestamp_row], df3[sd_block_id_row + '_exact'], c="g")
        # plt.scatter(df3['timestamp'], df3['min'],c="b")
        plt.title("Uptime over timestamp  from InfluxDB")
        plt.xlabel("Timestamp")
        plt.ylabel("SCByte Count")
        plt.grid(True)
        plt.plot(df3[timestamp_row], intercept + slope*df3[timestamp_row], 'r')
        plt.show()


if __name__ == '__main__':
    run()
