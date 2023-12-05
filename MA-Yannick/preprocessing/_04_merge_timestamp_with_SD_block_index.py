import pandas
import pandas as pd
import matplotlib.pyplot as plt
import config_preprocessing as cfg

pd.options.mode.chained_assignment = None  # default='warn'


def run():
    # read SD data of slave n# in chunks
    slave = "S01"
    timestamp_row = cfg.CSV_FILE_HEADER_UNIX_TIMESTAMP

    sd_df = pd.read_csv(cfg.CSV_WORKING_DIR + (cfg.CSV_FILENAME_01_SD_UPTIME % slave), header=0, sep=";")
    
    # read the fixed timestamps from the 3.rd script:
    # Peak values (highest uptime counter) are the known timestamp references
    timestamp_df = pd.read_csv(cfg.CSV_WORKING_DIR + (cfg.CSV_FILENAME_03_TIMESTAMP_AND_SD_BLOCK_ID_PEAKS % slave),
                               sep=";", header=0, index_col=0)
    timestamp_df[timestamp_row] = (timestamp_df[timestamp_row] * 1000).astype('int64')
    # timestamp_df[timestamp_row] = timestamp_df[timestamp_row].astype(int)

    # print("\ntimestamp_df (1):\n", timestamp_df)
    
    # Merge the timestamp df with the datadf, so that the few peak values of the sd card get a timestamp
    merged_df = pd.merge(left=sd_df, right=timestamp_df, how="left", on=cfg.CSV_FILE_HEADER_SD_BLOCK_ID)
    # merged_df.to_csv("temp.csv")
    
    # Dataframe with only sd card rows with a fixed timestamp (peak values)
    # The index of this df is the row number of the sdcard dataframe
    values_with_time_stamp_df = merged_df[merged_df[timestamp_row].notnull()]

    values_with_time_stamp_df[timestamp_row + '_str'] =\
        pandas.to_datetime(values_with_time_stamp_df.loc[:, timestamp_row], unit="ms")

    # values_with_time_stamp_df.to_csv("temp.csv")
    print("\nvalues_with_time_stamp_df:\n", values_with_time_stamp_df)
    # SD_block_ID   stm_tim_uptime              unixtimestamp
    # 28137               81549824    5233034   1.665648e+09
    # 296909             220015104   50705140   1.666186e+09
    # 301985             222614016     968123   1.666196e+09
    
    # Counter for lower boundary
    row_index_before = 0
    old_timestamp = 0
    for index, row in values_with_time_stamp_df.iterrows():
        row_start_index = row_index_before
        offset_list_index = row_index_before
        row_stop_index = index
        timestamp_stop = int(row.unixtimestamp)  # (timestamp to milliseconds)
        if old_timestamp == 0:
            timestamp_start = timestamp_stop - (
                1999 * (row_stop_index - row_start_index)
            )  # (timestamp to milliseconds)
        else:
            timestamp_start = old_timestamp
    
        duration_of_one_row = int(
            ((timestamp_stop - timestamp_start) / (row_stop_index - row_start_index))
        )  # (duration from seconds to milliseconds and float to int)
        # print("Duration: ",duration_of_one_row)
    
        # print("Anfang ist : {block_start} Ende ist {block_stop},"
        #       "timestamp Anfang ist: {t_a} timestamp Ende ist: {t_b}, "
        #       .format(block_start=row_index_before, block_stop=index, t_a= timestamp_start, t_b=timestamp_stop))

        # create a list of numbers increasing by 2
        timestamps = list(range(timestamp_start, timestamp_stop, duration_of_one_row))
        # print(timestamps)

        # create a dataframe with the numbers as a column
        timestamp_df = pd.DataFrame(
            timestamps,
            index=range(offset_list_index, len(timestamps) + offset_list_index),
            columns=[timestamp_row],
        )
        #         unixtimestamp
        # 28137   1665648461000
        # 28138   1665648462999
        # 28139   1665648464998

        # print("\ntimestamp_df (2):\n", timestamp_df)
        # print(merged_df)
        merged_df = pd.merge(
            left=merged_df,
            right=timestamp_df,
            left_index=True,
            right_index=True,
            how="left",
        )
        # print(merged_df)
        # merged_df.loc[merged_df[timestamp_row + '_y'].notnull(), timestamp_row + '_y'] *= 1000
        # print(merged_df)
        merged_df[timestamp_row + "_x"] = merged_df[timestamp_row + "_x"].fillna(merged_df[timestamp_row + "_y"])
        # print(merged_df)
        merged_df.drop(timestamp_row + "_y", axis=1, inplace=True)
        # print(merged_df)
        merged_df.rename(columns={timestamp_row + "_x": timestamp_row}, inplace=True)
        # print(merged_df)
    
        old_timestamp = timestamp_stop
        row_index_before = index
    
    # Handle the case if there is Data behind the last Max Point, assuming the frequency of measurement is 1999 ms
    
    length_sd_data = int(merged_df.tail(1).index[0])
    length_timestamp_data = int(values_with_time_stamp_df.tail(1).index[0])
    start_timestamp = int(values_with_time_stamp_df.tail(1).unixtimestamp)
    end_timestamp = int((length_sd_data - length_timestamp_data) * 1999) + start_timestamp + 1000
    print("length_sd_data: %u, length_timestamp_data: %u, start_timestamp: %u, end_timestamp: %u\n"
          % (length_sd_data, length_timestamp_data, start_timestamp, end_timestamp))
    
    if length_sd_data > length_timestamp_data:
        timestamps = list(range(start_timestamp, end_timestamp, 1999))
        timestamp_df = pd.DataFrame(
            timestamps,
            index=range(length_timestamp_data, len(timestamps) + length_timestamp_data),
            columns=[timestamp_row],
        )

        # print("\ntimestamp_df (3):\n", timestamp_df)
        merged_df = pd.merge(
            left=merged_df,
            right=timestamp_df,
            left_index=True,
            right_index=True,
            how="left",
        )
        # print("\nmerged_df (1):\n", merged_df)
        merged_df[timestamp_row + "_x"] = merged_df[timestamp_row + "_x"].fillna(merged_df[timestamp_row + "_y"])
        # print("\nmerged_df (2):\n", merged_df)
        merged_df.drop(timestamp_row + "_y", axis=1, inplace=True)
        # print("\nmerged_df (3):\n", merged_df)
        merged_df.rename(columns={timestamp_row + "_x": timestamp_row}, inplace=True)
        # print("\nmerged_df (4):\n", merged_df)

    # print("\nmerged_df (5):\n", merged_df)
    plot_df = merged_df
    plot_df[timestamp_row] = plot_df[timestamp_row].astype('int64')
    print("\nEmpty cells:\n", plot_df[plot_df[timestamp_row].isna()])
    
    plot_df.to_csv(cfg.CSV_WORKING_DIR + (cfg.CSV_FILENAME_04_SD_BLOCK_ID_UPTIME_TIMESTAMP % slave), sep=";")

    plot_df[timestamp_row + '_str'] = pandas.to_datetime(plot_df.loc[:, timestamp_row], unit="ms")
    print("\nDataframe:\n", plot_df)
    if cfg.PLOT:
        plot_df.plot(x=timestamp_row, y=cfg.CSV_FILE_HEADER_SD_BLOCK_ID)
        plt.show()
    # synckit


if __name__ == '__main__':
    run()
