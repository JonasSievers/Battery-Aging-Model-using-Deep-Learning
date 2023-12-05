import pandas
import pandas as pd
import matplotlib.pyplot as plt
import config_preprocessing as cfg

# constants and definitions
DELTA_UPTIME_REBOOT = cfg.UPTIME_MIN_REBOOT_THRESHOLD
pd.options.mode.chained_assignment = None  # default='warn'
timestamp_row = cfg.CSV_FILE_HEADER_UNIX_TIMESTAMP
sd_block_id_row = cfg.CSV_FILE_HEADER_SD_BLOCK_ID
uptime_row = cfg.CSV_FILE_HEADER_STM_TIM_UPTIME


def run():
    # read SD data of slave n# in chunks
    slave = "S01"

    sd_df = pd.read_csv(cfg.CSV_WORKING_DIR + (cfg.CSV_FILENAME_01_SD_UPTIME % slave), header=0, sep=cfg.CSV_SEP)
    
    # read the fixed timestamps from the 3.rd script:
    # Peak values (highest uptime counter) are the known timestamp references
    timestamp_df = pd.read_csv(cfg.CSV_WORKING_DIR + (cfg.CSV_FILENAME_03_TIMESTAMP_AND_SD_BLOCK_ID_PEAKS % slave),
                               sep=cfg.CSV_SEP, header=0, index_col=0)
    timestamp_df[timestamp_row + '_ms'] = (timestamp_df[timestamp_row] * 1000).astype('int64')
    # timestamp_df[timestamp_row] = timestamp_df[timestamp_row].astype(int)

    # print("\ntimestamp_df (1):\n", timestamp_df)
    
    # Merge the timestamp df with the datadf, so that the few peak values of the sd card get a timestamp
    merged_df = pd.merge(left=sd_df, right=timestamp_df, how="left", on=sd_block_id_row)
    # merged_df.to_csv("temp.csv")
    
    # Dataframe with only sd card rows with a fixed timestamp (peak values)
    # The index of this df is the row number of the sdcard dataframe
    values_with_time_stamp_df = merged_df[merged_df[timestamp_row + '_ms'].notnull()]

    values_with_time_stamp_df[timestamp_row + '_str'] =\
        pandas.to_datetime(values_with_time_stamp_df.loc[:, timestamp_row + '_ms'], unit="ms")

    # values_with_time_stamp_df.to_csv("temp.csv")
    print("\nvalues_with_time_stamp_df:\n", values_with_time_stamp_df.to_string())
    # SD_block_ID   stm_tim_uptime              unixtimestamp
    # 28137               81549824    5233034   1.665648e+09
    # 296909             220015104   50705140   1.666186e+09
    # 301985             222614016     968123   1.666196e+09

    influx_csv_file_path = cfg.CSV_WORKING_DIR + (cfg.CSV_FILENAME_02_INFLUX_UPTIME % slave)
    influx_df = pd.read_csv(influx_csv_file_path, sep=cfg.CSV_SEP)

    # assign timestamps to interim influx values
    i_session_start_min_ifx = 0
    i_session_start_min_sd = 0
    sd_df_merged = sd_df
    for index, row in values_with_time_stamp_df.iterrows():
        # Here, a "session" is defined as the runtime of a slave board.
        # If the slave board reboots, a new session starts.

        # 1. in the complete influx data set, find first item of current session
        #    a. find last unixtimestamp of session (as in values_with_time_stamp_df) in complete influx data set
        #    b. go to left until the first value arrives that is significantly larger
        # 2. in the complete SD data set, find the first item of the current session
        #    a. find last unixtimestamp of session (as in values_with_time_stamp_df) in complete SD data set
        #    b. go to left until the first value arrives that is significantly larger

        # influx -------------------------------------------------------------------------------------------------------
        i_session_end_ifx = influx_df[timestamp_row][influx_df[timestamp_row] == row[timestamp_row]].index.values[0]
        # print("i_session_end_ifx: ", i_session_end_ifx)

        # select the window from the influx data set in which the current session starts ()
        ifx_sess_win_df = influx_df.iloc[i_session_start_min_ifx:(i_session_end_ifx + 1), :]
        # print("\nifx_sess_win_df:\n", ifx_sess_win_df)
        ifx_sess_win_df['min'] = ifx_sess_win_df[uptime_row][
            (ifx_sess_win_df[uptime_row].shift(-1) > ifx_sess_win_df[uptime_row])
            & (ifx_sess_win_df[uptime_row].shift(1) > (ifx_sess_win_df[uptime_row] + DELTA_UPTIME_REBOOT))
            ]
        ifx_sess_win_df_min = ifx_sess_win_df.loc[~pd.isna(ifx_sess_win_df['min']), 'min']
        # print("\nifx_sess_win_df_min\n", ifx_sess_win_df_min)
        i_session_start_ifx = i_session_start_min_ifx
        ifx_sess_win_df_min_length = len(ifx_sess_win_df_min.index)
        if ifx_sess_win_df_min_length > 0:
            # one or more minima found in between - select the last minimum as the start of the relevant session
            # (this occurs if the data frame starts with small sessions that were ignored in the peak detection)
            i_session_start_ifx = ifx_sess_win_df_min.index[-1]
        # else: no minima found - the first index is the start of the current session (ifx_sess_win_df is unchanged)
        ifx_sess_win_df = influx_df.iloc[i_session_start_ifx:(i_session_end_ifx + 1), :]
        # ifx_sess_win_df.drop('min', inplace=True, axis=1)  # drop 'min' column
        # ifx_sess_win_df[timestamp_row + '_str'] = pandas.to_datetime(ifx_sess_win_df.loc[:, timestamp_row], unit="s")
        # print("\nifx_sess_win_df\n", ifx_sess_win_df)

        # SD -----------------------------------------------------------------------------------------------------------
        i_session_end_sd = sd_df[sd_block_id_row][sd_df[sd_block_id_row] == row[sd_block_id_row]].index.values[0]
        # print("i_session_end_sd: ", i_session_end_sd)

        # select the window from the SD data set in which the current session starts ()
        sd_sess_win_df = sd_df.iloc[i_session_start_min_sd:(i_session_end_sd + 1), :]
        # print("\nsd_sess_win_df:\n", sd_sess_win_df)
        sd_sess_win_df['min'] = sd_sess_win_df[uptime_row][
            (sd_sess_win_df[uptime_row].shift(-1) > sd_sess_win_df[uptime_row])
            & (sd_sess_win_df[uptime_row].shift(1) > (sd_sess_win_df[uptime_row] + DELTA_UPTIME_REBOOT))
            ]
        sd_sess_win_df_min = sd_sess_win_df.loc[~pd.isna(sd_sess_win_df['min']), 'min']
        # print("\nsd_sess_win_df_min\n", sd_sess_win_df_min)
        i_session_start_sd = i_session_start_min_sd
        sd_sess_win_df_min_length = len(sd_sess_win_df_min.index)
        if sd_sess_win_df_min_length > 0:
            # one or more minima found in between - select the last minimum as the start of the relevant session
            # (this occurs if the data frame starts with small sessions that were ignored in the peak detection)
            i_session_start_sd = sd_sess_win_df_min.index[-1]
        # else: no minima found - the first index is the start of the current session (sd_sess_win_df is unchanged)
        sd_sess_win_df = sd_df.iloc[i_session_start_sd:(i_session_end_sd + 1), :]
        # sd_sess_win_df.drop('min', inplace=True, axis=1)  # drop 'min' column
        # print("\nsd_sess_win_df\n", sd_sess_win_df)

        merged_df = pd.merge(
            left=sd_sess_win_df,
            right=ifx_sess_win_df,
            on=uptime_row,
            left_index=False,
            right_index=False,
            how="outer",
        )
        # FIXME: what happens if uptime values occur twice in InfluxDB because they were resent by the Raspi/router?
        # e.g. Slave 14, 07.12.2022, 04:42:08 and 04:42:22

        # print("\nmerged_df\n", merged_df)

        # if (timestamp_row + '_x') in merged_df.columns:
        #     merged_df[timestamp_row + '_x'].combine_first(merged_df[timestamp_row + '_y'])
        #     merged_df.drop(timestamp_row + "_y", inplace=True, axis=1)
        #     merged_df.rename(columns={timestamp_row + "_x": timestamp_row}, inplace=True)

        ifx_sess_length = len(ifx_sess_win_df.index)
        if ifx_sess_length > 0:
            nan_count = merged_df.isna().sum()
            date_row_index = ifx_sess_win_df.columns.get_loc(timestamp_row)
            start_date = pandas.to_datetime(ifx_sess_win_df.iloc[0, date_row_index],  unit="s")
            end_date = pandas.to_datetime(ifx_sess_win_df.iloc[-1, date_row_index],  unit="s")
            print("\nSession from:   %s   to   %s" % (start_date, end_date))
            print("  Missing influx rows: %u" % nan_count[timestamp_row])
            print("  Missing SD rows    : %u" % nan_count[sd_block_id_row])
        else:
            print("\nEmpty session")

        sd_df_merged = pd.merge(
            left=sd_df_merged,
            right=merged_df,
            on=sd_block_id_row,
            left_index=False,
            right_index=False,
            how="left",
            # how="outer", if we would merge influx and SD data logs, we could also use "outer".
            # Assume nothing special happens if the SD card data is missing, so we just use the SD card data
        )
        sd_df_merged.drop(uptime_row + "_y", inplace=True, axis=1)  # drop 'stm_tim_uptime_y' column (from merged_df)
        sd_df_merged.rename(columns={uptime_row + "_x": uptime_row}, inplace=True)
        if (timestamp_row + '_x') in sd_df_merged.columns:
            sd_df_merged[timestamp_row] =\
                sd_df_merged[timestamp_row + '_x'].combine_first(sd_df_merged[timestamp_row + '_y'])
            sd_df_merged.drop(timestamp_row + "_x", inplace=True, axis=1)
            sd_df_merged.drop(timestamp_row + "_y", inplace=True, axis=1)
            # sd_df_merged.rename(columns={timestamp_row + "_x": timestamp_row}, inplace=True)

        i_session_start_min_ifx = i_session_end_ifx + 1
        i_session_start_min_sd = i_session_end_sd + 1

    # sd_df_merged[timestamp_row] = sd_df_merged[timestamp_row].astype(int) -> doesn't work because of missing values
    sd_df_merged.to_csv(cfg.CSV_WORKING_DIR + (cfg.CSV_FILENAME_04_SD_BLOCK_ID_UPTIME_TIMESTAMP % slave),
                        index_label="index", sep=cfg.CSV_SEP)
    if cfg.PLOT:
        sd_df_merged.plot(x=timestamp_row, y=sd_block_id_row)
        plt.show()

    # # Counter for lower boundary
    # row_index_before = 0
    # old_timestamp = 0
    # for index, row in values_with_time_stamp_df.iterrows():
    #     row_start_index = row_index_before
    #     offset_list_index = row_index_before
    #     row_stop_index = index
    #     timestamp_stop = int(row[timestamp_row])  # (timestamp to milliseconds)
    #     if old_timestamp == 0:
    #         timestamp_start = timestamp_stop - (
    #             1999 * (row_stop_index - row_start_index)
    #         )  # (timestamp to milliseconds)
    #     else:
    #         timestamp_start = old_timestamp
    #
    #     duration_of_one_row = int(
    #         ((timestamp_stop - timestamp_start) / (row_stop_index - row_start_index))
    #     )  # (duration from seconds to milliseconds and float to int)
    #     # print("Duration: ",duration_of_one_row)
    #
    #     # print("Anfang ist : {block_start} Ende ist {block_stop},"
    #     #       "timestamp Anfang ist: {t_a} timestamp Ende ist: {t_b}, "
    #     #       .format(block_start=row_index_before, block_stop=index, t_a= timestamp_start, t_b=timestamp_stop))
    #
    #     # create a list of numbers increasing by 2
    #     timestamps = list(range(timestamp_start, timestamp_stop, duration_of_one_row))
    #     # print(timestamps)
    #
    #     # create a dataframe with the numbers as a column
    #     timestamp_df = pd.DataFrame(
    #         timestamps,
    #         index=range(offset_list_index, len(timestamps) + offset_list_index),
    #         columns=[timestamp_row],
    #     )
    #     #         unixtimestamp
    #     # 28137   1665648461000
    #     # 28138   1665648462999
    #     # 28139   1665648464998
    #
    #     # print("\ntimestamp_df (2):\n", timestamp_df)
    #     # print(merged_df)
    #     merged_df = pd.merge(
    #         left=merged_df,
    #         right=timestamp_df,
    #         left_index=True,
    #         right_index=True,
    #         how="left",
    #     )
    #     # print(merged_df)
    #     # merged_df.loc[merged_df[timestamp_row + '_y'].notnull(), timestamp_row + '_y'] *= 1000
    #     # print(merged_df)
    #     merged_df[timestamp_row + "_x"] = merged_df[timestamp_row + "_x"].fillna(merged_df[timestamp_row + "_y"])
    #     # print(merged_df)
    #     merged_df.drop(timestamp_row + "_y", axis=1, inplace=True)
    #     # print(merged_df)
    #     merged_df.rename(columns={timestamp_row + "_x": timestamp_row}, inplace=True)
    #     # print(merged_df)
    #
    #     old_timestamp = timestamp_stop
    #     row_index_before = index
    #
    # # Handle the case if there is Data behind the last Max Point, assuming the frequency of measurement is 1999 ms
    #
    # length_sd_data = int(merged_df.tail(1).index[0])
    # length_timestamp_data = int(values_with_time_stamp_df.tail(1).index[0])
    # start_timestamp = int(values_with_time_stamp_df.tail(1)[timestamp_row])
    # end_timestamp = int((length_sd_data - length_timestamp_data) * 1999) + start_timestamp + 1000
    # print("length_sd_data: %u, length_timestamp_data: %u, start_timestamp: %u, end_timestamp: %u\n"
    #       % (length_sd_data, length_timestamp_data, start_timestamp, end_timestamp))
    #
    # if length_sd_data > length_timestamp_data:
    #     timestamps = list(range(start_timestamp, end_timestamp, 1999))
    #     timestamp_df = pd.DataFrame(
    #         timestamps,
    #         index=range(length_timestamp_data, len(timestamps) + length_timestamp_data),
    #         columns=[timestamp_row],
    #     )
    #
    #     # print("\ntimestamp_df (3):\n", timestamp_df)
    #     merged_df = pd.merge(
    #         left=merged_df,
    #         right=timestamp_df,
    #         left_index=True,
    #         right_index=True,
    #         how="left",
    #     )
    #     # print("\nmerged_df (1):\n", merged_df)
    #     merged_df[timestamp_row + "_x"] = merged_df[timestamp_row + "_x"].fillna(merged_df[timestamp_row + "_y"])
    #     # print("\nmerged_df (2):\n", merged_df)
    #     merged_df.drop(timestamp_row + "_y", axis=1, inplace=True)
    #     # print("\nmerged_df (3):\n", merged_df)
    #     merged_df.rename(columns={timestamp_row + "_x": timestamp_row}, inplace=True)
    #     # print("\nmerged_df (4):\n", merged_df)
    #
    # # print("\nmerged_df (5):\n", merged_df)
    # plot_df = merged_df
    # plot_df[timestamp_row] = plot_df[timestamp_row].astype('int64')
    # print("\nEmpty cells:\n", plot_df[plot_df[timestamp_row].isna()])
    #
    # plot_df.to_csv(cfg.CSV_WORKING_DIR + (cfg.CSV_FILENAME_04_SD_BLOCK_ID_UPTIME_TIMESTAMP % slave), sep=cfg.CSV_SEP)
    #
    # plot_df[timestamp_row + '_str'] = pandas.to_datetime(plot_df.loc[:, timestamp_row], unit="ms")
    # print("\nDataframe:\n", plot_df)
    # if cfg.PLOT:
    #     plot_df.plot(x=timestamp_row, y=sd_block_id_row)
    #     plt.show()
    # # synckit


if __name__ == '__main__':
    run()
