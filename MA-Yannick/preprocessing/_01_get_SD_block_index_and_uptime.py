import os
import config_preprocessing as cfg
import pandas as pd
import matplotlib.pyplot as plt


# declaration of global variables and composite constants
global csv_file_uptime
CSV_FILE_HEADER = cfg.CSV_FILE_HEADER_SD_BLOCK_ID + ";" + cfg.CSV_FILE_HEADER_STM_TIM_UPTIME
pd.options.mode.chained_assignment = None  # default='warn'


def decode_block(block, sd_block_id):
    stm_tim_uptime = int.from_bytes(block[8:12], byteorder='big', signed=False)
    content_string = str(sd_block_id) + ";" + str(stm_tim_uptime) + "\n"
    csv_file_uptime.write(content_string)


def run():
    global csv_file_uptime

    slave = "S01"

    # check if file size is a multiple of the SD block size
    size = os.path.getsize(cfg.SD_IMAGE_FILE)
    print("Size of file: ", size)
    if size == 0 or (size % cfg.SD_BLOCK_SIZE_BYTES) != 0:
        print("Block error, size not n*%u" % cfg.SD_BLOCK_SIZE_BYTES)
        raise Exception

    # open file, write header
    file = open(cfg.SD_IMAGE_FILE, 'rb')
    csv_file_path = cfg.CSV_WORKING_DIR + (cfg.CSV_FILENAME_01_SD_UPTIME % slave)
    csv_file_uptime = open(csv_file_path, 'w')
    csv_file_uptime.write(CSV_FILE_HEADER + "\n")

    # write SD block ID and uptime to CSV file
    for i in range(0, size, cfg.SD_BLOCK_SIZE_BYTES):
        data = file.read(cfg.SD_BLOCK_SIZE_BYTES)

        # if (data[0:4] == b'MLB1')\
        #         or (data[0:4] == b'MLI1')\
        #         or (data[0:4] == b'MLI2')\
        #         or (data[0:4] == b'MLE1')\
        #         or (data[0:4] == b'MLE2'):
        if data[0:4] == b'MLB1':
            # if i%1000000 == 0:
            #     print(i/size*100,"%")
            # print("==== New Block ==== at (dec bytes number): ",i)
            decode_block(data, i)

    csv_file_uptime.close()

    # find maxima/peaks of SD card uptime and store to csv
    df = pd.read_csv(csv_file_path, sep=cfg.CSV_SEP, header=0)
    # SD_block_ID;stm_tim_uptime
    uptime_row = cfg.CSV_FILE_HEADER_STM_TIM_UPTIME
    df['max'] = df[uptime_row][
        (df[uptime_row].shift(1) < df[uptime_row]) & (df[uptime_row].shift(-1) < df[uptime_row])
    ]
    df['min'] = df[uptime_row][
        (df[uptime_row].shift(1) > df[uptime_row]) & (df[uptime_row].shift(-1) > df[uptime_row])
    ]

    print(df.loc[~pd.isna(df['max']), 'max'])
    csv_file_path_peaks = cfg.CSV_WORKING_DIR + (cfg.CSV_FILENAME_01_SD_UPTIME_PEAKS % slave)
    df_peaks = df[df['max'] > cfg.UPTIME_MIN_PEAK_THRESHOLD]
    print("\ndf_peaks (1):\n", df_peaks)

    last_row = df.tail(1)
    last_row['max'] = last_row[cfg.CSV_FILE_HEADER_STM_TIM_UPTIME]
    df.iloc[-1, df.columns.get_loc('max')] = last_row['max']
    print("\nlast_row:\n", last_row)
    df_peaks = pd.concat([df_peaks, last_row])
    print("\ndf_peaks (2):\n", df_peaks)

    print(df_peaks.to_csv(csv_file_path_peaks, index_label="index", sep=cfg.CSV_SEP))

    if cfg.PLOT:
        plt.plot(df.index, df[cfg.CSV_FILE_HEADER_STM_TIM_UPTIME])
        plt.scatter(df.index, df['max'], c="g")
        plt.scatter(df.index, df['min'], c="r")

        plt.title(cfg.CSV_FILE_HEADER_STM_TIM_UPTIME + " over index with local max detection")
        plt.xlabel("Index Count")
        plt.ylabel("Timer Count")
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    run()
