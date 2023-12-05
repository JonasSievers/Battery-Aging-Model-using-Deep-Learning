import config_preprocessing as cfg
import time
import struct
from ctypes import *
import numpy as np


data_block = b'MLB1\x003i \x00\x00\x02\xf9=\xccO\xebAo\xc5\x81A:\x90*;\xda$\x04\x00\xc0\x00\xfd\x00\x08\x00\x10\x00\x00\x00\x00:\x17\x86\x01\x7f\xc0\x00\x00\x00\x00\x00\x00?\x00\x00\x00\x7f\xc0\x00\x00\x03\x00\x10"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08\x00\x10\x00\x00\x00\x00\xba)\x1a\x01\x7f\xc0\x00\x00\x00\x00\x00\x00?\x00\x00\x00\x7f\xc0\x00\x00\x03\x00\x10"\x80\x00\x00\x00\x00\x00\x00\x00\x00\x08\x00\x10\x00\x00\x00\x00\xb9\xe7T\x01\x7f\xc0\x00\x00\x00\x00\x00\x00?\x00\x00\x00\x7f\xc0\x00\x00\x03\x00\x10"\x80\x00\x00\x00\x00\x00\x00\x00\x00\x08\x00\x10\x00\x00\x00\x00\xb9\xde0\x01\x7f\xc0\x00\x00\x00\x00\x00\x00?\x00\x00\x00\x7f\xc0\x00\x00\x03\x00\x10"\x80\x00\x00\x00\x00\x00\x00\x00\x00\x08\x00\x10:"\x00\x009\x17\xe0\x01\x7f\xc0\x00\x00\x00\x00\x00\x00?\x00\x00\x00\x7f\xc0\x00\x00\x03\x00\x10"3\xc07\x81\x00\x00\x00\x00\x00\x08\x00\x10:"\x00\x00\xba<\xca\x01\x7f\xc0\x00\x00\x00\x00\x00\x00?\x00\x00\x00\x7f\xc0\x00\x00\x03\x00\x10"\xb4\xee\xef\xa9\x00\x00\x00\x00\x00\x08\x00\x10:4\x00\x007\x9d\x80\x01\x7f\xc0\x00\x00\x00\x00\x00\x00?\x00\x00\x00\x7f\xc0\x00\x00\x03\x00\x10"2]|\x01\x00\x00\x00\x00\x00\x08\x00\x10:\x10\x00\x00\xb9\x9c\xcc\x01\x7f\xc0\x00\x00\x00\x00\x00\x00?\x00\x00\x00\x7f\xc0\x00\x00\x03\x00\x10"\xb40e\x81\x00\x00\x00\x00\x00\x08\x00\x10:j\x00\x00\xb8G\xb0\x01\x7f\xc0\x00\x00\x00\x00\x00\x00?\x00\x00\x00\x7f\xc0\x00\x00\x03\x00\x10"\xb36\x86\xe1\x00\x00\x00\x00\x00\x08\x00\x10:4\x00\x00\xb9?\xf4\x01\x7f\xc0\x00\x00\x00\x00\x00\x00?\x00\x00\x00\x7f\xc0\x00\x00\x03\x00\x10"\xb4\x06\xf7\x91\x00\x00\x00\x00\x00\x08\x00\x109\xb4\x00\x00\xb9\x90$\x01\x7f\xc0\x00\x00\x00\x00\x00\x00?\x00\x00\x00\x7f\xc0\x00\x00\x03\x00\x10"\xb3\xca\xb2\xa1\x00\x00\x00\x00\x00\x08\x00\x10:"\x00\x00\xb8\xc2\x10\x01\x7f\xc0\x00\x00\x00\x00\x00\x00?\x00\x00\x00\x7f\xc0\x00\x00\x03\x00\x10"\xb3u\x9cA\x00\x00\x00\x00'
timestamp = 1681236206
timestamp_origin = cfg.TimestampOrigin.INFLUX_EXACT
sd_block_id = 4096


def uint_from_bytes(blocks):
    return int.from_bytes(blocks, byteorder="big", signed=False)


def sint_from_bytes(blocks):
    return int.from_bytes(blocks, byteorder="big", signed=True)


def _convert_uint32_to_float(uint_val):
    # https://stackoverflow.com/a/1592200/2738240
    # if (uint_val & 0x7F800000) == 0x7F800000:
    #    return 0.0  # replace NaN, Infinity, -Infinity by 0 as it seems InfluxDB can't handle them
    cp = pointer(c_uint(uint_val))  # make this into a c unsigned integer
    fp = cast(cp, POINTER(c_float))  # cast the unsigned int pointer to a float pointer
    return fp.contents.value  # dereference the pointer, get the float


format_slave_log = "{timestamp}" + cfg.CSV_SEP + \
                   "{timestamp_origin}" + cfg.CSV_SEP + \
                   "{sd_block_id}" + cfg.CSV_SEP + \
                   "{high_level_state}" + cfg.CSV_SEP + \
                   "{scheduler_condition}" + cfg.CSV_SEP + \
                   "{controller_condition}" + cfg.CSV_SEP + \
                   "{v_main_V}" + cfg.CSV_SEP + \
                   "{i_main_A}" + cfg.CSV_SEP + \
                   "{p_main_W}" + cfg.CSV_SEP + \
                   "{v_aux_V}" + cfg.CSV_SEP + \
                   "{v_main_state}" + cfg.CSV_SEP + \
                   "{i_main_state}" + cfg.CSV_SEP + \
                   "{v_aux_state}" + cfg.CSV_SEP + \
                   "{monitor_enable}" + cfg.CSV_SEP + \
                   "{dc_state}" + cfg.CSV_SEP + \
                   "{log_state}" + cfg.CSV_SEP + \
                   "{eth_status}" + cfg.CSV_SEP + \
                   "{eth_fault}" + cfg.CSV_SEP + \
                   "{sd_fault}" + cfg.CSV_SEP + \
                   "{sd_fill}" + cfg.CSV_SEP + \
                   "{uptime_ticks}" + cfg.CSV_SEP + \
                   cfg.CSV_NEWLINE


dt_cell_log = np.dtype([('magic_word', '>i4'),
                        ('sch_ctrl_condition', 'u1'),
                        ('high_level_state', 'u1'),
                        ('log_dc_state', 'u1'),
                        ('uptime_ticks', '>u4'),
                        ('p_main_W', '>f'),
                        ('v_main_V', '>f'),
                        ('v_aux_V', '>f'),
                        ('i_main_A', '>f'),
                        ('eth_v_main_state', 'u1'),
                        ('eth_sd_fault_v_aux_state', 'u1'),
                        ('monitor_en_i_main_state', 'u1'),
                        ('sd_fill', '>i2'),
                        ])

time_start = time.time()

for i in range(0, 500000):
    # 3 runs: 14.8...15.3 seconds
    data_np = np.frombuffer(data_block, dtype=dt_cell_log, count=1)
    string_slave_log = str(timestamp) + cfg.CSV_SEP + \
                       str(timestamp_origin) + cfg.CSV_SEP + \
                       str(sd_block_id) + cfg.CSV_SEP + \
                       str(data_np['high_level_state'][0]) + cfg.CSV_SEP + \
                       str(data_np['sch_ctrl_condition'][0] >> 4) + cfg.CSV_SEP + \
                       str(data_np['sch_ctrl_condition'][0] & 0x0F) + cfg.CSV_SEP + \
                       format(data_np['v_main_V'][0], ".4f") + cfg.CSV_SEP + \
                       format(data_np['i_main_A'][0], ".4f") + cfg.CSV_SEP + \
                       format(data_np['p_main_W'][0], ".4f") + cfg.CSV_SEP + \
                       format(data_np['v_aux_V'][0], ".4f") + cfg.CSV_SEP + \
                       str(data_np['eth_v_main_state'][0] & 0x3F) + cfg.CSV_SEP + \
                       str(data_np['monitor_en_i_main_state'][0] & 0x3F) + cfg.CSV_SEP + \
                       str(data_np['eth_sd_fault_v_aux_state'][0] & 0x3F) + cfg.CSV_SEP + \
                       str(data_np['monitor_en_i_main_state'][0] >> 7) + cfg.CSV_SEP + \
                       str(data_np['log_dc_state'][0] & 0x0F) + cfg.CSV_SEP + \
                       str(data_np['log_dc_state'][0] >> 4) + cfg.CSV_SEP + \
                       str(data_np['eth_v_main_state'][0] >> 6) + cfg.CSV_SEP + \
                       str(data_np['eth_sd_fault_v_aux_state'][0] >> 7) + cfg.CSV_SEP + \
                       str((data_np['eth_sd_fault_v_aux_state'][0] >> 6) & 0x01) + cfg.CSV_SEP + \
                       str(data_np['sd_fill'][0]) + cfg.CSV_SEP + \
                       str(data_np['uptime_ticks'][0]) + cfg.CSV_SEP + \
                       cfg.CSV_NEWLINE

    # # 3 runs: 4.34...4.43 seconds
    # string_slave_log = format_slave_log.format(timestamp=timestamp,
    #                                            timestamp_origin=timestamp_origin,
    #                                            sd_block_id=sd_block_id,
    #                                            scheduler_condition=data_block[5] >> 4,
    #                                            controller_condition=data_block[5] & 0x0F,
    #                                            high_level_state=data_block[6],
    #                                            log_state=data_block[7] >> 4,
    #                                            dc_state=data_block[7] & 0x0F,
    #                                            uptime_ticks=uint_from_bytes(data_block[8:12]),
    #                                            p_main_W=struct.unpack(">f", data_block[12:16])[0],
    #                                            v_main_V=struct.unpack(">f", data_block[16:20])[0],
    #                                            v_aux_V=struct.unpack(">f", data_block[20:24])[0],
    #                                            i_main_A=struct.unpack(">f", data_block[24:28])[0],
    #                                            eth_status=data_block[28] >> 6,
    #                                            v_main_state=data_block[28] & 0x3F,
    #                                            eth_fault=data_block[29] >> 7,
    #                                            sd_fault=(data_block[29] >> 6) & 0x01,
    #                                            v_aux_state=data_block[29] & 0x3F,
    #                                            monitor_enable=data_block[30] >> 7,
    #                                            i_main_state=data_block[30] & 0x3F,
    #                                            sd_fill=sint_from_bytes(data_block[31:32]))

    # # 3 runs: 3.75...4.63 seconds
    # scheduler_condition = data_block[5] >> 4  # (data_block[5] >> 4) & 0x0F
    # controller_condition = data_block[5] & 0x0F
    # high_level_state = data_block[6]
    # log_state = data_block[7] >> 4  # (data_block[7] >> 4) & 0x0F
    # dc_state = data_block[7] & 0x0F
    # uptime_ticks = uint_from_bytes(data_block[8:12])
    # p_main_W = struct.unpack(">f", data_block[12:16])[0]
    # v_main_V = struct.unpack(">f", data_block[16:20])[0]
    # v_aux_V = struct.unpack(">f", data_block[20:24])[0]
    # i_main_A = struct.unpack(">f", data_block[24:28])[0]
    # eth_status = data_block[28] >> 6  # (data_block[28] >> 6) & 0x03
    # v_main_state = data_block[28] & 0x3F
    # eth_fault = data_block[29] >> 7  # data_block[29] >> 7 & 0x01
    # sd_fault = (data_block[29] >> 6) & 0x01
    # v_aux_state = data_block[29] & 0x3F
    # monitor_enable = data_block[30] >> 7  # (data_block[30] >> 7) & 0x01
    # i_main_state = data_block[30] & 0x3F
    # sd_fill = sint_from_bytes(data_block[31:32])
    #
    # string_slave_log = format_slave_log.format(timestamp=timestamp,
    #                                            timestamp_origin=timestamp_origin,
    #                                            sd_block_id=sd_block_id,
    #                                            scheduler_condition=scheduler_condition,
    #                                            controller_condition=controller_condition,
    #                                            high_level_state=high_level_state,
    #                                            log_state=log_state,
    #                                            dc_state=dc_state,
    #                                            uptime_ticks=uptime_ticks,
    #                                            p_main_W=p_main_W,
    #                                            v_main_V=v_main_V,
    #                                            v_aux_V=v_aux_V,
    #                                            i_main_A=i_main_A,
    #                                            eth_status=eth_status,
    #                                            v_main_state=v_main_state,
    #                                            eth_fault=eth_fault,
    #                                            sd_fault=sd_fault,
    #                                            v_aux_state=v_aux_state,
    #                                            monitor_enable=monitor_enable,
    #                                            i_main_state=i_main_state,
    #                                            sd_fill=sd_fill)

    # # 3 runs: 3.38...3.45 seconds
    # scheduler_condition = data_block[5] >> 4  # (data_block[5] >> 4) & 0x0F
    # controller_condition = data_block[5] & 0x0F
    # high_level_state = data_block[6]
    # log_state = data_block[7] >> 4  # (data_block[7] >> 4) & 0x0F
    # dc_state = data_block[7] & 0x0F
    # uptime_ticks = uint_from_bytes(data_block[8:12])
    # p_main_W = struct.unpack(">f", data_block[12:16])[0]
    # v_main_V = struct.unpack(">f", data_block[16:20])[0]
    # v_aux_V = struct.unpack(">f", data_block[20:24])[0]
    # i_main_A = struct.unpack(">f", data_block[24:28])[0]
    # eth_status = data_block[28] >> 6  # (data_block[28] >> 6) & 0x03
    # v_main_state = data_block[28] & 0x3F
    # eth_fault = data_block[29] >> 7  # data_block[29] >> 7 & 0x01
    # sd_fault = (data_block[29] >> 6) & 0x01
    # v_aux_state = data_block[29] & 0x3F
    # monitor_enable = data_block[30] >> 7  # (data_block[30] >> 7) & 0x01
    # i_main_state = data_block[30] & 0x3F
    # sd_fill = sint_from_bytes(data_block[31:32])
    #
    # string_slave_log = str(timestamp) + cfg.CSV_SEP + \
    #         str(timestamp_origin) + cfg.CSV_SEP + \
    #         str(sd_block_id) + cfg.CSV_SEP + \
    #         str(high_level_state) + cfg.CSV_SEP + \
    #         str(scheduler_condition) + cfg.CSV_SEP + \
    #         str(controller_condition) + cfg.CSV_SEP + \
    #         format(v_main_V, ".4f") + cfg.CSV_SEP + \
    #         format(i_main_A, ".4f") + cfg.CSV_SEP + \
    #         format(p_main_W, ".4f") + cfg.CSV_SEP + \
    #         format(v_aux_V, ".4f") + cfg.CSV_SEP + \
    #         str(v_main_state) + cfg.CSV_SEP + \
    #         str(i_main_state) + cfg.CSV_SEP + \
    #         str(v_aux_state) + cfg.CSV_SEP + \
    #         str(monitor_enable) + cfg.CSV_SEP + \
    #         str(dc_state) + cfg.CSV_SEP + \
    #         str(log_state) + cfg.CSV_SEP + \
    #         str(eth_status) + cfg.CSV_SEP + \
    #         str(eth_fault) + cfg.CSV_SEP + \
    #         str(sd_fault) + cfg.CSV_SEP + \
    #         str(sd_fill) + cfg.CSV_SEP + \
    #         str(uptime_ticks) + cfg.CSV_SEP + \
    #         cfg.CSV_NEWLINE
    #
    # # 3 runs: 3.53...3.55 seconds
    # scheduler_condition = data_block[5] >> 4  # (data_block[5] >> 4) & 0x0F
    # controller_condition = data_block[5] & 0x0F
    # high_level_state = data_block[6]
    # log_state = data_block[7] >> 4  # (data_block[7] >> 4) & 0x0F
    # dc_state = data_block[7] & 0x0F
    # uptime_ticks = uint_from_bytes(data_block[8:12])
    # p_main_W = struct.unpack(">f", data_block[12:16])[0]
    # v_main_V = struct.unpack(">f", data_block[16:20])[0]
    # v_aux_V = struct.unpack(">f", data_block[20:24])[0]
    # i_main_A = struct.unpack(">f", data_block[24:28])[0]
    # eth_status = data_block[28] >> 6  # (data_block[28] >> 6) & 0x03
    # v_main_state = data_block[28] & 0x3F
    # eth_fault = data_block[29] >> 7  # data_block[29] >> 7 & 0x01
    # sd_fault = (data_block[29] >> 6) & 0x01
    # v_aux_state = data_block[29] & 0x3F
    # monitor_enable = data_block[30] >> 7  # (data_block[30] >> 7) & 0x01
    # i_main_state = data_block[30] & 0x3F
    # sd_fill = sint_from_bytes(data_block[31:32])
    #
    # string_slave_log = str(timestamp) + cfg.CSV_SEP + \
    #         str(timestamp_origin) + cfg.CSV_SEP + \
    #         str(sd_block_id) + cfg.CSV_SEP + \
    #         str(high_level_state) + cfg.CSV_SEP + \
    #         str(scheduler_condition) + cfg.CSV_SEP + \
    #         str(controller_condition) + cfg.CSV_SEP + \
    #         str("{:.4f}".format(v_main_V)) + cfg.CSV_SEP + \
    #         str("{:.4f}".format(i_main_A)) + cfg.CSV_SEP + \
    #         str("{:.4f}".format(p_main_W)) + cfg.CSV_SEP + \
    #         str("{:.4f}".format(v_aux_V)) + cfg.CSV_SEP + \
    #         str(v_main_state) + cfg.CSV_SEP + \
    #         str(i_main_state) + cfg.CSV_SEP + \
    #         str(v_aux_state) + cfg.CSV_SEP + \
    #         str(monitor_enable) + cfg.CSV_SEP + \
    #         str(dc_state) + cfg.CSV_SEP + \
    #         str(log_state) + cfg.CSV_SEP + \
    #         str(eth_status) + cfg.CSV_SEP + \
    #         str(eth_fault) + cfg.CSV_SEP + \
    #         str(sd_fault) + cfg.CSV_SEP + \
    #         str(sd_fill) + cfg.CSV_SEP + \
    #         str(uptime_ticks) + cfg.CSV_SEP + \
    #         cfg.CSV_NEWLINE

    # # 3 runs: 4.04...4.32 seconds
    # scheduler_condition = data_block[5] >> 4  # (data_block[5] >> 4) & 0x0F
    # controller_condition = data_block[5] & 0x0F
    # high_level_state = data_block[6]
    # log_state = data_block[7] >> 4  # (data_block[7] >> 4) & 0x0F
    # dc_state = data_block[7] & 0x0F
    # uptime_ticks = uint_from_bytes(data_block[8:12])
    # p_main_W = struct.unpack(">f", data_block[12:16])[0]
    # v_main_V = struct.unpack(">f", data_block[16:20])[0]
    # v_aux_V = struct.unpack(">f", data_block[20:24])[0]
    # i_main_A = struct.unpack(">f", data_block[24:28])[0]
    # eth_status = data_block[28] >> 6  # (data_block[28] >> 6) & 0x03
    # v_main_state = data_block[28] & 0x3F
    # eth_fault = data_block[29] >> 7  # data_block[29] >> 7 & 0x01
    # sd_fault = (data_block[29] >> 6) & 0x01
    # v_aux_state = data_block[29] & 0x3F
    # monitor_enable = data_block[30] >> 7  # (data_block[30] >> 7) & 0x01
    # i_main_state = data_block[30] & 0x3F
    # sd_fill = sint_from_bytes(data_block[31:32])
    #
    # string_slave_log = str(timestamp) + cfg.CSV_SEP + \
    #         str(timestamp_origin) + cfg.CSV_SEP + \
    #         str(sd_block_id) + cfg.CSV_SEP + \
    #         str(high_level_state) + cfg.CSV_SEP + \
    #         str(scheduler_condition) + cfg.CSV_SEP + \
    #         str(controller_condition) + cfg.CSV_SEP + \
    #         str("{:.4f}".format(round(v_main_V, 4))) + cfg.CSV_SEP + \
    #         str("{:.4f}".format(round(i_main_A, 4))) + cfg.CSV_SEP + \
    #         str("{:.4f}".format(round(p_main_W, 4))) + cfg.CSV_SEP + \
    #         str("{:.4f}".format(round(v_aux_V, 4))) + cfg.CSV_SEP + \
    #         str(v_main_state) + cfg.CSV_SEP + \
    #         str(i_main_state) + cfg.CSV_SEP + \
    #         str(v_aux_state) + cfg.CSV_SEP + \
    #         str(monitor_enable) + cfg.CSV_SEP + \
    #         str(dc_state) + cfg.CSV_SEP + \
    #         str(log_state) + cfg.CSV_SEP + \
    #         str(eth_status) + cfg.CSV_SEP + \
    #         str(eth_fault) + cfg.CSV_SEP + \
    #         str(sd_fault) + cfg.CSV_SEP + \
    #         str(sd_fill) + cfg.CSV_SEP + \
    #         str(uptime_ticks) + cfg.CSV_SEP + \
    #         cfg.CSV_NEWLINE
    #
    # # 3 runs: 3.79...4.17 seconds
    # scheduler_condition = data_block[5] >> 4  # (data_block[5] >> 4) & 0x0F
    # controller_condition = data_block[5] & 0x0F
    # high_level_state = data_block[6]
    # log_state = data_block[7] >> 4  # (data_block[7] >> 4) & 0x0F
    # dc_state = data_block[7] & 0x0F
    # uptime_ticks = uint_from_bytes(data_block[8:12])
    # p_main_W = struct.unpack(">f", data_block[12:16])[0]
    # v_main_V = struct.unpack(">f", data_block[16:20])[0]
    # v_aux_V = struct.unpack(">f", data_block[20:24])[0]
    # i_main_A = struct.unpack(">f", data_block[24:28])[0]
    # eth_status = data_block[28] >> 6  # (data_block[28] >> 6) & 0x03
    # v_main_state = data_block[28] & 0x3F
    # eth_fault = data_block[29] >> 7  # data_block[29] >> 7 & 0x01
    # sd_fault = (data_block[29] >> 6) & 0x01
    # v_aux_state = data_block[29] & 0x3F
    # monitor_enable = data_block[30] >> 7  # (data_block[30] >> 7) & 0x01
    # i_main_state = data_block[30] & 0x3F
    # sd_fill = sint_from_bytes(data_block[31:32])
    #
    # string_slave_log = str(timestamp) + cfg.CSV_SEP + \
    #         str(timestamp_origin) + cfg.CSV_SEP + \
    #         str(sd_block_id) + cfg.CSV_SEP + \
    #         str(high_level_state) + cfg.CSV_SEP + \
    #         str(scheduler_condition) + cfg.CSV_SEP + \
    #         str(controller_condition) + cfg.CSV_SEP + \
    #         str(v_main_V) + cfg.CSV_SEP + \
    #         str(i_main_A) + cfg.CSV_SEP + \
    #         str(p_main_W) + cfg.CSV_SEP + \
    #         str(v_aux_V) + cfg.CSV_SEP + \
    #         str(v_main_state) + cfg.CSV_SEP + \
    #         str(i_main_state) + cfg.CSV_SEP + \
    #         str(v_aux_state) + cfg.CSV_SEP + \
    #         str(monitor_enable) + cfg.CSV_SEP + \
    #         str(dc_state) + cfg.CSV_SEP + \
    #         str(log_state) + cfg.CSV_SEP + \
    #         str(eth_status) + cfg.CSV_SEP + \
    #         str(eth_fault) + cfg.CSV_SEP + \
    #         str(sd_fault) + cfg.CSV_SEP + \
    #         str(sd_fill) + cfg.CSV_SEP + \
    #         str(uptime_ticks) + cfg.CSV_SEP + \
    #         cfg.CSV_NEWLINE

    # # 3 runs: 7.04...7.57 seconds
    # scheduler_condition = data_block[5] >> 4  # (data_block[5] >> 4) & 0x0F
    # controller_condition = data_block[5] & 0x0F
    # high_level_state = data_block[6]
    # log_state = data_block[7] >> 4  # (data_block[7] >> 4) & 0x0F
    # dc_state = data_block[7] & 0x0F
    # uptime_ticks = uint_from_bytes(data_block[8:12])
    # tmp_uint = (data_block[12] << 24) \
    #            | (data_block[13] << 16) \
    #            | (data_block[14] << 8) \
    #            | (data_block[15])
    # p_main_W = _convert_uint32_to_float(tmp_uint)
    # tmp_uint = (data_block[16] << 24) \
    #            | (data_block[17] << 16) \
    #            | (data_block[18] << 8) \
    #            | (data_block[19])
    # v_main_V = _convert_uint32_to_float(tmp_uint)
    # tmp_uint = (data_block[20] << 24) \
    #            | (data_block[21] << 16) \
    #            | (data_block[22] << 8) \
    #            | (data_block[23])
    # v_aux_V = _convert_uint32_to_float(tmp_uint)
    # tmp_uint = (data_block[24] << 24) \
    #            | (data_block[25] << 16) \
    #            | (data_block[26] << 8) \
    #            | (data_block[27])
    # i_main_A = _convert_uint32_to_float(tmp_uint)
    # eth_status = data_block[28] >> 6  # (data_block[28] >> 6) & 0x03
    # v_main_state = data_block[28] & 0x3F
    # eth_fault = data_block[29] >> 7  # data_block[29] >> 7 & 0x01
    # sd_fault = (data_block[29] >> 6) & 0x01
    # v_aux_state = data_block[29] & 0x3F
    # monitor_enable = data_block[30] >> 7  # (data_block[30] >> 7) & 0x01
    # i_main_state = data_block[30] & 0x3F
    # sd_fill = sint_from_bytes(data_block[31:32])
    #
    # string_slave_log = str(timestamp) + cfg.CSV_SEP + \
    #         str(timestamp_origin) + cfg.CSV_SEP + \
    #         str(sd_block_id) + cfg.CSV_SEP + \
    #         str(high_level_state) + cfg.CSV_SEP + \
    #         str(scheduler_condition) + cfg.CSV_SEP + \
    #         str(controller_condition) + cfg.CSV_SEP + \
    #         str(v_main_V) + cfg.CSV_SEP + \
    #         str(i_main_A) + cfg.CSV_SEP + \
    #         str(p_main_W) + cfg.CSV_SEP + \
    #         str(v_aux_V) + cfg.CSV_SEP + \
    #         str(v_main_state) + cfg.CSV_SEP + \
    #         str(i_main_state) + cfg.CSV_SEP + \
    #         str(v_aux_state) + cfg.CSV_SEP + \
    #         str(monitor_enable) + cfg.CSV_SEP + \
    #         str(dc_state) + cfg.CSV_SEP + \
    #         str(log_state) + cfg.CSV_SEP + \
    #         str(eth_status) + cfg.CSV_SEP + \
    #         str(eth_fault) + cfg.CSV_SEP + \
    #         str(sd_fault) + cfg.CSV_SEP + \
    #         str(sd_fill) + cfg.CSV_SEP + \
    #         str(uptime_ticks) + cfg.CSV_SEP + \
    #         cfg.CSV_NEWLINE

time_stop = time.time()
duration = time_stop - time_start
print("Duration: %.6f s" % duration)
