import _01_get_SD_block_index_and_uptime_mc as _01
import _02_get_InfluxDB_timestamp_and_uptime_mc as _02
import _03_merge_uptime_SD_Influx_exact_mc as _03
import _04_merge_timestamp_with_SD_block_index_exact_mc as _04
import _05_decode_SD_data_mc as _05
import _06_fix_issues_mc as _06

if __name__ == '__main__':
    _01.run()
    _02.run()
    _03.run()
    _04.run()
    _05.run()
    _06.run()
