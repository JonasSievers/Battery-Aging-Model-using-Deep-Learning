@ECHO OFF
setlocal enableDelayedExpansion
mkdir zip
ECHO collecting cfg filenames...
set filenames="C:\Program Files\7-Zip\7z.exe" a -t7z "zip/prepr_res_cfg.7z
FOR %%i IN (cell_cfg*.csv) DO (
	ECHO "%%i" | FIND /I "zip_others.bat" 1>NUL) || (
		SET filenames=!filenames!" "%%i
	)
)
FOR %%i IN (pool_cfg*.csv) DO (
	ECHO "%%i" | FIND /I "zip_others.bat" 1>NUL) || (
		SET filenames=!filenames!" "%%i
	)
)
FOR %%i IN (slave_cfg*.csv) DO (
	ECHO "%%i" | FIND /I "zip_others.bat" 1>NUL) || (
		SET filenames=!filenames!" "%%i
	)
)
set filenames=%filenames%"
%filenames%

ECHO collecting slave_log filenames...
set filenames="C:\Program Files\7-Zip\7z.exe" a -t7z "zip/prepr_res_slave_log.7z
FOR %%i IN (slave_log*.csv) DO (
	ECHO "%%i" | FIND /I "zip_others.bat" 1>NUL) || (
		SET filenames=!filenames!" "%%i
	)
)
set filenames=%filenames%"
%filenames%

ECHO collecting pool_log filenames...
set filenames="C:\Program Files\7-Zip\7z.exe" a -t7z "zip/prepr_res_pool_log.7z
FOR %%i IN (pool_log*.csv) DO (
	ECHO "%%i" | FIND /I "zip_others.bat" 1>NUL) || (
		SET filenames=!filenames!" "%%i
	)
)
set filenames=%filenames%"
%filenames%

ECHO collecting eoc filenames...
set filenames="C:\Program Files\7-Zip\7z.exe" a -t7z "zip/prepr_res_eoc.7z
FOR %%i IN (cell_eoc_*.csv) DO (
	ECHO "%%i" | FIND /I "zip_others.bat" 1>NUL) || (
		SET filenames=!filenames!" "%%i
	)
)
set filenames=%filenames%"
%filenames%

ECHO collecting eis filenames...
set filenames="C:\Program Files\7-Zip\7z.exe" a -t7z "zip/prepr_res_eis.7z
FOR %%i IN (cell_eis_*.csv) DO (
	ECHO "%%i" | FIND /I "zip_others.bat" 1>NUL) || (
		SET filenames=!filenames!" "%%i
	)
)
set filenames=%filenames%"
%filenames%

ECHO collecting eocv2 filenames...
set filenames="C:\Program Files\7-Zip\7z.exe" a -t7z "zip/prepr_res_eocv2.7z
FOR %%i IN (cell_eocv2_*.csv) DO (
	ECHO "%%i" | FIND /I "zip_others.bat" 1>NUL) || (
		SET filenames=!filenames!" "%%i
	)
)
set filenames=%filenames%"
%filenames%

ECHO collecting pls filenames...
set filenames="C:\Program Files\7-Zip\7z.exe" a -t7z "zip/prepr_res_pls.7z
FOR %%i IN (cell_pls_*.csv) DO (
	ECHO "%%i" | FIND /I "zip_others.bat" 1>NUL) || (
		SET filenames=!filenames!" "%%i
	)
)
set filenames=%filenames%"
%filenames%