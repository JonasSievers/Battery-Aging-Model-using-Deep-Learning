@ECHO OFF
mkdir "zip"
FOR %%i IN (cell_log*.csv) DO (
	ECHO "%%i" | FIND /I "zip_cell_log.bat" 1>NUL) || (
		:: Check if the selected filenames are correct:
		::ECHO "%%i"
		
		:: zip:
		::"C:\Program Files\7-Zip\7z.exe" a -tzip "zip/%%~ni.zip" "%%i"
		
		:: 7z:
		"C:\Program Files\7-Zip\7z.exe" a -t7z "zip/%%~ni.7z" "%%i"
	)
)
