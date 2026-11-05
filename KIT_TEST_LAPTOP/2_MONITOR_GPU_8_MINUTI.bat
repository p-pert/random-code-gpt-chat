@echo off
title Kit test laptop - Monitor GPU durante OCCT
echo.
echo  Avvia PRIMA il test 3D in OCCT sulla RTX 4060, poi lascia girare questo monitor.
echo  Durata 8 minuti. Premi Q per fermarlo prima.
echo.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Script\monitor_gpu.ps1" -Minuti 8
