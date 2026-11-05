@echo off
title Kit test laptop - Controllo automatico
echo.
echo  Controllo automatico in sola lettura (circa 1 minuto).
echo  Windows chiedera' i permessi di amministratore: rispondi SI.
echo.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Script\controllo_sistema.ps1"
