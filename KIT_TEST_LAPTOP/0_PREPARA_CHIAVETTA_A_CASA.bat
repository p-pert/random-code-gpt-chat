@echo off
title Kit test laptop - Preparazione chiavetta (serve Internet)
echo.
echo  Questo script scarica GPU-Z, HWiNFO64 e OCCT dai SITI UFFICIALI
echo  e li sistema nella cartella Programmi. Da usare A CASA, con Internet.
echo.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Script\prepara_chiavetta.ps1"
