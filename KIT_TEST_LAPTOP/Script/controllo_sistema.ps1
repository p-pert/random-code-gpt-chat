# controllo_sistema.ps1 - Controllo automatico SOLO LETTURA (circa 1 minuto)
# Non installa nulla, non modifica impostazioni. Funziona senza Internet.
# Verifica: modello, CPU, RAM, GPU/VRAM, SSD (salute e usura), batteria,
# Gestione dispositivi, registri errori di Windows (schermate blu, crash driver video, WHEA).

param([switch]$NoElevate)

. (Join-Path $PSScriptRoot 'comune.ps1')

$isAdmin = Test-Admin
if (-not $isAdmin -and -not $NoElevate) {
    Write-Host "Chiedo i permessi di amministratore (servono per leggere salute e usura dell'SSD)..." -ForegroundColor Yellow
    try {
        $argomenti = '-NoProfile -ExecutionPolicy Bypass -File "{0}" -NoElevate' -f $PSCommandPath
        Start-Process -FilePath 'powershell.exe' -Verb RunAs -ArgumentList $argomenti -ErrorAction Stop
        exit
    } catch {
        Write-Host "Permessi non concessi: proseguo con controlli ridotti (niente usura SSD)." -ForegroundColor Yellow
    }
}

$Risultati = New-Object System.Collections.Generic.List[object]

function Add-Ris([string]$Area, [string]$Voce, [string]$Valore, [string]$Esito, [string]$Nota = '') {
    $Risultati.Add([pscustomobject]@{ Area = $Area; Voce = $Voce; Valore = $Valore; Esito = $Esito; Nota = $Nota })
    $etichetta = ('[' + $Esito + ']').PadRight(13)
    Write-Host $etichetta -ForegroundColor (Get-ColoreEsito $Esito) -NoNewline
    Write-Host ("{0} - {1}: {2}" -f $Area, $Voce, $Valore)
    if ($Nota) { Write-Host ("             -> " + $Nota) -ForegroundColor DarkGray }
}

function Get-NumEventi([hashtable]$Filtro) {
    try { return @(Get-WinEvent -FilterHashtable $Filtro -ErrorAction Stop).Count } catch { return 0 }
}

Write-Titolo "CONTROLLO AUTOMATICO - ASUS TUF Gaming A15 (FA507NV)"
Write-Host ("Data: {0}   Amministratore: {1}" -f (Get-Date -Format 'dd/MM/yyyy HH:mm'), $(if ($isAdmin) { 'si' } else { 'no' }))
Write-Host ""

$seriale = 'sconosciuto'

# ---------------------------------------------------------------- SISTEMA
try {
    $cs   = Get-CimInstance Win32_ComputerSystem
    $bios = Get-CimInstance Win32_BIOS
    $modello = ("{0} {1}" -f $cs.Manufacturer, $cs.Model).Trim()
    if ($cs.Model -match 'FA507NU') {
        Add-Ris 'Sistema' 'Modello' $modello 'KO' "FA507NU e' la versione con RTX 4050 6 GB, NON la RTX 4060 promessa."
    } elseif ($cs.Model -match 'FA507NV') {
        Add-Ris 'Sistema' 'Modello' $modello 'OK' "Corrisponde alla versione con RTX 4060 (FA507NV / FA507NVR)."
    } elseif ($cs.Model -match 'FA507') {
        Add-Ris 'Sistema' 'Modello' $modello 'ATTENZIONE' "E' un TUF A15 2023 ma non la sigla FA507NV: controlla bene CPU e GPU qui sotto."
    } else {
        Add-Ris 'Sistema' 'Modello' $modello 'ATTENZIONE' "Modello diverso da quello atteso (FA507NV)."
    }
    $seriale = ("" + $bios.SerialNumber).Trim()
    $dataBios = ''
    if ($bios.ReleaseDate) { $dataBios = ' del ' + $bios.ReleaseDate.ToString('dd/MM/yyyy') }
    Add-Ris 'Sistema' 'Versione BIOS' ("{0}{1}" -f $bios.SMBIOSBIOSVersion, $dataBios) 'INFO' ''
    Add-Ris 'Sistema' 'Numero di serie' $seriale 'INFO' "Deve coincidere con l'etichetta sotto il portatile (e con scatola/fattura se presenti)."
} catch { Add-Ris 'Sistema' 'Lettura dati' $_.Exception.Message 'INFO' '' }

# ---------------------------------------------------------------- WINDOWS
try {
    $os = Get-CimInstance Win32_OperatingSystem
    Add-Ris 'Windows' 'Versione' ("{0} (build {1})" -f $os.Caption, $os.BuildNumber) 'INFO' ''
    $giorni = [int]((Get-Date) - $os.InstallDate).TotalDays
    $nota = ''
    if ($giorni -lt 14) { $nota = "Installazione/aggiornamento recente: i registri errori qui sotto coprono pochi giorni." }
    Add-Ris 'Windows' 'Data installazione' ("{0} ({1} giorni fa)" -f $os.InstallDate.ToString('dd/MM/yyyy'), $giorni) 'INFO' $nota
} catch { }

try {
    $lic = Get-CimInstance SoftwareLicensingProduct -Filter "ApplicationID='55c92734-d682-4d71-983e-d6ec3f16059f' AND PartialProductKey IS NOT NULL" -ErrorAction Stop |
           Select-Object -First 1
    if ($lic -and $lic.LicenseStatus -eq 1) {
        Add-Ris 'Windows' 'Attivazione' 'Windows attivato' 'OK' ''
    } else {
        Add-Ris 'Windows' 'Attivazione' 'NON attivato' 'ATTENZIONE' "La licenza OEM e' di solito legata alla scheda madre: chiedi spiegazioni."
    }
} catch { Add-Ris 'Windows' 'Attivazione' 'non verificabile' 'INFO' '' }

# ---------------------------------------------------------------- CPU
try {
    $cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
    $nomeCpu = ("" + $cpu.Name).Trim()
    $txt = "{0} - {1} core / {2} thread" -f $nomeCpu, $cpu.NumberOfCores, $cpu.NumberOfLogicalProcessors
    if ($nomeCpu -match '7735HS' -and $cpu.NumberOfCores -eq 8 -and $cpu.NumberOfLogicalProcessors -eq 16) {
        Add-Ris 'CPU' 'Processore' $txt 'OK' ''
    } elseif ($nomeCpu -match '7435HS') {
        Add-Ris 'CPU' 'Processore' $txt 'INFO' "Ryzen 7 7435HS: variante FA507NVR o di altri mercati, prestazioni quasi identiche al 7735HS."
    } else {
        Add-Ris 'CPU' 'Processore' $txt 'ATTENZIONE' "Atteso AMD Ryzen 7 7735HS, 8 core / 16 thread."
    }
} catch { }

# ---------------------------------------------------------------- RAM
try {
    $moduli = @(Get-CimInstance Win32_PhysicalMemory)
    $totGB = [math]::Round((($moduli | Measure-Object -Property Capacity -Sum).Sum) / 1GB)
    $dettagli = @()
    foreach ($m in $moduli) {
        $vel = $m.ConfiguredClockSpeed
        if (-not $vel) { $vel = $m.Speed }
        $dettagli += ("{0} GB @ {1} MT/s {2}" -f [math]::Round($m.Capacity / 1GB), $vel, ("" + $m.PartNumber).Trim())
    }
    $txt = "{0} GB totali in {1} modulo/i: {2}" -f $totGB, $moduli.Count, ($dettagli -join ' | ')
    if ($totGB -lt 15) {
        Add-Ris 'RAM' 'Memoria' $txt 'KO' "Attesi 16 GB."
    } elseif ($moduli.Count -eq 1) {
        Add-Ris 'RAM' 'Memoria' $txt 'ATTENZIONE' "Un solo modulo = single channel: nei giochi si perde fino al 15-30%. Risolvibile aggiungendo un modulo (30-50 euro): argomento per trattare."
    } else {
        Add-Ris 'RAM' 'Memoria' $txt 'OK' "Dual channel. Velocita' attesa 4800 MT/s."
    }
} catch { }

# ---------------------------------------------------------------- GPU
$nvTrovata = $false
try {
    $schede = @(Get-CimInstance Win32_VideoController)
    foreach ($v in $schede) {
        $info = "{0} (driver {1})" -f $v.Name, $v.DriverVersion
        $codice = [int]$v.ConfigManagerErrorCode
        if ($v.Name -match 'NVIDIA') {
            $nvTrovata = $true
            if ($codice -ne 0) {
                $nota = "Errore Gestione dispositivi codice $codice."
                if ($codice -eq 43) { $nota = "CODICE 43: tipico segnale di GPU guasta o driver danneggiato. Forte motivo per NON comprare." }
                Add-Ris 'GPU' 'Scheda dedicata' $info 'KO' $nota
            } elseif ($v.Name -match '4060') {
                Add-Ris 'GPU' 'Scheda dedicata' $info 'OK' ''
            } elseif ($v.Name -match '4050') {
                Add-Ris 'GPU' 'Scheda dedicata' $info 'KO' "E' una RTX 4050 (6 GB), non una RTX 4060!"
            } else {
                Add-Ris 'GPU' 'Scheda dedicata' $info 'ATTENZIONE' "Attesa NVIDIA GeForce RTX 4060 Laptop GPU."
            }
        } elseif ($v.Name -match 'Basic Display|Display di base|Microsoft') {
            Add-Ris 'GPU' 'Scheda video' $info 'ATTENZIONE' "Scheda senza driver: se e' la NVIDIA, il driver manca e il test GPU non e' possibile."
        } else {
            $hz = ''
            if ($v.CurrentRefreshRate) { $hz = " - schermo a {0} Hz" -f $v.CurrentRefreshRate }
            Add-Ris 'GPU' 'Scheda integrata' ($info + $hz) 'INFO' ''
        }
    }

    if (-not $nvTrovata) {
        $pciNv = @(Get-CimInstance Win32_PnPEntity | Where-Object { $_.DeviceID -like 'PCI\VEN_10DE*' })
        if ($pciNv.Count -gt 0) {
            $c = $pciNv | ForEach-Object { "{0} (codice {1})" -f $_.Name, $_.ConfigManagerErrorCode }
            Add-Ris 'GPU' 'Dispositivo NVIDIA' ($c -join '; ') 'KO' "La GPU NVIDIA c'e' ma non funziona come scheda video. Codice 43 = probabile guasto."
        } else {
            Add-Ris 'GPU' 'GPU NVIDIA' 'NON rilevata' 'KO' "Prima di concludere: in Armoury Crate controlla che la 'Modalita GPU' NON sia 'Eco' (GPU spenta), mettila su 'Standard', poi rilancia questo controllo."
        }
    }

    # VRAM dal registro (valore a 64 bit scritto dal driver)
    $vramMB = $null
    $chiave = 'HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}'
    Get-ChildItem -Path $chiave -ErrorAction SilentlyContinue | Where-Object { $_.PSChildName -match '^\d{4}$' } | ForEach-Object {
        $p = Get-ItemProperty -Path $_.PSPath -ErrorAction SilentlyContinue
        if ($p -and ("" + $p.DriverDesc) -match 'NVIDIA') {
            $q = $p.'HardwareInformation.qwMemorySize'
            if ($q -is [byte[]]) { $q = [BitConverter]::ToUInt64($q, 0) }
            if ($q) { $vramMB = [math]::Round([double]$q / 1MB) }
        }
    }

    $smi = Find-NvidiaSmi
    if ($smi) {
        $mem = ConvertTo-Num (Get-NvCampo $smi 'memory.total')
        if ($mem) { $vramMB = [math]::Round($mem) }
        $vbios = Get-NvCampo $smi 'vbios_version'
        $pMax  = Get-NvCampo $smi 'power.max_limit'
        $pDef  = Get-NvCampo $smi 'power.default_limit'
        $gen   = Get-NvCampo $smi 'pcie.link.gen.max'
        $width = Get-NvCampo $smi 'pcie.link.width.max'
        $temp  = Get-NvCampo $smi 'temperature.gpu'
        if ($vbios) { Add-Ris 'GPU' 'vBIOS' $vbios 'INFO' '' }
        if ($pMax -or $pDef) {
            Add-Ris 'GPU' 'Limite potenza (driver)' ("default {0} W / massimo {1} W" -f $pDef, $pMax) 'INFO' "Su questo modello: 115 W + 25 W Dynamic Boost = fino a 140 W. Sui portatili il valore puo' non essere riportato o essere diverso: conta il consumo reale sotto test."
        }
        if ($gen -and $width) {
            $esPcie = 'INFO'
            if ([int]$width -lt 8) { $esPcie = 'ATTENZIONE' }
            Add-Ris 'GPU' 'Collegamento PCIe' ("PCIe {0}.0 x{1}" -f $gen, $width) $esPcie "Atteso PCIe 4.0 x8 (normale per la RTX 4060 Laptop)."
        }
        if ($temp) {
            $esT = 'OK'
            if ([int]$temp -gt 65) { $esT = 'ATTENZIONE' }
            Add-Ris 'GPU' 'Temperatura a riposo' ("{0} C" -f $temp) $esT "A riposo normale 35-55 C."
        }
    } elseif ($nvTrovata) {
        Add-Ris 'GPU' 'nvidia-smi' 'non trovato' 'INFO' "Driver NVIDIA non standard: lo script di monitoraggio GPU non funzionera', usa HWiNFO."
    }

    if ($vramMB) {
        if ($vramMB -ge 7900) {
            Add-Ris 'GPU' 'Memoria video (VRAM)' ("{0} MB" -f $vramMB) 'OK' "Attesi 8 GB (circa 8188 MB)."
        } else {
            Add-Ris 'GPU' 'Memoria video (VRAM)' ("{0} MB" -f $vramMB) 'KO' "Attesi 8 GB. 6 GB = RTX 4050."
        }
    }
} catch { Add-Ris 'GPU' 'Lettura dati' $_.Exception.Message 'INFO' '' }

# ---------------------------------------------------------------- SSD
try {
    $dischi = @(Get-PhysicalDisk -ErrorAction Stop)
    foreach ($d in $dischi) {
        $gb = [math]::Round([double]$d.Size / 1e9)
        $salute = "" + $d.HealthStatus
        $txt = "{0} - {1} GB - {2} - stato: {3}" -f $d.FriendlyName, $gb, $d.BusType, $salute
        $es = 'OK'
        $nota = ''
        if ($salute -notmatch 'Healthy|^0$|Integro') { $es = 'KO'; $nota = "Windows segnala l'SSD come non integro." }
        elseif ($gb -lt 480 -or $gb -gt 540) { $es = 'INFO'; $nota = "Capacita' diversa dai 512 GB originali (SSD sostituito?)." }
        Add-Ris 'SSD' 'Disco' $txt $es $nota

        if ($isAdmin) {
            $rc = $d | Get-StorageReliabilityCounter -ErrorAction SilentlyContinue
            if ($rc) {
                if ($null -ne $rc.Wear) {
                    $esW = 'OK'
                    if ($rc.Wear -ge 80) { $esW = 'KO' } elseif ($rc.Wear -ge 40) { $esW = 'ATTENZIONE' }
                    Add-Ris 'SSD' 'Usura (percentuale consumata)' ("{0} %" -f $rc.Wear) $esW "0-10% normale per un portatile di 2-3 anni."
                }
                if ($null -ne $rc.PowerOnHours) {
                    Add-Ris 'SSD' 'Ore di accensione' ("{0} ore" -f $rc.PowerOnHours) 'INFO' "Uso tipico: 1000-4000 ore in 2-3 anni. Molto di piu' = uso intenso."
                }
                $errNC = 0
                if ($rc.ReadErrorsUncorrected)  { $errNC += $rc.ReadErrorsUncorrected }
                if ($rc.WriteErrorsUncorrected) { $errNC += $rc.WriteErrorsUncorrected }
                if ($errNC -gt 0) { Add-Ris 'SSD' 'Errori non corretti' "$errNC" 'KO' "L'SSD ha avuto errori di lettura/scrittura." }
                if ($rc.Temperature) {
                    $esT = 'OK'
                    if ($rc.Temperature -gt 70) { $esT = 'ATTENZIONE' }
                    Add-Ris 'SSD' 'Temperatura' ("{0} C (max registrata {1} C)" -f $rc.Temperature, $rc.TemperatureMax) $esT ''
                }
            } else {
                Add-Ris 'SSD' 'Dati SMART' 'non disponibili' 'INFO' "Usa la scheda 'Drives' di HWiNFO (Health / Percentage Used)."
            }
        }
    }
    $c = Get-PSDrive -Name C -ErrorAction SilentlyContinue
    if ($c) { Add-Ris 'SSD' 'Spazio libero C:' ("{0} GB liberi su {1} GB" -f [math]::Round($c.Free / 1GB), [math]::Round(($c.Used + $c.Free) / 1GB)) 'INFO' '' }
} catch { Add-Ris 'SSD' 'Lettura dati' $_.Exception.Message 'INFO' '' }

# ---------------------------------------------------------------- BATTERIA E ALIMENTAZIONE
try {
    Add-Type -AssemblyName System.Windows.Forms
    $ps = [System.Windows.Forms.SystemInformation]::PowerStatus
    if ("" + $ps.PowerLineStatus -eq 'Online') {
        Add-Ris 'Alimentazione' 'Caricabatterie' ("collegato - carica {0:P0}" -f $ps.BatteryLifePercent) 'OK' ''
    } else {
        Add-Ris 'Alimentazione' 'Caricabatterie' 'NON collegato' 'ATTENZIONE' "Collega l'alimentatore ORIGINALE (spinotto tondo) prima del test GPU."
    }
} catch { }

try {
    $xmlPath = Join-Path $env:TEMP 'kit_test_batteria.xml'
    & powercfg.exe /batteryreport /xml /output "$xmlPath" | Out-Null
    if (Test-Path $xmlPath) {
        $bx = New-Object System.Xml.XmlDocument
        $bx.Load($xmlPath)
        $bat = @($bx.BatteryReport.Batteries.Battery) | Select-Object -First 1
        if ($bat) {
            $design = ConvertTo-Num $bat.DesignCapacity
            $full   = ConvertTo-Num $bat.FullChargeCapacity
            if ($design -and $full -and $design -gt 0) {
                $salute = [math]::Round($full / $design * 100)
                $txt = "{0}% ({1} Wh su {2} Wh di progetto, cicli: {3})" -f $salute, [math]::Round($full / 1000, 1), [math]::Round($design / 1000, 1), $bat.CycleCount
                $es = 'OK'; $nota = "Sopra l'80%: ottima."
                if ($salute -lt 60) { $es = 'ATTENZIONE'; $nota = "Batteria molto consumata: da sostituire (circa 60-100 euro). Usalo per trattare." }
                elseif ($salute -lt 80) { $es = 'ATTENZIONE'; $nota = "Usura normale per l'eta', autonomia ridotta. Piccolo argomento per trattare." }
                Add-Ris 'Batteria' 'Salute' $txt $es $nota
            }
        }
        Remove-Item $xmlPath -ErrorAction SilentlyContinue
    }
} catch { Add-Ris 'Batteria' 'Salute' 'non verificabile' 'INFO' '' }

# ---------------------------------------------------------------- GESTIONE DISPOSITIVI
try {
    $problemi = @(Get-CimInstance Win32_PnPEntity | Where-Object { $_.ConfigManagerErrorCode -and $_.ConfigManagerErrorCode -ne 0 -and $_.ConfigManagerErrorCode -ne 45 })
    if ($problemi.Count -eq 0) {
        Add-Ris 'Dispositivi' 'Gestione dispositivi' 'nessun errore' 'OK' ''
    } else {
        foreach ($p in $problemi) {
            $es = 'ATTENZIONE'
            $nota = 'Driver mancante o dispositivo con problemi.'
            if ($p.ConfigManagerErrorCode -eq 22) { $es = 'INFO'; $nota = 'Dispositivo disabilitato manualmente.' }
            if ($p.ConfigManagerErrorCode -eq 43) { $es = 'KO'; $nota = 'Codice 43: il dispositivo ha segnalato un guasto.' }
            Add-Ris 'Dispositivi' ("" + $p.Name) ("codice errore {0}" -f $p.ConfigManagerErrorCode) $es $nota
        }
    }
} catch { }

# ---------------------------------------------------------------- REGISTRI ERRORI (ultimi 60 giorni)
$da = (Get-Date).AddDays(-60)
try {
    $bsod = (Get-NumEventi @{ LogName = 'System'; ProviderName = 'Microsoft-Windows-WER-SystemErrorReporting'; Id = 1001; StartTime = $da }) +
            (Get-NumEventi @{ LogName = 'System'; ProviderName = 'BugCheck'; Id = 1001; StartTime = $da })
    $dump = @(Get-ChildItem -Path (Join-Path $env:SystemRoot 'Minidump\*.dmp') -ErrorAction SilentlyContinue)
    $txt = "{0} eventi, {1} file minidump" -f $bsod, $dump.Count
    if ($dump.Count -gt 0) { $txt += " (ultimo: " + ($dump | Sort-Object LastWriteTime -Descending | Select-Object -First 1).LastWriteTime.ToString('dd/MM/yyyy') + ")" }
    if ($bsod -eq 0 -and $dump.Count -eq 0) { Add-Ris 'Registri' 'Schermate blu (BSOD)' $txt 'OK' '' }
    else { Add-Ris 'Registri' 'Schermate blu (BSOD)' $txt 'ATTENZIONE' "Il PC ha avuto crash di sistema. Chiedi al venditore quando e perche'." }

    $tdr = (Get-NumEventi @{ LogName = 'System'; ProviderName = 'Display'; Id = 4101; StartTime = $da }) +
           (Get-NumEventi @{ LogName = 'System'; ProviderName = 'nvlddmkm'; Level = 2; StartTime = $da })
    if ($tdr -eq 0) { Add-Ris 'Registri' 'Crash driver video' '0 eventi' 'OK' '' }
    elseif ($tdr -le 3) { Add-Ris 'Registri' 'Crash driver video' "$tdr eventi" 'ATTENZIONE' "Pochi eventi possono dipendere da un driver o da un gioco; osserva bene il test GPU." }
    else { Add-Ris 'Registri' 'Crash driver video' "$tdr eventi" 'KO' "Molti crash del driver NVIDIA: possibile GPU/VRAM instabile o surriscaldamento." }

    $whea = Get-NumEventi @{ LogName = 'System'; ProviderName = 'Microsoft-Windows-WHEA-Logger'; StartTime = $da }
    if ($whea -eq 0) { Add-Ris 'Registri' 'Errori hardware (WHEA)' '0 eventi' 'OK' '' }
    else { Add-Ris 'Registri' 'Errori hardware (WHEA)' "$whea eventi" 'ATTENZIONE' "Errori hardware registrati da Windows (CPU, RAM o PCIe). Pochi errori 'corretti' possono essere innocui, molti no." }

    $kp = Get-NumEventi @{ LogName = 'System'; ProviderName = 'Microsoft-Windows-Kernel-Power'; Id = 41; StartTime = $da }
    if ($kp -eq 0) { Add-Ris 'Registri' 'Spegnimenti improvvisi' '0 eventi' 'OK' '' }
    elseif ($kp -le 3) { Add-Ris 'Registri' 'Spegnimenti improvvisi' "$kp eventi" 'INFO' "Anche uno spegnimento forzato col tasto conta: pochi eventi sono normali." }
    else { Add-Ris 'Registri' 'Spegnimenti improvvisi' "$kp eventi" 'ATTENZIONE' "Molti riavvii/spegnimenti inattesi: possibile instabilita' o surriscaldamento." }
} catch { }

# ---------------------------------------------------------------- RIEPILOGO
$nKO  = @($Risultati | Where-Object { $_.Esito -eq 'KO' }).Count
$nATT = @($Risultati | Where-Object { $_.Esito -eq 'ATTENZIONE' }).Count

Write-Titolo "RIEPILOGO"
if ($nKO -gt 0) {
    Write-Host ("  {0} problemi ROSSI (KO), {1} punti GIALLI (ATTENZIONE)." -f $nKO, $nATT) -ForegroundColor Red
    Write-Host "  Leggi le righe rosse: se riguardano modello, GPU, VRAM o SSD e' meglio NON comprare." -ForegroundColor Red
} elseif ($nATT -gt 0) {
    Write-Host ("  Nessun problema grave. {0} punti GIALLI da valutare (o da usare per trattare)." -f $nATT) -ForegroundColor Yellow
} else {
    Write-Host "  Nessun problema rilevato. Prosegui con GPU-Z e il test GPU (OCCT)." -ForegroundColor Green
}

# ---------------------------------------------------------------- SALVATAGGIO RAPPORTO
try {
    $serialeFile = ($seriale -replace '[^A-Za-z0-9]', '')
    if (-not $serialeFile) { $serialeFile = 'PC' }
    $file = Join-Path $Global:RapportiDir ("Controllo_{0}_{1}.txt" -f (Get-Date -Format 'yyyyMMdd_HHmm'), $serialeFile)
    $righe = New-Object System.Collections.Generic.List[string]
    $righe.Add("CONTROLLO AUTOMATICO - " + (Get-Date -Format 'dd/MM/yyyy HH:mm'))
    $righe.Add("Amministratore: " + $isAdmin)
    $righe.Add("")
    foreach ($r in $Risultati) {
        $righe.Add(("[{0}] {1} - {2}: {3}" -f $r.Esito, $r.Area, $r.Voce, $r.Valore))
        if ($r.Nota) { $righe.Add("      -> " + $r.Nota) }
    }
    $righe.Add("")
    $righe.Add(("Totale: {0} KO, {1} ATTENZIONE" -f $nKO, $nATT))
    $righe | Out-File -FilePath $file -Encoding UTF8
    Write-Host ""
    Write-Host ("Rapporto salvato in: " + $file) -ForegroundColor Cyan
} catch {
    Write-Host ("Impossibile salvare il rapporto: " + $_.Exception.Message) -ForegroundColor Yellow
}

Wait-Invio
