# analizza_hwinfo.ps1 - Legge il file CSV registrato da HWiNFO (pulsante "Logging")
# e riassume temperature CPU/GPU/SSD, consumi, frequenze e throttling.
# Senza parametri usa il CSV piu' recente nella cartella Rapporti (o nella cartella di HWiNFO).

param([string]$Percorso = '')

. (Join-Path $PSScriptRoot 'comune.ps1')

Write-Titolo "ANALISI LOG HWiNFO"

if (-not $Percorso) {
    $cartelle = @($Global:RapportiDir, (Join-Path $Global:ProgrammiDir 'HWiNFO64'))
    $csvTrovati = @()
    foreach ($c in $cartelle) {
        $csvTrovati += @(Get-ChildItem -Path (Join-Path $c '*.csv') -ErrorAction SilentlyContinue | Where-Object { $_.Name -notlike 'MonitorGPU_*' })
    }
    $ultimo = $csvTrovati | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $ultimo) {
        Write-Host "Nessun file CSV di HWiNFO trovato in Rapporti." -ForegroundColor Red
        Write-Host "In HWiNFO (finestra Sensors) premi il pulsante di logging in basso e salva il file nella cartella Rapporti della chiavetta." -ForegroundColor Yellow
        Wait-Invio
        exit 1
    }
    $Percorso = $ultimo.FullName
}
Write-Host ("File: " + $Percorso)

# Rileva separatore (HWiNFO usa la virgola; alcune impostazioni usano il punto e virgola)
$primaRiga = Get-Content -Path $Percorso -TotalCount 1 -Encoding Default
$separatore = ','
if (($primaRiga.Split(';').Count) -gt ($primaRiga.Split(',').Count)) { $separatore = ';' }

Add-Type -AssemblyName Microsoft.VisualBasic
$parser = New-Object Microsoft.VisualBasic.FileIO.TextFieldParser($Percorso, [System.Text.Encoding]::Default)
$parser.TextFieldType = [Microsoft.VisualBasic.FileIO.FieldType]::Delimited
$parser.SetDelimiters($separatore)
$parser.HasFieldsEnclosedInQuotes = $true
$intestazione = $parser.ReadFields()
$dati = New-Object System.Collections.Generic.List[string[]]
$gruppi = $null
while (-not $parser.EndOfData) {
    try { $f = $parser.ReadFields() } catch { continue }
    if ($null -eq $f -or $f.Count -lt 3) { continue }
    if ($f[0] -match '^\d') { $dati.Add($f) }
    elseif ((($f -join ' ') -match 'GPU \[#|CPU \[#|NVIDIA|GeForce')) { $gruppi = $f }
}
$parser.Close()

if ($dati.Count -lt 5) {
    Write-Host "Il log contiene troppi pochi campioni ($($dati.Count)). Registra almeno 5 minuti." -ForegroundColor Red
    Wait-Invio
    exit 1
}

function Get-ColonneIdx([string]$Regex) {
    $r = @()
    for ($i = 0; $i -lt $intestazione.Count; $i++) { if ($intestazione[$i] -match $Regex) { $r += $i } }
    return ,$r
}
function Get-Valori([int]$Idx, [object[]]$Righe) {
    $v = New-Object System.Collections.Generic.List[double]
    foreach ($riga in $Righe) {
        if ($Idx -lt $riga.Count) { $n = ConvertTo-Num $riga[$Idx]; if ($null -ne $n) { $v.Add($n) } }
    }
    return ,$v
}
function Get-Stat([int]$Idx, [object[]]$Righe) {
    $v = Get-Valori $Idx $Righe
    if ($v.Count -eq 0) { return $null }
    $m = $v | Measure-Object -Maximum -Minimum -Average
    return [pscustomobject]@{ Max = [math]::Round($m.Maximum, 1); Min = [math]::Round($m.Minimum, 1); Avg = [math]::Round($m.Average, 1) }
}
function Get-PctSi([int]$Idx, [object[]]$Righe) {
    $tot = 0; $si = 0
    foreach ($riga in $Righe) {
        if ($Idx -lt $riga.Count) {
            $x = ("" + $riga[$Idx]).Trim()
            if ($x -match '^(Yes|No|Si|S.|1|0)$') { $tot++; if ($x -match '^(Yes|Si|S.|1)$') { $si++ } }
        }
    }
    if ($tot -eq 0) { return $null }
    return [math]::Round(100 * $si / $tot)
}
# Sceglie la colonna: se possibile quella del gruppo NVIDIA, altrimenti quella col massimo piu' alto
function Select-Colonna([string]$Regex, [string]$GruppoRegex = '') {
    $idx = Get-ColonneIdx $Regex
    if ($idx.Count -eq 0) { return -1 }
    if ($GruppoRegex -and $gruppi) {
        foreach ($i in $idx) { if ($i -lt $gruppi.Count -and $gruppi[$i] -match $GruppoRegex) { return $i } }
    }
    if ($idx.Count -eq 1) { return $idx[0] }
    $migliore = $idx[0]; $max = -1e9
    foreach ($i in $idx) { $s = Get-Stat $i $dati; if ($s -and $s.Max -gt $max) { $max = $s.Max; $migliore = $i } }
    return $migliore
}

# Righe "sotto carico" = utilizzo GPU NVIDIA >= 70% (se la colonna esiste)
$righeCarico = $dati.ToArray()
$iCaricoGpu = Select-Colonna '^GPU (Core Load|Utilization)' 'NVIDIA|GeForce'
if ($iCaricoGpu -ge 0) {
    $tmp = @($dati | Where-Object { $n = ConvertTo-Num $_[$iCaricoGpu]; $null -ne $n -and $n -ge 70 })
    if ($tmp.Count -ge 5) { $righeCarico = $tmp }
}
$durataMin = [math]::Round($dati.Count * 2 / 60, 1)
Write-Host ("Campioni: {0} (circa {1} minuti a 2 s), di cui sotto carico GPU: {2}" -f $dati.Count, $durataMin, $righeCarico.Count)
Write-Host ""

$esiti = New-Object System.Collections.Generic.List[object]
function Add-Esito([string]$Esito, [string]$Testo) {
    $esiti.Add([pscustomobject]@{ Esito = $Esito; Testo = $Testo })
    Write-Host (("[" + $Esito + "]").PadRight(13)) -ForegroundColor (Get-ColoreEsito $Esito) -NoNewline
    Write-Host $Testo
}

# ---------------- CPU
$i = Select-Colonna 'CPU \(Tctl/Tdie\)|^CPU Tctl|CPU Die \(average\)'
if ($i -ge 0) {
    $s = Get-Stat $i $dati
    $txt = "CPU temperatura: max {0} C, media {1} C, minima {2} C" -f $s.Max, $s.Avg, $s.Min
    if ($s.Max -gt 97) { Add-Esito 'KO' ($txt + " - oltre il limite di 95 C del Ryzen: raffreddamento insufficiente.") }
    elseif ($s.Avg -ge 93) { Add-Esito 'ATTENZIONE' ($txt + " - CPU sempre al limite termico: probabile polvere o pasta termica da rifare.") }
    else { Add-Esito 'OK' ($txt + " - normale per questo portatile (fino a 95 C sotto carico e' previsto).") }
    if ($s.Min -ge 70) { Add-Esito 'ATTENZIONE' "CPU mai sotto i 70 C, nemmeno a riposo: dissipazione scarsa." }
}
$i = Select-Colonna 'Thermal Throttling \(HTC\)'
if ($i -ge 0) {
    $p = Get-PctSi $i $dati
    if ($null -ne $p) {
        if ($p -ge 30) { Add-Esito 'ATTENZIONE' "CPU in throttling termico (HTC) per il $p% del tempo." }
        elseif ($p -gt 0) { Add-Esito 'INFO' "CPU in throttling termico (HTC) per il $p% del tempo: breve, accettabile." }
        else { Add-Esito 'OK' "Nessun throttling termico della CPU (HTC)." }
    }
}
$i = Select-Colonna 'PROCHOT'
if ($i -ge 0) {
    $p = Get-PctSi $i $dati
    if ($p -gt 0) { Add-Esito 'ATTENZIONE' "Segnale PROCHOT attivo per il $p% del tempo (rallentamento d'emergenza per calore o alimentazione)." }
}
$i = Select-Colonna 'CPU Package Power|CPU PPT'
if ($i -ge 0) { $s = Get-Stat $i $dati; Add-Esito 'INFO' ("CPU consumo: max {0} W, medio {1} W (atteso 35-80 W sotto carico a seconda della modalita')" -f $s.Max, $s.Avg) }
$i = Select-Colonna 'Core Clocks \(avg\)|Average Effective Clock'
if ($i -ge 0) { $s = Get-Stat $i $dati; Add-Esito 'INFO' ("CPU frequenza media: max {0} MHz, media {1} MHz" -f $s.Max, $s.Avg) }

# ---------------- GPU NVIDIA
$i = Select-Colonna '^GPU Temperature' 'NVIDIA|GeForce'
if ($i -ge 0) {
    $s = Get-Stat $i $righeCarico
    $txt = "GPU temperatura sotto carico: max {0} C, media {1} C" -f $s.Max, $s.Avg
    if ($s.Max -ge 92) { Add-Esito 'KO' ($txt + " - troppo alta.") }
    elseif ($s.Max -ge 88) { Add-Esito 'ATTENZIONE' ($txt + " - al limite termico (87 C).") }
    else { Add-Esito 'OK' ($txt + " - nella norma.") }
}
$i = Select-Colonna 'GPU Hot ?Spot' 'NVIDIA|GeForce'
if ($i -ge 0) {
    $s = Get-Stat $i $righeCarico
    if ($s) {
        $es = 'OK'; if ($s.Max -ge 105) { $es = 'KO' } elseif ($s.Max -ge 100) { $es = 'ATTENZIONE' }
        Add-Esito $es ("GPU Hot Spot: max {0} C (normale fino a circa 95-100 C)" -f $s.Max)
    }
}
$i = Select-Colonna 'GPU Memory Junction' 'NVIDIA|GeForce'
if ($i -ge 0) { $s = Get-Stat $i $righeCarico; if ($s) { Add-Esito 'INFO' ("GPU Memory Junction: max {0} C" -f $s.Max) } }
$i = Select-Colonna '^GPU Power( \[W\])?$|^GPU Power \[' 'NVIDIA|GeForce'
if ($i -ge 0) {
    $s = Get-Stat $i $righeCarico
    $es = 'OK'; if ($s.Avg -lt 60) { $es = 'ATTENZIONE' }
    Add-Esito $es ("GPU consumo sotto carico: medio {0} W, max {1} W (atteso 80-140 W in Prestazioni/Turbo)" -f $s.Avg, $s.Max)
}
$i = Select-Colonna '^GPU Clock' 'NVIDIA|GeForce'
if ($i -ge 0) {
    $s = Get-Stat $i $righeCarico
    $es = 'OK'; if ($s.Avg -lt 1500) { $es = 'ATTENZIONE' }
    Add-Esito $es ("GPU frequenza sotto carico: media {0} MHz, minima {1} MHz (atteso 1800-2400 MHz)" -f $s.Avg, $s.Min)
}
$i = Select-Colonna 'Performance Limit - Thermal' 'NVIDIA|GeForce'
if ($i -ge 0) {
    $p = Get-PctSi $i $righeCarico
    if ($null -ne $p) {
        if ($p -ge 25) { Add-Esito 'ATTENZIONE' "GPU limitata dal calore per il $p% del tempo sotto carico." }
        else { Add-Esito 'OK' "GPU limitata dal calore: $p% del tempo (va bene sotto il 25%)." }
    }
}
$i = Select-Colonna 'Performance Limit - Power' 'NVIDIA|GeForce'
if ($i -ge 0) {
    $p = Get-PctSi $i $righeCarico
    if ($null -ne $p) { Add-Esito 'INFO' "GPU al limite di potenza per il $p% del tempo: NORMALE sotto carico." }
}

# ---------------- SSD, ventole, errori
$idxSsd = Get-ColonneIdx 'Drive Temperature|Drive Airflow Temperature'
if ($idxSsd.Count -gt 0) {
    $maxSsd = -1
    foreach ($j in $idxSsd) { $s = Get-Stat $j $dati; if ($s -and $s.Max -gt $maxSsd) { $maxSsd = $s.Max } }
    $es = 'OK'; if ($maxSsd -ge 80) { $es = 'KO' } elseif ($maxSsd -ge 70) { $es = 'ATTENZIONE' }
    Add-Esito $es ("SSD temperatura massima: {0} C (normale sotto 70 C)" -f $maxSsd)
}
$idxFan = Get-ColonneIdx '\[RPM\]|RPM$'
foreach ($j in $idxFan) {
    $s = Get-Stat $j $righeCarico
    if ($s) {
        $es = 'INFO'; if ($s.Max -le 0) { $es = 'ATTENZIONE' }
        $nota = ''; if ($s.Max -le 0) { $nota = " - sempre 0: ventola ferma o sensore non letto. Verifica ascoltando/toccando l'uscita d'aria!" }
        Add-Esito $es ("Ventola '{0}': max {1} RPM{2}" -f $intestazione[$j], $s.Max, $nota)
    }
}
$i = Select-Colonna '^Total Errors'
if ($i -ge 0) {
    $s = Get-Stat $i $dati
    if ($s -and $s.Max -gt 0) { Add-Esito 'KO' ("Errori hardware WHEA durante la registrazione: {0}" -f $s.Max) }
    elseif ($s) { Add-Esito 'OK' "Nessun errore hardware WHEA durante la registrazione." }
}
$i = Select-Colonna 'Wear Level'
if ($i -ge 0) { $s = Get-Stat $i $dati; if ($s) { Add-Esito 'INFO' ("Batteria, usura (Wear Level): {0}%" -f $s.Max) } }

# ---------------- Riepilogo
$nKO = @($esiti | Where-Object { $_.Esito -eq 'KO' }).Count
$nAT = @($esiti | Where-Object { $_.Esito -eq 'ATTENZIONE' }).Count
Write-Titolo "RIEPILOGO HWiNFO"
if ($nKO -gt 0) { Write-Host "  $nKO problemi ROSSI: raffreddamento o hardware da approfondire seriamente." -ForegroundColor Red }
elseif ($nAT -gt 0) { Write-Host "  Nessun problema grave, $nAT punti gialli da valutare." -ForegroundColor Yellow }
else { Write-Host "  Temperature, consumi e frequenze nella norma." -ForegroundColor Green }
if ($esiti.Count -eq 0) { Write-Host "  Nessun sensore riconosciuto nel log: leggi i valori a mano in HWiNFO (vedi README)." -ForegroundColor Yellow }

$fileOut = Join-Path $Global:RapportiDir ("AnalisiHWiNFO_{0}.txt" -f (Get-Date -Format 'yyyyMMdd_HHmm'))
$out = @("ANALISI LOG HWiNFO - " + (Get-Date -Format 'dd/MM/yyyy HH:mm'), "File: $Percorso", "")
$out += ($esiti | ForEach-Object { "[{0}] {1}" -f $_.Esito, $_.Testo })
$out | Out-File -FilePath $fileOut -Encoding UTF8
Write-Host ("  Riepilogo salvato in: " + $fileOut) -ForegroundColor Cyan

Wait-Invio
