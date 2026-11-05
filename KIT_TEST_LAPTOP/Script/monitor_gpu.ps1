# monitor_gpu.ps1 - Registra temperatura, consumo, frequenza e motivi di rallentamento
# della GPU NVIDIA mentre OCCT e' in esecuzione, poi da' un giudizio automatico.
# Usa nvidia-smi (installato col driver NVIDIA). Non serve Internet. Non modifica nulla.
#
# Uso: avvia PRIMA il test 3D in OCCT, poi questo script. Premi Q per terminare prima.

param(
    [int]$Minuti = 8,
    [int]$IntervalloSec = 2
)

. (Join-Path $PSScriptRoot 'comune.ps1')

Write-Titolo "MONITOR GPU - durata $Minuti minuti (premi Q per fermare prima)"

$smi = Find-NvidiaSmi
if (-not $smi) {
    Write-Host "nvidia-smi non trovato: il driver NVIDIA sembra mancante." -ForegroundColor Red
    Write-Host "Senza driver NVIDIA il test GPU non e' affidabile. Usa solo HWiNFO e valuta con prudenza." -ForegroundColor Red
    Wait-Invio
    exit 1
}

# Campi richiesti (verifico quali sono supportati da questo driver)
$campiDesiderati = @('temperature.gpu', 'power.draw', 'clocks.gr', 'clocks.mem', 'utilization.gpu', 'fan.speed')
$campi = @()
foreach ($c in $campiDesiderati) { if (Test-NvCampo $smi $c) { $campi += $c } }
$campoMotivi = $null
foreach ($c in @('clocks_event_reasons.active', 'clocks_throttle_reasons.active')) {
    if (Test-NvCampo $smi $c) { $campoMotivi = $c; break }
}
if ($campoMotivi) { $campi += $campoMotivi }
if ($campi.Count -eq 0) {
    Write-Host "nvidia-smi non risponde correttamente." -ForegroundColor Red
    Wait-Invio
    exit 1
}
$query = $campi -join ','
$nomeGpu = Get-NvCampo $smi 'name'
Write-Host ("GPU: {0}" -f $nomeGpu)
Write-Host "Legenda motivi: PWR = limite di potenza (NORMALE sotto carico), TERM = rallenta per calore," -ForegroundColor DarkGray
Write-Host "                HW = rallentamento hardware (calore estremo o alimentatore insufficiente)" -ForegroundColor DarkGray
Write-Host ""

$csv = Join-Path $Global:RapportiDir ("MonitorGPU_{0}.csv" -f (Get-Date -Format 'yyyyMMdd_HHmmss'))
("secondi," + $query) | Out-File -FilePath $csv -Encoding ASCII

function Get-Motivi([UInt64]$m) {
    $l = @()
    if ($m -band 0x4)  { $l += 'PWR' }
    if ($m -band 0x60) { $l += 'TERM' }
    if ($m -band 0x88) { $l += 'HW' }
    return ($l -join '+')
}

$campioni = New-Object System.Collections.Generic.List[object]
$inizio = Get-Date
$fine = $inizio.AddMinutes($Minuti)

Write-Host (" {0,6} {1,6} {2,8} {3,8} {4,6} {5,6}  {6}" -f 'tempo', 'temp', 'potenza', 'freq', 'carico', 'ventol', 'motivi')
while ((Get-Date) -lt $fine) {
    $t0 = Get-Date
    $riga = & $smi "--query-gpu=$query" '--format=csv,noheader,nounits' 2>$null | Select-Object -First 1
    if ($riga) {
        $sec = [int]((Get-Date) - $inizio).TotalSeconds
        ("{0},{1}" -f $sec, $riga) | Out-File -FilePath $csv -Append -Encoding ASCII
        $val = @{}
        $parti = ("" + $riga).Split(',')
        for ($i = 0; $i -lt $campi.Count -and $i -lt $parti.Count; $i++) { $val[$campi[$i]] = $parti[$i].Trim() }

        $mot = [UInt64]0
        if ($campoMotivi -and $val[$campoMotivi] -match '^0x([0-9A-Fa-f]+)$') { $mot = [Convert]::ToUInt64($Matches[1], 16) }

        $s = [pscustomobject]@{
            Sec    = $sec
            Temp   = ConvertTo-Num $val['temperature.gpu']
            Pot    = ConvertTo-Num $val['power.draw']
            Freq   = ConvertTo-Num $val['clocks.gr']
            Carico = ConvertTo-Num $val['utilization.gpu']
            Ventola = $val['fan.speed']
            Motivi = $mot
        }
        $campioni.Add($s)

        $colore = 'Gray'
        if ($s.Temp -ge 88 -or ($mot -band 0xE8)) { $colore = 'Yellow' }
        if ($s.Temp -ge 92) { $colore = 'Red' }
        $min = "{0}:{1:00}" -f [int][math]::Floor($sec / 60), ($sec % 60)
        Write-Host (" {0,6} {1,4} C {2,6} W {3,5} MHz {4,4} % {5,6}  {6}" -f $min, $s.Temp, $s.Pot, $s.Freq, $s.Carico, $s.Ventola, (Get-Motivi $mot)) -ForegroundColor $colore
    }

    try {
        if ([Console]::KeyAvailable) {
            $k = [Console]::ReadKey($true)
            if ($k.Key -eq 'Q') { Write-Host "Interrotto dall'utente." -ForegroundColor Yellow; break }
        }
    } catch { }

    $attesa = $IntervalloSec * 1000 - [int]((Get-Date) - $t0).TotalMilliseconds
    if ($attesa -gt 0) { Start-Sleep -Milliseconds $attesa }
}

# ---------------------------------------------------------------- ANALISI
$carico = @($campioni | Where-Object { $_.Carico -ge 70 })
$righe = New-Object System.Collections.Generic.List[string]
$esiti = New-Object System.Collections.Generic.List[object]
function Add-Esito([string]$Esito, [string]$Testo) { $esiti.Add([pscustomobject]@{ Esito = $Esito; Testo = $Testo }) }

if ($carico.Count -lt 10) {
    Add-Esito 'ATTENZIONE' "La GPU NVIDIA non e' stata sotto carico (meno di 20 secondi con utilizzo >= 70%). OCCT era avviato sulla RTX 4060? Ripeti il test."
} else {
    $durataCarico = $carico.Count * $IntervalloSec
    $ultimoSec = ($carico | Measure-Object -Property Sec -Maximum).Maximum
    $primoSec  = ($carico | Measure-Object -Property Sec -Minimum).Minimum
    $finali  = @($carico | Where-Object { $_.Sec -ge ($ultimoSec - 120) })
    $iniziali = @($carico | Where-Object { $_.Sec -le ($primoSec + 60) })

    $tMax   = ($carico | Measure-Object -Property Temp -Maximum).Maximum
    $tFinale = [math]::Round(($finali | Measure-Object -Property Temp -Average).Average)
    $pMed   = [math]::Round(($carico | Measure-Object -Property Pot -Average).Average)
    $pMax   = [math]::Round(($carico | Measure-Object -Property Pot -Maximum).Maximum)
    $fIni   = ($iniziali | Measure-Object -Property Freq -Average).Average
    $fFin   = ($finali | Measure-Object -Property Freq -Average).Average
    $calo   = 0
    if ($fIni -gt 0) { $calo = [math]::Round((1 - $fFin / $fIni) * 100) }
    $pctTerm = [math]::Round(100 * @($carico | Where-Object { $_.Motivi -band 0x60 }).Count / $carico.Count)
    $pctHw   = [math]::Round(100 * @($carico | Where-Object { $_.Motivi -band 0x88 }).Count / $carico.Count)

    $righe.Add(("Tempo sotto carico: {0} s" -f $durataCarico))
    $righe.Add(("Temperatura massima: {0} C - media ultimi 2 minuti: {1} C" -f $tMax, $tFinale))
    $righe.Add(("Consumo GPU: medio {0} W, massimo {1} W" -f $pMed, $pMax))
    $righe.Add(("Frequenza GPU: primo minuto {0} MHz, ultimi 2 minuti {1} MHz (calo {2}%)" -f [math]::Round($fIni), [math]::Round($fFin), $calo))
    $righe.Add(("Campioni con rallentamento termico: {0}% - rallentamento hardware: {1}%" -f $pctTerm, $pctHw))

    if ($durataCarico -lt 300) { Add-Esito 'INFO' "Test breve (meno di 5 minuti sotto carico): le temperature potrebbero non essersi stabilizzate." }

    if ($tMax -ge 92) { Add-Esito 'KO' "Temperatura GPU troppo alta ($tMax C): raffreddamento inefficiente (polvere, pasta termica secca, ventola)." }
    elseif ($tMax -ge 88) { Add-Esito 'ATTENZIONE' "GPU al limite termico ($tMax C). Accettabile solo se frequenza e consumo restano stabili." }
    else { Add-Esito 'OK' "Temperatura GPU nella norma ($tMax C max)." }

    if ($pctTerm -ge 25) { Add-Esito 'ATTENZIONE' "La GPU rallenta spesso per il calore ($pctTerm% del tempo)." }
    if ($pctHw -ge 5) { Add-Esito 'ATTENZIONE' "Rallentamenti HARDWARE ($pctHw% del tempo): alimentatore non originale/insufficiente o protezione termica. Verifica l'alimentatore." }

    if ($calo -ge 20) { Add-Esito 'ATTENZIONE' "La frequenza cala del $calo% durante il test: raffreddamento al limite." }
    else { Add-Esito 'OK' "Frequenza stabile durante il test (calo $calo%)." }

    if ($pMed -lt 60) { Add-Esito 'ATTENZIONE' "Consumo medio basso ($pMed W): modalita' Silenziosa attiva, alimentatore scollegato/sbagliato o GPU limitata. In 'Prestazioni'/'Turbo' ci si aspetta 80-140 W." }
    elseif ($pMed -ge 80) { Add-Esito 'OK' "Consumo GPU in linea con una RTX 4060 Laptop ($pMed W medi)." }
    else { Add-Esito 'INFO' "Consumo medio $pMed W: plausibile in modalita' Prestazioni con carico misto." }
}

Write-Titolo "RISULTATO MONITOR GPU"
foreach ($r in $righe) { Write-Host ("  " + $r) }
Write-Host ""
foreach ($e in $esiti) { Write-Host ("  [{0}] {1}" -f $e.Esito, $e.Testo) -ForegroundColor (Get-ColoreEsito $e.Esito) }
Write-Host ""
Write-Host "  Ricorda: in OCCT gli ERRORI devono essere 0 e non devono comparire artefatti a schermo." -ForegroundColor Cyan

$riepilogo = $csv -replace '\.csv$', '_riepilogo.txt'
$out = @("MONITOR GPU - " + (Get-Date -Format 'dd/MM/yyyy HH:mm'), "GPU: $nomeGpu", "")
$out += $righe
$out += ""
$out += ($esiti | ForEach-Object { "[{0}] {1}" -f $_.Esito, $_.Testo })
$out | Out-File -FilePath $riepilogo -Encoding UTF8
Write-Host ("  Dati salvati in: " + $csv) -ForegroundColor Cyan
Write-Host ("  Riepilogo in:    " + $riepilogo) -ForegroundColor Cyan

Wait-Invio
