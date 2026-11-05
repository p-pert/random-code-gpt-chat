# prepara_chiavetta.ps1 - DA ESEGUIRE A CASA, con Internet, sul TUO PC Windows.
# Scarica GPU-Z, HWiNFO64 Portable e OCCT SOLO dai siti ufficiali, li mette nelle
# cartelle giuste e verifica la firma digitale dei programmi.
# Se un download automatico non riesce, apre la pagina ufficiale nel browser:
# scarichi tu il file e lo script lo trova da solo nella cartella Download.

. (Join-Path $PSScriptRoot 'comune.ps1')

try { [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12 } catch { }
$ProgressPreference = 'SilentlyContinue'

$dirGpuz   = Join-Path $Global:ProgrammiDir 'GPU-Z'
$dirHwinfo = Join-Path $Global:ProgrammiDir 'HWiNFO64'
$dirOcct   = Join-Path $Global:ProgrammiDir 'OCCT'
$dirTmp    = Join-Path $Global:KitRoot '_download_temporanei'
foreach ($d in @($dirGpuz, $dirHwinfo, $dirOcct, $dirTmp)) { New-Item -ItemType Directory -Path $d -Force | Out-Null }

$download = Join-Path $env:USERPROFILE 'Downloads'
try {
    $k = Get-ItemProperty 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders' -ErrorAction Stop
    $v = $k.'{374DE290-123F-4565-9164-39C4925E467B}'
    if ($v) { $download = [Environment]::ExpandEnvironmentVariables($v) }
} catch { }

$UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) KitTestLaptop/1.0'

Write-Titolo "PREPARAZIONE CHIAVETTA - download dai siti ufficiali"
Write-Host "Cartella del kit: $($Global:KitRoot)"
Write-Host "Cartella Download controllata: $download"

# Sblocca i file del kit (rimuove il segno 'scaricato da Internet' che fa comparire avvisi)
Get-ChildItem -Path $Global:KitRoot -Recurse -File -ErrorAction SilentlyContinue | Unblock-File -ErrorAction SilentlyContinue

function Test-Exe([string]$File) {
    if (-not (Test-Path $File)) { return $false }
    if ((Get-Item $File).Length -lt 100KB) { return $false }
    $fs = [System.IO.File]::OpenRead($File)
    try { $b1 = $fs.ReadByte(); $b2 = $fs.ReadByte() } finally { $fs.Close() }
    return ($b1 -eq 0x4D -and $b2 -eq 0x5A)   # 'MZ'
}
function Test-Zip([string]$File) {
    if (-not (Test-Path $File)) { return $false }
    $fs = [System.IO.File]::OpenRead($File)
    try { $b1 = $fs.ReadByte(); $b2 = $fs.ReadByte() } finally { $fs.Close() }
    return ($b1 -eq 0x50 -and $b2 -eq 0x4B)   # 'PK'
}
function Save-Url([string]$Url, [string]$Dest) {
    try {
        Invoke-WebRequest -Uri $Url -OutFile $Dest -UseBasicParsing -UserAgent $UA -TimeoutSec 180 -ErrorAction Stop
        return $true
    } catch {
        Write-Host ("   download automatico non riuscito: " + $_.Exception.Message) -ForegroundColor DarkGray
        return $false
    }
}
# Cerca nella cartella Download un file scaricato a mano (il piu' recente)
function Find-InDownload([string]$Pattern) {
    return Get-ChildItem -Path (Join-Path $download $Pattern) -ErrorAction SilentlyContinue |
           Sort-Object LastWriteTime -Descending | Select-Object -First 1
}
function Request-Manuale([string]$Nome, [string]$Pagina, [string]$Istruzioni, [string]$Pattern) {
    Write-Host ""
    Write-Host ("   Apro la pagina UFFICIALE di {0}:" -f $Nome) -ForegroundColor Yellow
    Write-Host ("   " + $Pagina) -ForegroundColor Yellow
    Write-Host ("   " + $Istruzioni) -ForegroundColor Yellow
    Start-Process $Pagina
    while ($true) {
        $r = Read-Host "   Quando il download e' FINITO premi INVIO (oppure scrivi S per saltare)"
        if ($r -match '^[sS]') { return $null }
        $f = Find-InDownload $Pattern
        if ($f) { Write-Host ("   Trovato: " + $f.FullName) -ForegroundColor Green; return $f }
        Write-Host ("   Non trovo file '{0}' in {1}. Controlla che il download sia terminato." -f $Pattern, $download) -ForegroundColor Red
    }
}

$riepilogo = New-Object System.Collections.Generic.List[string]

# =============================================================== HWiNFO64 Portable
Write-Titolo "1/3 HWiNFO64 Portable (www.hwinfo.com)"
$hwExe = Join-Path $dirHwinfo 'HWiNFO64.exe'
if (Test-Exe $hwExe) {
    Write-Host "   Gia' presente, salto il download. (Cancella la cartella per riscaricarlo.)" -ForegroundColor Green
} else {
    $zip = Join-Path $dirTmp 'hwinfo_portable.zip'
    $ok = $false
    try {
        $pag = Invoke-WebRequest -Uri 'https://www.hwinfo.com/download/' -UseBasicParsing -UserAgent $UA -TimeoutSec 60 -ErrorAction Stop
        $link = [regex]::Matches($pag.Content, 'https?://[^"''\s]*hwinfo\.com/files/hwi_?\d+\.zip') | ForEach-Object { $_.Value } | Select-Object -First 1
        if (-not $link) {
            $rel = [regex]::Match($pag.Content, '/files/hwi_?\d+\.zip').Value
            if ($rel) { $link = 'https://www.hwinfo.com' + $rel }
        }
        if ($link) {
            Write-Host "   Scarico $link"
            $ok = (Save-Url $link $zip) -and (Test-Zip $zip)
        }
    } catch { Write-Host ("   Pagina download non raggiungibile: " + $_.Exception.Message) -ForegroundColor DarkGray }
    if (-not $ok) {
        $f = Request-Manuale 'HWiNFO' 'https://www.hwinfo.com/download/' "Scegli la versione 'Portable' (file ZIP, es. hwi_852.zip) dal server 'Local (U.S.)' o 'Local (EU)'." 'hwi*.zip'
        if ($f) { Copy-Item $f.FullName $zip -Force; $ok = Test-Zip $zip }
    }
    if ($ok) {
        Expand-Archive -Path $zip -DestinationPath $dirHwinfo -Force
        $trovato = Get-ChildItem -Path $dirHwinfo -Recurse -Filter 'HWiNFO64.exe' | Select-Object -First 1
        if ($trovato -and $trovato.DirectoryName -ne $dirHwinfo) {
            Get-ChildItem -Path $trovato.DirectoryName | Move-Item -Destination $dirHwinfo -Force
        }
    }
}
$ini = Join-Path $dirHwinfo 'HWiNFO64.INI'
$iniModello = Join-Path $PSScriptRoot 'HWiNFO64.INI.modello'
if ((Test-Path $hwExe) -and -not (Test-Path $ini) -and (Test-Path $iniModello)) { Copy-Item $iniModello $ini }

# =============================================================== GPU-Z
Write-Titolo "2/3 GPU-Z (www.techpowerup.com)"
$gpuzExe = Join-Path $dirGpuz 'GPU-Z.exe'
if (Test-Exe $gpuzExe) {
    Write-Host "   Gia' presente, salto il download." -ForegroundColor Green
} else {
    # TechPowerUp richiede la scelta del server di download dal browser: procedura guidata
    $f = Request-Manuale 'GPU-Z' 'https://www.techpowerup.com/download/techpowerup-gpu-z/' "Premi 'Download' sulla versione standard (NON 'ASUS ROG'), poi scegli un server (es. 'TechPowerUp Germany')." 'GPU-Z*.exe'
    if ($f) { Copy-Item $f.FullName $gpuzExe -Force }
}

# =============================================================== OCCT
Write-Titolo "3/3 OCCT Personal (www.ocbase.com)"
$occtExe = Join-Path $dirOcct 'OCCT.exe'
if (Test-Exe $occtExe) {
    Write-Host "   Gia' presente, salto il download." -ForegroundColor Green
} else {
    $tmpOcct = Join-Path $dirTmp 'OCCT.exe'
    $ok = $false
    foreach ($u in @('https://www.ocbase.com/download/edition:Personal/os:Windows')) {
        Write-Host "   Provo $u"
        if ((Save-Url $u $tmpOcct) -and (Test-Exe $tmpOcct)) { $ok = $true; break }
    }
    if ($ok) {
        Move-Item $tmpOcct $occtExe -Force
    } else {
        $f = Request-Manuale 'OCCT' 'https://www.ocbase.com/download' "Scegli 'Personal' (gratuito, nessun account) e 'Windows'. Il file e' un unico .exe portatile." 'OCCT*.exe'
        if ($f) { Copy-Item $f.FullName $occtExe -Force }
    }
}

# =============================================================== VERIFICA FIRME E VERSIONI
Write-Titolo "VERIFICA FIRMA DIGITALE E VERSIONI"
$attesi = @(
    @{ Nome = 'GPU-Z';    File = $gpuzExe;  Firma = 'TechPowerUp' },
    @{ Nome = 'HWiNFO64'; File = $hwExe;    Firma = 'Mal.k|REALiX|HWiNFO' },
    @{ Nome = 'OCCT';     File = $occtExe;  Firma = 'OCBASE|Ocbase|OCCT' }
)
$tuttoOk = $true
foreach ($a in $attesi) {
    if (-not (Test-Exe $a.File)) {
        Write-Host ("[MANCA]      {0}: file non presente ({1})" -f $a.Nome, $a.File) -ForegroundColor Red
        $riepilogo.Add(("{0}: MANCANTE" -f $a.Nome)); $tuttoOk = $false
        continue
    }
    $sig = Get-AuthenticodeSignature -FilePath $a.File
    $firmatario = ''
    if ($sig.SignerCertificate) { $firmatario = $sig.SignerCertificate.Subject }
    $ver = (Get-Item $a.File).VersionInfo
    $versione = ("" + $ver.ProductVersion).Trim()
    if (-not $versione) { $versione = ("" + $ver.FileVersion).Trim() }
    $hash = (Get-FileHash -Path $a.File -Algorithm SHA256).Hash
    $esito = 'OK'; $colore = 'Green'
    if ("" + $sig.Status -ne 'Valid') { $esito = 'FIRMA NON VALIDA'; $colore = 'Red'; $tuttoOk = $false }
    elseif ($firmatario -notmatch $a.Firma) { $esito = 'FIRMATARIO INATTESO (controlla)'; $colore = 'Yellow' }
    Write-Host ("[{0}] {1} versione {2}" -f $esito, $a.Nome, $versione) -ForegroundColor $colore
    Write-Host ("      Firmato da: {0}" -f $firmatario) -ForegroundColor DarkGray
    $riepilogo.Add(("{0} {1} - firma: {2} - {3}" -f $a.Nome, $versione, $esito, $firmatario))
    $riepilogo.Add(("    SHA256 {0}" -f $hash))
}

Remove-Item -Path $dirTmp -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path $Global:ProgrammiDir -Recurse -File -ErrorAction SilentlyContinue | Unblock-File -ErrorAction SilentlyContinue

$fileVer = Join-Path $Global:RapportiDir 'VERSIONI_PROGRAMMI.txt'
$testa = @("Programmi preparati il " + (Get-Date -Format 'dd/MM/yyyy HH:mm') + " su " + $env:COMPUTERNAME, "")
($testa + $riepilogo) | Out-File -FilePath $fileVer -Encoding UTF8

Write-Titolo "FINE"
if ($tuttoOk) {
    Write-Host "  Tutto pronto! La chiavetta funziona anche SENZA Internet." -ForegroundColor Green
    Write-Host "  Consiglio: prova una volta GPU-Z, HWiNFO e OCCT sul tuo PC per prendere confidenza." -ForegroundColor Green
} else {
    Write-Host "  Qualcosa manca o non e' verificato: rilancia questo script o scarica a mano (vedi README)." -ForegroundColor Yellow
}
Write-Host "  Elenco versioni: $fileVer"
Wait-Invio
