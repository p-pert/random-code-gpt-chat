# comune.ps1 - funzioni condivise dagli script del kit (Windows PowerShell 5.1+)
# Questo file non fa nulla da solo: viene caricato dagli altri script.

$Global:KitRoot     = Split-Path -Parent $PSScriptRoot
$Global:RapportiDir = Join-Path $Global:KitRoot 'Rapporti'
$Global:ProgrammiDir = Join-Path $Global:KitRoot 'Programmi'
if (-not (Test-Path $Global:RapportiDir)) { New-Item -ItemType Directory -Path $Global:RapportiDir -Force | Out-Null }

$Global:Inv = [System.Globalization.CultureInfo]::InvariantCulture

function Test-Admin {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    $pr = New-Object Security.Principal.WindowsPrincipal($id)
    return $pr.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Write-Titolo([string]$Testo) {
    Write-Host ""
    Write-Host ("=" * 72) -ForegroundColor Cyan
    Write-Host ("  " + $Testo) -ForegroundColor Cyan
    Write-Host ("=" * 72) -ForegroundColor Cyan
}

function Get-ColoreEsito([string]$Esito) {
    switch ($Esito) {
        'OK'         { return 'Green' }
        'ATTENZIONE' { return 'Yellow' }
        'KO'         { return 'Red' }
        default      { return 'Gray' }
    }
}

# Converte una stringa in numero (accetta punto o virgola), restituisce $null se non numerica
function ConvertTo-Num($Valore) {
    if ($null -eq $Valore) { return $null }
    $s = ("$Valore").Trim() -replace ',', '.'
    $d = 0.0
    if ([double]::TryParse($s, [System.Globalization.NumberStyles]::Float, $Global:Inv, [ref]$d)) { return $d }
    return $null
}

# Trova nvidia-smi.exe (installato insieme al driver NVIDIA)
function Find-NvidiaSmi {
    $candidati = @(
        (Join-Path $env:SystemRoot 'System32\nvidia-smi.exe'),
        (Join-Path $env:ProgramFiles 'NVIDIA Corporation\NVSMI\nvidia-smi.exe')
    )
    foreach ($p in $candidati) { if (Test-Path $p) { return $p } }
    $ds = Get-ChildItem -Path (Join-Path $env:SystemRoot 'System32\DriverStore\FileRepository\nv*\nvidia-smi.exe') -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($ds) { return $ds.FullName }
    return $null
}

# True se nvidia-smi accetta il campo richiesto
function Test-NvCampo([string]$Exe, [string]$Campo) {
    try {
        $null = & $Exe "--query-gpu=$Campo" '--format=csv,noheader,nounits' 2>$null
        return ($LASTEXITCODE -eq 0)
    } catch { return $false }
}

# Legge un singolo campo da nvidia-smi; $null se non disponibile
function Get-NvCampo([string]$Exe, [string]$Campo) {
    try {
        $out = & $Exe "--query-gpu=$Campo" '--format=csv,noheader,nounits' 2>$null
        if ($LASTEXITCODE -ne 0 -or $null -eq $out) { return $null }
        $v = ("" + ($out | Select-Object -First 1)).Trim()
        if ($v -eq '' -or $v -match 'N/A' -or $v -match 'Not Supported' -or $v -match 'not a valid') { return $null }
        return $v
    } catch { return $null }
}

function Wait-Invio {
    Write-Host ""
    Read-Host "Premi INVIO per chiudere questa finestra" | Out-Null
}
