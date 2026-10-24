# ACE LOCAL — Memurai user instruction dialog and availability state (dot-sourced).
Set-StrictMode -Version Latest

$script:AceLocalMemuraiScenarioPongOk = 'PongOk'
$script:AceLocalMemuraiScenarioServiceStopped = 'ServiceStopped'
$script:AceLocalMemuraiScenarioServiceRunningNoPong = 'ServiceRunningNoPong'
$script:AceLocalMemuraiScenarioServiceNotInstalled = 'ServiceNotInstalled'
$script:AceLocalMemuraiScenarioAmbiguousCheckError = 'AmbiguousCheckError'

$script:AceLocalMemuraiInstructionScenarios = @(
    $script:AceLocalMemuraiScenarioServiceStopped
    $script:AceLocalMemuraiScenarioServiceRunningNoPong
)

function Get-AceLocalMemuraiLocalDevStateDir {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [switch]$AllowCreate
    )
    $dir = Join-Path $AceRoot '_backups\local-dev'
    if ($AllowCreate -and -not (Test-Path -LiteralPath $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
    return $dir
}

function Get-AceLocalMemuraiAvailabilityPath {
    param([Parameter(Mandatory = $true)][string]$AceRoot)
    return Join-Path (Get-AceLocalMemuraiLocalDevStateDir -AceRoot $AceRoot) 'memurai-availability.json'
}

function Get-AceLocalMemuraiInstructionStatePath {
    param([Parameter(Mandatory = $true)][string]$AceRoot)
    return Join-Path (Get-AceLocalMemuraiLocalDevStateDir -AceRoot $AceRoot) 'memurai-instruction-state.json'
}

function Get-AceLocalMemuraiInstructionDialogBody {
    $utf8 = [System.Text.Encoding]::UTF8
    return $utf8.GetString([Convert]::FromBase64String(
            '16nXmdeo15XXqiBNZW11cmFpINeg16LXpteoLiDXm9eT15kg15zXlNee16nXmdeaINec16LXkdeV15M6CgoxLiDXnNeX16Ug16LXnCBXaW4gKyBSLgoyLiDXlNen15zXkyBzZXJ2aWNlcy5tc2Mg15XXnNeX16UgRW50ZXIuCjMuINeX16TXqSDXkNeqINeU16nXmdeo15XXqiBNZW11cmFpLgo0LiDXnNeX16Ug16LXnNeZ15Ug15HXkNee16bXoteV16og15TXm9ek16rXldeoINeU15nXnteg15kuCjUuINeR15fXqCBTdGFydCAo15DXlSBSZXN0YXJ0INeQ150g15TXqdeZ16jXldeqINeb15HXqCDXpNeV16LXnCDXkNeaINeQ15nXoNeVINee15LXmdeRKS4KNi4g16HXkteV16gg15DXqiDXl9ec15XXnyDXlNep15nXqNeV16rXmdedINeV15TXpNei15wg157Xl9eT16kg15DXqiBMT0NBTCDXnten15nXpteV16gg15TXk9eo15og15HXqdeV15zXl9efINeU16LXkdeV15PXlC4='))
}

function Show-AceLocalMemuraiInstructionDialog {
    param(
        [scriptblock]$ShowDialog = $null
    )
    $body = Get-AceLocalMemuraiInstructionDialogBody
    if ($ShowDialog) {
        & $ShowDialog $body | Out-Null
        return @{ Ok = $true; Method = 'mock' }
    }
    try {
        Add-Type -AssemblyName System.Windows.Forms -ErrorAction Stop
        [void][System.Windows.Forms.MessageBox]::Show(
            $body,
            'MEMURAI',
            [System.Windows.Forms.MessageBoxButtons]::OK,
            [System.Windows.Forms.MessageBoxIcon]::Warning)
        return @{ Ok = $true; Method = 'messagebox' }
    }
    catch {
        return @{ Ok = $false; Method = 'none'; Error = $_.Exception.Message }
    }
}

function Get-AceLocalMemuraiDiagnosis {
    param(
        [Parameter(Mandatory = $true)][string]$RedisUrl,
        [scriptblock]$PingFn = $null,
        [scriptblock]$ServiceFn = $null
    )
    if (-not $PingFn) {
        $PingFn = { param($Url) Test-LocalMemuraiPing -RedisUrl $Url }
    }
    if (-not $ServiceFn) {
        $ServiceFn = { Get-Service -Name 'Memurai' -ErrorAction SilentlyContinue }
    }

    $ping = & $PingFn $RedisUrl
    $pingOk = [bool]$ping.Ok
    $pingErr = if ($ping.Error) { [string]$ping.Error } else { '' }

    if ($pingOk) {
        return @{
            Scenario             = $script:AceLocalMemuraiScenarioPongOk
            PingOk               = $true
            PingError            = ''
            WindowsServiceStatus = ''
            ShowInstruction      = $false
        }
    }

    $ambiguousTokens = @('memurai_cli_not_found', 'redis_url_invalid', 'redis_host_not_loopback')
    foreach ($tok in $ambiguousTokens) {
        if ($pingErr -like "*$tok*") {
            return @{
                Scenario             = $script:AceLocalMemuraiScenarioAmbiguousCheckError
                PingOk               = $false
                PingError            = $pingErr
                WindowsServiceStatus = ''
                ShowInstruction      = $false
            }
        }
    }

    $svc = & $ServiceFn
    if (-not $svc) {
        return @{
            Scenario             = $script:AceLocalMemuraiScenarioServiceNotInstalled
            PingOk               = $false
            PingError            = $pingErr
            WindowsServiceStatus = 'NotInstalled'
            ShowInstruction      = $false
        }
    }

    $status = [string]$svc.Status
    if ($status -eq 'Running') {
        return @{
            Scenario             = $script:AceLocalMemuraiScenarioServiceRunningNoPong
            PingOk               = $false
            PingError            = $pingErr
            WindowsServiceStatus = $status
            ShowInstruction      = $true
        }
    }

    return @{
        Scenario             = $script:AceLocalMemuraiScenarioServiceStopped
        PingOk               = $false
        PingError            = $pingErr
        WindowsServiceStatus = $status
        ShowInstruction      = $true
    }
}

function Read-AceLocalMemuraiInstructionState {
    param([Parameter(Mandatory = $true)][string]$AceRoot)
    $path = Get-AceLocalMemuraiInstructionStatePath -AceRoot $AceRoot
    if (-not (Test-Path -LiteralPath $path)) {
        return @{
            outageId                 = ''
            outageActive             = $false
            instructionAcknowledged  = $false
        }
    }
    try {
        $raw = Get-Content -LiteralPath $path -Raw -Encoding UTF8
        $data = $raw | ConvertFrom-Json
    }
    catch {
        return @{
            outageId                 = ''
            outageActive             = $false
            instructionAcknowledged  = $false
        }
    }
    return @{
        outageId                = [string]$data.outageId
        outageActive            = [bool]$data.outageActive
        instructionAcknowledged = [bool]$data.instructionAcknowledged
    }
}

function Write-AceLocalMemuraiInstructionState {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [Parameter(Mandatory = $true)][string]$OutageId,
        [Parameter(Mandatory = $true)][bool]$OutageActive,
        [Parameter(Mandatory = $true)][bool]$InstructionAcknowledged,
        [string]$Scenario = ''
    )
    $dir = Get-AceLocalMemuraiLocalDevStateDir -AceRoot $AceRoot -AllowCreate
    $path = Join-Path $dir 'memurai-instruction-state.json'
    $payload = [ordered]@{
        schemaVersion           = 1
        outageId                = $OutageId
        outageActive            = $OutageActive
        instructionAcknowledged = $InstructionAcknowledged
        scenario                = $Scenario
        updatedAt               = (Get-Date).ToUniversalTime().ToString('o')
    }
    $json = ($payload | ConvertTo-Json -Compress)
    $tmp = Join-Path $dir ("memurai-instruction-state.json.tmp.{0}" -f [guid]::NewGuid().ToString('n'))
    Set-Content -LiteralPath $tmp -Value $json -Encoding UTF8
    Move-Item -LiteralPath $tmp -Destination $path -Force
}

function Test-ShouldShowAceLocalMemuraiInstruction {
    param(
        [Parameter(Mandatory = $true)][hashtable]$Diagnosis,
        [Parameter(Mandatory = $true)][hashtable]$InstructionState,
        [Parameter(Mandatory = $true)][bool]$IsNotifyDownTransition,
        [switch]$StartupAttempt
    )
    if (-not $Diagnosis.ShowInstruction) {
        return $false
    }
    if ($StartupAttempt) {
        return $true
    }
    if (-not $IsNotifyDownTransition) {
        return $false
    }
    if ($InstructionState.outageActive -and $InstructionState.instructionAcknowledged) {
        return $false
    }
    return $true
}

function Write-AceLocalMemuraiAvailabilityState {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [Parameter(Mandatory = $true)][hashtable]$Diagnosis
    )
    $dir = Get-AceLocalMemuraiLocalDevStateDir -AceRoot $AceRoot -AllowCreate
    $path = Join-Path $dir 'memurai-availability.json'
    $blocks = -not [bool]$Diagnosis.PingOk
    $payload = [ordered]@{
        schemaVersion          = 1
        pingOk                 = [bool]$Diagnosis.PingOk
        scenario               = [string]$Diagnosis.Scenario
        pingError              = [string]$Diagnosis.PingError
        windowsServiceStatus   = [string]$Diagnosis.WindowsServiceStatus
        blocksNewRedisWork     = $blocks
        updatedAt              = (Get-Date).ToUniversalTime().ToString('o')
    }
    $json = ($payload | ConvertTo-Json -Compress)
    $tmp = Join-Path $dir ("memurai-availability.json.tmp.{0}" -f [guid]::NewGuid().ToString('n'))
    Set-Content -LiteralPath $tmp -Value $json -Encoding UTF8
    Move-Item -LiteralPath $tmp -Destination $path -Force
}

function Invoke-AceLocalMemuraiUserInstruction {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [Parameter(Mandatory = $true)][string]$RedisUrl,
        [switch]$StartupAttempt,
        [switch]$IsNotifyDownTransition,
        [scriptblock]$PingFn = $null,
        [scriptblock]$ServiceFn = $null,
        [scriptblock]$ShowDialog = $null
    )
    $diag = Get-AceLocalMemuraiDiagnosis -RedisUrl $RedisUrl -PingFn $PingFn -ServiceFn $ServiceFn
    Write-AceLocalMemuraiAvailabilityState -AceRoot $AceRoot -Diagnosis $diag | Out-Null

    $state = Read-AceLocalMemuraiInstructionState -AceRoot $AceRoot
    $shouldShow = Test-ShouldShowAceLocalMemuraiInstruction `
        -Diagnosis $diag `
        -InstructionState $state `
        -IsNotifyDownTransition:$IsNotifyDownTransition `
        -StartupAttempt:$StartupAttempt

    if (-not $shouldShow) {
        return @{
            Shown           = $false
            Diagnosis       = $diag
            DialogResult    = $null
        }
    }

    $outageId = $state.outageId
    if ($IsNotifyDownTransition -or $StartupAttempt -or -not $state.outageActive) {
        $outageId = [guid]::NewGuid().ToString('n')
    }
    Write-AceLocalMemuraiInstructionState -AceRoot $AceRoot `
        -OutageId $outageId -OutageActive $true -InstructionAcknowledged $false `
        -Scenario ([string]$diag.Scenario) | Out-Null

    $dialog = Show-AceLocalMemuraiInstructionDialog -ShowDialog $ShowDialog

    Write-AceLocalMemuraiInstructionState -AceRoot $AceRoot `
        -OutageId $outageId -OutageActive $true -InstructionAcknowledged $true `
        -Scenario ([string]$diag.Scenario) | Out-Null

    return @{
        Shown        = $dialog.Ok
        Diagnosis    = $diag
        DialogResult = $dialog
    }
}

function Clear-AceLocalMemuraiOutageInstructionState {
    param([Parameter(Mandatory = $true)][string]$AceRoot)
    Write-AceLocalMemuraiInstructionState -AceRoot $AceRoot `
        -OutageId 'cleared' -OutageActive $false -InstructionAcknowledged $false `
        -Scenario '' | Out-Null
}
