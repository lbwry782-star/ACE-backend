# ACE LOCAL — managed process registry, logs, and safe stop helpers (Windows).
Set-StrictMode -Version Latest

$script:AceLocalManagedRegistrySchemaVersion = 2
$script:ManagedComponentInactiveReasons = @(
    'tree_not_running',
    'empty_tree',
    'missing_root_pid',
    'root_not_in_tree',
    'not_started_by_stack'
)
$script:ManagedProcessCreationTimeToleranceSeconds = 2

function Get-AceLocalDevStateDir {
    param([Parameter(Mandatory = $true)][string]$AceRoot)
    $dir = Join-Path $AceRoot '_backups\local-dev'
    if (-not (Test-Path -LiteralPath $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
    return $dir
}

function Get-AceLocalLogDir {
    param([Parameter(Mandatory = $true)][string]$AceRoot)
    $dir = Join-Path (Get-AceLocalDevStateDir -AceRoot $AceRoot) 'logs'
    if (-not (Test-Path -LiteralPath $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
    return $dir
}

function Get-AceLocalManagedRegistryPath {
    param([Parameter(Mandatory = $true)][string]$AceRoot)
    return Join-Path (Get-AceLocalDevStateDir -AceRoot $AceRoot) 'managed-processes.json'
}

function Read-AceLocalManagedRegistry {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [switch]$AllowCreateStateDir
    )
    if ($AllowCreateStateDir) {
        $path = Get-AceLocalManagedRegistryPath -AceRoot $AceRoot
    }
    else {
        $path = Join-Path $AceRoot '_backups\local-dev\managed-processes.json'
    }
    if (-not (Test-Path -LiteralPath $path)) {
        return $null
    }
    try {
        $raw = Get-Content -LiteralPath $path -Raw -Encoding UTF8
        if ([string]::IsNullOrWhiteSpace($raw)) {
            return $null
        }
        return ($raw | ConvertFrom-Json)
    }
    catch {
        return $null
    }
}

function Write-AceLocalManagedRegistry {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [Parameter(Mandatory = $true)]$RegistryObject
    )
    $path = Get-AceLocalManagedRegistryPath -AceRoot $AceRoot
    $json = $RegistryObject | ConvertTo-Json -Depth 6
    $tmp = Join-Path (Split-Path -Parent $path) ("managed-processes.json.tmp.{0}" -f [guid]::NewGuid().ToString('N'))
    Set-Content -LiteralPath $tmp -Value $json -Encoding UTF8
    Move-Item -LiteralPath $tmp -Destination $path -Force
}

function New-AceLocalManagedRegistryShell {
    param(
        [Parameter(Mandatory = $true)][string]$BackendRoot,
        [Parameter(Mandatory = $true)][string]$FrontendRoot,
        [string]$LauncherSessionId = ""
    )
    if (-not $LauncherSessionId) {
        $LauncherSessionId = (Get-Date).ToString('yyyyMMdd-HHmmss')
    }
    return [PSCustomObject]@{
        schemaVersion   = $script:AceLocalManagedRegistrySchemaVersion
        backendRoot     = (Resolve-Path -LiteralPath $BackendRoot).ProviderPath
        frontendRoot    = (Resolve-Path -LiteralPath $FrontendRoot).ProviderPath
        launcherSession = $LauncherSessionId
        updatedAt       = (Get-Date).ToUniversalTime().ToString('o')
        components      = [PSCustomObject]@{
            backend  = $null
            worker   = $null
            frontend = $null
        }
    }
}

function New-AceLocalManagedComponentRecord {
    param(
        [Parameter(Mandatory = $true)][int]$ProcId,
        [Parameter(Mandatory = $true)][string]$LogPath,
        [string]$StartedAt = ""
    )
    if (-not $StartedAt) {
        $StartedAt = (Get-Date).ToUniversalTime().ToString('o')
    }
    return [PSCustomObject]@{
        pid            = $ProcId
        rootPid        = $ProcId
        childPids      = @()
        listenerPid    = $null
        treePids       = @($ProcId)
        wrapperPid     = $null
        launcherSession = ''
        startedByStack = $true
        logPath        = $LogPath
        startedAt      = $StartedAt
    }
}

function Get-Win32ProcessCreationTimeUtc {
    param([Parameter(Mandatory = $true)][int]$ProcId)
    if ($ProcId -le 0) {
        return $null
    }
    $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$ProcId" -ErrorAction SilentlyContinue
    if ($proc -and -not [string]::IsNullOrWhiteSpace([string]$proc.CreationDate)) {
        try {
            $local = [Management.ManagementDateTimeConverter]::ToDateTime($proc.CreationDate)
            return $local.ToUniversalTime().ToString('o')
        }
        catch {
            # fall through to Get-Process StartTime
        }
    }
    $fallback = Get-Process -Id $ProcId -ErrorAction SilentlyContinue
    if ($fallback -and $fallback.StartTime) {
        try {
            return $fallback.StartTime.ToUniversalTime().ToString('o')
        }
        catch {
            return $null
        }
    }
    return $null
}

function ConvertTo-ManagedProcessCreatedAtObject {
    param([hashtable]$Map)
    if (-not $Map -or $Map.Count -eq 0) {
        return $null
    }
    $ordered = [ordered]@{}
    foreach ($key in ($Map.Keys | Sort-Object)) {
        $ordered[[string]$key] = [string]$Map[$key]
    }
    return [PSCustomObject]$ordered
}

function Get-ManagedTreeProcessCreationTimes {
    param(
        [Parameter(Mandatory = $true)][int[]]$TreePids,
        [int]$WrapperPid = 0
    )
    $map = @{}
    foreach ($pidVal in @($TreePids | Sort-Object -Unique)) {
        $pidInt = [int]$pidVal
        if ($pidInt -le 0) { continue }
        $created = Get-Win32ProcessCreationTimeUtc -ProcId $pidInt
        if ($created) {
            $map[[string]$pidInt] = $created
        }
    }
    if ($WrapperPid -gt 0) {
        $created = Get-Win32ProcessCreationTimeUtc -ProcId $WrapperPid
        if ($created) {
            $map[[string]$WrapperPid] = $created
        }
    }
    return $map
}

function Test-IsoUtcTimesMatchWithinTolerance {
    param(
        [Parameter(Mandatory = $true)][string]$ExpectedIso,
        [Parameter(Mandatory = $true)][string]$ActualIso,
        [int]$ToleranceSeconds = 2
    )
    if ([string]::IsNullOrWhiteSpace($ExpectedIso) -or [string]::IsNullOrWhiteSpace($ActualIso)) {
        return $false
    }
    try {
        $expected = [DateTime]::Parse($ExpectedIso, $null, [System.Globalization.DateTimeStyles]::RoundtripKind).ToUniversalTime()
        $actual = [DateTime]::Parse($ActualIso, $null, [System.Globalization.DateTimeStyles]::RoundtripKind).ToUniversalTime()
        $delta = [Math]::Abs(($actual - $expected).TotalSeconds)
        return ($delta -le $ToleranceSeconds)
    }
    catch {
        return $false
    }
}

function Get-ManagedRecordProcessCreatedAtMap {
    param([Parameter(Mandatory = $true)]$Record)
    $map = @{}
    if (-not $Record -or -not $Record.processCreatedAt) {
        return $map
    }
    $Record.processCreatedAt.PSObject.Properties | ForEach-Object {
        if ($_.Value) {
            $map[[string]$_.Name] = [string]$_.Value
        }
    }
    return $map
}

function Test-ManagedProcessCreationTimesValid {
    param(
        [Parameter(Mandatory = $true)]$Record,
        [Parameter(Mandatory = $true)][int[]]$TreePids,
        [int]$WrapperPid = 0
    )
    $stored = Get-ManagedRecordProcessCreatedAtMap -Record $Record
    if ($stored.Count -eq 0) {
        return [PSCustomObject]@{ Ok = $false; Reason = 'missing_process_creation_times' }
    }
    $verifyPids = @($TreePids | ForEach-Object { [int]$_ } | Where-Object { $_ -gt 0 } | Sort-Object -Unique)
    if ($WrapperPid -gt 0 -and ($verifyPids -notcontains $WrapperPid)) {
        $verifyPids += [int]$WrapperPid
    }
    foreach ($pidVal in $verifyPids) {
        if (-not (Test-ProcessAlive $pidVal)) {
            continue
        }
        $key = [string]$pidVal
        if (-not $stored.ContainsKey($key)) {
            return [PSCustomObject]@{ Ok = $false; Reason = ("process_creation_time_missing_pid_{0}" -f $pidVal) }
        }
        $live = Get-Win32ProcessCreationTimeUtc -ProcId $pidVal
        if (-not $live) {
            return [PSCustomObject]@{ Ok = $false; Reason = ("process_creation_time_unverifiable_pid_{0}" -f $pidVal) }
        }
        if (-not (Test-IsoUtcTimesMatchWithinTolerance -ExpectedIso $stored[$key] -ActualIso $live `
                    -ToleranceSeconds $script:ManagedProcessCreationTimeToleranceSeconds)) {
            return [PSCustomObject]@{ Ok = $false; Reason = ("process_creation_time_mismatch_pid_{0}" -f $pidVal) }
        }
    }
    return [PSCustomObject]@{ Ok = $true; Reason = 'ok' }
}

function New-AceLocalManagedComponentTreeRecord {
    param(
        [Parameter(Mandatory = $true)][int]$RootPid,
        [Parameter(Mandatory = $true)][int[]]$TreePids,
        [Parameter(Mandatory = $true)][string]$LogPath,
        [Parameter(Mandatory = $true)][string]$LauncherSession,
        [int]$WrapperPid = 0,
        [int]$ListenerPid = 0,
        [string]$StartedAt = "",
        $ProcessCreatedAt = $null
    )
    if (-not $StartedAt) {
        $StartedAt = (Get-Date).ToUniversalTime().ToString('o')
    }
    $tree = @($TreePids | Sort-Object -Unique)
    $root = [int]$RootPid
    $children = @($tree | Where-Object { $_ -ne $root })
    $listener = if ($ListenerPid -gt 0) { [int]$ListenerPid } else { $null }
    $wrapper = if ($WrapperPid -gt 0) { [int]$WrapperPid } else { $null }
    $createdObj = $null
    if ($ProcessCreatedAt -is [hashtable] -and $ProcessCreatedAt.Count -gt 0) {
        $createdObj = ConvertTo-ManagedProcessCreatedAtObject -Map $ProcessCreatedAt
    }
    elseif ($ProcessCreatedAt) {
        $createdObj = $ProcessCreatedAt
    }
    return [PSCustomObject]@{
        pid               = $root
        rootPid           = $root
        childPids         = @($children)
        listenerPid       = $listener
        treePids          = $tree
        wrapperPid        = $wrapper
        launcherSession   = $LauncherSession
        startedByStack    = $true
        logPath           = $LogPath
        startedAt         = $StartedAt
        processCreatedAt  = $createdObj
    }
}

function Test-ProcessDescendsFromAncestor {
    param(
        [Parameter(Mandatory = $true)][int]$ProcId,
        [Parameter(Mandatory = $true)][int]$AncestorPid,
        [int]$MaxDepth = 24
    )
    if ($ProcId -le 0 -or $AncestorPid -le 0) {
        return $false
    }
    $current = $ProcId
    for ($i = 0; $i -lt $MaxDepth; $i++) {
        if ($current -eq $AncestorPid) {
            return $true
        }
        $parent = Get-Win32ParentProcessId -ProcessId $current
        if ($parent -le 0) {
            return $false
        }
        $current = $parent
    }
    return $false
}

function Get-AceIndependentTreeRootIds {
    param(
        [AllowEmptyCollection()]
        [array]$Processes = @(),
        [hashtable]$ParentProcessIdByPid = $null
    )
    if ($null -eq $Processes -or $Processes.Count -eq 0) {
        return @()
    }
    $pidSet = @{}
    foreach ($p in $Processes) {
        if ($null -eq $p) { continue }
        $pidSet[[int]$p.ProcId] = $true
    }
    if ($pidSet.Count -eq 0) {
        return @()
    }
    function Get-ResolvedParent([int]$ProcId) {
        if ($ParentProcessIdByPid -and $ParentProcessIdByPid.ContainsKey($ProcId)) {
            return [int]$ParentProcessIdByPid[$ProcId]
        }
        return Get-Win32ParentProcessId -ProcessId $ProcId
    }
    $roots = @{}
    foreach ($p in $Processes) {
        if ($null -eq $p) { continue }
        $current = [int]$p.ProcId
        if (-not $pidSet.ContainsKey($current)) { continue }
        while ($true) {
            $parent = Get-ResolvedParent -ProcId $current
            if ($parent -gt 0 -and $pidSet.ContainsKey($parent)) {
                $current = $parent
                continue
            }
            break
        }
        $roots[[int]$current] = $true
    }
    return @($roots.Keys | Sort-Object)
}

function Get-AceProcessTreeMemberIds {
    param(
        [Parameter(Mandatory = $true)][int]$RootPid,
        [Parameter(Mandatory = $true)][array]$Processes,
        [hashtable]$ParentProcessIdByPid = $null
    )
    $root = [int]$RootPid
    $members = @($Processes | Where-Object {
            $pidVal = [int]$_.ProcId
            $current = $pidVal
            for ($i = 0; $i -lt 24; $i++) {
                if ($current -eq $root) { return $true }
                if ($ParentProcessIdByPid -and $ParentProcessIdByPid.ContainsKey($current)) {
                    $parent = [int]$ParentProcessIdByPid[$current]
                }
                else {
                    $parent = Get-Win32ParentProcessId -ProcessId $current
                }
                if ($parent -le 0) { break }
                $current = $parent
            }
            return $false
        } | ForEach-Object { [int]$_.ProcId })
    return @($members | Sort-Object -Unique)
}

function Find-NewAceIndependentTreeRoot {
    param(
        [AllowEmptyCollection()][array]$BeforeProcesses = @(),
        [AllowEmptyCollection()][array]$AfterProcesses = @(),
        [hashtable]$ParentProcessIdByPid = $null
    )
    $beforeRoots = @(Get-AceIndependentTreeRootIds -Processes $BeforeProcesses -ParentProcessIdByPid $ParentProcessIdByPid)
    $afterRoots = @(Get-AceIndependentTreeRootIds -Processes $AfterProcesses -ParentProcessIdByPid $ParentProcessIdByPid)
    $newRoots = @($afterRoots | Where-Object { $beforeRoots -notcontains $_ })
    if ($newRoots.Count -ne 1) {
        return $null
    }
    return [int]$newRoots[0]
}

function Get-AceManagedProcessesFromWrapper {
    param(
        [Parameter(Mandatory = $true)][int]$WrapperPid,
        [Parameter(Mandatory = $true)][array]$CandidateProcesses
    )
    if ($WrapperPid -le 0) {
        return @()
    }
    return @($CandidateProcesses | Where-Object {
            Test-ProcessDescendsFromAncestor -ProcId ([int]$_.ProcId) -AncestorPid $WrapperPid
        })
}

function Discover-AceLocalManagedBackendTree {
    param(
        [int]$WrapperPid = 0,
        [AllowEmptyCollection()][array]$BeforeProcesses = @(),
        [Parameter(Mandatory = $true)][string]$BackendRoot,
        [int]$ListenerPort = 5000
    )
    $after = @(Find-LocalAceBackendProcesses -BackendRoot $BackendRoot -ListenerPort $ListenerPort)
    $scoped = @(Get-AceManagedProcessesFromWrapper -WrapperPid $WrapperPid -CandidateProcesses $after)
    $pool = if ($scoped.Count -gt 0) { $scoped } else { $after }
    $root = $null
    if ($scoped.Count -gt 0) {
        $roots = @(Get-AceIndependentTreeRootIds -Processes $scoped)
        if ($roots.Count -eq 1) {
            $root = [int]$roots[0]
        }
    }
    if ($null -eq $root) {
        $root = Find-NewAceIndependentTreeRoot -BeforeProcesses $BeforeProcesses -AfterProcesses $after
    }
    if ($null -eq $root) {
        return $null
    }
    $treePids = @(Get-AceProcessTreeMemberIds -RootPid $root -Processes $pool)
    if ($treePids.Count -eq 0) {
        $treePids = @(Get-AceProcessTreeMemberIds -RootPid $root -Processes $after)
    }
    if ($treePids.Count -eq 0) {
        return $null
    }
    $listeners = @(Get-AceLoopbackListenerProcessIds -Port $ListenerPort)
    $listenerPid = 0
    foreach ($lp in $listeners) {
        if ($treePids -contains [int]$lp) {
            $listenerPid = [int]$lp
            break
        }
    }
    return [PSCustomObject]@{
        RootPid     = [int]$root
        TreePids    = $treePids
        ListenerPid = $listenerPid
        WrapperPid  = [int]$WrapperPid
    }
}

function Discover-AceLocalManagedWorkerTree {
    param(
        [int]$WrapperPid = 0,
        [AllowEmptyCollection()][array]$BeforeProcesses = @(),
        [Parameter(Mandatory = $true)][string]$BackendRoot
    )
    $after = @(Find-LocalAceWorkerProcesses -BackendRoot $BackendRoot)
    $scoped = @(Get-AceManagedProcessesFromWrapper -WrapperPid $WrapperPid -CandidateProcesses $after)
    $pool = if ($scoped.Count -gt 0) { $scoped } else { $after }
    $root = $null
    if ($scoped.Count -gt 0) {
        $roots = @(Get-AceIndependentTreeRootIds -Processes $scoped)
        if ($roots.Count -eq 1) {
            $root = [int]$roots[0]
        }
    }
    if ($null -eq $root) {
        $root = Find-NewAceIndependentTreeRoot -BeforeProcesses $BeforeProcesses -AfterProcesses $after
    }
    if ($null -eq $root) {
        return $null
    }
    $treePids = @(Get-AceProcessTreeMemberIds -RootPid $root -Processes $pool)
    if ($treePids.Count -eq 0) {
        $treePids = @(Get-AceProcessTreeMemberIds -RootPid $root -Processes $after)
    }
    if ($treePids.Count -eq 0) {
        return $null
    }
    return [PSCustomObject]@{
        RootPid     = [int]$root
        TreePids    = $treePids
        ListenerPid = 0
        WrapperPid  = [int]$WrapperPid
    }
}

function Discover-AceLocalManagedFrontendTree {
    param(
        [int]$WrapperPid = 0,
        [AllowEmptyCollection()][array]$BeforeProcesses = @(),
        [Parameter(Mandatory = $true)][string]$FrontendRoot
    )
    $after = @(Find-LocalAceFrontendViteProcesses -FrontendRoot $FrontendRoot)
    $scoped = @(Get-AceManagedProcessesFromWrapper -WrapperPid $WrapperPid -CandidateProcesses $after)
    $pool = if ($scoped.Count -gt 0) { $scoped } else { $after }
    $root = $null
    if ($scoped.Count -gt 0) {
        $roots = @(Get-AceIndependentTreeRootIds -Processes $scoped)
        if ($roots.Count -eq 1) {
            $root = [int]$roots[0]
        }
    }
    if ($null -eq $root) {
        $root = Find-NewAceIndependentTreeRoot -BeforeProcesses $BeforeProcesses -AfterProcesses $after
    }
    if ($null -eq $root) {
        return $null
    }
    $treePids = @(Get-AceProcessTreeMemberIds -RootPid $root -Processes $pool)
    if ($treePids.Count -eq 0) {
        $treePids = @(Get-AceProcessTreeMemberIds -RootPid $root -Processes $after)
    }
    if ($treePids.Count -eq 0) {
        return $null
    }
    return [PSCustomObject]@{
        RootPid     = [int]$root
        TreePids    = $treePids
        ListenerPid = 0
        WrapperPid  = [int]$WrapperPid
    }
}

function Get-ManagedComponentTreePids {
    param([Parameter(Mandatory = $true)]$Record)
    if (-not $Record) {
        return @()
    }
    if ($Record.treePids -and @($Record.treePids).Count -gt 0) {
        return @($Record.treePids | ForEach-Object { [int]$_ } | Sort-Object -Unique)
    }
    if ($Record.rootPid) {
        $ids = @([int]$Record.rootPid)
        if ($Record.childPids) {
            $ids += @($Record.childPids | ForEach-Object { [int]$_ })
        }
        elseif ($Record.pid) {
            $ids += [int]$Record.pid
        }
        return @($ids | Sort-Object -Unique)
    }
    if ($Record.pid) {
        return @([int]$Record.pid)
    }
    return @()
}

function Test-ManagedWrapperPid {
    param(
        [Parameter(Mandatory = $true)][int]$ProcId,
        [Parameter(Mandatory = $true)][ValidateSet('backend', 'worker', 'frontend')][string]$Component
    )
    if (-not (Test-ProcessAlive $ProcId)) {
        return $false
    }
    $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$ProcId" -ErrorAction SilentlyContinue
    if (-not $proc) {
        return $false
    }
    $cmd = [string]$proc.CommandLine
    if ($cmd -notmatch 'powershell\.exe' -and $cmd -notmatch 'powershell') {
        return $false
    }
    switch ($Component) {
        'backend' { return ($cmd -match 'Start-AceBackendLocal\.ps1') }
        'worker' { return ($cmd -match 'Start-AceWorkerLocal\.ps1') }
        'frontend' { return ($cmd -match 'Start-AceFrontendLocal\.ps1') }
    }
    return $false
}

function Test-ManagedComponentRecordValid {
    param(
        [Parameter(Mandatory = $true)]$Record,
        [Parameter(Mandatory = $true)][ValidateSet('backend', 'worker', 'frontend')][string]$Component,
        [Parameter(Mandatory = $true)][string]$BackendRoot,
        [Parameter(Mandatory = $true)][string]$FrontendRoot,
        [string]$ExpectedLauncherSession = '',
        [int]$BackendPort = 5000,
        [switch]$RequireProcessCreationTimes
    )
    if (-not $Record -or -not $Record.startedByStack) {
        return [PSCustomObject]@{ Ok = $false; Reason = 'not_started_by_stack' }
    }
    if ($ExpectedLauncherSession -and $Record.launcherSession -and ([string]$Record.launcherSession -ne $ExpectedLauncherSession)) {
        return [PSCustomObject]@{ Ok = $false; Reason = 'launcher_session_mismatch' }
    }
    $rootPid = if ($Record.rootPid) { [int]$Record.rootPid } else { [int]$Record.pid }
    if ($rootPid -le 0) {
        return [PSCustomObject]@{ Ok = $false; Reason = 'missing_root_pid' }
    }
    $treePids = @(Get-ManagedComponentTreePids -Record $Record)
    if ($treePids.Count -eq 0) {
        return [PSCustomObject]@{ Ok = $false; Reason = 'empty_tree' }
    }
    if ($treePids -notcontains $rootPid) {
        return [PSCustomObject]@{ Ok = $false; Reason = 'root_not_in_tree' }
    }
    $alive = @($treePids | Where-Object { Test-ProcessAlive $_ })
    if ($alive.Count -eq 0) {
        return [PSCustomObject]@{ Ok = $false; Reason = 'tree_not_running' }
    }
    $rootOk = $false
    switch ($Component) {
        'backend' { $rootOk = Test-ManagedBackendPid -ProcId $rootPid -BackendRoot $BackendRoot }
        'worker' { $rootOk = Test-ManagedWorkerPid -ProcId $rootPid -BackendRoot $BackendRoot }
        'frontend' { $rootOk = Test-ManagedFrontendPid -ProcId $rootPid -FrontendRoot $FrontendRoot }
    }
    if (-not $rootOk) {
        return [PSCustomObject]@{ Ok = $false; Reason = 'root_fingerprint_mismatch' }
    }
    foreach ($pidVal in $treePids) {
        if (-not (Test-ProcessAlive $pidVal)) {
            continue
        }
        $match = $false
        switch ($Component) {
            'backend' { $match = Test-ManagedBackendPid -ProcId $pidVal -BackendRoot $BackendRoot }
            'worker' { $match = Test-ManagedWorkerPid -ProcId $pidVal -BackendRoot $BackendRoot }
            'frontend' { $match = Test-ManagedFrontendPid -ProcId $pidVal -FrontendRoot $FrontendRoot }
        }
        if (-not $match) {
            return [PSCustomObject]@{ Ok = $false; Reason = ("tree_pid_{0}_fingerprint_mismatch" -f $pidVal) }
        }
    }
    if ($Component -eq 'backend' -and $Record.listenerPid) {
        $lp = [int]$Record.listenerPid
        if (Test-ProcessAlive $lp) {
            $listeners = @(Get-AceLoopbackListenerProcessIds -Port $BackendPort)
            if ($listeners -notcontains $lp) {
                return [PSCustomObject]@{ Ok = $false; Reason = 'listener_not_on_port' }
            }
        }
    }
    $wrapperPid = if ($Record.wrapperPid) { [int]$Record.wrapperPid } else { 0 }
    if ($RequireProcessCreationTimes) {
        $creationCheck = Test-ManagedProcessCreationTimesValid -Record $Record -TreePids $treePids -WrapperPid $wrapperPid
        if (-not $creationCheck.Ok) {
            return [PSCustomObject]@{
                Ok         = $false
                Reason     = $creationCheck.Reason
                RootPid    = $rootPid
                TreePids   = $treePids
                WrapperPid = $wrapperPid
            }
        }
    }
    return [PSCustomObject]@{
        Ok       = $true
        Reason   = 'ok'
        RootPid  = $rootPid
        TreePids = $treePids
        WrapperPid = $wrapperPid
    }
}

function Confirm-ManagedComponentStopAllowed {
    param(
        [Parameter(Mandatory = $true)]$Record,
        [Parameter(Mandatory = $true)][ValidateSet('backend', 'worker', 'frontend')][string]$Component,
        [Parameter(Mandatory = $true)][string]$BackendRoot,
        [Parameter(Mandatory = $true)][string]$FrontendRoot,
        [Parameter(Mandatory = $true)][string]$ExpectedLauncherSession,
        [int]$BackendPort = 5000
    )
    return Test-ManagedComponentRecordValid -Record $Record -Component $Component `
        -BackendRoot $BackendRoot -FrontendRoot $FrontendRoot `
        -ExpectedLauncherSession $ExpectedLauncherSession -BackendPort $BackendPort `
        -RequireProcessCreationTimes
}

function Test-ProcessAlive([int]$ProcId) {
    if ($ProcId -le 0) {
        return $false
    }
    return $null -ne (Get-Process -Id $ProcId -ErrorAction SilentlyContinue)
}

function Find-LocalAceFrontendViteProcesses {
    param([Parameter(Mandatory = $true)][string]$FrontendRoot)
    $rootNorm = (Resolve-Path -LiteralPath $FrontendRoot).ProviderPath
    $found = @()
    Get-CimInstance Win32_Process -Filter "Name='node.exe'" -ErrorAction SilentlyContinue | ForEach-Object {
        $cmd = [string]$_.CommandLine
        if (-not $cmd) {
            return
        }
        if ($cmd -notmatch 'vite') {
            return
        }
        if ($cmd -notmatch [regex]::Escape($rootNorm)) {
            return
        }
        $found += [PSCustomObject]@{
            ProcId      = $_.ProcessId
            CommandLine = $cmd
        }
    }
    return @($found)
}

function Test-ManagedBackendPid {
    param(
        [Parameter(Mandatory = $true)][int]$ProcId,
        [Parameter(Mandatory = $true)][string]$BackendRoot
    )
    if (-not (Test-ProcessAlive $ProcId)) {
        return $false
    }
    $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$ProcId" -ErrorAction SilentlyContinue
    if (-not $proc) {
        return $false
    }
    return (Test-Win32ProcessIsAceBackendApp -Win32Process $proc -BackendRoot $BackendRoot -ListenerPort 0)
}

function Test-ManagedWorkerPid {
    param(
        [Parameter(Mandatory = $true)][int]$ProcId,
        [Parameter(Mandatory = $true)][string]$BackendRoot
    )
    if (-not (Test-ProcessAlive $ProcId)) {
        return $false
    }
    $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$ProcId" -ErrorAction SilentlyContinue
    if (-not $proc) {
        return $false
    }
    $cmd = [string]$proc.CommandLine
    if (-not ($cmd -match 'worker_video\.py')) {
        return $false
    }
    $rootNorm = (Resolve-Path -LiteralPath $BackendRoot).ProviderPath
    return ($cmd -match [regex]::Escape($rootNorm))
}

function Test-ManagedFrontendPid {
    param(
        [Parameter(Mandatory = $true)][int]$ProcId,
        [Parameter(Mandatory = $true)][string]$FrontendRoot
    )
    if (-not (Test-ProcessAlive $ProcId)) {
        return $false
    }
    $matches = @(Find-LocalAceFrontendViteProcesses -FrontendRoot $FrontendRoot | Where-Object { $_.ProcId -eq $ProcId })
    return ($matches.Count -eq 1)
}

function Get-NewProcessIds {
    param(
        [AllowEmptyCollection()]
        [Parameter(Mandatory = $true)]
        [int[]]$Before,
        [AllowEmptyCollection()]
        [Parameter(Mandatory = $true)]
        [int[]]$After
    )
    $beforeIds = @($Before | ForEach-Object { [int]$_ })
    $afterIds = @($After | ForEach-Object { [int]$_ })
    if ($afterIds.Count -eq 0) {
        return @()
    }
    if ($beforeIds.Count -eq 0) {
        return @($afterIds)
    }
    return @($afterIds | Where-Object { $beforeIds -notcontains $_ })
}

function Resolve-SingleNewProcessId {
    param(
        [AllowEmptyCollection()]
        [Parameter(Mandatory = $true)]
        [int[]]$Before,
        [AllowEmptyCollection()]
        [Parameter(Mandatory = $true)]
        [int[]]$After
    )
    $beforeIds = @($Before | ForEach-Object { [int]$_ })
    $afterIds = @($After | ForEach-Object { [int]$_ })
    if ($afterIds.Count -eq 0) {
        return $null
    }
    if ($beforeIds.Count -eq 0) {
        if ($afterIds.Count -eq 1) {
            return [int]$afterIds[0]
        }
        return $null
    }
    $newIds = @(Get-NewProcessIds -Before $beforeIds -After $afterIds)
    if ($newIds.Count -eq 1) {
        return [int]$newIds[0]
    }
    return $null
}

function Get-StaleManagedComponentsPreview {
    param(
        [Parameter(Mandatory = $true)]$Registry,
        [Parameter(Mandatory = $true)][string]$BackendRoot,
        [Parameter(Mandatory = $true)][string]$FrontendRoot,
        [int]$BackendPort = 5000
    )
    $preview = @()
    $session = [string]$Registry.launcherSession
    foreach ($name in @('backend', 'worker', 'frontend')) {
        $rec = $Registry.components.$name
        if (-not $rec) {
            continue
        }
        $check = Test-ManagedComponentRecordValid -Record $rec -Component $name `
            -BackendRoot $BackendRoot -FrontendRoot $FrontendRoot `
            -ExpectedLauncherSession $session -BackendPort $BackendPort
        if (-not $check.Ok -and ($script:ManagedComponentInactiveReasons -contains $check.Reason)) {
            $preview += [PSCustomObject]@{
                Component = $name
                Reason    = $check.Reason
            }
        }
    }
    return @($preview)
}

function Copy-AceLocalManagedRegistryObject {
    param([Parameter(Mandatory = $true)]$Registry)
    return ($Registry | ConvertTo-Json -Depth 12 | ConvertFrom-Json)
}

function Get-AceLocalManagedStopWhatIfReport {
    param(
        [Parameter(Mandatory = $true)]$Registry,
        [Parameter(Mandatory = $true)][string]$BackendRoot,
        [Parameter(Mandatory = $true)][string]$FrontendRoot,
        [int]$BackendPort = 5000
    )
    $stalePreview = @(Get-StaleManagedComponentsPreview -Registry $Registry `
            -BackendRoot $BackendRoot -FrontendRoot $FrontendRoot -BackendPort $BackendPort)
    $simRegistry = Copy-AceLocalManagedRegistryObject -Registry $Registry
    foreach ($item in $stalePreview) {
        $simRegistry.components.($item.Component) = $null
    }
    $actions = @(Get-ManagedComponentStopActions -Registry $simRegistry `
            -BackendRoot $BackendRoot -FrontendRoot $FrontendRoot -BackendPort $BackendPort)
    return [PSCustomObject]@{
        StaleComponents = $stalePreview
        StopActions     = $actions
        LauncherSession = [string]$Registry.launcherSession
    }
}

function Remove-StaleManagedComponents {
    param(
        [Parameter(Mandatory = $true)]$Registry,
        [Parameter(Mandatory = $true)][string]$BackendRoot,
        [Parameter(Mandatory = $true)][string]$FrontendRoot,
        [int]$BackendPort = 5000
    )
    $changed = $false
    $preview = @(Get-StaleManagedComponentsPreview -Registry $Registry `
            -BackendRoot $BackendRoot -FrontendRoot $FrontendRoot -BackendPort $BackendPort)
    foreach ($item in $preview) {
        $Registry.components.($item.Component) = $null
        $changed = $true
    }
    if ($changed) {
        $Registry.updatedAt = (Get-Date).ToUniversalTime().ToString('o')
    }
    return $changed
}

function Register-ManagedComponentTree {
    param(
        [Parameter(Mandatory = $true)]$Registry,
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [Parameter(Mandatory = $true)][ValidateSet('backend', 'worker', 'frontend')][string]$Name,
        [Parameter(Mandatory = $true)]$Discovery,
        [Parameter(Mandatory = $true)][string]$LogPath
    )
    if (-not $Discovery) {
        return $false
    }
    $wrapperPid = [int]$Discovery.WrapperPid
    $treePids = @($Discovery.TreePids)
    $creationMap = Get-ManagedTreeProcessCreationTimes -TreePids $treePids -WrapperPid $wrapperPid
    $Registry.components.$Name = New-AceLocalManagedComponentTreeRecord `
        -RootPid $Discovery.RootPid `
        -TreePids $treePids `
        -LogPath $LogPath `
        -LauncherSession ([string]$Registry.launcherSession) `
        -WrapperPid $wrapperPid `
        -ListenerPid ([int]$Discovery.ListenerPid) `
        -ProcessCreatedAt $creationMap
    $Registry.updatedAt = (Get-Date).ToUniversalTime().ToString('o')
    Write-AceLocalManagedRegistry -AceRoot $AceRoot -RegistryObject $Registry
    return $true
}

function Get-ManagedComponentStopActions {
    param(
        [Parameter(Mandatory = $true)]$Registry,
        [Parameter(Mandatory = $true)][string]$BackendRoot,
        [Parameter(Mandatory = $true)][string]$FrontendRoot,
        [int]$BackendPort = 5000
    )
    $actions = @()
    $session = [string]$Registry.launcherSession
    $order = @('frontend', 'worker', 'backend')
    foreach ($name in $order) {
        $rec = $Registry.components.$name
        if (-not $rec -or -not $rec.startedByStack) {
            continue
        }
        $check = Test-ManagedComponentRecordValid -Record $rec -Component $name `
            -BackendRoot $BackendRoot -FrontendRoot $FrontendRoot `
            -ExpectedLauncherSession $session -BackendPort $BackendPort `
            -RequireProcessCreationTimes
        if (-not $check.Ok) {
            $actions += [PSCustomObject]@{
                Component = $name
                Action    = 'skip'
                Reason    = $check.Reason
                Pids      = @()
            }
            continue
        }
        $stopPids = @([int]$check.RootPid)
        $wrapper = [int]$check.WrapperPid
        if ($wrapper -gt 0 -and (Test-ManagedWrapperPid -ProcId $wrapper -Component $name)) {
            $stopPids += $wrapper
        }
        $actions += [PSCustomObject]@{
            Component = $name
            Action    = 'stop'
            Reason    = 'verified_managed_tree'
            RootPid   = [int]$check.RootPid
            TreePids  = @($check.TreePids)
            WrapperPid = $wrapper
            Pids      = @($stopPids | Sort-Object -Unique)
        }
    }
    return @($actions)
}

function Stop-ProcessTreeSafe {
    param([Parameter(Mandatory = $true)][int]$ProcId)
    if (-not (Test-ProcessAlive $ProcId)) {
        return
    }
    Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object { $_.ParentProcessId -eq $ProcId } | ForEach-Object {
        Stop-ProcessTreeSafe -ProcId $_.ProcessId
    }
    Stop-Process -Id $ProcId -Force -ErrorAction SilentlyContinue
}

function Show-AceLocalUserAlert {
    param(
        [Parameter(Mandatory = $true)][string]$Message,
        [string]$LogPath = ""
    )
    $body = $Message
    if ($LogPath) {
        $body = "{0}`n`nLog: {1}" -f $Message, $LogPath
    }
    try {
        Add-Type -AssemblyName System.Windows.Forms -ErrorAction Stop
        [System.Windows.Forms.MessageBox]::Show($body, 'ACE LOCAL', [System.Windows.Forms.MessageBoxButtons]::OK, [System.Windows.Forms.MessageBoxIcon]::Error) | Out-Null
    }
    catch {
        try {
            $short = if ($LogPath) { "{0} (log: {1})" -f $Message, $LogPath } else { $Message }
            if ($short.Length -gt 240) {
                $short = $short.Substring(0, 237) + '...'
            }
            & msg.exe $env:USERNAME $short 2>$null
        }
        catch {
            # Silent launcher: nothing else to do.
        }
    }
}

function Start-AceLocalHiddenPowerShellLogPump {
    param(
        [Parameter(Mandatory = $true)][System.IO.TextReader]$Reader,
        [Parameter(Mandatory = $true)][string]$LogPath
    )
    $rs = [runspacefactory]::CreateRunspace()
    $rs.ThreadOptions = 'ReuseThread'
    $rs.Open()
    $ps = [powershell]::Create()
    $ps.Runspace = $rs
    [void]$ps.AddScript({
            param($InputReader, $Path)
            $writer = [System.IO.StreamWriter]::new($Path, $true, [System.Text.UTF8Encoding]::new($false))
            try {
                while ($null -ne ($line = $InputReader.ReadLine())) {
                    $writer.WriteLine($line)
                    $writer.Flush()
                }
            }
            finally {
                try { $writer.Flush() } catch { }
                try { $writer.Dispose() } catch { }
            }
        }).AddArgument($Reader).AddArgument($LogPath)
    $async = $ps.BeginInvoke()
    return [PSCustomObject]@{
        Runspace   = $rs
        PowerShell = $ps
        Async      = $async
    }
}

function Wait-AceLocalHiddenPowerShellLogPump {
    param(
        $Pump,
        [int]$JoinMs = 15000
    )
    if (-not $Pump) {
        return
    }
    try {
        if ($Pump.Async -and -not $Pump.Async.IsCompleted) {
            $deadline = [datetime]::UtcNow.AddMilliseconds($JoinMs)
            while (-not $Pump.Async.IsCompleted -and [datetime]::UtcNow -lt $deadline) {
                Start-Sleep -Milliseconds 50
            }
        }
        if ($Pump.Async) {
            $null = $Pump.PowerShell.EndInvoke($Pump.Async)
        }
    }
    catch {
        # Pump errors must not break launcher teardown.
    }
    finally {
        if ($Pump.PowerShell) {
            try { $Pump.PowerShell.Dispose() } catch { }
        }
        if ($Pump.Runspace) {
            try { $Pump.Runspace.Close() } catch { }
            try { $Pump.Runspace.Dispose() } catch { }
        }
    }
}

function Start-AceLocalHiddenPowerShell {
    param(
        [Parameter(Mandatory = $true)][string]$ScriptPath,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory,
        [Parameter(Mandatory = $true)][string]$StdoutLog,
        [Parameter(Mandatory = $true)][string]$StderrLog,
        [hashtable]$ExtraEnv = $null
    )
    $psExe = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
    if (-not (Test-Path -LiteralPath $psExe)) {
        $psExe = 'powershell.exe'
    }
    $psArgs = @(
        '-NoProfile',
        '-ExecutionPolicy', 'Bypass',
        '-WindowStyle', 'Hidden',
        '-File', $ScriptPath
    )
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $psExe
    $psi.Arguments = ($psArgs | ForEach-Object {
            if ($_ -match '\s') { '"{0}"' -f $_ } else { $_ }
        }) -join ' '
    $psi.WorkingDirectory = $WorkingDirectory
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    foreach ($key in @('ACE_LOCAL_DEV_ENABLED', 'ACE_SECURITY_ENABLED', 'PORT', 'REDIS_URL', 'ACE_PUBLIC_BASE_URL')) {
        $val = Get-EnvString $key
        if ($val) {
            $psi.EnvironmentVariables[$key] = $val
        }
    }
    if ($ExtraEnv) {
        foreach ($key in $ExtraEnv.Keys) {
            $psi.EnvironmentVariables[$key] = [string]$ExtraEnv[$key]
        }
    }
    $proc = New-Object System.Diagnostics.Process
    $proc.StartInfo = $psi
    $proc.EnableRaisingEvents = $true
    [void]$proc.Start()
    $stdoutPump = Start-AceLocalHiddenPowerShellLogPump -Reader $proc.StandardOutput -LogPath $StdoutLog
    $stderrPump = Start-AceLocalHiddenPowerShellLogPump -Reader $proc.StandardError -LogPath $StderrLog
    return [PSCustomObject]@{
        Process          = $proc
        WrapperProcessId = $proc.Id
        StdoutLog        = $StdoutLog
        StderrLog        = $StderrLog
        StdoutPump       = $stdoutPump
        StderrPump       = $stderrPump
        OutStream        = $null
        ErrStream        = $null
    }
}

function Close-AceLocalHiddenPowerShellStreams {
    param(
        $LaunchHandle,
        [int]$JoinMs = 15000
    )
    if (-not $LaunchHandle) {
        return
    }
    if ($LaunchHandle.Process -and -not $LaunchHandle.Process.HasExited) {
        return
    }
    Wait-AceLocalHiddenPowerShellLogPump -Pump $LaunchHandle.StdoutPump -JoinMs $JoinMs
    Wait-AceLocalHiddenPowerShellLogPump -Pump $LaunchHandle.StderrPump -JoinMs $JoinMs
    foreach ($stream in @($LaunchHandle.OutStream, $LaunchHandle.ErrStream)) {
        if ($stream) {
            try { $stream.Flush(); $stream.Dispose() } catch { }
        }
    }
}
