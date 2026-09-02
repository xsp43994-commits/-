param(
    [string]$PythonExe = "D:\anaconda3\envs\Deeplearning-gpu\python.exe",
    [string]$ExpectedProtocolHash = "92b25776749a9430e71e47e0882970c5ea149ed778906104ad38604b870860ba"
)

$ErrorActionPreference = "Stop"
$workspace = Split-Path -Parent $MyInvocation.MyCommand.Path
$protocolPath = Join-Path $workspace "paper_runs\protocols\multimap_generalization_v3_1\protocol.json"
$programPath = Join-Path $workspace "paper_multimap_experiments.py"
$outputRoot = Join-Path $workspace "paper_runs\multimap_v3_1"
$trainingRegistry = Join-Path $workspace "map_data\multimap_v3_1\procedural\training\map_registry.json"
$baseRecords = Join-Path $outputRoot "manifests\training\serial_base_for_merge.jsonl"
$shardRoot = Join-Path $outputRoot "manifests\training_shards"
$logRoot = Join-Path $outputRoot "logs"
$statusPath = Join-Path $outputRoot "monitoring\parallel_training_supervisor.json"

# 四个分片按地图编号切分；正式合并只使用旧串行结果中的地图0--3共36条。
$shards = @(
    [pscustomobject]@{ Name = "shard_00"; Start = 4; Stop = 22; Expected = 162 },
    [pscustomobject]@{ Name = "shard_01"; Start = 22; Stop = 39; Expected = 153 },
    [pscustomobject]@{ Name = "shard_02"; Start = 39; Stop = 56; Expected = 153 },
    [pscustomobject]@{ Name = "shard_03"; Start = 56; Stop = 72; Expected = 144 }
)

New-Item -ItemType Directory -Path $shardRoot, $logRoot, (Split-Path -Parent $statusPath) -Force | Out-Null

# 限制每个求解器进程只使用一个线程，避免四进程相互抢占造成反向减速。
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"

function Read-JsonIfPresent {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        return $null
    }
    return Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json
}

function Get-ShardCheckpoint {
    param([string]$Name)
    return Read-JsonIfPresent -Path (Join-Path $shardRoot "$Name\generation_checkpoint.json")
}

function Get-LiveShardProcess {
    param([string]$Name)
    return Get-CimInstance Win32_Process |
        Where-Object {
            $_.Name -like "python*" -and
            $_.CommandLine -like "*paper_multimap_experiments.py*prepare-tasks*" -and
            $_.CommandLine -like "*--shard-name*$Name*"
        } |
        Select-Object -First 1
}

function Start-Shard {
    param([pscustomobject]$Shard)
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $stdout = Join-Path $logRoot "training_$($Shard.Name)_$stamp.stdout.log"
    $stderr = Join-Path $logRoot "training_$($Shard.Name)_$stamp.stderr.log"
    $arguments = @(
        "-X", "utf8",
        $programPath,
        "prepare-tasks",
        "--split", "training",
        "--map-registry", $trainingRegistry,
        "--map-index-start", [string]$Shard.Start,
        "--map-index-stop", [string]$Shard.Stop,
        "--shard-name", $Shard.Name,
        "--screening-time-limit-s", "10",
        "--certification-time-limit-s", "60",
        "--max-attempts-per-task", "2000"
    )
    $recordsPath = Join-Path $shardRoot "$($Shard.Name)\records.jsonl"
    if (Test-Path -LiteralPath $recordsPath) {
        $arguments += "--resume-existing"
    }
    $process = Start-Process `
        -FilePath $PythonExe `
        -ArgumentList $arguments `
        -WorkingDirectory $workspace `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -PassThru
    # Write-Host不进入函数返回管道，确保调用方只接收到数值PID。
    Write-Host "Started $($Shard.Name): maps=$($Shard.Start)-$($Shard.Stop - 1) pid=$($process.Id)"
    return $process.Id
}

function Write-SupervisorStatus {
    param([array]$Rows, [string]$State)
    $payload = [ordered]@{
        schema_version = 1
        protocol_hash = $ExpectedProtocolHash
        state = $State
        updated_at = (Get-Date).ToString("o")
        shard_completed = [int](($Rows | Measure-Object -Property Completed -Sum).Sum)
        shard_expected = [int](($Rows | Measure-Object -Property Expected -Sum).Sum)
        shards = $Rows
    }
    $temporary = "$statusPath.$PID.tmp"
    [System.IO.File]::WriteAllText(
        $temporary,
        ($payload | ConvertTo-Json -Depth 8) + [Environment]::NewLine,
        [System.Text.UTF8Encoding]::new($false)
    )
    Move-Item -LiteralPath $temporary -Destination $statusPath -Force
}

function Invoke-CheckedPython {
    param([string[]]$Arguments)
    & $PythonExe -X utf8 $programPath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed: exit_code=$LASTEXITCODE args=$($Arguments -join ' ')"
    }
}

$protocol = Read-JsonIfPresent -Path $protocolPath
if ($null -eq $protocol -or $protocol.protocol_hash -ne $ExpectedProtocolHash) {
    $actual = if ($null -eq $protocol) { "missing" } else { $protocol.protocol_hash }
    throw "Protocol hash drift: expected=$ExpectedProtocolHash actual=$actual"
}

Set-Location -LiteralPath $workspace
$restartCounts = @{}
foreach ($shard in $shards) {
    $restartCounts[$shard.Name] = 0
}

while ($true) {
    $rows = @()
    $allCompleted = $true
    foreach ($shard in $shards) {
        $checkpoint = Get-ShardCheckpoint -Name $shard.Name
        $completed = if ($null -eq $checkpoint) { 0 } else { [int]$checkpoint.completed }
        $state = if ($null -eq $checkpoint) { "not_started" } else { [string]$checkpoint.state }
        $auditPassed = $null -ne $checkpoint -and [bool]$checkpoint.audit_passed
        if ($state -eq "completed" -and $auditPassed -and $completed -eq $shard.Expected) {
            $rows += [pscustomobject]@{
                Name = $shard.Name
                Completed = $completed
                Expected = $shard.Expected
                State = "completed"
                ProcessId = $null
                CurrentTask = $null
                Attempt = $null
            }
            continue
        }
        $allCompleted = $false
        if ($state -in @("failed", "audit_failed")) {
            throw "$($shard.Name) stopped: state=$state"
        }
        $live = Get-LiveShardProcess -Name $shard.Name
        if ($null -eq $live) {
            $restartCounts[$shard.Name] = [int]$restartCounts[$shard.Name] + 1
            if ($restartCounts[$shard.Name] -gt 5) {
                throw "$($shard.Name) exceeded five automatic restarts."
            }
            $pidValue = Start-Shard -Shard $shard
            $state = "running"
        }
        else {
            $pidValue = [int]$live.ProcessId
        }
        $rows += [pscustomobject]@{
            Name = $shard.Name
            Completed = $completed
            Expected = $shard.Expected
            State = $state
            ProcessId = $pidValue
            CurrentTask = if ($null -eq $checkpoint) { $null } else { $checkpoint.current_task_id }
            Attempt = if ($null -eq $checkpoint) { $null } else { $checkpoint.current_attempt }
        }
    }

    Write-SupervisorStatus -Rows $rows -State $(if ($allCompleted) { "merging" } else { "running" })
    $progress = ($rows | ForEach-Object { "$($_.Name)=$($_.Completed)/$($_.Expected)" }) -join ", "
    Write-Output "$(Get-Date -Format o) $progress"
    if ($allCompleted) {
        break
    }
    Start-Sleep -Seconds 30
}

# 合并输入按任务ID完全不重叠：串行36条覆盖地图0--3，分片从地图4开始。
Invoke-CheckedPython -Arguments @(
    "merge-task-shards",
    "--split", "training",
    "--map-registry", $trainingRegistry,
    "--base-records", $baseRecords
)

$mergedCheckpoint = Read-JsonIfPresent -Path (Join-Path $outputRoot "manifests\training\generation_checkpoint.json")
if (
    $null -eq $mergedCheckpoint -or
    $mergedCheckpoint.state -ne "completed" -or
    -not $mergedCheckpoint.audit_passed -or
    [int]$mergedCheckpoint.completed -ne 648
) {
    throw "Merged training manifest did not pass the 648-task gate."
}

Write-SupervisorStatus -Rows @() -State "merged"
Write-Output "Parallel certification merged and audited. Continuing to environment seal and the three-model pilot gate."
& (Join-Path $workspace "run_multimap_to_pilot.ps1") `
    -PythonExe $PythonExe `
    -ExpectedProtocolHash $ExpectedProtocolHash
if ($LASTEXITCODE -ne 0) {
    throw "Downstream pilot supervisor failed: exit_code=$LASTEXITCODE"
}
