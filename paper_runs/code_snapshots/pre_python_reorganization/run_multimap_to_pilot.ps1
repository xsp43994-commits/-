param(
    [string]$PythonExe = "D:\anaconda3\envs\Deeplearning-gpu\python.exe",
    [string]$ExpectedProtocolHash = "662766dcca7b964e67d9a49603c8d8634f468b4b02c19409927dfe358c14580e"
)

$ErrorActionPreference = "Stop"
$workspace = Split-Path -Parent $MyInvocation.MyCommand.Path
$protocolPath = Join-Path $workspace "paper_runs\protocols\multimap_generalization_v3_1\protocol.json"
$outputRoot = Join-Path $workspace "paper_runs\multimap_v3_1"
$mapRoot = Join-Path $workspace "map_data\multimap_v3_1"
$validationRoot = Join-Path $outputRoot "manifests\validation"
$trainingRoot = Join-Path $outputRoot "manifests\training"
$validationManifest = Join-Path $validationRoot "manifest.json"
$trainingManifest = Join-Path $trainingRoot "manifest.json"

function Invoke-CheckedPython {
    param([string[]]$Arguments)
    & $PythonExe -X utf8 (Join-Path $workspace "paper_multimap_experiments.py") @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed: exit_code=$LASTEXITCODE args=$($Arguments -join ' ')"
    }
}

function Read-Checkpoint {
    param([string]$Directory)
    $path = Join-Path $Directory "generation_checkpoint.json"
    if (-not (Test-Path -LiteralPath $path)) {
        return $null
    }
    return Get-Content -LiteralPath $path -Raw -Encoding UTF8 | ConvertFrom-Json
}

function Wait-Validation {
    while ($true) {
        $checkpoint = Read-Checkpoint -Directory $validationRoot
        if ($null -ne $checkpoint -and $checkpoint.state -eq "completed" -and $checkpoint.audit_passed) {
            Write-Output "Validation certification completed: $($checkpoint.completed)/$($checkpoint.expected)"
            return
        }
        if ($null -ne $checkpoint -and $checkpoint.state -in @("failed", "audit_failed")) {
            throw "Validation certification stopped: state=$($checkpoint.state)"
        }
        $process = Get-CimInstance Win32_Process |
            Where-Object {
                $_.Name -like "python*" -and
                $_.CommandLine -like "*paper_multimap_experiments.py*prepare-tasks*validation*"
            } |
            Select-Object -First 1
        if ($null -eq $process) {
            Write-Output "Validation process disappeared; resuming with frozen arguments."
            Invoke-CheckedPython -Arguments @(
                "prepare-tasks",
                "--split", "validation",
                "--map-registry", (Join-Path $mapRoot "procedural\validation\map_registry.json"),
                "--screening-time-limit-s", "10",
                "--certification-time-limit-s", "60",
                "--max-attempts-per-task", "2000",
                "--resume-existing"
            )
        }
        else {
            $completed = if ($null -eq $checkpoint) { 0 } else { $checkpoint.completed }
            Write-Output "Validation running: $completed/108 pid=$($process.ProcessId)"
            Start-Sleep -Seconds 30
        }
    }
}

$protocol = Get-Content -LiteralPath $protocolPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($protocol.protocol_hash -ne $ExpectedProtocolHash) {
    throw "Protocol hash drift: expected=$ExpectedProtocolHash actual=$($protocol.protocol_hash)"
}

Set-Location -LiteralPath $workspace
Wait-Validation

$trainingCheckpoint = Read-Checkpoint -Directory $trainingRoot
if ($null -eq $trainingCheckpoint -or $trainingCheckpoint.state -ne "completed" -or -not $trainingCheckpoint.audit_passed) {
    New-Item -ItemType Directory -Path $trainingRoot -Force | Out-Null
    $resumeArguments = @()
    if (Test-Path -LiteralPath (Join-Path $trainingRoot "records.jsonl")) {
        $resumeArguments = @("--resume-existing")
    }
    Write-Output "Starting or resuming model-independent MILP certification for 648 training tasks."
    Invoke-CheckedPython -Arguments (
        @(
            "prepare-tasks",
            "--split", "training",
            "--map-registry", (Join-Path $mapRoot "procedural\training\map_registry.json"),
            "--screening-time-limit-s", "10",
            "--certification-time-limit-s", "60",
            "--max-attempts-per-task", "2000"
        ) + $resumeArguments
    )
}

Invoke-CheckedPython -Arguments @("audit-tasks", "--manifest", $validationManifest)
Invoke-CheckedPython -Arguments @("audit-tasks", "--manifest", $trainingManifest)
Invoke-CheckedPython -Arguments @(
    "audit-splits",
    "--training-manifest", $trainingManifest,
    "--validation-manifest", $validationManifest
)
Invoke-CheckedPython -Arguments @(
    "seal-environment",
    "--training-manifest", $trainingManifest,
    "--validation-manifest", $validationManifest
)

Write-Output "Environment sealed. Starting three-model 600-episode pilot; pilot models are not paper eligible."
Invoke-CheckedPython -Arguments @(
    "train-grid",
    "--stage", "pilot",
    "--training-manifest", $trainingManifest,
    "--validation-manifest", $validationManifest,
    "--device", "cuda",
    "--resume-existing"
)
Invoke-CheckedPython -Arguments @(
    "assess-pilot",
    "--validation-manifest", $validationManifest,
    "--device", "cuda"
)
Write-Output "Pilot and automatic decision completed. This script does not start the 35 formal models."
