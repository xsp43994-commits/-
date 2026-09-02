param(
    [Parameter(Mandatory = $true)]
    [int]$SyntheticProcessId,
    [int]$RealProcessId = 0
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = 'D:\anaconda3\envs\Deeplearning-gpu\python.exe'
$protocol = 'paper_runs\protocols\multimap_generalization_v3_2\protocol.json'
$output = 'paper_runs\multimap_v3_2'
$mapRoot = 'map_data\multimap_v3_1'

# Synthetic and real task certification may run independently.  Wait for both
# before the resume/audit command touches the shared real-task manifest.
Wait-Process -Id $SyntheticProcessId
Set-Location -LiteralPath $root

& $python -X utf8 -B paper_multimap_experiments.py --protocol $protocol --map-root $mapRoot --output-root $output audit-tasks --manifest "$output\manifests\synthetic_test\manifest.json"
if ($LASTEXITCODE -ne 0) { throw 'Synthetic task audit failed.' }

if ($RealProcessId -gt 0) {
    Wait-Process -Id $RealProcessId
}

& $python -X utf8 -B paper_v3_2_experiments.py --protocol $protocol --output-root $output prepare-real-tasks --map-root $mapRoot --resume-existing
if ($LASTEXITCODE -ne 0) { throw 'Real DSM task certification failed.' }

& $python -X utf8 -B paper_v3_2_experiments.py --protocol $protocol --output-root $output freeze-matrix --synthetic-records "$output\manifests\synthetic_test\records.jsonl" --real-records "$output\formal_evaluation\real_tasks\records.jsonl"
if ($LASTEXITCODE -ne 0) { throw 'Formal evaluation matrix freeze failed.' }
