param([switch]$Apply)

# Chỉ dọn baseline cũ đã bỏ; không xoá dataset, model mới hoặc gói chuyển PC.
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$pythonPath = Join-Path $projectRoot '.venv-ocr\Scripts\python.exe'
$relativeTargets = @(
    'runs\vietlpr_v1',
    'runs\paddle_suite',
    'runs\public_vn_ocr',
    'runs\ocr_benchmark',
    'runs\comparison_ranked.csv',
    'third_party\VietLPR_v1',
    'runs\pc_setup'
)
$targets = @()
foreach ($relative in $relativeTargets) {
    $candidate = Join-Path $projectRoot $relative
    if (!(Test-Path -LiteralPath $candidate)) { continue }
    $resolved = (Resolve-Path -LiteralPath $candidate).Path
    if (!$resolved.StartsWith($projectRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw "Outside workspace: $resolved" }
    $item = Get-Item -LiteralPath $resolved -Force
    $entries = if ($item.PSIsContainer) { @(Get-ChildItem -LiteralPath $resolved -Recurse -Force) } else { @() }
    if (@($item) + $entries | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }) { throw "Linked entry: $resolved" }
    $files = if ($item.PSIsContainer) { @($entries | Where-Object { !$_.PSIsContainer }) } else { @($item) }
    $targets += [PSCustomObject]@{ Path=$resolved; Bytes=($files | Measure-Object Length -Sum).Sum }
}
$targets | Select-Object Path,@{Name='MiB';Expression={[math]::Round($_.Bytes/1MB,2)}} | Format-Table -AutoSize
if (!$Apply) { 'Preview only. Use -Apply after checking this exact list.'; return }
if (!(Test-Path -LiteralPath $pythonPath)) { throw 'Run cleanup on the ThinkPad with its existing .venv-ocr.' }
$verify = @'
from pathlib import Path
import hashlib, zipfile, json
from ocr.adaptation.common import validate_frozen
from ocr.benchmark_display import display_table
validate_frozen('VietLPR_v1')
bundle=Path('runs/pc_transfer/current_baselines.zip')
with zipfile.ZipFile(bundle) as archive:
    manifest=json.loads(archive.read('transfer_manifest.json'))
    for name,expected in manifest.items():
        if hashlib.sha256(archive.read(name)).hexdigest()!=expected:
            raise RuntimeError('Transfer ZIP changed: '+name)
for group in ('quantization','detection'):
    text=display_table(group)
    if '| Rank |' not in text: raise RuntimeError('Quantization archive is incomplete')
print('Verified current VietLPR, YOLO transfer and both archived quantization tables.')
'@
Push-Location -LiteralPath $projectRoot
try {
    & $pythonPath -B -c $verify
    if ($LASTEXITCODE -ne 0) { throw 'Verification failed; no files removed.' }
    foreach ($target in $targets) { Remove-Item -LiteralPath $target.Path -Recurse -Force }
    & $pythonPath -B -c $verify
    if ($LASTEXITCODE -ne 0) { throw 'Post-cleanup check failed.' }
    'Removed retired local artifacts: {0:N2} MiB.' -f (($targets | Measure-Object Bytes -Sum).Sum/1MB)
} finally { Pop-Location }
