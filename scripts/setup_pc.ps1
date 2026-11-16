param([switch]$SkipPaddle)

# Cài môi trường riêng trên PC; không sửa các venv của ThinkPad.
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $projectRoot
function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed: $Program $Arguments" }
}
Invoke-Checked 'nvidia-smi' @()
foreach ($folder in @('.venv-train-torch', '.venv-benchmark')) {
    if (!(Test-Path -LiteralPath "$folder\Scripts\python.exe")) {
        Invoke-Checked 'py' @('-3.11', '-m', 'venv', $folder)
    }
    Invoke-Checked "$folder\Scripts\python.exe" @('-m', 'pip', 'install', '--upgrade', 'pip')
}
$torchPy = Join-Path $projectRoot '.venv-train-torch\Scripts\python.exe'
Invoke-Checked $torchPy @('-m','pip','install','torch==2.8.0','torchvision==0.23.0','--index-url','https://download.pytorch.org/whl/cu128')
Invoke-Checked $torchPy @('-m','pip','install','-r','requirements/pc-torch.txt')
Invoke-Checked '.venv-benchmark\Scripts\python.exe' @('-m','pip','install','-r','requirements/pc-benchmark.txt')
Invoke-Checked $torchPy @('-m','ocr.adaptation.preflight','torch')

if (!$SkipPaddle) {
    if (!(Test-Path -LiteralPath '.venv-train-paddle\Scripts\python.exe')) {
        Invoke-Checked 'py' @('-3.11','-m','venv','.venv-train-paddle')
    }
    $paddlePy = Join-Path $projectRoot '.venv-train-paddle\Scripts\python.exe'
    Invoke-Checked $paddlePy @('-m','pip','install','--upgrade','pip')
    Invoke-Checked $paddlePy @('-m','pip','install','paddlepaddle-gpu==3.1.1','--index-url','https://www.paddlepaddle.org.cn/packages/stable/cu129/','--extra-index-url','https://pypi.org/simple')
    Invoke-Checked $paddlePy @('-m','pip','install','-r','requirements/pc-paddle.txt')
    Invoke-Checked $paddlePy @('-m','ocr.adaptation.preflight','paddle')
}
$env:KERAS_BACKEND = 'torch'
Invoke-Checked $torchPy @('-m','ocr.adaptation.preflight','sources')
Invoke-Checked $torchPy @('-m','ipykernel','install','--user','--name','vn-lpr-train-pc','--display-name','VN-LPR PC training')
Invoke-Checked '.venv-benchmark\Scripts\python.exe' @('-m','ipykernel','install','--user','--name','vn-lpr-benchmark-pc','--display-name','VN-LPR CPU benchmark')
Write-Output 'Setup complete. Copy dataset and baseline transfer ZIP before training.'
