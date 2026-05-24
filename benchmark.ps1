param(
    [string]$Model = "F:\inference-engine\models\Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    [string]$Label = "baseline",
    [int]$Threads = 4,
    [int]$Context = 2048,
    [int]$BatchSize = 512,
    [int]$NPredict = 128,
    [string]$Prompt = "Explain the theory of relativity in simple terms for a high school student."
)

$BinDir = "F:\inference-engine\llama.cpp\build\bin"
$ResultsDir = "F:\inference-engine\llama.cpp\benchmarks"
$Timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$ResultFile = "$ResultsDir\${Label}_${Timestamp}.txt"

if (-not (Test-Path $ResultsDir)) { New-Item -ItemType Directory -Path $ResultsDir | Out-Null }

Write-Host "============================================"
Write-Host " llama.cpp CPU Benchmark"
Write-Host "============================================"
Write-Host "Label:      $Label"
Write-Host "Model:      $Model"
Write-Host "Threads:    $Threads"
Write-Host "Context:    $Context"
Write-Host "Batch Size: $BatchSize"
Write-Host "N Predict:  $NPredict"
Write-Host "Timestamp:  $Timestamp"
Write-Host "============================================"
Write-Host ""

# --- Test 1: llama-bench (structured benchmark) ---
Write-Host ">>> Running llama-bench..."
$BenchOutput = & "$BinDir\llama-bench.exe" -m $Model -t $Threads -c $Context -b $BatchSize -n $NPredict 2>&1 | Out-String

Write-Host $BenchOutput

# --- Test 2: llama-cli generation (real-world feel) ---
Write-Host ""
Write-Host ">>> Running llama-cli generation test..."
$CliOutput = & "$BinDir\llama-cli.exe" -m $Model -t $Threads -c $Context -b $BatchSize -n $NPredict -p $Prompt --no-display-prompt 2>&1 | Out-String

# Extract timing lines
$TimingLines = ($CliOutput -split "`n") | Where-Object { $_ -match "token|eval|timing" }

Write-Host ""
Write-Host "--- Timing Summary ---"
$TimingLines | ForEach-Object { Write-Host $_ }

# --- Save results ---
$Report = @"
============================================
llama.cpp Benchmark Results
============================================
Label:      $Label
Model:      $Model
Threads:    $Threads
Context:    $Context
Batch Size: $BatchSize
N Predict:  $NPredict
Timestamp:  $Timestamp
============================================

=== llama-bench output ===
$BenchOutput

=== llama-cli timing ===
$($TimingLines -join "`n")

=== Full cli output ===
$CliOutput
"@

$Report | Out-File -FilePath $ResultFile -Encoding utf8
Write-Host ""
Write-Host "Results saved to: $ResultFile"
Write-Host ""

# --- Quick summary extraction ---
$ppMatch = [regex]::Match($BenchOutput, "pp\d+.*?(\d+\.\d+)\s*\±")
$tgMatch = [regex]::Match($BenchOutput, "tg\d+.*?(\d+\.\d+)\s*\±")

if ($ppMatch.Success -and $tgMatch.Success) {
    Write-Host "============================================"
    Write-Host " SUMMARY"
    Write-Host "============================================"
    Write-Host "  Prompt Processing: $($ppMatch.Groups[1].Value) tokens/sec"
    Write-Host "  Text Generation:   $($tgMatch.Groups[1].Value) tokens/sec"
    Write-Host "============================================"
}
