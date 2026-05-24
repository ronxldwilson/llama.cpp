param(
    [string]$Model = "F:\inference-engine\models\Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    [string]$Prompt = "",
    [int]$Context = 2048,
    [int]$NPredict = 256,
    [switch]$Server,
    [switch]$Bench
)

$BinDir = "F:\inference-engine\llama.cpp\build\bin"
$Threads = 4

if ($Bench) {
    Write-Host "Running optimized benchmark..."
    & "$BinDir\llama-bench.exe" -m $Model -t $Threads -p 512 -n 128 -r 5
    exit
}

if ($Server) {
    Write-Host "Starting optimized server on port 8080..."
    & "$BinDir\llama-server.exe" `
        -m $Model `
        -t $Threads `
        -c $Context `
        --spec-type ngram-mod `
        --port 8080
    exit
}

$args_list = @(
    "-m", $Model,
    "-t", $Threads,
    "-c", $Context,
    "-n", $NPredict,
    "--spec-type", "ngram-simple",
    "-cnv"
)

if ($Prompt) {
    $args_list += @("-p", $Prompt, "--no-display-prompt")
}

Write-Host "Running with optimized settings (ngram speculation enabled)..."
Write-Host "Model: $Model"
Write-Host "Threads: $Threads | Context: $Context | Spec: ngram-simple"
Write-Host "---"

& "$BinDir\llama-cli.exe" @args_list
