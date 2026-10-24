param(
    [string]$Voice = "Microsoft Zira Desktop",
    [int]$Rate = 3
)

$ErrorActionPreference = "Stop"
$labRoot = Split-Path $PSScriptRoot -Parent
$outDir = Join-Path $labRoot "artifacts\20260808\sensor_uncertainty_3d"
$silentMovie = Join-Path $outDir "sensor_uncertainty_rta_comparison.mp4"
$scriptFile = Join-Path $outDir "sensor_uncertainty_voiceover_script.txt"
$rawWave = Join-Path $outDir "sensor_uncertainty_voiceover_raw.wav"
$finalWave = Join-Path $outDir "sensor_uncertainty_voiceover.wav"
$narratedMovie = Join-Path $outDir "sensor_uncertainty_rta_comparison_narrated.mp4"

foreach ($path in @($silentMovie, $scriptFile)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required input is missing: $path"
    }
}

Add-Type -AssemblyName System.Speech
$synthesizer = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
    $synthesizer.SelectVoice($Voice)
    $synthesizer.Rate = $Rate
    $synthesizer.Volume = 100
    $synthesizer.SetOutputToWaveFile($rawWave)
    $synthesizer.Speak((Get-Content -Raw -LiteralPath $scriptFile))
}
finally {
    $synthesizer.Dispose()
}

& ffmpeg -y -i $rawWave -af "adelay=350:all=1,loudnorm=I=-18:TP=-2:LRA=7,aresample=48000,apad=pad_dur=2" -t 22 $finalWave -loglevel error
if ($LASTEXITCODE -ne 0) {
    throw "ffmpeg failed while preparing the narration track."
}

& ffmpeg -y -i $silentMovie -i $finalWave -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -ar 48000 -b:a 160k -t 22 -movflags +faststart $narratedMovie -loglevel error
if ($LASTEXITCODE -ne 0) {
    throw "ffmpeg failed while muxing the narrated movie."
}

$manifestItems = @(
    @{ Label = "sensor_uncertainty_rta_comparison_narrated.mp4"; Path = $narratedMovie },
    @{ Label = "sensor_uncertainty_rta_comparison.mp4"; Path = $silentMovie },
    @{ Label = "sensor_uncertainty_rta_comparison_poster.png"; Path = (Join-Path $outDir "sensor_uncertainty_rta_comparison_poster.png") },
    @{ Label = "sensor_uncertainty_3d_step_trace.csv"; Path = (Join-Path $outDir "sensor_uncertainty_3d_step_trace.csv") },
    @{ Label = "sensor_uncertainty_3d_summary.csv"; Path = (Join-Path $outDir "sensor_uncertainty_3d_summary.csv") },
    @{ Label = "nominal_lec_checkpoint.pt"; Path = (Join-Path $outDir "nominal_lec_checkpoint.pt") },
    @{ Label = "nominal_lec_training.json"; Path = (Join-Path $outDir "nominal_lec_training.json") },
    @{ Label = "sensor_uncertainty_rta_report.md"; Path = (Join-Path $outDir "sensor_uncertainty_rta_report.md") },
    @{ Label = "canvas_response_sensor_uncertainty.md"; Path = (Join-Path $outDir "canvas_response_sensor_uncertainty.md") },
    @{ Label = "sensor_uncertainty_voiceover_script.txt"; Path = $scriptFile },
    @{ Label = "sensor_uncertainty_voiceover.wav"; Path = $finalWave },
    @{ Label = "sensor_uncertainty_voiceover_raw.wav"; Path = $rawWave },
    @{ Label = "analysis/run_3d_sensor_uncertainty_rta_simulation.py"; Path = (Join-Path $PSScriptRoot "run_3d_sensor_uncertainty_rta_simulation.py") },
    @{ Label = "analysis/render_sensor_uncertainty_voiceover.ps1"; Path = $PSCommandPath }
)
$manifestLines = foreach ($item in $manifestItems) {
    if (-not (Test-Path -LiteralPath $item.Path)) {
        throw "Manifest input is missing: $($item.Path)"
    }
    $digest = (Get-FileHash -Algorithm SHA256 -LiteralPath $item.Path).Hash.ToLowerInvariant()
    "$digest  $($item.Label)"
}
$manifest = Join-Path $outDir "SENSOR_UNCERTAINTY_3D_MANIFEST.sha256"
[System.IO.File]::WriteAllLines($manifest, $manifestLines, [System.Text.Encoding]::ASCII)

Write-Output $narratedMovie
