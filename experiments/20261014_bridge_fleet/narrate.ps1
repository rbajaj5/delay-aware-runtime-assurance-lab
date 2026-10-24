param(
    [string]$OutputDirectory = (Join-Path $PSScriptRoot 'output'),
    [string]$Voice = 'Microsoft Zira Desktop'
)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$segments = Get-Content -Raw -Encoding UTF8 -LiteralPath (Join-Path $OutputDirectory 'narration_segments.json') | ConvertFrom-Json
$audioDirectory = Join-Path $OutputDirectory 'narration_parts'
New-Item -ItemType Directory -Force -Path $audioDirectory | Out-Null
Copy-Item -LiteralPath (Join-Path $OutputDirectory 'narration_segments.json') -Destination (Join-Path $audioDirectory 'narration_source.json')
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
    $synth.SelectVoice($Voice)
    $synth.Rate = 1
    $synth.Volume = 100
    for ($i = 0; $i -lt $segments.Count; $i++) {
        $path = Join-Path $audioDirectory ('segment_{0:D2}.wav' -f $i)
        $synth.SetOutputToWaveFile($path)
        $synth.Speak($segments[$i].text)
        $synth.SetOutputToNull()
    }
}
finally { $synth.Dispose() }
Write-Output "Generated $($segments.Count) local synthetic-voice segments."
