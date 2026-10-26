$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$build = Join-Path $PSScriptRoot 'build'
$segments = Get-Content -LiteralPath (Join-Path $build 'narration.json') -Raw | ConvertFrom-Json
$voice = New-Object System.Speech.Synthesis.SpeechSynthesizer
$voice.SelectVoice('Microsoft Zira Desktop')
$voice.Rate = 0
try {
    for ($i=0; $i -lt $segments.Count; $i++) {
        $voice.SetOutputToWaveFile((Join-Path $build ('voice_{0:D2}.wav' -f $i)))
        $voice.Speak($segments[$i].text)
        $voice.SetOutputToNull()
    }
} finally { $voice.Dispose() }
