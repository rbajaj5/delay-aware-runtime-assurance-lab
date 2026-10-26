$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$taskBuild = Join-Path $PSScriptRoot 'build'
$taskSegments = Get-Content -LiteralPath (Join-Path $taskBuild 'narration.json') -Raw | ConvertFrom-Json
$taskVoice = New-Object System.Speech.Synthesis.SpeechSynthesizer
$taskVoice.SelectVoice('Microsoft Zira Desktop')
$taskVoice.Rate = 0
try {
    for ($i=0; $i -lt $taskSegments.Count; $i++) {
        $taskVoice.SetOutputToWaveFile((Join-Path $taskBuild ('voice_{0:D2}.wav' -f $i)))
        $taskVoice.Speak($taskSegments[$i].text)
        $taskVoice.SetOutputToNull()
    }
} finally { $taskVoice.Dispose() }
