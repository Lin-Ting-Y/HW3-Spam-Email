<#
Download dataset for SMS spam baseline into data/ directory.
Usage (PowerShell):
  .\scripts\download_dataset.ps1
#>
$outDir = Join-Path $PSScriptRoot '..\data' | Resolve-Path -Relative
if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }
$url = 'https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/master/Chapter03/datasets/sms_spam_no_header.csv'
$out = Join-Path $outDir 'sms_spam_no_header.csv'
Write-Host "Downloading dataset to $out..."
Invoke-WebRequest -Uri $url -OutFile $out -UseBasicParsing
Write-Host "Done."
