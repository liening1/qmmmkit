# PowerShell version of sync_to_hpc.sh for pure-Windows hosts.
# Uses scp/ssh (built into Windows 10+ via OpenSSH client) to ship the
# source tree to an HPC. Slower than rsync because it doesn't do
# incremental copy, but works without WSL or third-party tools.
#
#   .\scripts\sync_to_hpc.ps1 -Destination user@hpc.example.edu:/home/user/qmmmkit
#   .\scripts\sync_to_hpc.ps1 -Destination user@hpc:/scratch/user/qmmmkit -SshArgs "-p 2222"

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [string]$Destination,

    [string]$SshArgs = "",

    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

# Reject things we'd never want on a remote: bytecode, vcs internals, build artefacts.
$excludeDirs = @(
    "__pycache__", ".git\objects", ".git\lfs", ".venv", "build", "dist",
    ".ruff_cache", ".mypy_cache", ".pytest_cache", "qmmmkit_run"
)
$excludeExts = @(".pyc", ".pyo")

if (-not $Destination -match "^[^@]+@[^:]+:.+") {
    Write-Error "Destination must be of the form user@host:/remote/path"
    exit 2
}

$parts = $Destination -split ":", 2
$remote = $parts[0]
$remotePath = $parts[1]

Write-Host "[sync] from: $repoRoot"
Write-Host "[sync] to:   $Destination"
if ($DryRun) { Write-Host "[sync] DRY RUN" }

# Build a list of files to copy.
$files = Get-ChildItem -Recurse -File | Where-Object {
    $rel = $_.FullName.Substring($repoRoot.Length + 1)
    $skip = $false
    foreach ($d in $excludeDirs) {
        if ($rel -like "$d\*" -or $rel -like "*\$d\*") { $skip = $true; break }
    }
    if (-not $skip -and $excludeExts -contains $_.Extension.ToLower()) { $skip = $true }
    -not $skip
}

Write-Host "[sync] $($files.Count) files to transfer"

if ($DryRun) {
    $files | ForEach-Object { Write-Host "  $($_.FullName.Substring($repoRoot.Length + 1))" }
    return
}

# Make the remote directory.
$sshCmd = "ssh $SshArgs $remote `"mkdir -p '$remotePath'`""
Invoke-Expression $sshCmd

# scp -r the whole tree; the includes/excludes above limited which files exist,
# but scp doesn't have rsync's --exclude. To stay accurate we tar+ssh through stdin.
Write-Host "[sync] streaming tar over ssh..."

# Build a temp tar on Windows. Windows 10+ ships tar.exe.
$tarPath = Join-Path $env:TEMP "qmmmkit-sync-$([System.IO.Path]::GetRandomFileName()).tar"
$relFiles = $files | ForEach-Object { $_.FullName.Substring($repoRoot.Length + 1).Replace('\','/') }
$listFile = Join-Path $env:TEMP "qmmmkit-sync-list-$([System.IO.Path]::GetRandomFileName()).txt"
[System.IO.File]::WriteAllLines($listFile, $relFiles, [System.Text.Encoding]::ASCII)

try {
    & tar -cf $tarPath -T $listFile
    if ($LASTEXITCODE -ne 0) { throw "tar failed" }

    # Upload + extract on the remote.
    $remoteCmd = "tar -xf - -C '$remotePath'"
    Get-Content $tarPath -Raw -AsByteStream | & ssh $SshArgs.Split(" ") $remote $remoteCmd
    if ($LASTEXITCODE -ne 0) { throw "ssh tar pipeline failed" }
} finally {
    Remove-Item -Force -ErrorAction SilentlyContinue $tarPath, $listFile
}

Write-Host ""
Write-Host "[sync] DONE."
Write-Host ""
Write-Host "Next steps on the HPC:"
Write-Host "  ssh $remote"
Write-Host "  cd $remotePath"
Write-Host "  bash scripts/deploy_hpc.sh"
