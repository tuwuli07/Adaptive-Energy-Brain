param(
    [string]$StorageRoot,
    [string]$SplitRoot,
    [ValidateRange(1, 100000)]
    [double]$MinimumFreeGB = 5
)

$ErrorActionPreference = "Stop"

$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$localEnvironmentPath = Join-Path $repositoryRoot ".env.storage"
$configuredValues = @{}
$managedNames = @(
    "AEB_STORAGE_ROOT",
    "AEB_DATA_ROOT",
    "AEB_SPLIT_ROOT",
    "AEB_EXPERIMENTS_ROOT",
    "AEB_CHECKPOINTS_ROOT",
    "PIP_CACHE_DIR",
    "TORCH_HOME",
    "HF_HOME",
    "MPLCONFIGDIR",
    "TMP",
    "TEMP",
    "PYTHONPYCACHEPREFIX",
    "COVERAGE_FILE"
)

if (Test-Path -LiteralPath $localEnvironmentPath) {
    foreach ($line in Get-Content -LiteralPath $localEnvironmentPath) {
        if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$') {
            $name = $Matches[1]
            $value = $Matches[2]
            if ($value.Length -ge 2 -and $value.StartsWith("'") -and $value.EndsWith("'")) {
                $value = $value.Substring(1, $value.Length - 2)
            }
            $configuredValues[$name] = $value
        }
    }
}

$requestedRoot = $StorageRoot
if ([string]::IsNullOrWhiteSpace($requestedRoot)) {
    $requestedRoot = $env:AEB_STORAGE_ROOT
}
if ([string]::IsNullOrWhiteSpace($requestedRoot)) {
    $requestedRoot = $configuredValues["AEB_STORAGE_ROOT"]
}

$availableDrives = @{}
foreach ($letter in @("E", "F")) {
    $drive = [System.IO.DriveInfo]::new("$letter`:\")
    if ($drive.IsReady) {
        $availableDrives[$letter] = [pscustomobject]@{
            Root = $drive.RootDirectory.FullName
            FreeGB = $drive.AvailableFreeSpace / 1GB
        }
    }
}

if ([string]::IsNullOrWhiteSpace($requestedRoot)) {
    $selectedLetter = $null
    foreach ($letter in @("E", "F")) {
        if ($availableDrives.ContainsKey($letter) -and $availableDrives[$letter].FreeGB -ge $MinimumFreeGB) {
            $selectedLetter = $letter
            break
        }
    }
    if ($null -eq $selectedLetter) {
        throw "Neither E: nor F: is available with at least $MinimumFreeGB GB free. No storage directories were created."
    }
    $requestedRoot = Join-Path $availableDrives[$selectedLetter].Root "AdaptiveEnergyBrainStorage"
}

$resolvedRoot = [System.IO.Path]::GetFullPath($requestedRoot)
$rootDrive = [System.IO.Path]::GetPathRoot($resolvedRoot).TrimEnd("\", ":").ToUpperInvariant()
if ($rootDrive -notin @("E", "F") -or -not $availableDrives.ContainsKey($rootDrive)) {
    throw "StorageRoot must be on an available E: or F: drive; received '$resolvedRoot'."
}
if ($availableDrives[$rootDrive].FreeGB -lt $MinimumFreeGB) {
    throw "$rootDrive`: has $([math]::Round($availableDrives[$rootDrive].FreeGB, 2)) GB free; at least $MinimumFreeGB GB is required."
}

$directoryPaths = @{
    "AEB_STORAGE_ROOT" = $resolvedRoot
    "AEB_VENV_ROOT" = Join-Path $resolvedRoot "venv"
    "AEB_DATA_ROOT" = Join-Path $resolvedRoot "datasets"
    "AEB_CACHE_ROOT" = Join-Path $resolvedRoot "cache"
    "PIP_CACHE_DIR" = Join-Path $resolvedRoot "pip"
    "TORCH_HOME" = Join-Path $resolvedRoot "torch"
    "HF_HOME" = Join-Path $resolvedRoot "huggingface"
    "MPLCONFIGDIR" = Join-Path $resolvedRoot "matplotlib"
    "TMP" = Join-Path $resolvedRoot "temp"
    "TEMP" = Join-Path $resolvedRoot "temp"
    "PYTHONPYCACHEPREFIX" = Join-Path $resolvedRoot "pycache"
    "AEB_CHECKPOINTS_ROOT" = Join-Path $resolvedRoot "checkpoints"
    "AEB_EXPERIMENTS_ROOT" = Join-Path $resolvedRoot "experiments"
    "AEB_LOGS_ROOT" = Join-Path $resolvedRoot "logs"
}

$defaultSplitRoot = Join-Path $repositoryRoot "data\processed\splits_hypothetical_campus_v1"
$configuredSplitRoot = $SplitRoot
if ([string]::IsNullOrWhiteSpace($configuredSplitRoot)) {
    $configuredSplitRoot = $env:AEB_SPLIT_ROOT
}
if ([string]::IsNullOrWhiteSpace($configuredSplitRoot)) {
    $configuredSplitRoot = $configuredValues["AEB_SPLIT_ROOT"]
}
if ([string]::IsNullOrWhiteSpace($configuredSplitRoot)) {
    if (Test-Path -LiteralPath (Join-Path $defaultSplitRoot "preprocessing.json")) {
        $configuredSplitRoot = $defaultSplitRoot
    } else {
        $configuredSplitRoot = Join-Path $directoryPaths["AEB_DATA_ROOT"] "processed\splits_hypothetical_campus_v1"
    }
}
$directoryPaths["AEB_SPLIT_ROOT"] = [System.IO.Path]::GetFullPath($configuredSplitRoot)

$createPaths = @(
    $directoryPaths["AEB_VENV_ROOT"],
    $directoryPaths["AEB_DATA_ROOT"],
    (Join-Path $directoryPaths["AEB_CACHE_ROOT"] "misc"),
    $directoryPaths["PIP_CACHE_DIR"],
    $directoryPaths["TORCH_HOME"],
    $directoryPaths["HF_HOME"],
    $directoryPaths["MPLCONFIGDIR"],
    $directoryPaths["TMP"],
    $directoryPaths["PYTHONPYCACHEPREFIX"],
    $directoryPaths["AEB_CHECKPOINTS_ROOT"],
    $directoryPaths["AEB_EXPERIMENTS_ROOT"],
    $directoryPaths["AEB_LOGS_ROOT"],
    (Split-Path -Parent $directoryPaths["AEB_SPLIT_ROOT"])
)
foreach ($path in $createPaths) {
    New-Item -ItemType Directory -Path $path -Force | Out-Null
}

$directoryPaths["COVERAGE_FILE"] = Join-Path $directoryPaths["AEB_LOGS_ROOT"] ".coverage"
foreach ($name in $managedNames) {
    Set-Item -Path "Env:$name" -Value $directoryPaths[$name]
}

$updatedLines = [System.Collections.Generic.List[string]]::new()
$replacedNames = @{}
if (Test-Path -LiteralPath $localEnvironmentPath) {
    foreach ($line in Get-Content -LiteralPath $localEnvironmentPath) {
        if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=') {
            $name = $Matches[1]
            if ($managedNames -contains $name) {
                if (-not $replacedNames.ContainsKey($name)) {
                    $updatedLines.Add("$name='$($directoryPaths[$name])'")
                    $replacedNames[$name] = $true
                }
                continue
            }
        }
        $updatedLines.Add($line)
    }
}
foreach ($name in $managedNames) {
    if (-not $replacedNames.ContainsKey($name)) {
        $updatedLines.Add("$name='$($directoryPaths[$name])'")
    }
}
[System.IO.File]::WriteAllLines(
    $localEnvironmentPath,
    $updatedLines,
    [System.Text.UTF8Encoding]::new($false)
)

Write-Host "Adaptive Energy Brain storage root: $resolvedRoot"
Write-Host ("Selected drive free space: {0:N2} GB" -f $availableDrives[$rootDrive].FreeGB)
Write-Host "Managed storage environment:"
foreach ($name in $managedNames) {
    Write-Host ("  {0}={1}" -f $name, $directoryPaths[$name])
}
Write-Host "Local VS Code environment file: $localEnvironmentPath"

$projectPython = Join-Path $repositoryRoot ".venv\Scripts\python.exe"
if (Test-Path -LiteralPath $projectPython) {
    Write-Host "Existing project interpreter (preserved): $projectPython"
    $projectEnvironmentDrive = [System.IO.Path]::GetPathRoot($projectPython).TrimEnd("\", ":").ToUpperInvariant()
    if ($projectEnvironmentDrive -in @("E", "F")) {
        . (Join-Path $repositoryRoot ".venv\Scripts\Activate.ps1")
        python -c "import sys; print('  Active Python', sys.version.split()[0], '| prefix:', sys.prefix)"
    } elseif (Test-Path -LiteralPath (Join-Path $directoryPaths["AEB_VENV_ROOT"] "Scripts\Activate.ps1")) {
        Write-Warning "The project .venv is on C:. Preserving it and activating the existing external environment instead."
        . (Join-Path $directoryPaths["AEB_VENV_ROOT"] "Scripts\Activate.ps1")
    } else {
        Write-Warning "The existing project .venv is on C: and has not been moved. Create and test a fresh environment on the selected drive before switching VS Code."
        & $projectPython -c "import sys; print('  Existing Python', sys.version.split()[0], '| prefix:', sys.prefix)"
    }
} else {
    $externalActivate = Join-Path $directoryPaths["AEB_VENV_ROOT"] "Scripts\Activate.ps1"
    if (Test-Path -LiteralPath $externalActivate) {
        Write-Host "Activating the external project environment: $externalActivate"
        . $externalActivate
    } else {
        Write-Host "No project virtual environment found. To create a new environment on the selected drive, use an installed compatible Python 3.12 interpreter:"
        Write-Host "  py -3.12 -m venv `"$($directoryPaths["AEB_VENV_ROOT"])`""
        Write-Host "Then install the repository requirements into that environment and select its interpreter in VS Code."
    }
}

if (-not (Test-Path -LiteralPath $directoryPaths["AEB_SPLIT_ROOT"])) {
    Write-Warning "Configured dataset split path does not exist yet: $($directoryPaths["AEB_SPLIT_ROOT"]). No dataset was moved or generated."
}
Write-Host "Dot-source this script in each standalone PowerShell terminal to apply variables there:"
Write-Host "  . .\scripts\setup_windows_storage.ps1"
