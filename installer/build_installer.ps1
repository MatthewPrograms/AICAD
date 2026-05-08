param(
    [string]$PythonExe = "python",
    [switch]$SkipInstaller
)

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Resolve-Path (Join-Path $scriptRoot "..")
$specFile = Join-Path $scriptRoot "aicad_gui.spec"
$innoScript = Join-Path $scriptRoot "AICAD.iss"
$distAppDir = Join-Path $projectRoot "dist\\AICAD"
$buildDir = Join-Path $projectRoot "build"

$innoCandidates = @(
    "${env:ProgramFiles(x86)}\\Inno Setup 6\\ISCC.exe",
    "${env:ProgramFiles}\\Inno Setup 6\\ISCC.exe",
    "${env:LocalAppData}\\Programs\\Inno Setup 6\\ISCC.exe"
)

Write-Host "Project root: $projectRoot"

if (-not (Test-Path $specFile)) {
    throw "Missing PyInstaller spec file: $specFile"
}

$pyInstallerCheck = & $PythonExe -m PyInstaller --version 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing PyInstaller..."
    & $PythonExe -m pip install pyinstaller
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install PyInstaller."
    }
}

if (Test-Path $distAppDir) {
    Remove-Item -Recurse -Force $distAppDir
}
if (Test-Path $buildDir) {
    Remove-Item -Recurse -Force $buildDir
}

Write-Host "Building standalone GUI app with PyInstaller..."
& $PythonExe -m PyInstaller --noconfirm $specFile
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller build failed."
}

$mainExe = Join-Path $distAppDir "AICAD.exe"
if (-not (Test-Path $mainExe)) {
    throw "Expected executable not found: $mainExe"
}

if ($SkipInstaller) {
    Write-Host "SkipInstaller enabled. Built app folder: $distAppDir"
    exit 0
}

$iscc = $null
foreach ($candidate in $innoCandidates) {
    if (Test-Path $candidate) {
        $iscc = $candidate
        break
    }
}

if (-not $iscc) {
    throw "Inno Setup compiler (ISCC.exe) not found. Install Inno Setup 6 and rerun."
}

$version = "3.0.0"
$pyprojectPath = Join-Path $projectRoot "pyproject.toml"
if (Test-Path $pyprojectPath) {
    $versionLine = Get-Content $pyprojectPath | Where-Object { $_ -match '^version\s*=\s*".+"' } | Select-Object -First 1
    if ($versionLine -match '"(.+)"') {
        $version = $Matches[1]
    }
}

$env:AICAD_VERSION = $version
Write-Host "Building installer wizard EXE with Inno Setup (version $version)..."
& $iscc $innoScript
if ($LASTEXITCODE -ne 0) {
    throw "Inno Setup build failed."
}

$installerOutDir = Join-Path $projectRoot "dist\\installer"
Write-Host "Installer build complete. Output directory: $installerOutDir"
