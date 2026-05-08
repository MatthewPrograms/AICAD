#define MyAppName "AICAD"
#define MyAppVersion GetEnv("AICAD_VERSION")
#if MyAppVersion == ""
  #define MyAppVersion "3.0.0"
#endif
#define MyAppPublisher "MatthewPrograms"
#define MyAppExeName "AICAD.exe"

[Setup]
AppId={{8D509924-6B3B-4D99-A443-9E9E6974A1A8}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\AICAD
DefaultGroupName=AICAD
DisableProgramGroupPage=no
LicenseFile=..\LICENSE
OutputDir=..\dist\installer
OutputBaseFilename=AICAD-Setup-{#MyAppVersion}
Compression=lzma
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Files]
Source: "..\dist\AICAD\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\lisp-code\*"; DestDir: "{app}\lisp-code"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\AICAD"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall AICAD"; Filename: "{uninstallexe}"
Name: "{autodesktop}\AICAD"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch AICAD"; Flags: nowait postinstall skipifsilent
