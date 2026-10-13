; Inno Setup 6 script: turns packaging\dist\MRStudio\ (PyInstaller one-folder build) into a
; one-click Windows installer  packaging\dist\MR-Studio-<version>-Windows-x64-setup.exe
;
;   iscc /DMyAppVersion=1.0.0 packaging\mrstudio.iss        (build_windows.ps1 does this for you)
;
; Installs per user (no administrator rights needed) into %LOCALAPPDATA%\Programs\MR Studio,
; adds Start-menu and optional desktop shortcuts and an uninstaller.

#ifndef MyAppVersion
  #define MyAppVersion "1.0.0"
#endif
#define MyAppName "MR Studio"
#define MyAppExe "MRStudio.exe"

[Setup]
AppId={{6C1E2F4A-3B7D-4C59-9A0E-5D2B8F1C7A31}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher=MR Studio
AppComments=Kinematics and dynamics of robot chains following Modern Robotics (Lynch & Park)
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=dist
OutputBaseFilename=MR-Studio-{#MyAppVersion}-Windows-x64-setup
#if FileExists(AddBackslash(SourcePath) + "build\icon.ico")
SetupIconFile=build\icon.ico
#endif
UninstallDisplayIcon={app}\{#MyAppExe}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "dist\MRStudio\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExe}"
Name: "{group}\{#MyAppName} (projector mode)"; Filename: "{app}\{#MyAppExe}"; Parameters: "--projector"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExe}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; Flags: nowait postinstall skipifsilent
