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

; compression level and LZMA2 block threads (override: iscc /DLzmaLevel=normal /DLzmaThreads=1 ...)
#ifndef LzmaLevel
  #define LzmaLevel "max"
#endif
#ifndef LzmaThreads
  #define LzmaThreads "2"
#endif

[Setup]
AppId={{6C1E2F4A-3B7D-4C59-9A0E-5D2B8F1C7A31}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher=Muhammad A. M. Islam
AppPublisherURL=https://github.com/MIslam1010/mr-studio
AppSupportURL=https://github.com/MIslam1010/mr-studio/issues
CloseApplications=yes
AppComments=Kinematics and dynamics of robot chains following Modern Robotics (Lynch & Park)
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=dist
OutputBaseFilename=MR-Studio-{#MyAppVersion}-Windows-x64-setup
#if FileExists(AddBackslash(SourcePath) + "build\icon.ico")
SetupIconFile=build\icon.ico
#endif
UninstallDisplayIcon={app}\{#MyAppExe}
Compression=lzma2/{#LzmaLevel}
SolidCompression=yes
; two LZMA2 blocks compressed in parallel (each block thread uses two cores; the CI runner has four)
LZMANumBlockThreads={#LzmaThreads}
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "dist\MRStudio\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\THIRD_PARTY_NOTICES.md"; DestDir: "{app}"; Flags: ignoreversion

[Registry]
; mrstudio://open?tab=...&robot=...&preset=... links (Knowledge Map "Open in MR Studio"): Windows starts
; MRStudio.exe with the URL as its argument (launcher.py -> gui/app.py main). HKA = per-user with PrivilegesRequired=lowest.
Root: HKA; Subkey: "Software\Classes\mrstudio"; ValueType: string; ValueName: ""; ValueData: "URL:MR Studio link"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\mrstudio"; ValueType: string; ValueName: "URL Protocol"; ValueData: ""
Root: HKA; Subkey: "Software\Classes\mrstudio\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\{#MyAppExe},0"
Root: HKA; Subkey: "Software\Classes\mrstudio\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#MyAppExe}"" ""%1"""

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExe}"
Name: "{group}\{#MyAppName} (projector mode)"; Filename: "{app}\{#MyAppExe}"; Parameters: "--projector"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExe}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; Flags: nowait postinstall skipifsilent
