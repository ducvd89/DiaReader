; Inno Setup script for DiaReader.
;
; Two things this installer deliberately does NOT ship:
;
;   The model (~6.3 GB). It is downloaded from Hugging Face on first run and cached
;   under %APPDATA%, so it survives reinstalls and upgrades instead of being copied
;   into every one of them.
;
;   FFmpeg. torchcodec needs FFmpeg's *shared* libraries to read voice reference
;   audio, and they are a system component with their own installer. Without them,
;   every sentence in a voice-conditioned export silently produces no audio — a
;   multi-hour export finishes as a file of silence. So the installer checks for
;   ffmpeg and offers to fetch it rather than leaving the user to discover that.

#define AppName "DiaReader"
#define AppVersion "1.0.0"
#define AppPublisher "DiaReader"
#define AppExe "DiaReader.exe"

[Setup]
AppId={{7C4A9E12-3B5D-4F8A-9C21-6D0E4A7B1F93}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
OutputDir=..\build\installer
OutputBaseFilename=DiaReader-{#AppVersion}-setup
Compression=lzma2/max
SolidCompression=yes
; The payload is several GB of DLLs; 64-bit only, which is also all CUDA supports.
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern
DisableProgramGroupPage=yes
PrivilegesRequiredOverridesAllowed=dialog
UninstallDisplayIcon={app}\{#AppExe}
SetupIconFile=assets\icon.ico
; Refuses to install into a folder without room for the unpacked app.
ExtraDiskSpaceRequired=0

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"

[Files]
Source: "..\build\dist\DiaReader\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "Start {#AppName}"; Flags: nowait postinstall skipifsilent

[Code]
function FfmpegSharedPresent(): Boolean;
var
  ResultCode: Integer;
begin
  { avcodec-*.dll is what torchcodec actually loads; ffmpeg.exe alone does not
    satisfy it, which is exactly the trap the bundled imageio-ffmpeg falls into. }
  Result := Exec('cmd.exe', '/C where avcodec-*.dll >nul 2>&1', '', SW_HIDE,
                 ewWaitUntilTerminated, ResultCode) and (ResultCode = 0);
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
begin
  if CurStep = ssPostInstall then
  begin
    if not FfmpegSharedPresent() then
    begin
      if MsgBox('DiaReader needs FFmpeg''s shared libraries to read voice reference' + #13#10 +
                'audio. They were not found on this machine.' + #13#10 + #13#10 +
                'Without them, exports that use a reference voice produce silence' + #13#10 +
                'rather than an error.' + #13#10 + #13#10 +
                'Install FFmpeg now with winget?', mbConfirmation, MB_YESNO) = IDYES then
      begin
        Exec('winget.exe',
             'install --id Gyan.FFmpeg.Shared -e --source winget ' +
             '--accept-package-agreements --accept-source-agreements',
             '', SW_SHOW, ewWaitUntilTerminated, ResultCode);
      end;
    end;
  end;
end;
