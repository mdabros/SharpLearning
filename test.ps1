$vstestconsolepath = $env:VS140COMNTOOLS + "..\IDE\CommonExtensions\Microsoft\TestWindow\vstest.console.exe"
$testRoot = "BuildTest\"

$testDlls = Get-ChildItem -Path $testRoot -File -Recurse -Include *Test.dll

& $vstestconsolepath $testDlls '/Platform:x86' # '/Parallel'
& $vstestconsolepath $testDlls '/Platform:x64' # '/Parallel'