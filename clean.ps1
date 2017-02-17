$buildPath = "./Build"
$buildTestPath = "./BuildTest"
$testResultsPath = "./TestResults"
If (Test-Path $buildPath)
{ Remove-Item -Confirm -Recurse -Path $buildPath }
If (Test-Path $buildTestPath)
{ Remove-Item -Confirm -Recurse -Path $buildTestPath }
If (Test-Path $testResultsPath)
{ Remove-Item -Confirm -Recurse -Path  $testResultsPath }