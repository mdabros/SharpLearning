$slnPath = "SharpLearning.sln"
$sourceNugetExe = "https://dist.nuget.org/win-x86-commandline/latest/nuget.exe"
# We download into packages so it is not checked-in since this is ignored in git
$packagesPath = ".\packages\"
$targetNugetExe = $packagesPath + "nuget.exe"
# Download if it does not exist
If (!(Test-Path $targetNugetExe))
{
   If (!(Test-Path $packagesPath))
   {
     mkdir $packagesPath
   }
   "Downloading nuget to: " + $targetNugetExe
   Invoke-WebRequest $sourceNugetExe -OutFile $targetNugetExe
}

# Install VSTS nuget bootstrapper https://github.com/Microsoft/vsts-nuget-bootstrapper 
#iex ($targetNugetExe + " install -OutputDirectory " + $packagesPath + " Microsoft.VisualStudio.Services.NuGet.Bootstrap")
#iex ($packagesPath + "Microsoft.VisualStudio.Services.NuGet.Bootstrap.*\tools\Bootstrap.ps1")

# Restore
iex ($targetNugetExe + " restore " + $slnPath)
