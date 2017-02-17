$projects = Get-ChildItem -Path $testRoot -File -Recurse -Include *.csproj -Exclude *Test.csproj
Foreach ($p in $projects)
{
    #Push-Location $p.Directory
    #..\..\packages\nuget.exe spec
    #Pop-Location
    .\packages\nuget.exe pack $p -IncludeReferencedProjects -Properties Configuration=Release -OutputDirectory "nupkgs/"
}
