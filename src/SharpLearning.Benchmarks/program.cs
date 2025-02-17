// Type 'Program' can be sealed because it has no subtypes in its containing assembly and is not externally visible
#pragma warning disable CA1852
using System;
using System.Diagnostics;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Reports;
using BenchmarkDotNet.Running;
using SharpLearning.Benchmarks;

[assembly: System.Runtime.InteropServices.ComVisible(false)]

Action<string> log = t => { Console.WriteLine(t); Trace.WriteLine(t); };

log($"{Environment.Version} args: {args.Length}");

var config = (Debugger.IsAttached ? new DebugInProcessConfig() : DefaultConfig.Instance)
    .WithSummaryStyle(SummaryStyle.Default.WithMaxParameterColumnWidth(200));

BenchmarkRunner.Run(typeof(Benchmarks.Regression.DecisionTreeLearner), config, args);
BenchmarkRunner.Run(typeof(Benchmarks.Regression.AdaboostLearner), config, args);
BenchmarkRunner.Run(typeof(Benchmarks.Regression.RandomForestLearner), config, args);
BenchmarkRunner.Run(typeof(Benchmarks.Regression.ExtremelyRandomizedTreeLearner), config, args);
BenchmarkRunner.Run(typeof(Benchmarks.Regression.SquareLossGradientBoostLearner), config, args);
