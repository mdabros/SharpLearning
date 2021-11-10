using System.Linq;
using System.Diagnostics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Optimization.Test.StagedOptimization.OptimizerBuilders;
using static SharpLearning.Optimization.Test.ObjectiveUtilities;
using System;
using System.Collections.Generic;

namespace SharpLearning.Optimization.Test.StagedOptimization
{
    [TestClass]
    public class StagedOptimizationBrainStorm
    {
        [TestMethod]
        public void RunExample_3()
        {
            Func<IControlFlowDoer> inititFlow = () => new ControlFlowDoer();
            var stageScheduler = new ControlFlowScheduler(inititFlow);
            stageScheduler.Initialize()            
                .Do(r =>
                {
                    Trace.WriteLine("Running RandomStage");

                    var parameterSpecs = new IParameterSpec[] { new MinMaxParameterSpec(10, 100) };
                    var randomSearch = new RandomSearchOptimizer(parameterSpecs, iterations: 10);
                    var results = randomSearch.Optimize(MinimizeWeightFromHeight);

                    r.Add("randomResults", results);
                })
                .Do(r =>
                {
                    Trace.WriteLine("Running GridStage");

                    var parameterSpecs = new IParameterSpec[] { new GridParameterSpec(1, 5, 10, 20, 40, 80, 100) };
                    var gridSearch = new GridSearchOptimizer(parameterSpecs);
                    var results = gridSearch.Optimize(MinimizeWeightFromHeight);

                    r.Add("gridResults", results);
                })
                .Do(r =>
                {
                    Trace.WriteLine("Running SmacStage");

                    var randomResults = r.Get<OptimizerResult[]>("randomResults");
                    var gridResults = r.Get<OptimizerResult[]>("gridResults");

                    var previousResults = new List<OptimizerResult>();
                    previousResults.AddRange(randomResults);
                    previousResults.AddRange(gridResults);

                    var parameterSpecs = new IParameterSpec[] { new MinMaxParameterSpec(10, 100) };
                    var smac = new SmacOptimizer(parameterSpecs, iterations: 10);

                    var iterations = 10;
                    var smacResults = new List<OptimizerResult>();

                    for (int iteration = 0; iteration < iterations; iteration++)
                    {
                        var suggestedParameters = smac.ProposeParameterSets(1, previousResults);
                        var results = smac.RunParameterSets(MinimizeWeightFromHeight, suggestedParameters);
                        smacResults.AddRange(results);
                        previousResults.AddRange(results);
                    }
                    r.Add("smacResults", smacResults);
                });

            var repository = stageScheduler.Execute();
            var randomBest = repository.Get<OptimizerResult[]>("randomResults");
            var gridBest = repository.Get<OptimizerResult[]>("gridResults");
            var smacBest = repository.Get<List<OptimizerResult>> ("smacResults");

            Trace.WriteLine("RandomBest: " + randomBest.OrderBy(r => r.Error).First().Error);
            Trace.WriteLine("GridBest: " + gridBest.OrderBy(r => r.Error).First().Error);
            Trace.WriteLine("SmacBest: " + smacBest.OrderBy(r => r.Error).First().Error);
        }

        [TestMethod]
        public void RunExample_2()
        {

            // StagedOptimization.New()
            //  .AddStage()
            //  .AddSearchRange(new MinMaxParameterSpec)
            //  .AddSearchrange(new MinMaxParameterSpec)
            //  .AddOptimizer(r => new RandomSearch(r))
            //  .AddStage()
            //  .AddSearchRange(new GridParameterSpec())
            //  .AddOptimizer(r => new GridSearch(r))
            //  .AddStage()
            //  .AddRange(new GridParameterSpec())
            //  .AddOptimier(r => 
            //  {
            //    for... //custom optimizer
            //  }
            //  .AsStagedOptimization() // .AsCompositeOptimizer;
            //
            //

            var runOptimizer = CreateOptimizerBuilder.New()
                .AddParameterSpec(new MinMaxParameterSpec(10, 100, Transform.Linear))
                .BuildOptimizer()
                .AddOptimizer(r => new RandomSearchOptimizer(r, iterations: 100))
                .BuildObjectiveFunction()
                .AddObjective(MinimizeWeightFromHeight)
                .BuildRunOptimizer();

            var results = runOptimizer();
            var best = results.OrderBy(r => r.Error).First();

            Trace.WriteLine($"Error: {best.Error}. Parameters: {string.Join(", ", best.ParameterSet)}");
        }
    }
}
