using System.Linq;
using System.Diagnostics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using static SharpLearning.Optimization.Test.ObjectiveUtilities;

namespace SharpLearning.Optimization.Test.StagedOptimization
{
    [TestClass]
    public class StagedOptimizationBrainStorm
    {
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

            var runOptimizer = StagedOptimization.New()
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
