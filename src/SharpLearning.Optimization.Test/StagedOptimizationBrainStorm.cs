using System;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using static SharpLearning.Optimization.Test.ObjectiveUtilities;

namespace SharpLearning.Optimization.Test
{
    [TestClass]
    public class StagedOptimizationBrainStorm
    {
        [TestMethod]
        public void RunExample()
        {
            var stage1Optimizer = new RandomSearchOptimizer(new MinMaxParameterSpec[]
            {
                new MinMaxParameterSpec(0.0, 100.0, Transform.Linear)
            }, 100);
            Stage stage1 = () => stage1Optimizer.Optimize(MinimizeWeightFromHeight);

            var stage2Optimizer = new BayesianOptimizer(new MinMaxParameterSpec[]
            {
                new MinMaxParameterSpec(0.0, 100.0, Transform.Linear)
            }, 100);
            Stage stage2 = () => stage2Optimizer.Optimize(MinimizeWeightFromHeight);

            var stages = new List<Stage> { stage1, stage2 };

            var stageExecutor = new StageExecutor(stages);

            var results = stageExecutor.Run();
        }
    }

    public delegate OptimizerResult[] Stage();

    public class StageExecutor
    {
        readonly IReadOnlyList<Stage> m_stages;

        public StageExecutor(IReadOnlyList<Stage> stages)
        {
            m_stages = stages ?? throw new ArgumentNullException(nameof(stages));
        }

        public List<OptimizerResult> Run()
        {
            var results = new List<OptimizerResult>();
            foreach (var stage in m_stages)
            {
                var stageResults = stage();
                results.AddRange(stageResults);
            }
            return results;
        }
    }
}
