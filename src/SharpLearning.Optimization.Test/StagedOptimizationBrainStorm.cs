using System;
using System.Linq;
using System.Collections.Generic;
using System.Diagnostics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using static SharpLearning.Optimization.Test.ObjectiveUtilities;

namespace SharpLearning.Optimization.Test
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

        public static class StagedOptimization
        {
            public static IParameterSpecsBuilder New()
            {
                return new AddParameterSpecBuilder();
            }
        }

        public delegate IOptimizer CreateOptimizer(IParameterSpec[] parameterSpecs);
        public delegate OptimizerResult ObjectiveFunction(double[] parameters);
        public delegate OptimizerResult[] RunOptimizer();

        public interface IRunOptimizerBuilder
        {
            IRunOptimizerBuilder AddObjective(ObjectiveFunction objectiveFunction);
            RunOptimizer BuildRunOptimizer();
        }

        public class RunOptimizerBuilder : IRunOptimizerBuilder
        {
            readonly IOptimizer m_optimizer;
            ObjectiveFunction m_objectiveFunction;
            
            public RunOptimizerBuilder(IOptimizer optimizer)
            {
                m_optimizer = optimizer ?? throw new ArgumentNullException(nameof(optimizer));
            }

            public IRunOptimizerBuilder AddObjective(ObjectiveFunction objectiveFunction)
            {
                m_objectiveFunction = objectiveFunction;
                return this;
            }

            public RunOptimizer BuildRunOptimizer() => 
                () => m_optimizer.Optimize(p => m_objectiveFunction(p));
        }

        public interface IOptimizerBuilder
        {
            IOptimizerBuilder AddOptimizer(CreateOptimizer createOptimizer);
            IOptimizer BuildOptimizer();
            IRunOptimizerBuilder BuildObjectiveFunction();
        }

        public class OptimizerBuilder : IOptimizerBuilder
        {
            readonly IParameterSpec[] m_parameterSpecs;
            CreateOptimizer m_createOptimizer;
            
            public OptimizerBuilder(IParameterSpec[] parameterSpecs)
            {
                m_parameterSpecs = parameterSpecs ?? throw new ArgumentNullException(nameof(parameterSpecs));
            }
            
            public IOptimizerBuilder AddOptimizer(CreateOptimizer createOptimizer)
            {
                m_createOptimizer = createOptimizer;
                return this;
            }

            public IOptimizer BuildOptimizer() => m_createOptimizer(m_parameterSpecs);
            public IRunOptimizerBuilder BuildObjectiveFunction() => new RunOptimizerBuilder(BuildOptimizer());
        }

        public interface IParameterSpecsBuilder
        {
            IParameterSpecsBuilder AddParameterSpec(IParameterSpec parameterSpec);
            IParameterSpec[] BuildParameterSpecs();
            IOptimizerBuilder BuildOptimizer();
        }

        public class AddParameterSpecBuilder : IParameterSpecsBuilder
        {
            readonly List<IParameterSpec> m_parameterSpecs = new List<IParameterSpec>();

            public IParameterSpecsBuilder AddParameterSpec(IParameterSpec parameterSpec)
            {
                m_parameterSpecs.Add(parameterSpec);
                return this;
            }

            public IParameterSpec[] BuildParameterSpecs() => m_parameterSpecs.ToArray();
            public IOptimizerBuilder BuildOptimizer() => new OptimizerBuilder(BuildParameterSpecs());
        }

        [TestMethod]
        public void RunExample_1()
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
