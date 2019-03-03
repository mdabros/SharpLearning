using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using static SharpLearning.Optimization.Test.ObjectiveUtilities;

namespace SharpLearning.Optimization.Test
{
    [TestClass]
    public class SmacOptimizerTest
    {
        [TestMethod]
        public void SmacOptimizer_OptimizeBest_SingleParameter()
        {
            var parameters = new MinMaxParameterSpec[]
            {
                new MinMaxParameterSpec(0.0, 100.0, Transform.Linear)
            };

            var sut = new SmacOptimizer(parameters,
                iterations: 80,
                randomStartingPointCount: 20,
                functionEvaluationsPerIterationCount: 1,
                localSearchPointCount: 10,
                randomSearchPointCount: 1000,
                seed: 42);

            var actual = sut.OptimizeBest(MinimizeWeightFromHeight);

            Assert.AreEqual(109.616853578648, actual.Error, Delta);
            Assert.AreEqual(37.6315924979893, actual.ParameterSet.Single(), Delta);
        }

        [TestMethod]
        public void SmacOptimizer_OptimizeBest_MultipleParameters()
        {
            var parameters = new MinMaxParameterSpec[]
            {
                new MinMaxParameterSpec(-10.0, 10.0, Transform.Linear),
                new MinMaxParameterSpec(-10.0, 10.0, Transform.Linear),
                new MinMaxParameterSpec(-10.0, 10.0, Transform.Linear),
            };

            var sut = new SmacOptimizer(parameters,
                iterations: 80,
                randomStartingPointCount: 20,
                functionEvaluationsPerIterationCount: 1,
                localSearchPointCount: 10,
                randomSearchPointCount: 1000,
                seed: 42);

            var actual = sut.OptimizeBest(Minimize);

            Assert.AreEqual(-0.964878416222769, actual.Error, Delta);
            Assert.AreEqual(actual.ParameterSet.Length, 3);

            Assert.AreEqual(-7.8487638560350819, actual.ParameterSet[0], Delta);
            Assert.AreEqual(6.2840940040927826, actual.ParameterSet[1], Delta);
            Assert.AreEqual(0.036385473812179825, actual.ParameterSet[2], Delta);
        }

        [TestMethod]
        public void SmacOptimizer_Optimize()
        {
            var parameters = new MinMaxParameterSpec[]
            {
                new MinMaxParameterSpec(0.0, 100.0, Transform.Linear)
            };

            var sut = new SmacOptimizer(parameters,
                iterations: 80,
                randomStartingPointCount: 20,
                functionEvaluationsPerIterationCount: 1,
                localSearchPointCount: 10,
                randomSearchPointCount: 1000,
                seed: 42);

            var actual = sut.Optimize(MinimizeWeightFromHeight);

            var expected = new OptimizerResult[]
            {
                new OptimizerResult(new double[] { 90.513222660177 }, 114559.431919558),
                new OptimizerResult(new double[] { 41.8333740634068 },  806.274612132759),
            };

            Assert.AreEqual(expected.First().Error, actual.First().Error, Delta);
            Assert.AreEqual(expected.First().ParameterSet.First(), actual.First().ParameterSet.First(), Delta);

            Assert.AreEqual(expected.Last().Error, actual.Last().Error, Delta);
            Assert.AreEqual(expected.Last().ParameterSet.First(), actual.Last().ParameterSet.First(), Delta);
        }

        [TestMethod]
        public void SmacOptimizer_OptimizeBest_MultipleParameters_Open_Loop()
        {
            var emptyResults = new List<OptimizerResult>();

            OptimizerResult actual = RunOpenLoopOptimizationTest(emptyResults);

            Assert.AreEqual(-0.964878416222769, actual.Error, Delta);
            Assert.AreEqual(actual.ParameterSet.Length, 3);

            Assert.AreEqual(-7.8487638560350819, actual.ParameterSet[0], Delta);
            Assert.AreEqual(6.2840940040927826, actual.ParameterSet[1], Delta);
            Assert.AreEqual(0.036385473812179825, actual.ParameterSet[2], Delta);
        }

        [TestMethod]
        public void SmacOptimizer_OptimizeBest_MultipleParameters_Open_Loop_Using_PreviousResults()
        {
            var previousResults = new List<OptimizerResult>()
            {
                new OptimizerResult(new[] {-6.83357586936726,6.0834837966056,-0.0766206300242906}, -0.476143174040315),
                new OptimizerResult(new[] {-7.29391428515963,6.0834837966056,1.01057317620636}, -0.41300737879641),
                new OptimizerResult(new[] {-7.29391428515963,6.0834837966056,1.01057317620636}, -0.41300737879641),
                new OptimizerResult(new[] {-8.05557010604794,-5.14662256238359,0.0363854738121798}, -0.397724266113204),
                new OptimizerResult(new[] {-8.06241082868651,5.88012208038947,-1.5210571566229}, -0.356975377788698),
                new OptimizerResult(new[] {4.42408777513732,0.472018332440413,1.7076749781648}, -0.315360461074171),
                new OptimizerResult(new[] {-8.14483470197061,7.54724840519356,0.0363854738121798}, -0.279108605472165),
                new OptimizerResult(new[] {-6.64746686660101,6.7109944004151,-0.214493549528761}, -0.266917186594653),
                new OptimizerResult(new[] {5.34224593795009,-6.45170816986435,-2.1147669628797}, -0.255769932489526),
                new OptimizerResult(new[] {-7.84876385603508,6.28409400409278,3.1447921661403}, -0.241263236969342),
                new OptimizerResult(new[] {-7.84876385603508,4.96554990995934,-0.0766206300242906}, -0.232637166385485),
                new OptimizerResult(new[] {-8.14041409554911,7.16927772256047,1.75166608381628}, -0.220476103560048),
                new OptimizerResult(new[] {-7.84876385603508,6.0834837966056,-3.60210045874217}, -0.212970686239402),
                new OptimizerResult(new[] {-7.29391428515963,5.22505613752876,1.01057317620636}, -0.206689239504653),
                new OptimizerResult(new[] {-9.20479206331297,6.0834837966056,-0.0766206300242906}, -0.198657722521128),
                new OptimizerResult(new[] {-8.25145286426481,5.27274844947865,-1.82163462593296}, -0.17367847378187),
                new OptimizerResult(new[] {-7.84876385603508,6.0834837966056,5.3824106023565}, -0.153564625328103),
                new OptimizerResult(new[] {-1.37364300497511,-1.35665034472786,-0.585322245296707}, -0.131453543138338),
                new OptimizerResult(new[] {-7.84876385603508,7.74187722138216,-0.0766206300242906}, -0.103906821017427),
                new OptimizerResult(new[] {9.20868899636375,-9.38389458664874,1.51842798642741}, -0.0850657757130275),
                new OptimizerResult(new[] {-7.72406242681856,5.70825177044992,9.95585092341334}, -0.0759553721161318),
                new OptimizerResult(new[] {1.65093947744506,-4.37866264692445,-4.29402069854272}, -0.0616761163702651),
                new OptimizerResult(new[] {-9.37414173938993,6.28409400409278,0.0363854738121798}, -0.0488375857853505),
                new OptimizerResult(new[] {3.38691201684387,5.42095644186295,-5.71318443664964}, -0.0235423806080941),
                new OptimizerResult(new[] {-6.48224856540665,-7.13935053774125,7.05507751417117}, -0.0160884883078408),
                new OptimizerResult(new[] {-9.68539061941457,7.96346846873102,-0.990608674935348}, -0.0141441279734299),
                new OptimizerResult(new[] {-9.41382774124566,5.12580713030221,0.630654976996897}, -0.00269773409680873),
                new OptimizerResult(new[] {6.7694738305963,1.56629731485913,-2.12145430600338}, 0.000673595210828553),
                new OptimizerResult(new[] {-0.0282478006688169,2.87566112022645,-4.84997700660023}, 0.00465834522866944),
                new OptimizerResult(new[] {3.50054986472267,8.01269467827524,7.36471213277649}, 0.00663762309484885),
                new OptimizerResult(new[] {3.05129390817662,-6.16640157819092,7.49125691013935}, 0.0105475373675896),
            };

            OptimizerResult actual = RunOpenLoopOptimizationTest(previousResults);

            Assert.AreEqual(-0.96958858653084612, actual.Error, Delta);
            Assert.AreEqual(actual.ParameterSet.Length, 3);

            Assert.AreEqual(-1.5411651281365977, actual.ParameterSet[0], Delta);
            Assert.AreEqual(6.3397464232238683, actual.ParameterSet[1], Delta);
            Assert.AreEqual(-0.029263948104000903, actual.ParameterSet[2], Delta);
        }

        OptimizerResult RunOpenLoopOptimizationTest(List<OptimizerResult> results)
        {
            var parameters = new MinMaxParameterSpec[]
            {
                new MinMaxParameterSpec(-10.0, 10.0, Transform.Linear),
                new MinMaxParameterSpec(-10.0, 10.0, Transform.Linear),
                new MinMaxParameterSpec(-10.0, 10.0, Transform.Linear),
            };

            var iterations = 80;
            var randomStartingPointsCount = 20;
            var functionEvaluationsPerIterationCount = 1;

            var sut = new SmacOptimizer(parameters,
                iterations: iterations,
                randomStartingPointCount: randomStartingPointsCount,
                functionEvaluationsPerIterationCount: functionEvaluationsPerIterationCount,
                localSearchPointCount: 10,
                randomSearchPointCount: 1000,
                seed: 42);

            // Using SmacOptimizer in an open loop.
            var initialParameterSets = sut.ProposeParameterSets(randomStartingPointsCount, results);
            var initializationResults = sut.RunParameterSets(Minimize, initialParameterSets);
            results.AddRange(initializationResults);

            for (int i = 0; i < iterations; i++)
            {
                var parameterSets = sut.ProposeParameterSets(functionEvaluationsPerIterationCount, results);
                var iterationResults = sut.RunParameterSets(Minimize, parameterSets);
                results.AddRange(iterationResults);
            }

            return results.Where(v => !double.IsNaN(v.Error)).OrderBy(r => r.Error).First();
        }
    }
}
