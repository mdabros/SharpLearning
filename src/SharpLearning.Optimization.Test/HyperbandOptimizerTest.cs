using System;
using System.Linq;
using System.Diagnostics;
using System.IO;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using static SharpLearning.Optimization.Test.ObjectiveUtilities;

namespace SharpLearning.Optimization.Test
{
    [TestClass]
    public class HyperbandOptimizerTest
    {
        [TestMethod]
        public void HyperbandOptimizer_test()
        {
            // Use StreamReader(filepath) when running from filesystem
            var parser = new CsvParser(() => new StreamReader(@"E:\Git\SharpLearning.Examples\src\Resources\winequality-white.csv"));
            var targetName = "quality";

            // read feature matrix
            var observations = parser.EnumerateRows(c => c != targetName)
                .ToF64Matrix();

            // read regression targets
            var targets = parser.EnumerateRows(targetName)
                .ToF64Vector();

            // creates training test splitter, 
            // Since this is a regression problem, we use the random training/test set splitter.
            // 30 % of the data is used for the test set. 
            var splitter = new RandomTrainingTestIndexSplitter<double>(trainingPercentage: 0.7, seed: 24);

            var trainingTestSplit = splitter.SplitSet(observations, targets);
            var trainSet = trainingTestSplit.TrainingSet;
            var testSet = trainingTestSplit.TestSet;

            // since this is a regression problem we are using square error as metric
            // for evaluating how well the model performs.
            var metric = new MeanSquaredErrorRegressionMetric();

            // Usually better results can be achieved by tuning a gradient boost learner

            var numberOfFeatures = trainSet.Observations.ColumnCount;

            // Parameter ranges for the optimizer
            // best parameter to tune on random forest is featuresPrSplit.
            var parameters = new IParameterSpec[]
            {
                //new MinMaxParameterSpec(min: 80, max: 300, transform: Transform.Linear), // iterations
                new MinMaxParameterSpec(min: 0.02, max:  0.2, transform: Transform.Log10), // learning rate
                new MinMaxParameterSpec(min: 8, max: 15, transform: Transform.Linear), // maximumTreeDepth
                new MinMaxParameterSpec(min: 0.5, max: 0.9, transform: Transform.Linear), // subSampleRatio
                new MinMaxParameterSpec(min: numberOfFeatures * 0.5, max: numberOfFeatures, transform: Transform.Linear), // featuresPrSplit
            };

            // Further split the training data to have a validation set to measure
            // how well the model generalizes to unseen data during the optimization.
            var validationSplit = new RandomTrainingTestIndexSplitter<double>(trainingPercentage: 0.7, seed: 24)
                .SplitSet(trainSet.Observations, trainSet.Targets);

            var treesPrIteration = 3;
            var random = new Random(343);
            // Define optimizer objective (function to minimize)
            HyperbandObjectiveFunction minimize = (p, r) =>
            {
                //return new OptimizerResult(p, random.NextDouble());
                // create the candidate learner using the current optimization parameters.
                var candidateLearner = new RegressionSquareLossGradientBoostLearner(
                        iterations: (int)(treesPrIteration * r),
                        learningRate: p[0],
                        maximumTreeDepth: (int)p[1],
                        subSampleRatio: p[2],
                        featuresPrSplit: (int)p[3],
                        runParallel: false);

                var candidateModel = candidateLearner.Learn(validationSplit.TrainingSet.Observations,
                   validationSplit.TrainingSet.Targets);
                var validationPredictions = candidateModel.Predict(validationSplit.TestSet.Observations);
                var candidateError = metric.Error(validationSplit.TestSet.Targets, validationPredictions);

                return new OptimizerResult(p, candidateError);
            };

            // create random search optimizer
            var maximunIterationsPrConfiguration = 81;
            var optimizer = new BOHPOptimizer(parameters, maximunIterationsPrConfiguration, eta: 3);

            var timer = new Stopwatch();

            timer.Start();
            // find best hyperparameters
            var result = optimizer.Optimize(minimize);
            timer.Stop();

            var best = result.OrderBy(r => r.Error).First();

            Trace.WriteLine($"Best Error: {best.Error} | Parameters: {string.Join(", ", best.ParameterSet)}");
            Trace.WriteLine($"Total configuration count: {result.Count()}");
            Trace.WriteLine($"Total time (ms): {timer.ElapsedMilliseconds}");

            // create the final learner using the best hyperparameters.
            var b = best.ParameterSet;
            var learner = new RegressionSquareLossGradientBoostLearner(
                        iterations: (int)(treesPrIteration * maximunIterationsPrConfiguration),
                        learningRate: b[0],
                        maximumTreeDepth: (int)b[1],
                        subSampleRatio: b[2],
                        featuresPrSplit: (int)b[3],
                        runParallel: false);

            // learn model with found parameters
            var model = learner.Learn(trainSet.Observations, trainSet.Targets);

            // predict the training and test set.
            var trainPredictions = model.Predict(trainSet.Observations);
            var testPredictions = model.Predict(testSet.Observations);

            // measure the error on training and test set.
            var trainError = metric.Error(trainSet.Targets, trainPredictions);
            var testError = metric.Error(testSet.Targets, testPredictions);

            // Optimizer found hyperparameters.
            Trace.WriteLine(string.Format($"Test Error: {testError:0.00000}."));

            throw new NotImplementedException();
        }
    

        [TestMethod]
        public void HyperbandOptimizer_OptimizeBest()
        {
            var parameters = new IParameterSpec[]
            {
                new MinMaxParameterSpec(min: 80, max: 300, transform: Transform.Linear),
                new MinMaxParameterSpec(min: 0.02, max:  0.2, transform: Transform.Log10),
                new MinMaxParameterSpec(min: 8, max: 15, transform: Transform.Linear),
            };

            var random = new Random(343);
            HyperbandObjectiveFunction minimize = (p, r) =>
            {
                var error = random.NextDouble();
                return new OptimizerResult(p, error);
            };

            var sut = new HyperbandOptimizer(
                parameters,
                maximumBudget: 81,
                eta: 5,
                skipLastIterationOfEachRound: false,
                seed: 34);

            var actual = sut.OptimizeBest(minimize);
            var expected = new OptimizerResult(new[] { 278.337940, 0.098931, 13.177449 }, 0.009549);

            AssertOptimizerResult(expected, actual);
        }

        [TestMethod]
        public void HyperbandOptimizer_Optimize()
        {
            var parameters = new IParameterSpec[]
            {
                new MinMaxParameterSpec(min: 80, max: 300, transform: Transform.Linear),
                new MinMaxParameterSpec(min: 0.02, max:  0.2, transform: Transform.Log10),
                new MinMaxParameterSpec(min: 8, max: 15, transform: Transform.Linear),
            };

            var random = new Random(343);
            HyperbandObjectiveFunction minimize = (p, r) =>
            {
                var error = random.NextDouble();
                return new OptimizerResult(p, error);
            };
        
            var sut = new HyperbandOptimizer(
                parameters, 
                maximumBudget: 81, 
                eta: 5, 
                skipLastIterationOfEachRound: false,
                seed: 34);

            var actual = sut.Optimize(minimize);

            AssertOptimizerResults(Expected, actual);
        }

        static void AssertOptimizerResults(OptimizerResult[] expected, OptimizerResult[] actual)
        {
            Assert.AreEqual(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
            {
                AssertOptimizerResult(expected[i], actual[i]);
            }
        }

        static void AssertOptimizerResult(OptimizerResult expected, OptimizerResult actual)
        {
            Assert.AreEqual(expected.Error, actual.Error, Delta);

            var expectedParameterSet = expected.ParameterSet;
            var actualParameterSet = actual.ParameterSet;

            Assert.AreEqual(expectedParameterSet.Length, actualParameterSet.Length);

            for (int i = 0; i < expectedParameterSet.Length; i++)
            {
                Assert.AreEqual(expectedParameterSet[i], actualParameterSet[i], Delta);
            }
        }

        static OptimizerResult[] Expected => new OptimizerResult[]
        {
            new OptimizerResult(new [] { 183.050454, 0.103778, 10.521202 }, 0.815090),
            new OptimizerResult(new [] { 242.035154, 0.160589, 14.928944 }, 0.319450),
            new OptimizerResult(new [] { 217.110439, 0.121371, 9.134293 }, 0.287873),
            new OptimizerResult(new [] { 205.828006, 0.026428, 13.831848 }, 0.213150),
            new OptimizerResult(new [] { 81.318916, 0.028789, 13.468363 }, 0.833401),
            new OptimizerResult(new [] { 183.050454, 0.103778, 10.521202 }, 0.138057),
            new OptimizerResult(new [] { 280.115839, 0.043236, 14.109365 }, 0.315902),
            new OptimizerResult(new [] { 199.842478, 0.023487, 12.218300 }, 0.858262),
            new OptimizerResult(new [] { 89.288205, 0.029247, 12.503943 }, 0.960621),
            new OptimizerResult(new [] { 238.527937, 0.023610, 14.521096 }, 0.998539),
            new OptimizerResult(new [] { 103.184215, 0.048606, 11.929732 }, 0.391503),
            new OptimizerResult(new [] { 217.110439, 0.121371, 9.134293 }, 0.125866),
            new OptimizerResult(new [] { 80.598836, 0.039832, 8.388401 }, 0.962324),
            new OptimizerResult(new [] { 89.359300, 0.042719, 10.902781 }, 0.655116),
            new OptimizerResult(new [] { 183.050454, 0.103778, 10.521202 }, 0.045531),
            new OptimizerResult(new [] { 242.035154, 0.160589, 14.928944 }, 0.241034),
            new OptimizerResult(new [] { 205.828006, 0.026428, 13.831848 }, 0.072501),
            new OptimizerResult(new [] { 137.807164, 0.080876, 9.133881 }, 0.917069),
            new OptimizerResult(new [] { 122.739555, 0.071284, 9.159947 }, 0.428372),
            new OptimizerResult(new [] { 265.007895, 0.065434, 9.655193 }, 0.252369),
            new OptimizerResult(new [] { 242.616914, 0.051308, 14.785707 }, 0.990477),
            new OptimizerResult(new [] { 245.944001, 0.173415, 11.243352 }, 0.755331),
            new OptimizerResult(new [] { 87.069973, 0.049606, 9.162192 }, 0.412378),
            new OptimizerResult(new [] { 121.689890, 0.109421, 14.372696 }, 0.519928),
            new OptimizerResult(new [] { 211.466343, 0.060338, 10.341543 }, 0.589474),
            new OptimizerResult(new [] { 138.097042, 0.028550, 8.527269 }, 0.305832),
            new OptimizerResult(new [] { 81.318916, 0.028789, 13.468363 }, 0.065642),
            new OptimizerResult(new [] { 258.473191, 0.043830, 8.081241 }, 0.769086),
            new OptimizerResult(new [] { 110.790052, 0.063165, 9.287423 }, 0.520903),
            new OptimizerResult(new [] { 259.348583, 0.072041, 9.899872 }, 0.459911),
            new OptimizerResult(new [] { 187.514870, 0.124334, 11.735301 }, 0.918126),
            new OptimizerResult(new [] { 80.806287, 0.028735, 9.547892 }, 0.824839),
            new OptimizerResult(new [] { 212.130398, 0.142035, 8.342675 }, 0.713911),
            new OptimizerResult(new [] { 212.130398, 0.142035, 8.342675 }, 0.082547),
            new OptimizerResult(new [] { 80.806287, 0.028735, 9.547892, }, 0.135099),
            new OptimizerResult(new [] { 119.813471, 0.074485, 13.382158 }, 0.154206),
            new OptimizerResult(new [] { 202.034806, 0.137801, 9.508964 },0.627903),
            new OptimizerResult(new [] { 102.696143, 0.099462, 8.557010 },0.410965),
            new OptimizerResult(new [] { 118.759207, 0.038629, 9.560888 },0.587768),
            new OptimizerResult(new [] { 96.998060, 0.039504, 11.428746 },0.225692),
            new OptimizerResult(new [] { 117.955108, 0.082906, 12.319315 }, 0.801867),
            new OptimizerResult(new [] { 246.662655, 0.027162, 14.963403 }, 0.088704),
            new OptimizerResult(new [] { 156.214348, 0.167765, 12.516866 }, 0.365275),
            new OptimizerResult(new [] { 278.337940, 0.098931, 13.177449 }, 0.009549),
        };
    }
}
