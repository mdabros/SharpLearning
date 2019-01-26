using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.CrossValidation.TrainingTestSplitters;
using SharpLearning.GradientBoost.Learners;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Metrics.Regression;

namespace SharpLearning.Optimization.Test
{
    [TestClass]
    public class HyperbandOptimizerTest
    {
        [TestMethod]
        public void HyperbandOptimizer_Optimize()
        {
            // Use StreamReader(filepath) when running from filesystem
            var parser = new CsvParser(() => new StreamReader(@"E:\Git\open-source\SharpLearning.Examples\src\Resources\winequality-white.csv"));
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

            var treesPrIteration = 5;
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
            var maximunIterationsPrConfiguration = 40;
            var optimizer = new HyperbandOptimizer(parameters, maximunIterationsPrConfiguration, eta: 3);

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
    }
}
