using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers;
using SharpLearning.Containers.Matrices;
using SharpLearning.CrossValidation.BiasVarianceAnalysis;
using SharpLearning.CrossValidation.Shufflers;
using SharpLearning.CrossValidation.Test.Properties;
using SharpLearning.CrossValidation.TrainingValidationSplitters;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Metrics.Regression;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace SharpLearning.CrossValidation.Test.BiasVarianceAnalysis
{
    [TestClass]
    public class BiasVarianceLearningCurvesCalculatorTest
    {
        [TestMethod]
        public void BiasVarianceLearningCurvesCalculator_Calculate()
        {
            var sut = new BiasVarianceLearningCurvesCalculator<double>(new RandomTrainingValidationIndexSplitter<double>(0.8, 42),
                new RandomCrossValidationShuffler<double>(42), new MeanSquaredErrorRegressionMetric(), new double[] { 0.2, 0.8 });

            var targetName = "T";
            var parser = new CsvParser(() => new StringReader(Resources.DecisionTreeData));
            var observations = parser.EnumerateRows(v => !v.Contains(targetName)).ToF64Matrix();
            var targets = parser.EnumerateRows(targetName).ToF64Vector();

            var actual = sut.Calculate(new RegressionDecisionTreeLearner(),
                observations, targets);

            var expected = new List<BiasVarianceLearningCurvePoint>() { new BiasVarianceLearningCurvePoint(32, 0, 0.19281116525022002), 
                new BiasVarianceLearningCurvePoint(128, 0.0, 0.09414342143248)};
            
            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void BiasVarianceLearningCurvesCalculator_Calculate_Indices_Provided()
        {
            var splitter = new RandomTrainingValidationIndexSplitter<double>(0.8, 42);
            
            var sut = new BiasVarianceLearningCurvesCalculator<double>(splitter, new RandomCrossValidationShuffler<double>(42),
                new MeanSquaredErrorRegressionMetric(), new double[] { 0.2, 0.8 });

            var targetName = "T";
            var parser = new CsvParser(() => new StringReader(Resources.DecisionTreeData));
            var observations = parser.EnumerateRows(v => !v.Contains(targetName)).ToF64Matrix();
            var targets = parser.EnumerateRows(targetName).ToF64Vector();
            var indexSplits = splitter.Split(targets);

            var actual = sut.Calculate(new RegressionDecisionTreeLearner(),
                observations, targets, indexSplits.TrainingIndices, indexSplits.ValidationIndices);

            var expected = new List<BiasVarianceLearningCurvePoint>() { new BiasVarianceLearningCurvePoint(32, 0, 0.19281116525022002), 
                new BiasVarianceLearningCurvePoint(128, 0.0, 0.09414342143248)};

            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void BiasVarianceLearningCurvesCalculator_Calculate_Metric_Null()
        {
            new BiasVarianceLearningCurvesCalculator<double>(new RandomTrainingValidationIndexSplitter<double>(0.8, 42),
                new RandomCrossValidationShuffler<double>(42),
                null, new double[] { 0.2, 0.8 } );
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void BiasVarianceLearningCurvesCalculator_Calculate_Sample_Percentages_Null()
        {
            new BiasVarianceLearningCurvesCalculator<double>(new RandomTrainingValidationIndexSplitter<double>(0.8, 42),
                new RandomCrossValidationShuffler<double>(42),
                new MeanSquaredErrorRegressionMetric(), null);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void BiasVarianceLearningCurvesCalculator_Calculate_Sample_Percentages_Empty()
        {
            new BiasVarianceLearningCurvesCalculator<double>(new RandomTrainingValidationIndexSplitter<double>(0.8, 42),
                new RandomCrossValidationShuffler<double>(42),
                new MeanSquaredErrorRegressionMetric(), new double[] { });
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void BiasVarianceLearningCurvesCalculator_Calculate_Sample_Percentage_Too_Low()
        {
            var sut = new BiasVarianceLearningCurvesCalculator<double>(new RandomTrainingValidationIndexSplitter<double>(0.8, 42),
                new RandomCrossValidationShuffler<double>(42),
                new MeanSquaredErrorRegressionMetric(), new double[] { 0.0, 0.8 });

            var observations = new F64Matrix(10, 10);
            var targets = new double[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

            sut.Calculate(new RegressionDecisionTreeLearner(),
                observations, targets);

        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void BiasVarianceLearningCurvesCalculator_Calculate_Sample_Percentage_Too_High()
        {
            var sut = new BiasVarianceLearningCurvesCalculator<double>(new RandomTrainingValidationIndexSplitter<double>(0.8, 42),
                new RandomCrossValidationShuffler<double>(42),
                new MeanSquaredErrorRegressionMetric(), new double[] { 1.1, 0.8 });

            var observations = new F64Matrix(10, 10);
            var targets = new double[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

            sut.Calculate(new RegressionDecisionTreeLearner(),
                observations, targets);
        } 
    }
}
