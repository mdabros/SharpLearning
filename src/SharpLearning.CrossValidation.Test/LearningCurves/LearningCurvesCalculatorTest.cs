using System;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Matrices;
using SharpLearning.CrossValidation.LearningCurves;
using SharpLearning.CrossValidation.Samplers;
using SharpLearning.CrossValidation.TrainingTestSplitters;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.Metrics.Regression;

namespace SharpLearning.CrossValidation.Test.LearningCurves
{
    [TestClass]
    public class LearningCurvesCalculatorTest
    {
        [TestMethod]
        public void LearningCurvesCalculator_Calculate()
        {
            var sut = new LearningCurvesCalculator<double>(
                new RandomTrainingTestIndexSplitter<double>(0.8, 42),
                new RandomIndexSampler<double>(42), 
                new MeanSquaredErrorRegressionMetric(), 
                new double[] { 0.2, 0.8 });

            var (observations, targets) = DataSetUtilities.LoadDecisionTreeDataSet();

            var actual = sut.Calculate(new RegressionDecisionTreeLearner(),
                observations, targets);

            var expected = new List<LearningCurvePoint>()
            {
                new LearningCurvePoint(32, 0, 0.141565953928265), 
                new LearningCurvePoint(128, 0.0, 0.068970597423950036)
            };
            
            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void LearningCurvesCalculator_Calculate_Indices_Provided()
        {
            var splitter = new RandomTrainingTestIndexSplitter<double>(0.8, 42);
            
            var sut = new LearningCurvesCalculator<double>(
                splitter, 
                new RandomIndexSampler<double>(42),
                new MeanSquaredErrorRegressionMetric(), 
                new double[] { 0.2, 0.8 });

            var (observations, targets) = DataSetUtilities.LoadDecisionTreeDataSet();
            var indexSplits = splitter.Split(targets);

            var actual = sut.Calculate(new RegressionDecisionTreeLearner(),
                observations, targets, indexSplits.TrainingIndices, indexSplits.TestIndices);

            var expected = new List<LearningCurvePoint>()
            {
                new LearningCurvePoint(32, 0, 0.141565953928265), 
                new LearningCurvePoint(128, 0.0, 0.068970597423950036)
            };

            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void LearningCurvesCalculator_Calculate_Metric_Null()
        {
            new LearningCurvesCalculator<double>(
                new RandomTrainingTestIndexSplitter<double>(0.8, 42),
                new RandomIndexSampler<double>(42),
                null, 
                new double[] { 0.2, 0.8 } );
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void LearningCurvesCalculator_Calculate_Sample_Percentages_Null()
        {
            new LearningCurvesCalculator<double>(
                new RandomTrainingTestIndexSplitter<double>(0.8, 42),
                new RandomIndexSampler<double>(42),
                new MeanSquaredErrorRegressionMetric(), 
                null);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void LearningCurvesCalculator_Calculate_Sample_Percentages_Empty()
        {
            new LearningCurvesCalculator<double>(
                new RandomTrainingTestIndexSplitter<double>(0.8, 42),
                new RandomIndexSampler<double>(42),
                new MeanSquaredErrorRegressionMetric(), 
                new double[] { });
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void LearningCurvesCalculator_Calculate_Sample_Percentage_Too_Low()
        {
            var sut = new LearningCurvesCalculator<double>(
                new RandomTrainingTestIndexSplitter<double>(0.8, 42),
                new RandomIndexSampler<double>(42),
                new MeanSquaredErrorRegressionMetric(), 
                new double[] { 0.0, 0.8 });

            var observations = new F64Matrix(10, 10);
            var targets = new double[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

            sut.Calculate(new RegressionDecisionTreeLearner(),
                observations, targets);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void LearningCurvesCalculator_Calculate_Sample_Percentage_Too_High()
        {
            var sut = new LearningCurvesCalculator<double>(
                new RandomTrainingTestIndexSplitter<double>(0.8, 42),
                new RandomIndexSampler<double>(42),
                new MeanSquaredErrorRegressionMetric(), 
                new double[] { 1.1, 0.8 });

            var observations = new F64Matrix(10, 10);
            var targets = new double[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

            sut.Calculate(new RegressionDecisionTreeLearner(),
                observations, targets);
        } 
    }
}
