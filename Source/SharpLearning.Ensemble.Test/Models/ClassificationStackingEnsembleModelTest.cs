using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.InputOutput.Csv;
using System.IO;
using SharpLearning.Ensemble.Test.Properties;
using SharpLearning.Containers;
using SharpLearning.Common.Interfaces;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.Ensemble.Learners;
using SharpLearning.Metrics.Classification;
using SharpLearning.CrossValidation.CrossValidators;
using System.Collections.Generic;
using System.Linq;
using System.Diagnostics;

namespace SharpLearning.Ensemble.Test.Models
{
    [TestClass]
    public class ClassificationStackingEnsembleModelTest
    {
        [TestMethod]
        public void ClassificationStackingEnsembleModel_Predict_single()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var learners = new IIndexedLearner<ProbabilityPrediction>[]
            {
                new ClassificationDecisionTreeLearner(2),
                new ClassificationDecisionTreeLearner(5),
                new ClassificationDecisionTreeLearner(7),
                new ClassificationDecisionTreeLearner(9)
            };

            var learner = new ClassificationStackingEnsembleLearner(learners, new ClassificationDecisionTreeLearner(9),
                new RandomCrossValidation<ProbabilityPrediction>(5, 23), false); 
            
            var sut = learner.Learn(observations, targets);

            var predictions = new double[rows];
            for (int i = 0; i < rows; i++)
            {
                predictions[i] = sut.Predict(observations.Row(i));
            }

            var metric = new TotalErrorClassificationMetric<double>();
            var actual = metric.Error(targets, predictions);

            Assert.AreEqual(0.34615384615384615, actual, 0.0000001);
        }

        [TestMethod]
        public void ClassificationStackingEnsembleModel_Predict_Multiple()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var learners = new IIndexedLearner<ProbabilityPrediction>[]
            {
                new ClassificationDecisionTreeLearner(2),
                new ClassificationDecisionTreeLearner(5),
                new ClassificationDecisionTreeLearner(7),
                new ClassificationDecisionTreeLearner(9)
            };

            var learner = new ClassificationStackingEnsembleLearner(learners, new ClassificationDecisionTreeLearner(9),
                new RandomCrossValidation<ProbabilityPrediction>(5, 23), false);
            
            var sut = learner.Learn(observations, targets);

            var predictions = sut.Predict(observations);

            var metric = new TotalErrorClassificationMetric<double>();
            var actual = metric.Error(targets, predictions);

            Assert.AreEqual(0.34615384615384615, actual, 0.0000001);
        }

        [TestMethod]
        public void ClassificationStackingEnsembleModel_PredictProbability_single()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var learners = new IIndexedLearner<ProbabilityPrediction>[]
            {
                new ClassificationDecisionTreeLearner(2),
                new ClassificationDecisionTreeLearner(5),
                new ClassificationDecisionTreeLearner(7),
                new ClassificationDecisionTreeLearner(9)
            };

            var learner = new ClassificationStackingEnsembleLearner(learners, new ClassificationDecisionTreeLearner(9),
                new RandomCrossValidation<ProbabilityPrediction>(5, 23), false);
            
            var sut = learner.Learn(observations, targets);

            var predictions = new ProbabilityPrediction[rows];
            for (int i = 0; i < rows; i++)
            {
                predictions[i] = sut.PredictProbability(observations.Row(i));
            }

            var metric = new LogLossClassificationProbabilityMetric();
            var actual = metric.Error(targets, predictions);

            Assert.AreEqual(0.6696598716465223, actual, 0.0000001);
        }

        [TestMethod]
        public void ClassificationStackingEnsembleModel_PredictProbability_Multiple()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var learners = new IIndexedLearner<ProbabilityPrediction>[]
            {
                new ClassificationDecisionTreeLearner(2),
                new ClassificationDecisionTreeLearner(5),
                new ClassificationDecisionTreeLearner(7),
                new ClassificationDecisionTreeLearner(9)
            };

            var learner = new ClassificationStackingEnsembleLearner(learners, new ClassificationDecisionTreeLearner(9),
                new RandomCrossValidation<ProbabilityPrediction>(5, 23), false);
            
            var sut = learner.Learn(observations, targets);

            var predictions = sut.PredictProbability(observations);

            var metric = new LogLossClassificationProbabilityMetric();
            var actual = metric.Error(targets, predictions);

            Assert.AreEqual(0.6696598716465223, actual, 0.0000001);
        }

        [TestMethod]
        public void ClassificationStackingEnsembleModel_GetVariableImportance()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var featureNameToIndex = new Dictionary<string, int> { { "AptitudeTestScore", 0 }, 
                { "PreviousExperience_month", 1 } };

            var learners = new IIndexedLearner<ProbabilityPrediction>[]
            {
                new ClassificationDecisionTreeLearner(2),
                new ClassificationDecisionTreeLearner(5),
                new ClassificationDecisionTreeLearner(7),
                new ClassificationDecisionTreeLearner(9)
            };

            var learner = new ClassificationStackingEnsembleLearner(learners, new ClassificationDecisionTreeLearner(9),
                new RandomCrossValidation<ProbabilityPrediction>(5, 23), false);
            
            var sut = learner.Learn(observations, targets);

            var actual = sut.GetVariableImportance(featureNameToIndex);

            WriteImportances(actual);
            var expected = new Dictionary<string, double> { { "ClassificationDecisionTreeModel_1_Class_Probability_0", 100 }, { "ClassificationDecisionTreeModel_2_Class_Probability_0", 92.2443379072288 }, { "ClassificationDecisionTreeModel_0_Class_Probability_0", 76.9658783620323 }, { "ClassificationDecisionTreeModel_1_Class_Probability_1", 21.1944454897829 }, { "ClassificationDecisionTreeModel_0_Class_Probability_1", 0 }, { "ClassificationDecisionTreeModel_2_Class_Probability_1", 0 }, { "ClassificationDecisionTreeModel_3_Class_Probability_0", 0 }, { "ClassificationDecisionTreeModel_3_Class_Probability_1", 0 } }; 

            Assert.AreEqual(expected.Count, actual.Count);
            var zip = expected.Zip(actual, (e, a) => new { Expected = e, Actual = a });

            foreach (var item in zip)
            {
                Assert.AreEqual(item.Expected.Key, item.Actual.Key);
                Assert.AreEqual(item.Expected.Value, item.Actual.Value, 0.000001);
            }
        }

        [TestMethod]
        public void ClassificationStackingEnsembleModel_GetRawVariableImportance()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();

            var learners = new IIndexedLearner<ProbabilityPrediction>[]
            {
                new ClassificationDecisionTreeLearner(2),
                new ClassificationDecisionTreeLearner(5),
                new ClassificationDecisionTreeLearner(7),
                new ClassificationDecisionTreeLearner(9)
            };

            var learner = new ClassificationStackingEnsembleLearner(learners, new ClassificationDecisionTreeLearner(9),
                new RandomCrossValidation<ProbabilityPrediction>(5, 23), false);
            
            var sut = learner.Learn(observations, targets);

            var actual = sut.GetRawVariableImportance();
            var expected = new double[] { 0.12545787545787546, 0, 0.16300453932032882, 0.0345479082321188, 0.15036245805476572, 0, 0, 0 }; 

            Assert.AreEqual(expected.Length, actual.Length);

            for (int i = 0; i < expected.Length; i++)
            {
                Assert.AreEqual(expected[i], actual[i], 0.000001);
            }
        }

        void WriteImportances(Dictionary<string, double> featureImportance)
        {
            var result = "new Dictionary<string, double> {";
            foreach (var item in featureImportance)
            {
                result += "{" + "\"" + item.Key + "\"" + ", " + item.Value + "}, ";
            }

            Trace.WriteLine(result);
        }
    }
}
