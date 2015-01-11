using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.DecisionTrees.Models;
using SharpLearning.DecisionTrees.Test.Properties;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Metrics.Classification;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace SharpLearning.DecisionTrees.Test.suts
{
    [TestClass]
    public class ClassificationDecisionTreeModelTest
    {
        [TestMethod]
        public void ClassificationDecisionTreeModel_Predict_Single()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var learner = new ClassificationDecisionTreeLearner(100, 1, 2, 0.001, 42);
            var sut = learner.Learn(observations, targets);

            var predictions = new double[rows];
            for (int i = 0; i < rows; i++)
            {
                predictions[i] = sut.Predict(observations.GetRow(i));
            }

            var evaluator = new TotalErrorClassificationMetric<double>();
            var error = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.038461538461538464, error, 0.0000001);
        }

        [TestMethod]
        public void ClassificationDecisionTreeModel_Precit_Multiple()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var learner = new ClassificationDecisionTreeLearner(100, 1, 2, 0.001, 42);
            var sut = learner.Learn(observations, targets);

            var predictions = sut.Predict(observations);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var error = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.038461538461538464, error, 0.0000001);
        }

        [TestMethod]
        public void ClassificationDecisionTreeModel_Predict_Multiple_Indexed()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var learner = new ClassificationDecisionTreeLearner(100, 5, 2, 0.001, 42);
            var sut = learner.Learn(observations, targets);

            var indices = new int[] { 0, 3, 4, 5, 6, 7, 8, 9, 20, 21 };
            var predictions = sut.Predict(observations, indices);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var indexedTargets = targets.GetIndices(indices);
            var error = evaluator.Error(indexedTargets, predictions);

            Assert.AreEqual(0.1, error, 0.0000001);
        }


        [TestMethod]
        public void ClassificationDecisionTreeModel_PredictProbability_Single()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var learner = new ClassificationDecisionTreeLearner(100, 5, 2, 0.001, 42);
            var sut = learner.Learn(observations, targets);

            var actual = new ProbabilityPrediction[rows];
            for (int i = 0; i < rows; i++)
            {
                actual[i] = sut.PredictProbability(observations.GetRow(i));
            }

            var evaluator = new TotalErrorClassificationMetric<double>();
            var error = evaluator.Error(targets, actual.Select(p => p.Prediction).ToArray());

            Assert.AreEqual(0.23076923076923078, error, 0.0000001);

            var expected = new ProbabilityPrediction[] { new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.571428571428571 }, { 1, 0.428571428571429 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.428571428571429 }, { 1, 0.571428571428571 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.857142857142857 }, { 1, 0.142857142857143 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.571428571428571 }, { 1, 0.428571428571429 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.428571428571429 }, { 1, 0.571428571428571 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.75 }, { 1, 0.25 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.285714285714286 }, { 1, 0.714285714285714 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.75 }, { 1, 0.25 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.857142857142857 }, { 1, 0.142857142857143 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.285714285714286 }, { 1, 0.714285714285714 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.571428571428571 }, { 1, 0.428571428571429 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.857142857142857 }, { 1, 0.142857142857143 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.428571428571429 }, { 1, 0.571428571428571 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.571428571428571 }, { 1, 0.428571428571429 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.428571428571429 }, { 1, 0.571428571428571 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.75 }, { 1, 0.25 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.75 }, { 1, 0.25 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.285714285714286 }, { 1, 0.714285714285714 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.285714285714286 }, { 1, 0.714285714285714 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.571428571428571 }, { 1, 0.428571428571429 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.857142857142857 }, { 1, 0.142857142857143 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.285714285714286 }, { 1, 0.714285714285714 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.428571428571429 }, { 1, 0.571428571428571 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.75 }, { 1, 0.25 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.75 }, { 1, 0.25 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.857142857142857 }, { 1, 0.142857142857143 }, }), };
            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void ClassificationDecisionTreeModel_PredictProbability_Multiple()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var learner = new ClassificationDecisionTreeLearner(100, 5, 2, 0.001, 42);
            var sut = learner.Learn(observations, targets);

            var actual = sut.PredictProbability(observations);
            var evaluator = new TotalErrorClassificationMetric<double>();
            var error = evaluator.Error(targets, actual.Select(p => p.Prediction).ToArray());

            Assert.AreEqual(0.23076923076923078, error, 0.0000001);
 
            var expected = new ProbabilityPrediction[] { new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.571428571428571 }, { 1, 0.428571428571429 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.428571428571429 }, { 1, 0.571428571428571 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.857142857142857 }, { 1, 0.142857142857143 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.571428571428571 }, { 1, 0.428571428571429 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.428571428571429 }, { 1, 0.571428571428571 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.75 }, { 1, 0.25 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.285714285714286 }, { 1, 0.714285714285714 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.75 }, { 1, 0.25 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.857142857142857 }, { 1, 0.142857142857143 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.285714285714286 }, { 1, 0.714285714285714 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.571428571428571 }, { 1, 0.428571428571429 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.857142857142857 }, { 1, 0.142857142857143 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.428571428571429 }, { 1, 0.571428571428571 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.571428571428571 }, { 1, 0.428571428571429 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.428571428571429 }, { 1, 0.571428571428571 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.75 }, { 1, 0.25 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.75 }, { 1, 0.25 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.285714285714286 }, { 1, 0.714285714285714 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.285714285714286 }, { 1, 0.714285714285714 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.571428571428571 }, { 1, 0.428571428571429 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.857142857142857 }, { 1, 0.142857142857143 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.285714285714286 }, { 1, 0.714285714285714 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.428571428571429 }, { 1, 0.571428571428571 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.75 }, { 1, 0.25 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.75 }, { 1, 0.25 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.857142857142857 }, { 1, 0.142857142857143 }, }), };
            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void ClassificationDecisionTreeModel_PredictProbability_Multiple_Indexed()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var learner = new ClassificationDecisionTreeLearner(100, 5, 2, 0.001, 42);
            var sut = learner.Learn(observations, targets);

            var indices = new int[] { 0, 3, 4, 5, 6, 7, 8, 9, 20, 21 };
            var actual = sut.PredictProbability(observations, indices);

            var indexedTargets = targets.GetIndices(indices);
            var evaluator = new TotalErrorClassificationMetric<double>();
            var error = evaluator.Error(indexedTargets, actual.Select(p => p.Prediction).ToArray());

            Assert.AreEqual(0.1, error, 0.0000001);

            var expected = new ProbabilityPrediction[] { new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.571428571428571 }, { 1, 0.428571428571429 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.571428571428571 }, { 1, 0.428571428571429 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.428571428571429 }, { 1, 0.571428571428571 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.75 }, { 1, 0.25 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.285714285714286 }, { 1, 0.714285714285714 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.75 }, { 1, 0.25 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.857142857142857 }, { 1, 0.142857142857143 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.285714285714286 }, { 1, 0.714285714285714 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.857142857142857 }, { 1, 0.142857142857143 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.285714285714286 }, { 1, 0.714285714285714 }, }), };
            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void ClassificationDecisionTreeModel_GetVariableImportance()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var featureNameToIndex = new Dictionary<string, int> { { "AptitudeTestScore", 0 }, { "PreviousExperience_month", 1 } };

            var learner = new ClassificationDecisionTreeLearner(100, 1, 2, 0.001, 42);
            var sut = learner.Learn(observations, targets);
            
            var actual = sut.GetVariableImportance(featureNameToIndex);
            var expected = new Dictionary<string, double> { { "PreviousExperience_month", 100.0 }, { "AptitudeTestScore", 19.5121951219512 } };

            Assert.AreEqual(expected.Count, actual.Count);
            var zip = expected.Zip(actual, (e, a) => new { Expected = e, Actual = a });

            foreach (var item in zip)
            {
                Assert.AreEqual(item.Expected.Key, item.Actual.Key);
                Assert.AreEqual(item.Expected.Value, item.Actual.Value, 0.000001);
            }
        }

        [TestMethod]
        public void ClassificationDecisionTreeModel_GetRawVariableImportance()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();

            var learner = new ClassificationDecisionTreeLearner(100, 1, 2, 0.001, 42);
            var sut = learner.Learn(observations, targets);

            var actual = sut.GetRawVariableImportance();
            var expected = new double[] { 0.071005917159763288, 0.36390532544378695 };

            Assert.AreEqual(expected.Length, actual.Length);

            for (int i = 0; i < expected.Length; i++)
            {
                Assert.AreEqual(expected[i], actual[i], 0.000001);                
            }
        }

        [TestMethod]
        public void ClassificationDecisionTreeModel_Save()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();

            var learner = new ClassificationDecisionTreeLearner(100, 1, 2, 0.001, 42);
            var sut = learner.Learn(observations, targets);
            
            var writer = new StringWriter();
            sut.Save(() => writer);
            
            Assert.AreEqual(ClassificationDecisionTreeModelString, writer.ToString());
        }

        [TestMethod]
        public void ClassificationDecisionTreeModel_Load()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();

            var reader = new StringReader(ClassificationDecisionTreeModelString);
            var sut = ClassificationDecisionTreeModel.Load(() => reader);

            var predictions = sut.Predict(observations);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var error = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.038461538461538464, error, 0.0000001);
        }


        readonly string ClassificationDecisionTreeModelString =
            "<?xml version=\"1.0\" encoding=\"utf-16\"?>\r\n<ClassificationDecisionTreeModel xmlns:i=\"http://www.w3.org/2001/XMLSchema-instance\" z:Id=\"1\" xmlns:z=\"http://schemas.microsoft.com/2003/10/Serialization/\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.DecisionTrees.Models\">\r\n  <Tree xmlns:d2p1=\"http://schemas.datacontract.org/2004/07/SharpLearning.DecisionTrees.Nodes\" z:Id=\"2\">\r\n    <d2p1:Nodes z:Id=\"3\" z:Size=\"23\">\r\n      <d2p1:Node>\r\n        <d2p1:FeatureIndex>1</d2p1:FeatureIndex>\r\n        <d2p1:LeafProbabilityIndex>-1</d2p1:LeafProbabilityIndex>\r\n        <d2p1:LeftIndex>1</d2p1:LeftIndex>\r\n        <d2p1:NodeIndex>0</d2p1:NodeIndex>\r\n        <d2p1:RightIndex>22</d2p1:RightIndex>\r\n        <d2p1:Value>20</d2p1:Value>\r\n      </d2p1:Node>\r\n      <d2p1:Node>\r\n        <d2p1:FeatureIndex>1</d2p1:FeatureIndex>\r\n        <d2p1:LeafProbabilityIndex>-1</d2p1:LeafProbabilityIndex>\r\n        <d2p1:LeftIndex>2</d2p1:LeftIndex>\r\n        <d2p1:NodeIndex>1</d2p1:NodeIndex>\r\n        <d2p1:RightIndex>3</d2p1:RightIndex>\r\n        <d2p1:Value>2.5</d2p1:Value>\r\n      </d2p1:Node>\r\n      <d2p1:Node>\r\n        <d2p1:FeatureIndex>-1</d2p1:FeatureIndex>\r\n        <d2p1:LeafProbabilityIndex>0</d2p1:LeafProbabilityIndex>\r\n        <d2p1:LeftIndex>-1</d2p1:LeftIndex>\r\n        <d2p1:NodeIndex>2</d2p1:NodeIndex>\r\n        <d2p1:RightIndex>-1</d2p1:RightIndex>\r\n        <d2p1:Value>1</d2p1:Value>\r\n      </d2p1:Node>\r\n      <d2p1:Node>\r\n        <d2p1:FeatureIndex>1</d2p1:FeatureIndex>\r\n        <d2p1:LeafProbabilityIndex>-1</d2p1:LeafProbabilityIndex>\r\n        <d2p1:LeftIndex>4</d2p1:LeftIndex>\r\n        <d2p1:NodeIndex>3</d2p1:NodeIndex>\r\n        <d2p1:RightIndex>15</d2p1:RightIndex>\r\n        <d2p1:Value>13.5</d2p1:Value>\r\n      </d2p1:Node>\r\n      <d2p1:Node>\r\n        <d2p1:FeatureIndex>1</d2p1:FeatureIndex>\r\n        <d2p1:LeafProbabilityIndex>-1</d2p1:LeafProbabilityIndex>\r\n        <d2p1:LeftIndex>5</d2p1:LeftIndex>\r\n        <d2p1:NodeIndex>4</d2p1:NodeIndex>\r\n        <d2p1:RightIndex>8</d2p1:RightIndex>\r\n        <d2p1:Value>4.5</d2p1:Value>\r\n      </d2p1:Node>\r\n      <d2p1:Node>\r\n        <d2p1:FeatureIndex>0</d2p1:FeatureIndex>\r\n        <d2p1:LeafProbabilityIndex>-1</d2p1:LeafProbabilityIndex>\r\n        <d2p1:LeftIndex>6</d2p1:LeftIndex>\r\n        <d2p1:NodeIndex>5</d2p1:NodeIndex>\r\n        <d2p1:RightIndex>7</d2p1:RightIndex>\r\n        <d2p1:Value>4</d2p1:Value>\r\n      </d2p1:Node>\r\n      <d2p1:Node>\r\n        <d2p1:FeatureIndex>-1</d2p1:FeatureIndex>\r\n        <d2p1:LeafProbabilityIndex>1</d2p1:LeafProbabilityIndex>\r\n        <d2p1:LeftIndex>-1</d2p1:LeftIndex>\r\n        <d2p1:NodeIndex>6</d2p1:NodeIndex>\r\n        <d2p1:RightIndex>-1</d2p1:RightIndex>\r\n        <d2p1:Value>0</d2p1:Value>\r\n      </d2p1:Node>\r\n      <d2p1:Node>\r\n        <d2p1:FeatureIndex>-1</d2p1:FeatureIndex>\r\n        <d2p1:LeafProbabilityIndex>2</d2p1:LeafProbabilityIndex>\r\n        <d2p1:LeftIndex>-1</d2p1:LeftIndex>\r\n        <d2p1:NodeIndex>7</d2p1:NodeIndex>\r\n        <d2p1:RightIndex>-1</d2p1:RightIndex>\r\n        <d2p1:Value>1</d2p1:Value>\r\n      </d2p1:Node>\r\n      <d2p1:Node>\r\n        <d2p1:FeatureIndex>0</d2p1:FeatureIndex>\r\n        <d2p1:LeafProbabilityIndex>-1</d2p1:LeafProbabilityIndex>\r\n        <d2p1:LeftIndex>9</d2p1:LeftIndex>\r\n        <d2p1:NodeIndex>8</d2p1:NodeIndex>\r\n        <d2p1:RightIndex>14</d2p1:RightIndex>\r\n        <d2p1:Value>1.5</d2p1:Value>\r\n      </d2p1:Node>\r\n      <d2p1:Node>\r\n        <d2p1:FeatureIndex>1</d2p1:FeatureIndex>\r\n        <d2p1:LeafProbabilityIndex>-1</d2p1:LeafProbabilityIndex>\r\n        <d2p1:LeftIndex>10</d2p1:LeftIndex>\r\n        <d2p1:NodeIndex>9</d2p1:NodeIndex>\r\n        <d2p1:RightIndex>11</d2p1:RightIndex>\r\n        <d2p1:Value>9.5</d2p1:Value>\r\n      </d2p1:Node>\r\n      <d2p1:Node>\r\n        <d2p1:FeatureIndex>-1</d2p1:FeatureIndex>\r\n        <d2p1:LeafProbabilityIndex>3</d2p1:LeafProbabilityIndex>\r\n        <d2p1:LeftIndex>-1</d2p1:LeftIndex>\r\n        <d2p1:NodeIndex>10</d2p1:NodeIndex>\r\n        <d2p1:RightIndex>-1</d2p1:RightIndex>\r\n        <d2p1:Value>0</d2p1:Value>\r\n      </d2p1:Node>\r\n      <d2p1:Node>\r\n        <d2p1:FeatureIndex>1</d2p1:FeatureIndex>\r\n        <d2p1:LeafProbabilityIndex>-1</d2p1:LeafProbabilityIndex>\r\n        <d2p1:LeftIndex>12</d2p1:LeftIndex>\r\n        <d2p1:NodeIndex>11</d2p1:NodeIndex>\r\n        <d2p1:RightIndex>13</d2p1:RightIndex>\r\n        <d2p1:Value>11</d2p1:Value>\r\n      </d2p1:Node>\r\n      <d2p1:Node>\r\n        <d2p1:FeatureIndex>-1</d2p1:FeatureIndex>\r\n        <d2p1:LeafProbabilityIndex>4</d2p1:LeafProbabilityIndex>\r\n        <d2p1:LeftIndex>-1</d2p1:LeftIndex>\r\n        <d2p1:NodeIndex>12</d2p1:NodeIndex>\r\n        <d2p1:RightIndex>-1</d2p1:RightIndex>\r\n        <d2p1:Value>1</d2p1:Value>\r\n      </d2p1:Node>\r\n      <d2p1:Node>\r\n        <d2p1:FeatureIndex>-1</d2p1:FeatureIndex>\r\n        <d2p1:LeafProbabilityIndex>5</d2p1:LeafProbabilityIndex>\r\n        <d2p1:LeftIndex>-1</d2p1:LeftIndex>\r\n        <d2p1:NodeIndex>13</d2p1:NodeIndex>\r\n        <d2p1:RightIndex>-1</d2p1:RightIndex>\r\n        <d2p1:Value>0</d2p1:Value>\r\n      </d2p1:Node>\r\n      <d2p1:Node>\r\n        <d2p1:FeatureIndex>-1</d2p1:FeatureIndex>\r\n        <d2p1:LeafProbabilityIndex>6</d2p1:LeafProbabilityIndex>\r\n        <d2p1:LeftIndex>-1</d2p1:LeftIndex>\r\n        <d2p1:NodeIndex>14</d2p1:NodeIndex>\r\n        <d2p1:RightIndex>-1</d2p1:RightIndex>\r\n        <d2p1:Value>0</d2p1:Value>\r\n      </d2p1:Node>\r\n      <d2p1:Node>\r\n        <d2p1:FeatureIndex>0</d2p1:FeatureIndex>\r\n        <d2p1:LeafProbabilityIndex>-1</d2p1:LeafProbabilityIndex>\r\n        <d2p1:LeftIndex>16</d2p1:LeftIndex>\r\n        <d2p1:NodeIndex>15</d2p1:NodeIndex>\r\n        <d2p1:RightIndex>19</d2p1:RightIndex>\r\n        <d2p1:Value>2.5</d2p1:Value>\r\n      </d2p1:Node>\r\n      <d2p1:Node>\r\n        <d2p1:FeatureIndex>0</d2p1:FeatureIndex>\r\n        <d2p1:LeafProbabilityIndex>-1</d2p1:LeafProbabilityIndex>\r\n        <d2p1:LeftIndex>17</d2p1:LeftIndex>\r\n        <d2p1:NodeIndex>16</d2p1:NodeIndex>\r\n        <d2p1:RightIndex>18</d2p1:RightIndex>\r\n        <d2p1:Value>1.5</d2p1:Value>\r\n      </d2p1:Node>\r\n      <d2p1:Node>\r\n        <d2p1:FeatureIndex>-1</d2p1:FeatureIndex>\r\n        <d2p1:LeafProbabilityIndex>7</d2p1:LeafProbabilityIndex>\r\n        <d2p1:LeftIndex>-1</d2p1:LeftIndex>\r\n        <d2p1:NodeIndex>17</d2p1:NodeIndex>\r\n        <d2p1:RightIndex>-1</d2p1:RightIndex>\r\n        <d2p1:Value>0</d2p1:Value>\r\n      </d2p1:Node>\r\n      <d2p1:Node>\r\n        <d2p1:FeatureIndex>-1</d2p1:FeatureIndex>\r\n        <d2p1:LeafProbabilityIndex>8</d2p1:LeafProbabilityIndex>\r\n        <d2p1:LeftIndex>-1</d2p1:LeftIndex>\r\n        <d2p1:NodeIndex>18</d2p1:NodeIndex>\r\n        <d2p1:RightIndex>-1</d2p1:RightIndex>\r\n        <d2p1:Value>1</d2p1:Value>\r\n      </d2p1:Node>\r\n      <d2p1:Node>\r\n        <d2p1:FeatureIndex>1</d2p1:FeatureIndex>\r\n        <d2p1:LeafProbabilityIndex>-1</d2p1:LeafProbabilityIndex>\r\n        <d2p1:LeftIndex>20</d2p1:LeftIndex>\r\n        <d2p1:NodeIndex>19</d2p1:NodeIndex>\r\n        <d2p1:RightIndex>21</d2p1:RightIndex>\r\n        <d2p1:Value>17</d2p1:Value>\r\n      </d2p1:Node>\r\n      <d2p1:Node>\r\n        <d2p1:FeatureIndex>-1</d2p1:FeatureIndex>\r\n        <d2p1:LeafProbabilityIndex>9</d2p1:LeafProbabilityIndex>\r\n        <d2p1:LeftIndex>-1</d2p1:LeftIndex>\r\n        <d2p1:NodeIndex>20</d2p1:NodeIndex>\r\n        <d2p1:RightIndex>-1</d2p1:RightIndex>\r\n        <d2p1:Value>1</d2p1:Value>\r\n      </d2p1:Node>\r\n      <d2p1:Node>\r\n        <d2p1:FeatureIndex>-1</d2p1:FeatureIndex>\r\n        <d2p1:LeafProbabilityIndex>10</d2p1:LeafProbabilityIndex>\r\n        <d2p1:LeftIndex>-1</d2p1:LeftIndex>\r\n        <d2p1:NodeIndex>21</d2p1:NodeIndex>\r\n        <d2p1:RightIndex>-1</d2p1:RightIndex>\r\n        <d2p1:Value>0</d2p1:Value>\r\n      </d2p1:Node>\r\n      <d2p1:Node>\r\n        <d2p1:FeatureIndex>-1</d2p1:FeatureIndex>\r\n        <d2p1:LeafProbabilityIndex>11</d2p1:LeafProbabilityIndex>\r\n        <d2p1:LeftIndex>-1</d2p1:LeftIndex>\r\n        <d2p1:NodeIndex>22</d2p1:NodeIndex>\r\n        <d2p1:RightIndex>-1</d2p1:RightIndex>\r\n        <d2p1:Value>1</d2p1:Value>\r\n      </d2p1:Node>\r\n    </d2p1:Nodes>\r\n    <d2p1:Probabilities xmlns:d3p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"4\" z:Size=\"12\">\r\n      <d3p1:ArrayOfdouble z:Id=\"5\" z:Size=\"2\">\r\n        <d3p1:double>0.33333333333333331</d3p1:double>\r\n        <d3p1:double>0.66666666666666663</d3p1:double>\r\n      </d3p1:ArrayOfdouble>\r\n      <d3p1:ArrayOfdouble z:Id=\"6\" z:Size=\"2\">\r\n        <d3p1:double>0.66666666666666663</d3p1:double>\r\n        <d3p1:double>0.33333333333333331</d3p1:double>\r\n      </d3p1:ArrayOfdouble>\r\n      <d3p1:ArrayOfdouble z:Id=\"7\" z:Size=\"2\">\r\n        <d3p1:double>0.33333333333333331</d3p1:double>\r\n        <d3p1:double>0.66666666666666663</d3p1:double>\r\n      </d3p1:ArrayOfdouble>\r\n      <d3p1:ArrayOfdouble z:Id=\"8\" z:Size=\"2\">\r\n        <d3p1:double>0.8</d3p1:double>\r\n        <d3p1:double>0.2</d3p1:double>\r\n      </d3p1:ArrayOfdouble>\r\n      <d3p1:ArrayOfdouble z:Id=\"9\" z:Size=\"2\">\r\n        <d3p1:double>0.33333333333333331</d3p1:double>\r\n        <d3p1:double>0.66666666666666663</d3p1:double>\r\n      </d3p1:ArrayOfdouble>\r\n      <d3p1:ArrayOfdouble z:Id=\"10\" z:Size=\"2\">\r\n        <d3p1:double>0.75</d3p1:double>\r\n        <d3p1:double>0.25</d3p1:double>\r\n      </d3p1:ArrayOfdouble>\r\n      <d3p1:ArrayOfdouble z:Id=\"11\" z:Size=\"2\">\r\n        <d3p1:double>0.88888888888888884</d3p1:double>\r\n        <d3p1:double>0.1111111111111111</d3p1:double>\r\n      </d3p1:ArrayOfdouble>\r\n      <d3p1:ArrayOfdouble z:Id=\"12\" z:Size=\"2\">\r\n        <d3p1:double>0.5</d3p1:double>\r\n        <d3p1:double>0.5</d3p1:double>\r\n      </d3p1:ArrayOfdouble>\r\n      <d3p1:ArrayOfdouble z:Id=\"13\" z:Size=\"2\">\r\n        <d3p1:double>0.33333333333333331</d3p1:double>\r\n        <d3p1:double>0.66666666666666663</d3p1:double>\r\n      </d3p1:ArrayOfdouble>\r\n      <d3p1:ArrayOfdouble z:Id=\"14\" z:Size=\"2\">\r\n        <d3p1:double>0.33333333333333331</d3p1:double>\r\n        <d3p1:double>0.66666666666666663</d3p1:double>\r\n      </d3p1:ArrayOfdouble>\r\n      <d3p1:ArrayOfdouble z:Id=\"15\" z:Size=\"2\">\r\n        <d3p1:double>0.75</d3p1:double>\r\n        <d3p1:double>0.25</d3p1:double>\r\n      </d3p1:ArrayOfdouble>\r\n      <d3p1:ArrayOfdouble z:Id=\"16\" z:Size=\"2\">\r\n        <d3p1:double>0.16666666666666666</d3p1:double>\r\n        <d3p1:double>0.83333333333333326</d3p1:double>\r\n      </d3p1:ArrayOfdouble>\r\n    </d2p1:Probabilities>\r\n    <d2p1:TargetNames xmlns:d3p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"17\" z:Size=\"2\">\r\n      <d3p1:double>0</d3p1:double>\r\n      <d3p1:double>1</d3p1:double>\r\n    </d2p1:TargetNames>\r\n    <d2p1:VariableImportance xmlns:d3p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"18\" z:Size=\"2\">\r\n      <d3p1:double>0.071005917159763288</d3p1:double>\r\n      <d3p1:double>0.36390532544378695</d3p1:double>\r\n    </d2p1:VariableImportance>\r\n  </Tree>\r\n  <m_variableImportance xmlns:d2p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Ref=\"18\" i:nil=\"true\" />\r\n</ClassificationDecisionTreeModel>";

        void Write(ProbabilityPrediction[] predictions)
        {
            var value = "new ProbabilityPrediction[] {";
            foreach (var item in predictions)
            {
                value += "new ProbabilityPrediction(" + item.Prediction + ", new Dictionary<double, double> {";
                foreach (var prob in item.Probabilities)
                {
                    value += "{" + prob.Key + ", " + prob.Value + "}, ";
                }
                value += "}),";
            }
            value += "};";

            Trace.WriteLine(value);
        }
    }
}
