using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers;
using SharpLearning.FeatureTransformations.MatrixTransforms;
using SharpLearning.InputOutput.Csv;
using SharpLearning.InputOutput.Serialization;
using SharpLearning.Metrics.Regression;
using SharpLearning.Neural.Activations;
using SharpLearning.Neural.Layers;
using SharpLearning.Neural.Learners;
using SharpLearning.Neural.Loss;
using SharpLearning.Neural.Models;
using SharpLearning.Neural.Test.Properties;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace SharpLearning.Neural.Test.Models
{
    [TestClass]
    public class RegressionNeuralNetModelTest
    {
        [TestMethod]
        public void RegressionNeuralNetModel_Predict_Single()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var features = parser.EnumerateRows(v => v != "Target").First().ColumnNameToIndex.Keys.ToArray();
            var normalizer = new MinMaxTransformer(0.0, 1.0);
            var observations = parser.EnumerateRows(features)
                .ToF64Matrix();

            normalizer.Transform(observations, observations);

            var targets = parser.EnumerateRows("Target").ToF64Vector();
            var rows = targets.Length;

            var learner = new RegressionMomentumNeuralNetLearner(new HiddenLayer[] { HiddenLayer.New(50) }, new ReluActivation(), new LogLoss(),
                100, 0.1f, 20, 0, 0.0, LearningRateSchedule.InvScaling);

            var sut = learner.Learn(observations, targets);

            var predictions = new double[rows];
            for (int i = 0; i < rows; i++)
            {
                predictions[i] = sut.Predict(observations.GetRow(i));
            }

            var evaluator = new MeanSquaredErrorRegressionMetric();
            var error = evaluator.Error(targets, predictions);

            Assert.AreEqual(1.4383101999692078, error, 1e-6);
        }

        [TestMethod]
        public void RegressionNeuralNetModel_Predict_Multiple()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var features = parser.EnumerateRows(v => v != "Target").First().ColumnNameToIndex.Keys.ToArray();
            var normalizer = new MinMaxTransformer(0.0, 1.0);
            var observations = parser.EnumerateRows(features)
                .ToF64Matrix();

            normalizer.Transform(observations, observations);

            var targets = parser.EnumerateRows("Target").ToF64Vector();
            var rows = targets.Length;

            var learner = new RegressionMomentumNeuralNetLearner(new HiddenLayer[] { HiddenLayer.New(50) }, new ReluActivation(), new LogLoss(),
                100, 0.1f, 20, 0, 0.0, LearningRateSchedule.InvScaling);

            var sut = learner.Learn(observations, targets);

            var predictions = sut.Predict(observations);

            var evaluator = new MeanSquaredErrorRegressionMetric();
            var error = evaluator.Error(targets, predictions);

            Assert.AreEqual(1.4383101999692078, error, 1e-6);
        }

      
        [TestMethod]
        public void RegressionNeuralNetModel_GetVariableImportance()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var features = parser.EnumerateRows(v => v != "Target").First().ColumnNameToIndex.Keys.ToArray();
            var normalizer = new MinMaxTransformer(0.0, 1.0);
            var observations = parser.EnumerateRows(features)
                .ToF64Matrix();

            normalizer.Transform(observations, observations);

            var targets = parser.EnumerateRows("Target").ToF64Vector();
            var featureNameToIndex = parser.EnumerateRows(v => v != "Target").First().ColumnNameToIndex;

            var learner = new RegressionMomentumNeuralNetLearner(new HiddenLayer[] { HiddenLayer.New(50) }, new ReluActivation(), new LogLoss(),
                100, 0.1f, 20, 0, 0.0, LearningRateSchedule.InvScaling);

            var sut = learner.Learn(observations, targets);

            var actual = sut.GetVariableImportance(featureNameToIndex);
            var expected = new Dictionary<string, double>
            {
                { "F3", 100},
                { "F4", 43.1321746334981},
                { "F8", 39.260236291283},
                { "F2", 37.5314583985078},
                { "F1", 29.7158776391697},
                { "F10", 27.5597940593381},
                { "F7", 13.0020722086038},
                { "F5", 2.36354849213805},
                { "F6", 0.11205908824216},
            };

            Assert.AreEqual(expected.Count, actual.Count);
            var zip = expected.Zip(actual, (e, a) => new { Expected = e, Actual = a });

            foreach (var item in zip)
            {
                Assert.AreEqual(item.Expected.Key, item.Actual.Key);
                Assert.AreEqual(item.Expected.Value, item.Actual.Value, 1e-3);
            }
        }

        [TestMethod]
        public void RegressionNeuralNetModel_GetRawVariableImportance()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var features = parser.EnumerateRows(v => v != "Target").First().ColumnNameToIndex.Keys.ToArray();
            var normalizer = new MinMaxTransformer(0.0, 1.0);
            var observations = parser.EnumerateRows(features)
                .ToF64Matrix();

            normalizer.Transform(observations, observations);

            var targets = parser.EnumerateRows("Target").ToF64Vector();
            var featureNameToIndex = parser.EnumerateRows(v => v != "Target").First().ColumnNameToIndex;

            var learner = new RegressionMomentumNeuralNetLearner(new HiddenLayer[] { HiddenLayer.New(50) }, new ReluActivation(), new LogLoss(),
                100, 0.1f, 20, 0, 0.0, LearningRateSchedule.InvScaling);

            var sut = learner.Learn(observations, targets);

            var actual = sut.GetRawVariableImportance();
            var expected = new double[] 
            {
                2.5147202014923096,
                3.1761174201965332,
                8.4625473022460938,
                3.650080680847168,
                0.20001640915870667,
                0.009483053348958492,
                1.100306510925293,
                3.3224160671234131,
                2.3322606086730957,
            };

            Assert.AreEqual(expected.Length, actual.Length);

            for (int i = 0; i < expected.Length; i++)
            {
                Assert.AreEqual(expected[i], actual[i], 0.001);
            }
        }

        [TestMethod]
        public void RegressionNeuralNetModel_Save()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var features = parser.EnumerateRows(v => v != "Target").First().ColumnNameToIndex.Keys.ToArray();
            var normalizer = new MinMaxTransformer(0.0, 1.0);
            var observations = parser.EnumerateRows(features)
                .ToF64Matrix();

            normalizer.Transform(observations, observations);

            var targets = parser.EnumerateRows("Target").ToF64Vector();

            var learner = new RegressionMomentumNeuralNetLearner(new HiddenLayer[] { HiddenLayer.New(50) }, new ReluActivation(), new LogLoss(),
                100, 0.1f, 20, 0, 0.0, LearningRateSchedule.InvScaling);

            var sut = learner.Learn(observations, targets);

            var writer = new StringWriter();
            sut.Save(() => writer);

            var actual = writer.ToString();
            Assert.AreEqual(RegressionNeuralNetModelString, actual);
        }

        [TestMethod]
        public void RegressionNeuralNetModel_Load()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var features = parser.EnumerateRows(v => v != "Target").First().ColumnNameToIndex.Keys.ToArray();
            var normalizer = new MinMaxTransformer(0.0, 1.0);
            var observations = parser.EnumerateRows(features)
                .ToF64Matrix();

            normalizer.Transform(observations, observations);

            var targets = parser.EnumerateRows("Target").ToF64Vector();

            var reader = new StringReader(RegressionNeuralNetModelString);
            var sut = RegressionNeuralNetModel.Load(() => reader);

            var predictions = sut.Predict(observations);

            var evaluator = new MeanSquaredErrorRegressionMetric();
            var error = evaluator.Error(targets, predictions);

            Assert.AreEqual(1.4383103408006051, error, 0.0000001);
        }

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

        readonly string RegressionNeuralNetModelString =
        "<?xml version=\"1.0\" encoding=\"utf-16\"?>\r\n<RegressionNeuralNetModel xmlns:i=\"http://www.w3.org/2001/XMLSchema-instance\" z:Id=\"1\" xmlns:z=\"http://schemas.microsoft.com/2003/10/Serialization/\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.Neural.Models\">\r\n  <Iterations>4</Iterations>\r\n  <m_model z:Id=\"2\">\r\n    <Iterations>4</Iterations>\r\n    <m_hiddenActivation z:Id=\"3\" xmlns:d3p1=\"SharpLearning.Neural.Activations\" i:type=\"d3p1:ReluActivation\" />\r\n    <m_intercepts xmlns:d3p1=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra\" z:Id=\"4\" z:Size=\"2\">\r\n      <d3p1:VectorOffloat z:Id=\"5\" xmlns:d4p1=\"MathNet.Numerics.LinearAlgebra.Single\" i:type=\"d4p1:DenseVector\">\r\n        <d3p1:_x003C_Count_x003E_k__BackingField>50</d3p1:_x003C_Count_x003E_k__BackingField>\r\n        <d3p1:_x003C_Storage_x003E_k__BackingField xmlns:d5p1=\"urn:MathNet/Numerics/LinearAlgebra\" z:Id=\"6\" i:type=\"d5p1:DenseVectorStorageOffloat\">\r\n          <d5p1:Length>50</d5p1:Length>\r\n          <d5p1:Data xmlns:d6p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"7\" z:Size=\"50\">\r\n            <d6p1:float>-0.395941228</d6p1:float>\r\n            <d6p1:float>-0.183560222</d6p1:float>\r\n            <d6p1:float>-0.2701936</d6p1:float>\r\n            <d6p1:float>-0.337369025</d6p1:float>\r\n            <d6p1:float>-0.06353555</d6p1:float>\r\n            <d6p1:float>-0.372908354</d6p1:float>\r\n            <d6p1:float>-0.23732689</d6p1:float>\r\n            <d6p1:float>0.0256848373</d6p1:float>\r\n            <d6p1:float>-0.09050797</d6p1:float>\r\n            <d6p1:float>-0.287968844</d6p1:float>\r\n            <d6p1:float>0.275823563</d6p1:float>\r\n            <d6p1:float>-0.0241736</d6p1:float>\r\n            <d6p1:float>-0.3410131</d6p1:float>\r\n            <d6p1:float>0.139434427</d6p1:float>\r\n            <d6p1:float>0.117753468</d6p1:float>\r\n            <d6p1:float>-0.233445346</d6p1:float>\r\n            <d6p1:float>-0.198966652</d6p1:float>\r\n            <d6p1:float>-0.6055196</d6p1:float>\r\n            <d6p1:float>0.243298754</d6p1:float>\r\n            <d6p1:float>-0.0668216646</d6p1:float>\r\n            <d6p1:float>-0.8441365</d6p1:float>\r\n            <d6p1:float>-0.286678374</d6p1:float>\r\n            <d6p1:float>-0.3539871</d6p1:float>\r\n            <d6p1:float>0.0105249491</d6p1:float>\r\n            <d6p1:float>-0.0389125161</d6p1:float>\r\n            <d6p1:float>-0.360628784</d6p1:float>\r\n            <d6p1:float>-0.209838614</d6p1:float>\r\n            <d6p1:float>-0.0471423529</d6p1:float>\r\n            <d6p1:float>-0.570923865</d6p1:float>\r\n            <d6p1:float>0.00327232876</d6p1:float>\r\n            <d6p1:float>-0.655472934</d6p1:float>\r\n            <d6p1:float>0.0886516</d6p1:float>\r\n            <d6p1:float>0.399650156</d6p1:float>\r\n            <d6p1:float>0.0440019257</d6p1:float>\r\n            <d6p1:float>-0.0770989656</d6p1:float>\r\n            <d6p1:float>-0.279910564</d6p1:float>\r\n            <d6p1:float>-0.411004066</d6p1:float>\r\n            <d6p1:float>-0.167670563</d6p1:float>\r\n            <d6p1:float>0.171242058</d6p1:float>\r\n            <d6p1:float>-0.441906</d6p1:float>\r\n            <d6p1:float>0.0553927869</d6p1:float>\r\n            <d6p1:float>-0.3171428</d6p1:float>\r\n            <d6p1:float>-0.20809105</d6p1:float>\r\n            <d6p1:float>-0.1521698</d6p1:float>\r\n            <d6p1:float>-0.12269827</d6p1:float>\r\n            <d6p1:float>-0.2900371</d6p1:float>\r\n            <d6p1:float>-0.3807219</d6p1:float>\r\n            <d6p1:float>0.021986194</d6p1:float>\r\n            <d6p1:float>-0.254819244</d6p1:float>\r\n            <d6p1:float>0.05431803</d6p1:float>\r\n          </d5p1:Data>\r\n        </d3p1:_x003C_Storage_x003E_k__BackingField>\r\n        <_length xmlns=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra.Single\">50</_length>\r\n        <_values xmlns:d5p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Ref=\"7\" i:nil=\"true\" xmlns=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra.Single\" />\r\n      </d3p1:VectorOffloat>\r\n      <d3p1:VectorOffloat z:Id=\"8\" xmlns:d4p1=\"MathNet.Numerics.LinearAlgebra.Single\" i:type=\"d4p1:DenseVector\">\r\n        <d3p1:_x003C_Count_x003E_k__BackingField>1</d3p1:_x003C_Count_x003E_k__BackingField>\r\n        <d3p1:_x003C_Storage_x003E_k__BackingField xmlns:d5p1=\"urn:MathNet/Numerics/LinearAlgebra\" z:Id=\"9\" i:type=\"d5p1:DenseVectorStorageOffloat\">\r\n          <d5p1:Length>1</d5p1:Length>\r\n          <d5p1:Data xmlns:d6p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"10\" z:Size=\"1\">\r\n            <d6p1:float>1.99351025</d6p1:float>\r\n          </d5p1:Data>\r\n        </d3p1:_x003C_Storage_x003E_k__BackingField>\r\n        <_length xmlns=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra.Single\">1</_length>\r\n        <_values xmlns:d5p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Ref=\"10\" i:nil=\"true\" xmlns=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra.Single\" />\r\n      </d3p1:VectorOffloat>\r\n    </m_intercepts>\r\n    <m_layes>3</m_layes>\r\n    <m_outputActivation z:Id=\"11\" xmlns:d3p1=\"SharpLearning.Neural.Activations\" i:type=\"d3p1:IdentityActivation\" />\r\n    <m_weights xmlns:d3p1=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra\" z:Id=\"12\" z:Size=\"2\">\r\n      <d3p1:MatrixOffloat z:Id=\"13\" xmlns:d4p1=\"MathNet.Numerics.LinearAlgebra.Single\" i:type=\"d4p1:DenseMatrix\">\r\n        <d3p1:_x003C_ColumnCount_x003E_k__BackingField>50</d3p1:_x003C_ColumnCount_x003E_k__BackingField>\r\n        <d3p1:_x003C_RowCount_x003E_k__BackingField>9</d3p1:_x003C_RowCount_x003E_k__BackingField>\r\n        <d3p1:_x003C_Storage_x003E_k__BackingField xmlns:d5p1=\"urn:MathNet/Numerics/LinearAlgebra\" z:Id=\"14\" i:type=\"d5p1:DenseColumnMajorMatrixStorageOffloat\">\r\n          <d5p1:RowCount>9</d5p1:RowCount>\r\n          <d5p1:ColumnCount>50</d5p1:ColumnCount>\r\n          <d5p1:Data xmlns:d6p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"15\" z:Size=\"450\">\r\n            <d6p1:float>0.1298725</d6p1:float>\r\n            <d6p1:float>-0.184331164</d6p1:float>\r\n            <d6p1:float>-0.214240521</d6p1:float>\r\n            <d6p1:float>-0.208714</d6p1:float>\r\n            <d6p1:float>-0.292444021</d6p1:float>\r\n            <d6p1:float>-0.0586145744</d6p1:float>\r\n            <d6p1:float>-0.004819561</d6p1:float>\r\n            <d6p1:float>-0.00499042263</d6p1:float>\r\n            <d6p1:float>0.0298287719</d6p1:float>\r\n            <d6p1:float>-0.219470724</d6p1:float>\r\n            <d6p1:float>-0.242144614</d6p1:float>\r\n            <d6p1:float>0.181256011</d6p1:float>\r\n            <d6p1:float>-0.295535117</d6p1:float>\r\n            <d6p1:float>0.06332194</d6p1:float>\r\n            <d6p1:float>-0.0734836</d6p1:float>\r\n            <d6p1:float>0.196236551</d6p1:float>\r\n            <d6p1:float>-0.237337217</d6p1:float>\r\n            <d6p1:float>0.100919366</d6p1:float>\r\n            <d6p1:float>-0.268485218</d6p1:float>\r\n            <d6p1:float>0.509732068</d6p1:float>\r\n            <d6p1:float>-1.08534515</d6p1:float>\r\n            <d6p1:float>0.623451233</d6p1:float>\r\n            <d6p1:float>0.237136438</d6p1:float>\r\n            <d6p1:float>-0.135033548</d6p1:float>\r\n            <d6p1:float>-0.109316856</d6p1:float>\r\n            <d6p1:float>0.713104248</d6p1:float>\r\n            <d6p1:float>-0.1003805</d6p1:float>\r\n            <d6p1:float>0.160548016</d6p1:float>\r\n            <d6p1:float>-0.356597453</d6p1:float>\r\n            <d6p1:float>-0.139358118</d6p1:float>\r\n            <d6p1:float>-0.03432326</d6p1:float>\r\n            <d6p1:float>-0.268537343</d6p1:float>\r\n            <d6p1:float>-0.159462839</d6p1:float>\r\n            <d6p1:float>-0.0396008044</d6p1:float>\r\n            <d6p1:float>-0.256223053</d6p1:float>\r\n            <d6p1:float>0.148394153</d6p1:float>\r\n            <d6p1:float>-0.5277297</d6p1:float>\r\n            <d6p1:float>0.449144363</d6p1:float>\r\n            <d6p1:float>-1.42380381</d6p1:float>\r\n            <d6p1:float>0.530564</d6p1:float>\r\n            <d6p1:float>-0.03829569</d6p1:float>\r\n            <d6p1:float>-0.07169379</d6p1:float>\r\n            <d6p1:float>-0.0576367453</d6p1:float>\r\n            <d6p1:float>0.255332232</d6p1:float>\r\n            <d6p1:float>-0.40740034</d6p1:float>\r\n            <d6p1:float>-0.0318459943</d6p1:float>\r\n            <d6p1:float>-0.03271685</d6p1:float>\r\n            <d6p1:float>-0.380817115</d6p1:float>\r\n            <d6p1:float>-0.03413612</d6p1:float>\r\n            <d6p1:float>-0.169339746</d6p1:float>\r\n            <d6p1:float>0.0600786768</d6p1:float>\r\n            <d6p1:float>-0.3417103</d6p1:float>\r\n            <d6p1:float>-0.0165554788</d6p1:float>\r\n            <d6p1:float>0.274693847</d6p1:float>\r\n            <d6p1:float>0.09593287</d6p1:float>\r\n            <d6p1:float>-0.351209164</d6p1:float>\r\n            <d6p1:float>0.07293062</d6p1:float>\r\n            <d6p1:float>-0.350080341</d6p1:float>\r\n            <d6p1:float>0.09418991</d6p1:float>\r\n            <d6p1:float>0.0197674036</d6p1:float>\r\n            <d6p1:float>0.0251991954</d6p1:float>\r\n            <d6p1:float>0.165965065</d6p1:float>\r\n            <d6p1:float>-0.0235447828</d6p1:float>\r\n            <d6p1:float>-0.306240946</d6p1:float>\r\n            <d6p1:float>0.102759778</d6p1:float>\r\n            <d6p1:float>0.0352076441</d6p1:float>\r\n            <d6p1:float>0.145454392</d6p1:float>\r\n            <d6p1:float>-0.0266913548</d6p1:float>\r\n            <d6p1:float>-0.04380375</d6p1:float>\r\n            <d6p1:float>0.0258271825</d6p1:float>\r\n            <d6p1:float>0.179046869</d6p1:float>\r\n            <d6p1:float>-0.136932582</d6p1:float>\r\n            <d6p1:float>-0.2754411</d6p1:float>\r\n            <d6p1:float>0.103169568</d6p1:float>\r\n            <d6p1:float>-0.185759664</d6p1:float>\r\n            <d6p1:float>0.0673677847</d6p1:float>\r\n            <d6p1:float>-0.359315336</d6p1:float>\r\n            <d6p1:float>-0.06696837</d6p1:float>\r\n            <d6p1:float>0.2573058</d6p1:float>\r\n            <d6p1:float>-0.336355478</d6p1:float>\r\n            <d6p1:float>-0.242318779</d6p1:float>\r\n            <d6p1:float>0.124645934</d6p1:float>\r\n            <d6p1:float>-0.1913952</d6p1:float>\r\n            <d6p1:float>-0.109878771</d6p1:float>\r\n            <d6p1:float>0.274942</d6p1:float>\r\n            <d6p1:float>-0.08526974</d6p1:float>\r\n            <d6p1:float>0.181276247</d6p1:float>\r\n            <d6p1:float>-0.09019703</d6p1:float>\r\n            <d6p1:float>0.313561648</d6p1:float>\r\n            <d6p1:float>-0.221989527</d6p1:float>\r\n            <d6p1:float>0.285922855</d6p1:float>\r\n            <d6p1:float>-0.21642819</d6p1:float>\r\n            <d6p1:float>-0.2730878</d6p1:float>\r\n            <d6p1:float>-0.3131833</d6p1:float>\r\n            <d6p1:float>-0.241817176</d6p1:float>\r\n            <d6p1:float>0.09144679</d6p1:float>\r\n            <d6p1:float>-0.338000625</d6p1:float>\r\n            <d6p1:float>0.153928876</d6p1:float>\r\n            <d6p1:float>0.0286350045</d6p1:float>\r\n            <d6p1:float>0.0152038718</d6p1:float>\r\n            <d6p1:float>-0.253950447</d6p1:float>\r\n            <d6p1:float>-0.08397945</d6p1:float>\r\n            <d6p1:float>0.1558561</d6p1:float>\r\n            <d6p1:float>-0.17066738</d6p1:float>\r\n            <d6p1:float>0.311100483</d6p1:float>\r\n            <d6p1:float>-0.111156486</d6p1:float>\r\n            <d6p1:float>-0.267016321</d6p1:float>\r\n            <d6p1:float>0.129480779</d6p1:float>\r\n            <d6p1:float>0.0350427553</d6p1:float>\r\n            <d6p1:float>0.147410378</d6p1:float>\r\n            <d6p1:float>0.105370432</d6p1:float>\r\n            <d6p1:float>-0.114567086</d6p1:float>\r\n            <d6p1:float>-0.42718935</d6p1:float>\r\n            <d6p1:float>0.127437443</d6p1:float>\r\n            <d6p1:float>-0.137148708</d6p1:float>\r\n            <d6p1:float>0.2794065</d6p1:float>\r\n            <d6p1:float>0.0463777035</d6p1:float>\r\n            <d6p1:float>-0.221068949</d6p1:float>\r\n            <d6p1:float>-0.258843273</d6p1:float>\r\n            <d6p1:float>-0.1729572</d6p1:float>\r\n            <d6p1:float>-0.265909731</d6p1:float>\r\n            <d6p1:float>-0.162640661</d6p1:float>\r\n            <d6p1:float>-0.0289401077</d6p1:float>\r\n            <d6p1:float>-0.103276275</d6p1:float>\r\n            <d6p1:float>0.02620901</d6p1:float>\r\n            <d6p1:float>0.05948317</d6p1:float>\r\n            <d6p1:float>-0.3012638</d6p1:float>\r\n            <d6p1:float>0.042773433</d6p1:float>\r\n            <d6p1:float>-1.3186084</d6p1:float>\r\n            <d6p1:float>0.569534361</d6p1:float>\r\n            <d6p1:float>0.279612869</d6p1:float>\r\n            <d6p1:float>-0.172093391</d6p1:float>\r\n            <d6p1:float>0.126661375</d6p1:float>\r\n            <d6p1:float>0.437877625</d6p1:float>\r\n            <d6p1:float>0.0138841188</d6p1:float>\r\n            <d6p1:float>0.1334072</d6p1:float>\r\n            <d6p1:float>0.1928864</d6p1:float>\r\n            <d6p1:float>0.176881433</d6p1:float>\r\n            <d6p1:float>-0.144674331</d6p1:float>\r\n            <d6p1:float>-0.02853076</d6p1:float>\r\n            <d6p1:float>-0.153278857</d6p1:float>\r\n            <d6p1:float>-0.309471518</d6p1:float>\r\n            <d6p1:float>0.0555440225</d6p1:float>\r\n            <d6p1:float>-0.328682363</d6p1:float>\r\n            <d6p1:float>-0.23615922</d6p1:float>\r\n            <d6p1:float>-0.0127341338</d6p1:float>\r\n            <d6p1:float>-0.2701992</d6p1:float>\r\n            <d6p1:float>-0.129296318</d6p1:float>\r\n            <d6p1:float>0.0746537</d6p1:float>\r\n            <d6p1:float>-0.2218654</d6p1:float>\r\n            <d6p1:float>-0.2645677</d6p1:float>\r\n            <d6p1:float>0.226177678</d6p1:float>\r\n            <d6p1:float>0.104274139</d6p1:float>\r\n            <d6p1:float>-0.254475325</d6p1:float>\r\n            <d6p1:float>-0.286274254</d6p1:float>\r\n            <d6p1:float>-0.0236558579</d6p1:float>\r\n            <d6p1:float>0.02648165</d6p1:float>\r\n            <d6p1:float>-0.434683</d6p1:float>\r\n            <d6p1:float>0.0714041</d6p1:float>\r\n            <d6p1:float>-0.243464887</d6p1:float>\r\n            <d6p1:float>-0.3102106</d6p1:float>\r\n            <d6p1:float>0.283597559</d6p1:float>\r\n            <d6p1:float>-0.340422779</d6p1:float>\r\n            <d6p1:float>-0.0241487753</d6p1:float>\r\n            <d6p1:float>-0.238138482</d6p1:float>\r\n            <d6p1:float>0.268355966</d6p1:float>\r\n            <d6p1:float>0.244930059</d6p1:float>\r\n            <d6p1:float>0.0385601372</d6p1:float>\r\n            <d6p1:float>0.189353451</d6p1:float>\r\n            <d6p1:float>0.219016746</d6p1:float>\r\n            <d6p1:float>-0.3981245</d6p1:float>\r\n            <d6p1:float>-0.259380251</d6p1:float>\r\n            <d6p1:float>0.476791054</d6p1:float>\r\n            <d6p1:float>-1.68675625</d6p1:float>\r\n            <d6p1:float>0.7535355</d6p1:float>\r\n            <d6p1:float>-0.2555277</d6p1:float>\r\n            <d6p1:float>0.0172151718</d6p1:float>\r\n            <d6p1:float>-0.257071584</d6p1:float>\r\n            <d6p1:float>0.322277844</d6p1:float>\r\n            <d6p1:float>-0.6387765</d6p1:float>\r\n            <d6p1:float>-0.250993848</d6p1:float>\r\n            <d6p1:float>-0.08915873</d6p1:float>\r\n            <d6p1:float>-0.383543968</d6p1:float>\r\n            <d6p1:float>-0.341218442</d6p1:float>\r\n            <d6p1:float>-0.568374455</d6p1:float>\r\n            <d6p1:float>0.168397143</d6p1:float>\r\n            <d6p1:float>-0.210104138</d6p1:float>\r\n            <d6p1:float>-0.06649924</d6p1:float>\r\n            <d6p1:float>-0.3385036</d6p1:float>\r\n            <d6p1:float>-0.2909363</d6p1:float>\r\n            <d6p1:float>0.208440587</d6p1:float>\r\n            <d6p1:float>-0.0172635484</d6p1:float>\r\n            <d6p1:float>-0.155002341</d6p1:float>\r\n            <d6p1:float>0.260522246</d6p1:float>\r\n            <d6p1:float>0.0105446214</d6p1:float>\r\n            <d6p1:float>-0.119073637</d6p1:float>\r\n            <d6p1:float>-0.280730784</d6p1:float>\r\n            <d6p1:float>-0.0108599514</d6p1:float>\r\n            <d6p1:float>0.0112559414</d6p1:float>\r\n            <d6p1:float>-0.314122945</d6p1:float>\r\n            <d6p1:float>-0.127242252</d6p1:float>\r\n            <d6p1:float>-0.2441536</d6p1:float>\r\n            <d6p1:float>-0.429500043</d6p1:float>\r\n            <d6p1:float>0.25356105</d6p1:float>\r\n            <d6p1:float>-0.465707779</d6p1:float>\r\n            <d6p1:float>-0.0298487041</d6p1:float>\r\n            <d6p1:float>-0.310261071</d6p1:float>\r\n            <d6p1:float>-0.243010819</d6p1:float>\r\n            <d6p1:float>-0.09664463</d6p1:float>\r\n            <d6p1:float>-0.0394169651</d6p1:float>\r\n            <d6p1:float>-0.2185082</d6p1:float>\r\n            <d6p1:float>-0.0644020662</d6p1:float>\r\n            <d6p1:float>-0.1913584</d6p1:float>\r\n            <d6p1:float>0.0686918646</d6p1:float>\r\n            <d6p1:float>0.2035067</d6p1:float>\r\n            <d6p1:float>-0.265759349</d6p1:float>\r\n            <d6p1:float>-0.09426466</d6p1:float>\r\n            <d6p1:float>-0.123410724</d6p1:float>\r\n            <d6p1:float>-0.112710707</d6p1:float>\r\n            <d6p1:float>-0.184985653</d6p1:float>\r\n            <d6p1:float>-0.4274224</d6p1:float>\r\n            <d6p1:float>-0.193821862</d6p1:float>\r\n            <d6p1:float>-0.0436423868</d6p1:float>\r\n            <d6p1:float>0.2238575</d6p1:float>\r\n            <d6p1:float>0.185174912</d6p1:float>\r\n            <d6p1:float>-0.0189936273</d6p1:float>\r\n            <d6p1:float>-0.19701761</d6p1:float>\r\n            <d6p1:float>0.160796687</d6p1:float>\r\n            <d6p1:float>0.236768022</d6p1:float>\r\n            <d6p1:float>-0.049517408</d6p1:float>\r\n            <d6p1:float>0.2569422</d6p1:float>\r\n            <d6p1:float>-0.269887954</d6p1:float>\r\n            <d6p1:float>0.05858555</d6p1:float>\r\n            <d6p1:float>0.177909255</d6p1:float>\r\n            <d6p1:float>-0.297522873</d6p1:float>\r\n            <d6p1:float>-0.5362002</d6p1:float>\r\n            <d6p1:float>-0.196136758</d6p1:float>\r\n            <d6p1:float>-0.191429168</d6p1:float>\r\n            <d6p1:float>-0.362382323</d6p1:float>\r\n            <d6p1:float>-0.353060454</d6p1:float>\r\n            <d6p1:float>0.0508089028</d6p1:float>\r\n            <d6p1:float>-0.021608904</d6p1:float>\r\n            <d6p1:float>-0.303591669</d6p1:float>\r\n            <d6p1:float>-0.0368680134</d6p1:float>\r\n            <d6p1:float>0.35940966</d6p1:float>\r\n            <d6p1:float>-0.24904713</d6p1:float>\r\n            <d6p1:float>0.000863554655</d6p1:float>\r\n            <d6p1:float>0.177211434</d6p1:float>\r\n            <d6p1:float>-0.275904477</d6p1:float>\r\n            <d6p1:float>-0.120578267</d6p1:float>\r\n            <d6p1:float>0.3572673</d6p1:float>\r\n            <d6p1:float>-0.03699376</d6p1:float>\r\n            <d6p1:float>0.141692355</d6p1:float>\r\n            <d6p1:float>-0.171239525</d6p1:float>\r\n            <d6p1:float>-0.168649465</d6p1:float>\r\n            <d6p1:float>-0.1365483</d6p1:float>\r\n            <d6p1:float>0.0129009653</d6p1:float>\r\n            <d6p1:float>0.0507229939</d6p1:float>\r\n            <d6p1:float>-0.3102475</d6p1:float>\r\n            <d6p1:float>0.00270436332</d6p1:float>\r\n            <d6p1:float>-0.210485041</d6p1:float>\r\n            <d6p1:float>-0.201035917</d6p1:float>\r\n            <d6p1:float>-0.234051779</d6p1:float>\r\n            <d6p1:float>0.153915927</d6p1:float>\r\n            <d6p1:float>-0.0189715214</d6p1:float>\r\n            <d6p1:float>-0.302858025</d6p1:float>\r\n            <d6p1:float>0.20601581</d6p1:float>\r\n            <d6p1:float>-0.0415451936</d6p1:float>\r\n            <d6p1:float>-0.0796675</d6p1:float>\r\n            <d6p1:float>0.230153874</d6p1:float>\r\n            <d6p1:float>-0.134764418</d6p1:float>\r\n            <d6p1:float>-0.0442265235</d6p1:float>\r\n            <d6p1:float>-0.2975642</d6p1:float>\r\n            <d6p1:float>-0.343623966</d6p1:float>\r\n            <d6p1:float>-0.4676388</d6p1:float>\r\n            <d6p1:float>-0.0350814052</d6p1:float>\r\n            <d6p1:float>-0.452439517</d6p1:float>\r\n            <d6p1:float>0.17064856</d6p1:float>\r\n            <d6p1:float>0.274991751</d6p1:float>\r\n            <d6p1:float>-0.118333921</d6p1:float>\r\n            <d6p1:float>-0.08201331</d6p1:float>\r\n            <d6p1:float>-0.439332038</d6p1:float>\r\n            <d6p1:float>0.335536867</d6p1:float>\r\n            <d6p1:float>0.0187374838</d6p1:float>\r\n            <d6p1:float>0.138943061</d6p1:float>\r\n            <d6p1:float>-0.111089081</d6p1:float>\r\n            <d6p1:float>0.0498380847</d6p1:float>\r\n            <d6p1:float>-0.297383636</d6p1:float>\r\n            <d6p1:float>-0.323405</d6p1:float>\r\n            <d6p1:float>0.8012297</d6p1:float>\r\n            <d6p1:float>-1.3412801</d6p1:float>\r\n            <d6p1:float>0.895428538</d6p1:float>\r\n            <d6p1:float>0.2854663</d6p1:float>\r\n            <d6p1:float>0.21794042</d6p1:float>\r\n            <d6p1:float>0.0530578867</d6p1:float>\r\n            <d6p1:float>0.6871187</d6p1:float>\r\n            <d6p1:float>-0.456203878</d6p1:float>\r\n            <d6p1:float>-0.07781827</d6p1:float>\r\n            <d6p1:float>-0.232971445</d6p1:float>\r\n            <d6p1:float>0.304056257</d6p1:float>\r\n            <d6p1:float>0.0838349</d6p1:float>\r\n            <d6p1:float>0.313565731</d6p1:float>\r\n            <d6p1:float>-0.14796181</d6p1:float>\r\n            <d6p1:float>-0.114241846</d6p1:float>\r\n            <d6p1:float>0.230025828</d6p1:float>\r\n            <d6p1:float>0.32579422</d6p1:float>\r\n            <d6p1:float>-0.318974227</d6p1:float>\r\n            <d6p1:float>0.371884435</d6p1:float>\r\n            <d6p1:float>-1.43510425</d6p1:float>\r\n            <d6p1:float>0.7465604</d6p1:float>\r\n            <d6p1:float>0.182178751</d6p1:float>\r\n            <d6p1:float>-0.39135012</d6p1:float>\r\n            <d6p1:float>-0.146340042</d6p1:float>\r\n            <d6p1:float>0.7404711</d6p1:float>\r\n            <d6p1:float>-0.494367629</d6p1:float>\r\n            <d6p1:float>0.00187068759</d6p1:float>\r\n            <d6p1:float>-0.0640564039</d6p1:float>\r\n            <d6p1:float>0.0521159731</d6p1:float>\r\n            <d6p1:float>-0.302795142</d6p1:float>\r\n            <d6p1:float>-0.2385754</d6p1:float>\r\n            <d6p1:float>0.174789414</d6p1:float>\r\n            <d6p1:float>0.133823514</d6p1:float>\r\n            <d6p1:float>-0.254364878</d6p1:float>\r\n            <d6p1:float>0.124180645</d6p1:float>\r\n            <d6p1:float>0.05550214</d6p1:float>\r\n            <d6p1:float>-0.12980929</d6p1:float>\r\n            <d6p1:float>-0.184106216</d6p1:float>\r\n            <d6p1:float>-0.444153041</d6p1:float>\r\n            <d6p1:float>-0.1866802</d6p1:float>\r\n            <d6p1:float>0.143714622</d6p1:float>\r\n            <d6p1:float>-0.395727068</d6p1:float>\r\n            <d6p1:float>0.231310308</d6p1:float>\r\n            <d6p1:float>0.20325157</d6p1:float>\r\n            <d6p1:float>-0.373641</d6p1:float>\r\n            <d6p1:float>0.407272518</d6p1:float>\r\n            <d6p1:float>-0.97562623</d6p1:float>\r\n            <d6p1:float>0.224585339</d6p1:float>\r\n            <d6p1:float>0.123774312</d6p1:float>\r\n            <d6p1:float>0.07968864</d6p1:float>\r\n            <d6p1:float>0.08637614</d6p1:float>\r\n            <d6p1:float>0.490684032</d6p1:float>\r\n            <d6p1:float>0.0978663638</d6p1:float>\r\n            <d6p1:float>-0.2781967</d6p1:float>\r\n            <d6p1:float>-0.028520016</d6p1:float>\r\n            <d6p1:float>-0.149899781</d6p1:float>\r\n            <d6p1:float>-0.301622063</d6p1:float>\r\n            <d6p1:float>-0.195412874</d6p1:float>\r\n            <d6p1:float>0.00716368947</d6p1:float>\r\n            <d6p1:float>0.06935125</d6p1:float>\r\n            <d6p1:float>-0.105930425</d6p1:float>\r\n            <d6p1:float>-0.156158283</d6p1:float>\r\n            <d6p1:float>0.130988389</d6p1:float>\r\n            <d6p1:float>-0.290430218</d6p1:float>\r\n            <d6p1:float>-0.0436347</d6p1:float>\r\n            <d6p1:float>-0.3894385</d6p1:float>\r\n            <d6p1:float>0.08053618</d6p1:float>\r\n            <d6p1:float>0.160211891</d6p1:float>\r\n            <d6p1:float>-0.0259798244</d6p1:float>\r\n            <d6p1:float>-0.102052644</d6p1:float>\r\n            <d6p1:float>-0.243844017</d6p1:float>\r\n            <d6p1:float>-0.0813542455</d6p1:float>\r\n            <d6p1:float>-0.0368994027</d6p1:float>\r\n            <d6p1:float>0.150815338</d6p1:float>\r\n            <d6p1:float>-0.128221527</d6p1:float>\r\n            <d6p1:float>0.205252424</d6p1:float>\r\n            <d6p1:float>0.008552202</d6p1:float>\r\n            <d6p1:float>-0.183922619</d6p1:float>\r\n            <d6p1:float>-0.01730614</d6p1:float>\r\n            <d6p1:float>0.3628607</d6p1:float>\r\n            <d6p1:float>0.181478143</d6p1:float>\r\n            <d6p1:float>-0.256377339</d6p1:float>\r\n            <d6p1:float>0.201529145</d6p1:float>\r\n            <d6p1:float>0.238014027</d6p1:float>\r\n            <d6p1:float>-0.206200972</d6p1:float>\r\n            <d6p1:float>-0.115420535</d6p1:float>\r\n            <d6p1:float>0.14709048</d6p1:float>\r\n            <d6p1:float>0.06631154</d6p1:float>\r\n            <d6p1:float>0.06831525</d6p1:float>\r\n            <d6p1:float>-0.04481599</d6p1:float>\r\n            <d6p1:float>0.208589524</d6p1:float>\r\n            <d6p1:float>-0.009837012</d6p1:float>\r\n            <d6p1:float>-0.250684083</d6p1:float>\r\n            <d6p1:float>0.02933461</d6p1:float>\r\n            <d6p1:float>-0.0554737039</d6p1:float>\r\n            <d6p1:float>0.0296411142</d6p1:float>\r\n            <d6p1:float>-0.06326983</d6p1:float>\r\n            <d6p1:float>0.105000876</d6p1:float>\r\n            <d6p1:float>0.350730538</d6p1:float>\r\n            <d6p1:float>-0.1922931</d6p1:float>\r\n            <d6p1:float>0.104895346</d6p1:float>\r\n            <d6p1:float>0.215693623</d6p1:float>\r\n            <d6p1:float>0.260862917</d6p1:float>\r\n            <d6p1:float>-0.181588352</d6p1:float>\r\n            <d6p1:float>0.100032024</d6p1:float>\r\n            <d6p1:float>0.07455702</d6p1:float>\r\n            <d6p1:float>-0.0252032913</d6p1:float>\r\n            <d6p1:float>-0.0389834978</d6p1:float>\r\n            <d6p1:float>0.352986783</d6p1:float>\r\n            <d6p1:float>-0.171861887</d6p1:float>\r\n            <d6p1:float>0.359849274</d6p1:float>\r\n            <d6p1:float>0.2994789</d6p1:float>\r\n            <d6p1:float>0.114865407</d6p1:float>\r\n            <d6p1:float>0.0278566144</d6p1:float>\r\n            <d6p1:float>0.3797161</d6p1:float>\r\n            <d6p1:float>-0.204003692</d6p1:float>\r\n            <d6p1:float>-0.0258155726</d6p1:float>\r\n            <d6p1:float>0.242280319</d6p1:float>\r\n            <d6p1:float>-0.284357131</d6p1:float>\r\n            <d6p1:float>0.09279593</d6p1:float>\r\n            <d6p1:float>-0.182243586</d6p1:float>\r\n            <d6p1:float>-0.187591657</d6p1:float>\r\n            <d6p1:float>-0.259610981</d6p1:float>\r\n            <d6p1:float>-0.230751649</d6p1:float>\r\n            <d6p1:float>0.0139687965</d6p1:float>\r\n            <d6p1:float>0.0902573541</d6p1:float>\r\n            <d6p1:float>-0.263564825</d6p1:float>\r\n            <d6p1:float>0.08650616</d6p1:float>\r\n            <d6p1:float>-0.281678319</d6p1:float>\r\n            <d6p1:float>0.0264684539</d6p1:float>\r\n            <d6p1:float>-0.1339547</d6p1:float>\r\n            <d6p1:float>0.1585608</d6p1:float>\r\n            <d6p1:float>-0.172321334</d6p1:float>\r\n            <d6p1:float>-0.144456074</d6p1:float>\r\n            <d6p1:float>-0.3147687</d6p1:float>\r\n            <d6p1:float>-0.223699972</d6p1:float>\r\n            <d6p1:float>-0.169102177</d6p1:float>\r\n            <d6p1:float>0.0199153367</d6p1:float>\r\n            <d6p1:float>-0.06436743</d6p1:float>\r\n            <d6p1:float>0.0949862</d6p1:float>\r\n            <d6p1:float>0.271493226</d6p1:float>\r\n            <d6p1:float>-0.09103191</d6p1:float>\r\n            <d6p1:float>0.212133735</d6p1:float>\r\n            <d6p1:float>0.259684265</d6p1:float>\r\n            <d6p1:float>0.08976745</d6p1:float>\r\n            <d6p1:float>0.301833838</d6p1:float>\r\n            <d6p1:float>0.160189956</d6p1:float>\r\n            <d6p1:float>0.05225989</d6p1:float>\r\n            <d6p1:float>0.1753132</d6p1:float>\r\n            <d6p1:float>-0.206804022</d6p1:float>\r\n            <d6p1:float>-0.2865898</d6p1:float>\r\n            <d6p1:float>0.0247859117</d6p1:float>\r\n            <d6p1:float>0.06774576</d6p1:float>\r\n            <d6p1:float>0.0823864043</d6p1:float>\r\n            <d6p1:float>-0.681237161</d6p1:float>\r\n            <d6p1:float>0.6092618</d6p1:float>\r\n            <d6p1:float>0.255529761</d6p1:float>\r\n            <d6p1:float>0.110177852</d6p1:float>\r\n            <d6p1:float>0.22502242</d6p1:float>\r\n            <d6p1:float>0.4884047</d6p1:float>\r\n            <d6p1:float>-0.218172327</d6p1:float>\r\n          </d5p1:Data>\r\n        </d3p1:_x003C_Storage_x003E_k__BackingField>\r\n        <_columnCount xmlns=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra.Single\">50</_columnCount>\r\n        <_rowCount xmlns=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra.Single\">9</_rowCount>\r\n        <_values xmlns:d5p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Ref=\"15\" i:nil=\"true\" xmlns=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra.Single\" />\r\n      </d3p1:MatrixOffloat>\r\n      <d3p1:MatrixOffloat z:Id=\"16\" xmlns:d4p1=\"MathNet.Numerics.LinearAlgebra.Single\" i:type=\"d4p1:DenseMatrix\">\r\n        <d3p1:_x003C_ColumnCount_x003E_k__BackingField>1</d3p1:_x003C_ColumnCount_x003E_k__BackingField>\r\n        <d3p1:_x003C_RowCount_x003E_k__BackingField>50</d3p1:_x003C_RowCount_x003E_k__BackingField>\r\n        <d3p1:_x003C_Storage_x003E_k__BackingField xmlns:d5p1=\"urn:MathNet/Numerics/LinearAlgebra\" z:Id=\"17\" i:type=\"d5p1:DenseColumnMajorMatrixStorageOffloat\">\r\n          <d5p1:RowCount>50</d5p1:RowCount>\r\n          <d5p1:ColumnCount>1</d5p1:ColumnCount>\r\n          <d5p1:Data xmlns:d6p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"18\" z:Size=\"50\">\r\n            <d6p1:float>0.0249628462</d6p1:float>\r\n            <d6p1:float>0.156514</d6p1:float>\r\n            <d6p1:float>1.03811181</d6p1:float>\r\n            <d6p1:float>0.262249</d6p1:float>\r\n            <d6p1:float>0.732978463</d6p1:float>\r\n            <d6p1:float>0.716722548</d6p1:float>\r\n            <d6p1:float>-0.247894436</d6p1:float>\r\n            <d6p1:float>-0.209872156</d6p1:float>\r\n            <d6p1:float>0.10969153</d6p1:float>\r\n            <d6p1:float>-0.138593286</d6p1:float>\r\n            <d6p1:float>0.1309375</d6p1:float>\r\n            <d6p1:float>0.0745508745</d6p1:float>\r\n            <d6p1:float>0.004284214</d6p1:float>\r\n            <d6p1:float>0.23096478</d6p1:float>\r\n            <d6p1:float>-0.125683188</d6p1:float>\r\n            <d6p1:float>-0.163527057</d6p1:float>\r\n            <d6p1:float>-0.292192459</d6p1:float>\r\n            <d6p1:float>0.005708625</d6p1:float>\r\n            <d6p1:float>0.5455223</d6p1:float>\r\n            <d6p1:float>0.654336154</d6p1:float>\r\n            <d6p1:float>0.4065296</d6p1:float>\r\n            <d6p1:float>-0.07228146</d6p1:float>\r\n            <d6p1:float>0.554572761</d6p1:float>\r\n            <d6p1:float>-0.121918954</d6p1:float>\r\n            <d6p1:float>-0.145961672</d6p1:float>\r\n            <d6p1:float>-0.009522332</d6p1:float>\r\n            <d6p1:float>0.210615635</d6p1:float>\r\n            <d6p1:float>0.467155218</d6p1:float>\r\n            <d6p1:float>-0.114550292</d6p1:float>\r\n            <d6p1:float>0.06897701</d6p1:float>\r\n            <d6p1:float>-0.11806348</d6p1:float>\r\n            <d6p1:float>0.26305455</d6p1:float>\r\n            <d6p1:float>1.5766207</d6p1:float>\r\n            <d6p1:float>-0.147744715</d6p1:float>\r\n            <d6p1:float>1.27446258</d6p1:float>\r\n            <d6p1:float>-0.0689658746</d6p1:float>\r\n            <d6p1:float>0.5443362</d6p1:float>\r\n            <d6p1:float>0.428513139</d6p1:float>\r\n            <d6p1:float>-0.09295752</d6p1:float>\r\n            <d6p1:float>-0.476350367</d6p1:float>\r\n            <d6p1:float>-0.192215055</d6p1:float>\r\n            <d6p1:float>0.08837687</d6p1:float>\r\n            <d6p1:float>0.08589814</d6p1:float>\r\n            <d6p1:float>-0.6825381</d6p1:float>\r\n            <d6p1:float>0.559921563</d6p1:float>\r\n            <d6p1:float>0.165702358</d6p1:float>\r\n            <d6p1:float>-0.08204819</d6p1:float>\r\n            <d6p1:float>-0.0535372123</d6p1:float>\r\n            <d6p1:float>-0.229036182</d6p1:float>\r\n            <d6p1:float>-0.452031255</d6p1:float>\r\n          </d5p1:Data>\r\n        </d3p1:_x003C_Storage_x003E_k__BackingField>\r\n        <_columnCount xmlns=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra.Single\">1</_columnCount>\r\n        <_rowCount xmlns=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra.Single\">50</_rowCount>\r\n        <_values xmlns:d5p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Ref=\"18\" i:nil=\"true\" xmlns=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra.Single\" />\r\n      </d3p1:MatrixOffloat>\r\n    </m_weights>\r\n  </m_model>\r\n</RegressionNeuralNetModel>";
    }
}
