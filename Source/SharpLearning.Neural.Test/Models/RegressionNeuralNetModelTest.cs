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

            Assert.AreEqual(1.0586877414339495, error, 1e-6);
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

            Assert.AreEqual(1.0586877414339495, error, 1e-6);
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
                {"F2", 100},
                {"F3", 97.307936647518},
                {"F4", 75.2084048520556},
                {"F10",55.3195308540928},	
                {"F1", 48.9049215309262},	
                {"F8", 41.9399034166079},	
                {"F7", 40.2258638757031},	
                {"F5", 12.6275023801994},	
                {"F6", 10.5242336434216},    
            };

            Assert.AreEqual(expected.Count, actual.Count);
            var zip = expected.Zip(actual, (e, a) => new { Expected = e, Actual = a });

            foreach (var item in zip)
            {
                Assert.AreEqual(item.Expected.Key, item.Actual.Key);
                Assert.AreEqual(item.Expected.Value, item.Actual.Value, 1e-6);
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
                1.6547751426696777, 
                3.3836572170257568, 
                3.2925674915313721, 
                2.544795036315918,  
                0.42727169394493103,
                0.35610395669937134,
                1.3611053228378296, 
                1.419102668762207,  
                1.8718241453170776, 
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

            Assert.AreEqual(1.0586876986141716, error, 0.0000001);
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
        "<?xml version=\"1.0\" encoding=\"utf-16\"?>\r\n<RegressionNeuralNetModel xmlns:i=\"http://www.w3.org/2001/XMLSchema-instance\" z:Id=\"1\" xmlns:z=\"http://schemas.microsoft.com/2003/10/Serialization/\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.Neural.Models\">\r\n  <m_model z:Id=\"2\">\r\n    <m_hiddenActivation z:Id=\"3\" xmlns:d3p1=\"SharpLearning.Neural.Activations\" i:type=\"d3p1:ReluActivation\" />\r\n    <m_intercepts xmlns:d3p1=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra\" z:Id=\"4\" z:Size=\"2\">\r\n      <d3p1:VectorOffloat z:Id=\"5\" xmlns:d4p1=\"MathNet.Numerics.LinearAlgebra.Single\" i:type=\"d4p1:DenseVector\">\r\n        <d3p1:_x003C_Count_x003E_k__BackingField>50</d3p1:_x003C_Count_x003E_k__BackingField>\r\n        <d3p1:_x003C_Storage_x003E_k__BackingField xmlns:d5p1=\"urn:MathNet/Numerics/LinearAlgebra\" z:Id=\"6\" i:type=\"d5p1:DenseVectorStorageOffloat\">\r\n          <d5p1:Length>50</d5p1:Length>\r\n          <d5p1:Data xmlns:d6p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"7\" z:Size=\"50\">\r\n            <d6p1:float>-0.395941228</d6p1:float>\r\n            <d6p1:float>-0.183560222</d6p1:float>\r\n            <d6p1:float>-0.025522884</d6p1:float>\r\n            <d6p1:float>-0.337368965</d6p1:float>\r\n            <d6p1:float>-0.130507991</d6p1:float>\r\n            <d6p1:float>-0.372908354</d6p1:float>\r\n            <d6p1:float>-0.23732689</d6p1:float>\r\n            <d6p1:float>-0.0940536857</d6p1:float>\r\n            <d6p1:float>-0.0919243842</d6p1:float>\r\n            <d6p1:float>-0.185273111</d6p1:float>\r\n            <d6p1:float>0.20269002</d6p1:float>\r\n            <d6p1:float>-0.0436020344</d6p1:float>\r\n            <d6p1:float>-0.3410131</d6p1:float>\r\n            <d6p1:float>0.139434427</d6p1:float>\r\n            <d6p1:float>0.191736013</d6p1:float>\r\n            <d6p1:float>-0.266203433</d6p1:float>\r\n            <d6p1:float>-0.198966652</d6p1:float>\r\n            <d6p1:float>-0.6055196</d6p1:float>\r\n            <d6p1:float>0.148847371</d6p1:float>\r\n            <d6p1:float>0.419017971</d6p1:float>\r\n            <d6p1:float>-0.8441365</d6p1:float>\r\n            <d6p1:float>-0.276023</d6p1:float>\r\n            <d6p1:float>-0.353987038</d6p1:float>\r\n            <d6p1:float>0.0105249491</d6p1:float>\r\n            <d6p1:float>0.0208697822</d6p1:float>\r\n            <d6p1:float>-0.3443884</d6p1:float>\r\n            <d6p1:float>-0.209838614</d6p1:float>\r\n            <d6p1:float>-0.0163471065</d6p1:float>\r\n            <d6p1:float>-0.570923865</d6p1:float>\r\n            <d6p1:float>0.044704318</d6p1:float>\r\n            <d6p1:float>-0.655472934</d6p1:float>\r\n            <d6p1:float>0.187395379</d6p1:float>\r\n            <d6p1:float>0.364384323</d6p1:float>\r\n            <d6p1:float>-0.09523873</d6p1:float>\r\n            <d6p1:float>0.3010193</d6p1:float>\r\n            <d6p1:float>-0.279910564</d6p1:float>\r\n            <d6p1:float>-0.411004066</d6p1:float>\r\n            <d6p1:float>0.254441231</d6p1:float>\r\n            <d6p1:float>0.171242058</d6p1:float>\r\n            <d6p1:float>-0.4419059</d6p1:float>\r\n            <d6p1:float>-0.008835569</d6p1:float>\r\n            <d6p1:float>-0.3520199</d6p1:float>\r\n            <d6p1:float>-0.219318613</d6p1:float>\r\n            <d6p1:float>-0.331238955</d6p1:float>\r\n            <d6p1:float>-0.15115191</d6p1:float>\r\n            <d6p1:float>-0.2900371</d6p1:float>\r\n            <d6p1:float>-0.3807219</d6p1:float>\r\n            <d6p1:float>0.006857207</d6p1:float>\r\n            <d6p1:float>-0.294352263</d6p1:float>\r\n            <d6p1:float>-0.275658548</d6p1:float>\r\n          </d5p1:Data>\r\n        </d3p1:_x003C_Storage_x003E_k__BackingField>\r\n        <_length xmlns=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra.Single\">50</_length>\r\n        <_values xmlns:d5p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Ref=\"7\" i:nil=\"true\" xmlns=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra.Single\" />\r\n      </d3p1:VectorOffloat>\r\n      <d3p1:VectorOffloat z:Id=\"8\" xmlns:d4p1=\"MathNet.Numerics.LinearAlgebra.Single\" i:type=\"d4p1:DenseVector\">\r\n        <d3p1:_x003C_Count_x003E_k__BackingField>1</d3p1:_x003C_Count_x003E_k__BackingField>\r\n        <d3p1:_x003C_Storage_x003E_k__BackingField xmlns:d5p1=\"urn:MathNet/Numerics/LinearAlgebra\" z:Id=\"9\" i:type=\"d5p1:DenseVectorStorageOffloat\">\r\n          <d5p1:Length>1</d5p1:Length>\r\n          <d5p1:Data xmlns:d6p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"10\" z:Size=\"1\">\r\n            <d6p1:float>1.62016308</d6p1:float>\r\n          </d5p1:Data>\r\n        </d3p1:_x003C_Storage_x003E_k__BackingField>\r\n        <_length xmlns=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra.Single\">1</_length>\r\n        <_values xmlns:d5p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Ref=\"10\" i:nil=\"true\" xmlns=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra.Single\" />\r\n      </d3p1:VectorOffloat>\r\n    </m_intercepts>\r\n    <m_layes>3</m_layes>\r\n    <m_outputActivation z:Id=\"11\" xmlns:d3p1=\"SharpLearning.Neural.Activations\" i:type=\"d3p1:IdentityActivation\" />\r\n    <m_weights xmlns:d3p1=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra\" z:Id=\"12\" z:Size=\"2\">\r\n      <d3p1:MatrixOffloat z:Id=\"13\" xmlns:d4p1=\"MathNet.Numerics.LinearAlgebra.Single\" i:type=\"d4p1:DenseMatrix\">\r\n        <d3p1:_x003C_ColumnCount_x003E_k__BackingField>50</d3p1:_x003C_ColumnCount_x003E_k__BackingField>\r\n        <d3p1:_x003C_RowCount_x003E_k__BackingField>9</d3p1:_x003C_RowCount_x003E_k__BackingField>\r\n        <d3p1:_x003C_Storage_x003E_k__BackingField xmlns:d5p1=\"urn:MathNet/Numerics/LinearAlgebra\" z:Id=\"14\" i:type=\"d5p1:DenseColumnMajorMatrixStorageOffloat\">\r\n          <d5p1:RowCount>9</d5p1:RowCount>\r\n          <d5p1:ColumnCount>50</d5p1:ColumnCount>\r\n          <d5p1:Data xmlns:d6p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"15\" z:Size=\"450\">\r\n            <d6p1:float>0.1298725</d6p1:float>\r\n            <d6p1:float>-0.184331164</d6p1:float>\r\n            <d6p1:float>-0.214240521</d6p1:float>\r\n            <d6p1:float>-0.208714</d6p1:float>\r\n            <d6p1:float>-0.292444021</d6p1:float>\r\n            <d6p1:float>-0.0586145744</d6p1:float>\r\n            <d6p1:float>-0.004819561</d6p1:float>\r\n            <d6p1:float>-0.00499042263</d6p1:float>\r\n            <d6p1:float>0.0298287719</d6p1:float>\r\n            <d6p1:float>-0.219470724</d6p1:float>\r\n            <d6p1:float>-0.242144614</d6p1:float>\r\n            <d6p1:float>0.181256011</d6p1:float>\r\n            <d6p1:float>-0.295535117</d6p1:float>\r\n            <d6p1:float>0.06332194</d6p1:float>\r\n            <d6p1:float>-0.0734836</d6p1:float>\r\n            <d6p1:float>0.196236551</d6p1:float>\r\n            <d6p1:float>-0.237337217</d6p1:float>\r\n            <d6p1:float>0.100919366</d6p1:float>\r\n            <d6p1:float>-0.266568035</d6p1:float>\r\n            <d6p1:float>0.8799391</d6p1:float>\r\n            <d6p1:float>-0.9676782</d6p1:float>\r\n            <d6p1:float>0.937274754</d6p1:float>\r\n            <d6p1:float>0.409720838</d6p1:float>\r\n            <d6p1:float>-0.23349762</d6p1:float>\r\n            <d6p1:float>-0.159609824</d6p1:float>\r\n            <d6p1:float>0.6928156</d6p1:float>\r\n            <d6p1:float>-0.173752367</d6p1:float>\r\n            <d6p1:float>0.160548061</d6p1:float>\r\n            <d6p1:float>-0.356597453</d6p1:float>\r\n            <d6p1:float>-0.1393581</d6p1:float>\r\n            <d6p1:float>-0.03432324</d6p1:float>\r\n            <d6p1:float>-0.2685373</d6p1:float>\r\n            <d6p1:float>-0.159462839</d6p1:float>\r\n            <d6p1:float>-0.03960078</d6p1:float>\r\n            <d6p1:float>-0.256223053</d6p1:float>\r\n            <d6p1:float>0.148394167</d6p1:float>\r\n            <d6p1:float>-0.6462106</d6p1:float>\r\n            <d6p1:float>0.429979831</d6p1:float>\r\n            <d6p1:float>-1.10179043</d6p1:float>\r\n            <d6p1:float>0.4358337</d6p1:float>\r\n            <d6p1:float>0.111235015</d6p1:float>\r\n            <d6p1:float>-0.172258809</d6p1:float>\r\n            <d6p1:float>-0.18239522</d6p1:float>\r\n            <d6p1:float>0.09748658</d6p1:float>\r\n            <d6p1:float>-0.548909843</d6p1:float>\r\n            <d6p1:float>-0.0318459943</d6p1:float>\r\n            <d6p1:float>-0.03271685</d6p1:float>\r\n            <d6p1:float>-0.380817115</d6p1:float>\r\n            <d6p1:float>-0.03413612</d6p1:float>\r\n            <d6p1:float>-0.169339746</d6p1:float>\r\n            <d6p1:float>0.0600786768</d6p1:float>\r\n            <d6p1:float>-0.3417103</d6p1:float>\r\n            <d6p1:float>-0.0165554788</d6p1:float>\r\n            <d6p1:float>0.274693847</d6p1:float>\r\n            <d6p1:float>0.09593287</d6p1:float>\r\n            <d6p1:float>-0.351209164</d6p1:float>\r\n            <d6p1:float>0.07293062</d6p1:float>\r\n            <d6p1:float>-0.350080341</d6p1:float>\r\n            <d6p1:float>0.09418993</d6p1:float>\r\n            <d6p1:float>0.0197674036</d6p1:float>\r\n            <d6p1:float>0.02519921</d6p1:float>\r\n            <d6p1:float>0.165965065</d6p1:float>\r\n            <d6p1:float>-0.0235447828</d6p1:float>\r\n            <d6p1:float>-0.353425771</d6p1:float>\r\n            <d6p1:float>0.0489255264</d6p1:float>\r\n            <d6p1:float>-0.06267486</d6p1:float>\r\n            <d6p1:float>0.11040318</d6p1:float>\r\n            <d6p1:float>-0.08525629</d6p1:float>\r\n            <d6p1:float>-0.03752803</d6p1:float>\r\n            <d6p1:float>-0.0141265923</d6p1:float>\r\n            <d6p1:float>0.162686333</d6p1:float>\r\n            <d6p1:float>-0.139544725</d6p1:float>\r\n            <d6p1:float>-0.275566041</d6p1:float>\r\n            <d6p1:float>0.102681808</d6p1:float>\r\n            <d6p1:float>-0.185759664</d6p1:float>\r\n            <d6p1:float>0.0661543459</d6p1:float>\r\n            <d6p1:float>-0.359484762</d6p1:float>\r\n            <d6p1:float>-0.068384774</d6p1:float>\r\n            <d6p1:float>0.2571045</d6p1:float>\r\n            <d6p1:float>-0.336355478</d6p1:float>\r\n            <d6p1:float>-0.242318779</d6p1:float>\r\n            <d6p1:float>0.179052517</d6p1:float>\r\n            <d6p1:float>-0.17947495</d6p1:float>\r\n            <d6p1:float>-0.153407708</d6p1:float>\r\n            <d6p1:float>0.3648379</d6p1:float>\r\n            <d6p1:float>-0.07940045</d6p1:float>\r\n            <d6p1:float>0.229866087</d6p1:float>\r\n            <d6p1:float>-0.0349584259</d6p1:float>\r\n            <d6p1:float>0.393849432</d6p1:float>\r\n            <d6p1:float>-0.189260527</d6p1:float>\r\n            <d6p1:float>0.226842821</d6p1:float>\r\n            <d6p1:float>-0.2262467</d6p1:float>\r\n            <d6p1:float>-0.275009185</d6p1:float>\r\n            <d6p1:float>-0.34509322</d6p1:float>\r\n            <d6p1:float>-0.250653923</d6p1:float>\r\n            <d6p1:float>0.0859776661</d6p1:float>\r\n            <d6p1:float>-0.390028834</d6p1:float>\r\n            <d6p1:float>0.1012677</d6p1:float>\r\n            <d6p1:float>0.00188159733</d6p1:float>\r\n            <d6p1:float>0.0115556307</d6p1:float>\r\n            <d6p1:float>-0.25983268</d6p1:float>\r\n            <d6p1:float>-0.08434076</d6p1:float>\r\n            <d6p1:float>0.140674561</d6p1:float>\r\n            <d6p1:float>-0.1746854</d6p1:float>\r\n            <d6p1:float>0.295221061</d6p1:float>\r\n            <d6p1:float>-0.115970388</d6p1:float>\r\n            <d6p1:float>-0.26708436</d6p1:float>\r\n            <d6p1:float>0.126467749</d6p1:float>\r\n            <d6p1:float>0.03504274</d6p1:float>\r\n            <d6p1:float>0.147410378</d6p1:float>\r\n            <d6p1:float>0.105370417</d6p1:float>\r\n            <d6p1:float>-0.114567086</d6p1:float>\r\n            <d6p1:float>-0.42718935</d6p1:float>\r\n            <d6p1:float>0.127437443</d6p1:float>\r\n            <d6p1:float>-0.137148708</d6p1:float>\r\n            <d6p1:float>0.2794065</d6p1:float>\r\n            <d6p1:float>0.0463777073</d6p1:float>\r\n            <d6p1:float>-0.221068949</d6p1:float>\r\n            <d6p1:float>-0.258843273</d6p1:float>\r\n            <d6p1:float>-0.1729572</d6p1:float>\r\n            <d6p1:float>-0.265909731</d6p1:float>\r\n            <d6p1:float>-0.162640661</d6p1:float>\r\n            <d6p1:float>-0.0289401077</d6p1:float>\r\n            <d6p1:float>-0.103276275</d6p1:float>\r\n            <d6p1:float>0.02620901</d6p1:float>\r\n            <d6p1:float>0.05948317</d6p1:float>\r\n            <d6p1:float>-0.246091545</d6p1:float>\r\n            <d6p1:float>0.110480972</d6p1:float>\r\n            <d6p1:float>-1.30349672</d6p1:float>\r\n            <d6p1:float>0.617980361</d6p1:float>\r\n            <d6p1:float>0.2264075</d6p1:float>\r\n            <d6p1:float>-0.172567755</d6p1:float>\r\n            <d6p1:float>0.152556047</d6p1:float>\r\n            <d6p1:float>0.54723835</d6p1:float>\r\n            <d6p1:float>-0.0558111556</d6p1:float>\r\n            <d6p1:float>0.120305374</d6p1:float>\r\n            <d6p1:float>0.172308192</d6p1:float>\r\n            <d6p1:float>0.156616211</d6p1:float>\r\n            <d6p1:float>-0.163385257</d6p1:float>\r\n            <d6p1:float>-0.0338732935</d6p1:float>\r\n            <d6p1:float>-0.159584031</d6p1:float>\r\n            <d6p1:float>-0.313037634</d6p1:float>\r\n            <d6p1:float>0.0352666862</d6p1:float>\r\n            <d6p1:float>-0.328682363</d6p1:float>\r\n            <d6p1:float>-0.23615922</d6p1:float>\r\n            <d6p1:float>-0.0127341338</d6p1:float>\r\n            <d6p1:float>-0.2701992</d6p1:float>\r\n            <d6p1:float>-0.129296318</d6p1:float>\r\n            <d6p1:float>0.0746537</d6p1:float>\r\n            <d6p1:float>-0.2218654</d6p1:float>\r\n            <d6p1:float>-0.2645677</d6p1:float>\r\n            <d6p1:float>0.226177678</d6p1:float>\r\n            <d6p1:float>0.104274139</d6p1:float>\r\n            <d6p1:float>-0.254475325</d6p1:float>\r\n            <d6p1:float>-0.286274254</d6p1:float>\r\n            <d6p1:float>-0.0236558579</d6p1:float>\r\n            <d6p1:float>0.02648165</d6p1:float>\r\n            <d6p1:float>-0.434683</d6p1:float>\r\n            <d6p1:float>0.0714041</d6p1:float>\r\n            <d6p1:float>-0.243464887</d6p1:float>\r\n            <d6p1:float>-0.3102106</d6p1:float>\r\n            <d6p1:float>0.283597559</d6p1:float>\r\n            <d6p1:float>-0.3981646</d6p1:float>\r\n            <d6p1:float>0.03263278</d6p1:float>\r\n            <d6p1:float>-0.280015826</d6p1:float>\r\n            <d6p1:float>0.2848555</d6p1:float>\r\n            <d6p1:float>0.210756257</d6p1:float>\r\n            <d6p1:float>-0.0365558267</d6p1:float>\r\n            <d6p1:float>0.104563</d6p1:float>\r\n            <d6p1:float>0.264447331</d6p1:float>\r\n            <d6p1:float>-0.562566757</d6p1:float>\r\n            <d6p1:float>-0.223094791</d6p1:float>\r\n            <d6p1:float>0.824646</d6p1:float>\r\n            <d6p1:float>-0.967685461</d6p1:float>\r\n            <d6p1:float>0.944524348</d6p1:float>\r\n            <d6p1:float>0.07035799</d6p1:float>\r\n            <d6p1:float>-0.0136585282</d6p1:float>\r\n            <d6p1:float>-0.3074975</d6p1:float>\r\n            <d6p1:float>0.300046265</d6p1:float>\r\n            <d6p1:float>-0.8207432</d6p1:float>\r\n            <d6p1:float>-0.250993848</d6p1:float>\r\n            <d6p1:float>-0.08915873</d6p1:float>\r\n            <d6p1:float>-0.383543968</d6p1:float>\r\n            <d6p1:float>-0.341218442</d6p1:float>\r\n            <d6p1:float>-0.568374455</d6p1:float>\r\n            <d6p1:float>0.168397143</d6p1:float>\r\n            <d6p1:float>-0.210104138</d6p1:float>\r\n            <d6p1:float>-0.06649924</d6p1:float>\r\n            <d6p1:float>-0.3385036</d6p1:float>\r\n            <d6p1:float>-0.2909363</d6p1:float>\r\n            <d6p1:float>0.219095856</d6p1:float>\r\n            <d6p1:float>-0.0172635484</d6p1:float>\r\n            <d6p1:float>-0.154836357</d6p1:float>\r\n            <d6p1:float>0.271177441</d6p1:float>\r\n            <d6p1:float>0.0105446214</d6p1:float>\r\n            <d6p1:float>-0.117865488</d6p1:float>\r\n            <d6p1:float>-0.280730784</d6p1:float>\r\n            <d6p1:float>-0.0108599514</d6p1:float>\r\n            <d6p1:float>0.0112559414</d6p1:float>\r\n            <d6p1:float>-0.314122945</d6p1:float>\r\n            <d6p1:float>-0.127242252</d6p1:float>\r\n            <d6p1:float>-0.2441536</d6p1:float>\r\n            <d6p1:float>-0.429500043</d6p1:float>\r\n            <d6p1:float>0.25356105</d6p1:float>\r\n            <d6p1:float>-0.465707779</d6p1:float>\r\n            <d6p1:float>-0.0298487041</d6p1:float>\r\n            <d6p1:float>-0.310261071</d6p1:float>\r\n            <d6p1:float>-0.243010819</d6p1:float>\r\n            <d6p1:float>-0.09664463</d6p1:float>\r\n            <d6p1:float>-0.0394169651</d6p1:float>\r\n            <d6p1:float>-0.2185082</d6p1:float>\r\n            <d6p1:float>-0.0644020662</d6p1:float>\r\n            <d6p1:float>-0.1913584</d6p1:float>\r\n            <d6p1:float>0.0686918646</d6p1:float>\r\n            <d6p1:float>0.2035067</d6p1:float>\r\n            <d6p1:float>-0.265759349</d6p1:float>\r\n            <d6p1:float>-0.04151555</d6p1:float>\r\n            <d6p1:float>-0.123410724</d6p1:float>\r\n            <d6p1:float>-0.112710707</d6p1:float>\r\n            <d6p1:float>-0.151276678</d6p1:float>\r\n            <d6p1:float>-0.4274224</d6p1:float>\r\n            <d6p1:float>-0.188238323</d6p1:float>\r\n            <d6p1:float>8.314487E-05</d6p1:float>\r\n            <d6p1:float>0.283639848</d6p1:float>\r\n            <d6p1:float>0.217996612</d6p1:float>\r\n            <d6p1:float>-0.0170437563</d6p1:float>\r\n            <d6p1:float>-0.1905887</d6p1:float>\r\n            <d6p1:float>0.1646782</d6p1:float>\r\n            <d6p1:float>0.251462072</d6p1:float>\r\n            <d6p1:float>-0.0478973836</d6p1:float>\r\n            <d6p1:float>0.2689775</d6p1:float>\r\n            <d6p1:float>-0.268182933</d6p1:float>\r\n            <d6p1:float>0.06261141</d6p1:float>\r\n            <d6p1:float>0.177909255</d6p1:float>\r\n            <d6p1:float>-0.297522873</d6p1:float>\r\n            <d6p1:float>-0.5362002</d6p1:float>\r\n            <d6p1:float>-0.196136758</d6p1:float>\r\n            <d6p1:float>-0.191429168</d6p1:float>\r\n            <d6p1:float>-0.362382323</d6p1:float>\r\n            <d6p1:float>-0.353060454</d6p1:float>\r\n            <d6p1:float>0.0508089028</d6p1:float>\r\n            <d6p1:float>-0.021608904</d6p1:float>\r\n            <d6p1:float>-0.303591669</d6p1:float>\r\n            <d6p1:float>-0.118335292</d6p1:float>\r\n            <d6p1:float>0.485612</d6p1:float>\r\n            <d6p1:float>-0.257328749</d6p1:float>\r\n            <d6p1:float>0.0626799</d6p1:float>\r\n            <d6p1:float>0.2511581</d6p1:float>\r\n            <d6p1:float>-0.234781861</d6p1:float>\r\n            <d6p1:float>-0.215303421</d6p1:float>\r\n            <d6p1:float>0.347661883</d6p1:float>\r\n            <d6p1:float>-0.196484536</d6p1:float>\r\n            <d6p1:float>0.141692355</d6p1:float>\r\n            <d6p1:float>-0.171239525</d6p1:float>\r\n            <d6p1:float>-0.168649465</d6p1:float>\r\n            <d6p1:float>-0.1365483</d6p1:float>\r\n            <d6p1:float>0.0129009653</d6p1:float>\r\n            <d6p1:float>0.0507229939</d6p1:float>\r\n            <d6p1:float>-0.3102475</d6p1:float>\r\n            <d6p1:float>0.00270436332</d6p1:float>\r\n            <d6p1:float>-0.210485041</d6p1:float>\r\n            <d6p1:float>-0.174148515</d6p1:float>\r\n            <d6p1:float>-0.216657266</d6p1:float>\r\n            <d6p1:float>0.1931897</d6p1:float>\r\n            <d6p1:float>-0.009011732</d6p1:float>\r\n            <d6p1:float>-0.283731461</d6p1:float>\r\n            <d6p1:float>0.190702543</d6p1:float>\r\n            <d6p1:float>-0.0209164675</d6p1:float>\r\n            <d6p1:float>-0.076585196</d6p1:float>\r\n            <d6p1:float>0.282044321</d6p1:float>\r\n            <d6p1:float>-0.134764418</d6p1:float>\r\n            <d6p1:float>-0.0442265235</d6p1:float>\r\n            <d6p1:float>-0.2975642</d6p1:float>\r\n            <d6p1:float>-0.343623966</d6p1:float>\r\n            <d6p1:float>-0.4676388</d6p1:float>\r\n            <d6p1:float>-0.0350814052</d6p1:float>\r\n            <d6p1:float>-0.452439517</d6p1:float>\r\n            <d6p1:float>0.17064856</d6p1:float>\r\n            <d6p1:float>0.274991751</d6p1:float>\r\n            <d6p1:float>-0.09989672</d6p1:float>\r\n            <d6p1:float>0.0133191422</d6p1:float>\r\n            <d6p1:float>-0.346444547</d6p1:float>\r\n            <d6p1:float>0.408254981</d6p1:float>\r\n            <d6p1:float>0.07191229</d6p1:float>\r\n            <d6p1:float>0.111898221</d6p1:float>\r\n            <d6p1:float>-0.112343244</d6p1:float>\r\n            <d6p1:float>0.0775960162</d6p1:float>\r\n            <d6p1:float>-0.383678138</d6p1:float>\r\n            <d6p1:float>-0.491129816</d6p1:float>\r\n            <d6p1:float>1.158382</d6p1:float>\r\n            <d6p1:float>-1.31744611</d6p1:float>\r\n            <d6p1:float>1.121148</d6p1:float>\r\n            <d6p1:float>0.4022663</d6p1:float>\r\n            <d6p1:float>0.02243413</d6p1:float>\r\n            <d6p1:float>-0.169509</d6p1:float>\r\n            <d6p1:float>0.621958</d6p1:float>\r\n            <d6p1:float>-0.6706347</d6p1:float>\r\n            <d6p1:float>-0.09815408</d6p1:float>\r\n            <d6p1:float>-0.3154738</d6p1:float>\r\n            <d6p1:float>0.3889796</d6p1:float>\r\n            <d6p1:float>-0.0115597341</d6p1:float>\r\n            <d6p1:float>0.185815066</d6p1:float>\r\n            <d6p1:float>-0.158945337</d6p1:float>\r\n            <d6p1:float>-0.175630555</d6p1:float>\r\n            <d6p1:float>0.256600559</d6p1:float>\r\n            <d6p1:float>0.295713</d6p1:float>\r\n            <d6p1:float>-0.32948187</d6p1:float>\r\n            <d6p1:float>0.8336907</d6p1:float>\r\n            <d6p1:float>-1.09576941</d6p1:float>\r\n            <d6p1:float>1.065762</d6p1:float>\r\n            <d6p1:float>0.5188636</d6p1:float>\r\n            <d6p1:float>-0.5105563</d6p1:float>\r\n            <d6p1:float>-0.218767539</d6p1:float>\r\n            <d6p1:float>0.697285831</d6p1:float>\r\n            <d6p1:float>-0.705355346</d6p1:float>\r\n            <d6p1:float>0.00187068759</d6p1:float>\r\n            <d6p1:float>-0.0640564039</d6p1:float>\r\n            <d6p1:float>0.0521159731</d6p1:float>\r\n            <d6p1:float>-0.302795142</d6p1:float>\r\n            <d6p1:float>-0.2385754</d6p1:float>\r\n            <d6p1:float>0.174789414</d6p1:float>\r\n            <d6p1:float>0.133823514</d6p1:float>\r\n            <d6p1:float>-0.254364878</d6p1:float>\r\n            <d6p1:float>0.124180645</d6p1:float>\r\n            <d6p1:float>0.05550214</d6p1:float>\r\n            <d6p1:float>-0.12980929</d6p1:float>\r\n            <d6p1:float>-0.184106216</d6p1:float>\r\n            <d6p1:float>-0.444153041</d6p1:float>\r\n            <d6p1:float>-0.1866802</d6p1:float>\r\n            <d6p1:float>0.143714622</d6p1:float>\r\n            <d6p1:float>-0.395727068</d6p1:float>\r\n            <d6p1:float>0.231310308</d6p1:float>\r\n            <d6p1:float>0.20325157</d6p1:float>\r\n            <d6p1:float>-0.28306973</d6p1:float>\r\n            <d6p1:float>0.669231057</d6p1:float>\r\n            <d6p1:float>-0.5749554</d6p1:float>\r\n            <d6p1:float>0.44850412</d6p1:float>\r\n            <d6p1:float>0.370314747</d6p1:float>\r\n            <d6p1:float>0.06319777</d6p1:float>\r\n            <d6p1:float>0.134162918</d6p1:float>\r\n            <d6p1:float>0.4889652</d6p1:float>\r\n            <d6p1:float>0.103358962</d6p1:float>\r\n            <d6p1:float>-0.2781967</d6p1:float>\r\n            <d6p1:float>-0.028520016</d6p1:float>\r\n            <d6p1:float>-0.149899781</d6p1:float>\r\n            <d6p1:float>-0.301622063</d6p1:float>\r\n            <d6p1:float>-0.195412874</d6p1:float>\r\n            <d6p1:float>0.00716368947</d6p1:float>\r\n            <d6p1:float>0.06935125</d6p1:float>\r\n            <d6p1:float>-0.105930425</d6p1:float>\r\n            <d6p1:float>-0.156158283</d6p1:float>\r\n            <d6p1:float>0.130988389</d6p1:float>\r\n            <d6p1:float>-0.290430129</d6p1:float>\r\n            <d6p1:float>-0.0436347</d6p1:float>\r\n            <d6p1:float>-0.3894385</d6p1:float>\r\n            <d6p1:float>0.08053621</d6p1:float>\r\n            <d6p1:float>0.160211891</d6p1:float>\r\n            <d6p1:float>-0.0259797778</d6p1:float>\r\n            <d6p1:float>-0.102052644</d6p1:float>\r\n            <d6p1:float>-0.243844017</d6p1:float>\r\n            <d6p1:float>-0.100255184</d6p1:float>\r\n            <d6p1:float>-0.08163855</d6p1:float>\r\n            <d6p1:float>0.2806286</d6p1:float>\r\n            <d6p1:float>-0.24332042</d6p1:float>\r\n            <d6p1:float>0.197054192</d6p1:float>\r\n            <d6p1:float>-0.0134459035</d6p1:float>\r\n            <d6p1:float>-0.210188523</d6p1:float>\r\n            <d6p1:float>-0.08762046</d6p1:float>\r\n            <d6p1:float>0.302076072</d6p1:float>\r\n            <d6p1:float>0.15073441</d6p1:float>\r\n            <d6p1:float>-0.25656563</d6p1:float>\r\n            <d6p1:float>0.201441422</d6p1:float>\r\n            <d6p1:float>0.218471617</d6p1:float>\r\n            <d6p1:float>-0.206235662</d6p1:float>\r\n            <d6p1:float>-0.11866203</d6p1:float>\r\n            <d6p1:float>0.1215296</d6p1:float>\r\n            <d6p1:float>0.0319998041</d6p1:float>\r\n            <d6p1:float>0.0492798</d6p1:float>\r\n            <d6p1:float>-0.04481599</d6p1:float>\r\n            <d6p1:float>0.197361961</d6p1:float>\r\n            <d6p1:float>-0.009837012</d6p1:float>\r\n            <d6p1:float>-0.2508589</d6p1:float>\r\n            <d6p1:float>0.0181070957</d6p1:float>\r\n            <d6p1:float>-0.0554737039</d6p1:float>\r\n            <d6p1:float>0.0283681117</d6p1:float>\r\n            <d6p1:float>-0.06326983</d6p1:float>\r\n            <d6p1:float>0.105000876</d6p1:float>\r\n            <d6p1:float>0.404416472</d6p1:float>\r\n            <d6p1:float>-0.389522552</d6p1:float>\r\n            <d6p1:float>-0.125746712</d6p1:float>\r\n            <d6p1:float>0.07057016</d6p1:float>\r\n            <d6p1:float>0.122212358</d6p1:float>\r\n            <d6p1:float>-0.232888192</d6p1:float>\r\n            <d6p1:float>0.215391114</d6p1:float>\r\n            <d6p1:float>0.105804011</d6p1:float>\r\n            <d6p1:float>-0.0145762619</d6p1:float>\r\n            <d6p1:float>-0.07311204</d6p1:float>\r\n            <d6p1:float>0.440375835</d6p1:float>\r\n            <d6p1:float>-0.124586925</d6p1:float>\r\n            <d6p1:float>0.353323042</d6p1:float>\r\n            <d6p1:float>0.3362505</d6p1:float>\r\n            <d6p1:float>0.0596229248</d6p1:float>\r\n            <d6p1:float>-0.0489708073</d6p1:float>\r\n            <d6p1:float>0.383501977</d6p1:float>\r\n            <d6p1:float>-0.416376263</d6p1:float>\r\n            <d6p1:float>-0.0258155726</d6p1:float>\r\n            <d6p1:float>0.242280319</d6p1:float>\r\n            <d6p1:float>-0.284357131</d6p1:float>\r\n            <d6p1:float>0.09279593</d6p1:float>\r\n            <d6p1:float>-0.182243586</d6p1:float>\r\n            <d6p1:float>-0.187591657</d6p1:float>\r\n            <d6p1:float>-0.259610981</d6p1:float>\r\n            <d6p1:float>-0.230751649</d6p1:float>\r\n            <d6p1:float>0.0139687965</d6p1:float>\r\n            <d6p1:float>0.0902573541</d6p1:float>\r\n            <d6p1:float>-0.2635648</d6p1:float>\r\n            <d6p1:float>0.08650619</d6p1:float>\r\n            <d6p1:float>-0.281678319</d6p1:float>\r\n            <d6p1:float>0.0264684539</d6p1:float>\r\n            <d6p1:float>-0.133954689</d6p1:float>\r\n            <d6p1:float>0.158560812</d6p1:float>\r\n            <d6p1:float>-0.172321334</d6p1:float>\r\n            <d6p1:float>-0.144456074</d6p1:float>\r\n            <d6p1:float>-0.32501</d6p1:float>\r\n            <d6p1:float>-0.228999272</d6p1:float>\r\n            <d6p1:float>-0.180416211</d6p1:float>\r\n            <d6p1:float>0.0153479725</d6p1:float>\r\n            <d6p1:float>-0.07533081</d6p1:float>\r\n            <d6p1:float>0.105357856</d6p1:float>\r\n            <d6p1:float>0.262362242</d6p1:float>\r\n            <d6p1:float>-0.0930908546</d6p1:float>\r\n            <d6p1:float>0.18475239</d6p1:float>\r\n            <d6p1:float>0.191615924</d6p1:float>\r\n            <d6p1:float>0.0163119864</d6p1:float>\r\n            <d6p1:float>0.4817642</d6p1:float>\r\n            <d6p1:float>0.0878648162</d6p1:float>\r\n            <d6p1:float>0.06749572</d6p1:float>\r\n            <d6p1:float>0.222364739</d6p1:float>\r\n            <d6p1:float>-0.27572307</d6p1:float>\r\n            <d6p1:float>-0.3320215</d6p1:float>\r\n            <d6p1:float>-0.006265504</d6p1:float>\r\n            <d6p1:float>-0.019373415</d6p1:float>\r\n            <d6p1:float>-0.08909961</d6p1:float>\r\n            <d6p1:float>-0.9815528</d6p1:float>\r\n            <d6p1:float>0.486984283</d6p1:float>\r\n            <d6p1:float>0.0579448566</d6p1:float>\r\n            <d6p1:float>0.07908061</d6p1:float>\r\n            <d6p1:float>0.163397744</d6p1:float>\r\n            <d6p1:float>0.546362638</d6p1:float>\r\n            <d6p1:float>-0.233918771</d6p1:float>\r\n          </d5p1:Data>\r\n        </d3p1:_x003C_Storage_x003E_k__BackingField>\r\n        <_columnCount xmlns=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra.Single\">50</_columnCount>\r\n        <_rowCount xmlns=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra.Single\">9</_rowCount>\r\n        <_values xmlns:d5p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Ref=\"15\" i:nil=\"true\" xmlns=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra.Single\" />\r\n      </d3p1:MatrixOffloat>\r\n      <d3p1:MatrixOffloat z:Id=\"16\" xmlns:d4p1=\"MathNet.Numerics.LinearAlgebra.Single\" i:type=\"d4p1:DenseMatrix\">\r\n        <d3p1:_x003C_ColumnCount_x003E_k__BackingField>1</d3p1:_x003C_ColumnCount_x003E_k__BackingField>\r\n        <d3p1:_x003C_RowCount_x003E_k__BackingField>50</d3p1:_x003C_RowCount_x003E_k__BackingField>\r\n        <d3p1:_x003C_Storage_x003E_k__BackingField xmlns:d5p1=\"urn:MathNet/Numerics/LinearAlgebra\" z:Id=\"17\" i:type=\"d5p1:DenseColumnMajorMatrixStorageOffloat\">\r\n          <d5p1:RowCount>50</d5p1:RowCount>\r\n          <d5p1:ColumnCount>1</d5p1:ColumnCount>\r\n          <d5p1:Data xmlns:d6p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"18\" z:Size=\"50\">\r\n            <d6p1:float>0.0249628462</d6p1:float>\r\n            <d6p1:float>0.156514</d6p1:float>\r\n            <d6p1:float>0.732018769</d6p1:float>\r\n            <d6p1:float>0.262249</d6p1:float>\r\n            <d6p1:float>0.101700284</d6p1:float>\r\n            <d6p1:float>0.716722548</d6p1:float>\r\n            <d6p1:float>-0.247894436</d6p1:float>\r\n            <d6p1:float>-0.07615478</d6p1:float>\r\n            <d6p1:float>0.109591842</d6p1:float>\r\n            <d6p1:float>-0.3003684</d6p1:float>\r\n            <d6p1:float>0.0495668165</d6p1:float>\r\n            <d6p1:float>-0.00484267343</d6p1:float>\r\n            <d6p1:float>0.00428414159</d6p1:float>\r\n            <d6p1:float>0.23096478</d6p1:float>\r\n            <d6p1:float>-0.7000387</d6p1:float>\r\n            <d6p1:float>-0.16132088</d6p1:float>\r\n            <d6p1:float>-0.292192459</d6p1:float>\r\n            <d6p1:float>0.005708625</d6p1:float>\r\n            <d6p1:float>0.304064929</d6p1:float>\r\n            <d6p1:float>0.157961726</d6p1:float>\r\n            <d6p1:float>0.4065296</d6p1:float>\r\n            <d6p1:float>-0.09627282</d6p1:float>\r\n            <d6p1:float>0.554572761</d6p1:float>\r\n            <d6p1:float>-0.121918954</d6p1:float>\r\n            <d6p1:float>-0.198929176</d6p1:float>\r\n            <d6p1:float>-0.02777155</d6p1:float>\r\n            <d6p1:float>0.210615635</d6p1:float>\r\n            <d6p1:float>0.375739038</d6p1:float>\r\n            <d6p1:float>-0.114550292</d6p1:float>\r\n            <d6p1:float>0.08669782</d6p1:float>\r\n            <d6p1:float>-0.11806348</d6p1:float>\r\n            <d6p1:float>0.242206976</d6p1:float>\r\n            <d6p1:float>1.03469872</d6p1:float>\r\n            <d6p1:float>-0.2535996</d6p1:float>\r\n            <d6p1:float>0.945155263</d6p1:float>\r\n            <d6p1:float>-0.0689658746</d6p1:float>\r\n            <d6p1:float>0.5443362</d6p1:float>\r\n            <d6p1:float>0.3497127</d6p1:float>\r\n            <d6p1:float>-0.09295752</d6p1:float>\r\n            <d6p1:float>-0.476350367</d6p1:float>\r\n            <d6p1:float>-0.168838337</d6p1:float>\r\n            <d6p1:float>0.0172755867</d6p1:float>\r\n            <d6p1:float>0.08354507</d6p1:float>\r\n            <d6p1:float>-0.4797492</d6p1:float>\r\n            <d6p1:float>0.342321724</d6p1:float>\r\n            <d6p1:float>0.165702358</d6p1:float>\r\n            <d6p1:float>-0.08204821</d6p1:float>\r\n            <d6p1:float>-0.0509937555</d6p1:float>\r\n            <d6p1:float>-0.314429939</d6p1:float>\r\n            <d6p1:float>-0.479469776</d6p1:float>\r\n          </d5p1:Data>\r\n        </d3p1:_x003C_Storage_x003E_k__BackingField>\r\n        <_columnCount xmlns=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra.Single\">1</_columnCount>\r\n        <_rowCount xmlns=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra.Single\">50</_rowCount>\r\n        <_values xmlns:d5p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Ref=\"18\" i:nil=\"true\" xmlns=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra.Single\" />\r\n      </d3p1:MatrixOffloat>\r\n    </m_weights>\r\n  </m_model>\r\n</RegressionNeuralNetModel>";
    }
}
