﻿using System;
using System.IO;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers;
using SharpLearning.Containers.Matrices;
using SharpLearning.Metrics.Classification;
using SharpLearning.Neural.Layers;
using SharpLearning.Neural.Learners;
using SharpLearning.Neural.Loss;
using SharpLearning.Neural.Models;

namespace SharpLearning.Neural.Test.Models
{
    [TestClass]
    public class ClassificationNeuralNetModelTest
    {
        [TestMethod]
        public void ClassificationNeuralNetModel_Predict_Single()
        {
            var numberOfObservations = 500;
            var numberOfFeatures = 5;
            var numberOfClasses = 5;

            var random = new Random(32);
            var observations = new F64Matrix(numberOfObservations, numberOfFeatures);
            observations.Map(() => random.NextDouble());
            var targets = Enumerable.Range(0, numberOfObservations).Select(i => (double)random.Next(0, numberOfClasses)).ToArray();

            var sut = ClassificationNeuralNetModel.Load(() => new StringReader(ClassificationNeuralNetModelText));

            var predictions = new double[numberOfObservations];
            for (int i = 0; i < numberOfObservations; i++)
            {
                predictions[i] = sut.Predict(observations.Row(i));
            }

            var evaluator = new TotalErrorClassificationMetric<double>();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.762, actual);
        }

        [TestMethod]
        public void ClassificationNeuralNetModel_Predict_Multiple()
        {
            var numberOfObservations = 500;
            var numberOfFeatures = 5;
            var numberOfClasses = 5;

            var random = new Random(32);
            var observations = new F64Matrix(numberOfObservations, numberOfFeatures);
            observations.Map(() => random.NextDouble());
            var targets = Enumerable.Range(0, numberOfObservations).Select(i => (double)random.Next(0, numberOfClasses)).ToArray();

            var sut = ClassificationNeuralNetModel.Load(() => new StringReader(ClassificationNeuralNetModelText));

            var predictions = sut.Predict(observations);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.762, actual);
        }

        [TestMethod]
        public void ClassificationNeuralNetModel_PredictProbability_Single()
        {
            var numberOfObservations = 500;
            var numberOfFeatures = 5;
            var numberOfClasses = 5;

            var random = new Random(32);
            var observations = new F64Matrix(numberOfObservations, numberOfFeatures);
            observations.Map(() => random.NextDouble());
            var targets = Enumerable.Range(0, numberOfObservations).Select(i => (double)random.Next(0, numberOfClasses)).ToArray();

            var sut = ClassificationNeuralNetModel.Load(() => new StringReader(ClassificationNeuralNetModelText));

            var predictions = new ProbabilityPrediction[numberOfObservations];
            for (int i = 0; i < numberOfObservations; i++)
            {
                predictions[i] = sut.PredictProbability(observations.Row(i));
            }

            var evaluator = new TotalErrorClassificationMetric<double>();
            var actual = evaluator.Error(targets, predictions.Select(p => p.Prediction).ToArray());

            Assert.AreEqual(0.762, actual);
        }

        [TestMethod]
        public void ClassificationNeuralNetModel_PredictProbability_Multiple()
        {
            var numberOfObservations = 500;
            var numberOfFeatures = 5;
            var numberOfClasses = 5;

            var random = new Random(32);
            var observations = new F64Matrix(numberOfObservations, numberOfFeatures);
            observations.Map(() => random.NextDouble());
            var targets = Enumerable.Range(0, numberOfObservations).Select(i => (double)random.Next(0, numberOfClasses)).ToArray();

            var sut = ClassificationNeuralNetModel.Load(() => new StringReader(ClassificationNeuralNetModelText));

            var predictions = sut.PredictProbability(observations);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var actual = evaluator.Error(targets, predictions.Select(p => p.Prediction).ToArray());

            Assert.AreEqual(0.762, actual);
        }

        [TestMethod]
        public void ClassificationNeuralNetModel_Save()
        {
            var numberOfObservations = 500;
            var numberOfFeatures = 5;
            var numberOfClasses = 5;

            var random = new Random(32);
            var observations = new F64Matrix(numberOfObservations, numberOfFeatures);
            observations.Map(() => random.NextDouble());
            var targets = Enumerable.Range(0, numberOfObservations).Select(i => (double)random.Next(0, numberOfClasses)).ToArray();

            var net = new NeuralNet();
            net.Add(new InputLayer(numberOfFeatures));
            net.Add(new DenseLayer(10));
            net.Add(new SvmLayer(numberOfClasses));

            var learner = new ClassificationNeuralNetLearner(net, new AccuracyLoss());
            var sut = learner.Learn(observations, targets);

            // save model.
            var writer = new StringWriter();
            sut.Save(() => writer);

            // load model and assert prediction results.
            sut = ClassificationNeuralNetModel.Load(() => new StringReader(writer.ToString()));
            var predictions = sut.Predict(observations);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.762, actual, 0.0000001);
        }

        string ClassificationNeuralNetModelText = "<?xml version=\"1.0\" encoding=\"utf-16\"?>\r\n<ClassificationNeuralNetModel xmlns:i=\"http://www.w3.org/2001/XMLSchema-instance\" z:Id=\"1\" xmlns:z=\"http://schemas.microsoft.com/2003/10/Serialization/\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.Neural.Models\">\r\n  <m_neuralNet xmlns:d2p1=\"http://schemas.datacontract.org/2004/07/SharpLearning.Neural\" z:Id=\"2\">\r\n    <d2p1:Layers xmlns:d3p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"3\" z:Size=\"5\">\r\n      <d3p1:anyType z:Id=\"4\" xmlns:d4p1=\"SharpLearning.Neural.Layers\" i:type=\"d4p1:InputLayer\">\r\n        <_x003C_ActivationFunc_x003E_k__BackingField xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.Neural.Layers\">Undefined</_x003C_ActivationFunc_x003E_k__BackingField>\r\n        <_x003C_Depth_x003E_k__BackingField xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.Neural.Layers\">5</_x003C_Depth_x003E_k__BackingField>\r\n        <_x003C_Height_x003E_k__BackingField xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.Neural.Layers\">1</_x003C_Height_x003E_k__BackingField>\r\n        <_x003C_Width_x003E_k__BackingField xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.Neural.Layers\">1</_x003C_Width_x003E_k__BackingField>\r\n      </d3p1:anyType>\r\n      <d3p1:anyType z:Id=\"5\" xmlns:d4p1=\"SharpLearning.Neural.Layers\" i:type=\"d4p1:DenseLayer\">\r\n        <Bias xmlns:d5p1=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra\" z:Id=\"6\" xmlns:d5p2=\"MathNet.Numerics.LinearAlgebra.Single\" i:type=\"d5p2:DenseVector\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.Neural.Layers\">\r\n          <d5p1:_x003C_Count_x003E_k__BackingField>10</d5p1:_x003C_Count_x003E_k__BackingField>\r\n          <d5p1:_x003C_Storage_x003E_k__BackingField xmlns:d6p1=\"urn:MathNet/Numerics/LinearAlgebra\" z:Id=\"7\" i:type=\"d6p1:DenseVectorStorageOffloat\">\r\n            <d6p1:Length>10</d6p1:Length>\r\n            <d6p1:Data z:Id=\"8\" z:Size=\"10\">\r\n              <d3p1:float>0.0610481948</d3p1:float>\r\n              <d3p1:float>-0.124249</d3p1:float>\r\n              <d3p1:float>0.034037143</d3p1:float>\r\n              <d3p1:float>-0.170957983</d3p1:float>\r\n              <d3p1:float>-0.0293014776</d3p1:float>\r\n              <d3p1:float>0.07928534</d3p1:float>\r\n              <d3p1:float>0.0198179837</d3p1:float>\r\n              <d3p1:float>-0.0226761941</d3p1:float>\r\n              <d3p1:float>0.0533807278</d3p1:float>\r\n              <d3p1:float>0.05666069</d3p1:float>\r\n            </d6p1:Data>\r\n          </d5p1:_x003C_Storage_x003E_k__BackingField>\r\n          <_length xmlns=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra.Single\">10</_length>\r\n          <_values z:Ref=\"8\" i:nil=\"true\" xmlns=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra.Single\" />\r\n        </Bias>\r\n        <BiasGradients xmlns:d5p1=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra\" i:nil=\"true\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.Neural.Layers\" />\r\n        <OutputActivations xmlns:d5p1=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra\" z:Id=\"9\" xmlns:d5p2=\"MathNet.Numerics.LinearAlgebra.Single\" i:type=\"d5p2:DenseMatrix\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.Neural.Layers\">\r\n          <d5p1:_x003C_ColumnCount_x003E_k__BackingField>10</d5p1:_x003C_ColumnCount_x003E_k__BackingField>\r\n          <d5p1:_x003C_RowCount_x003E_k__BackingField>1</d5p1:_x003C_RowCount_x003E_k__BackingField>\r\n          <d5p1:_x003C_Storage_x003E_k__BackingField xmlns:d6p1=\"urn:MathNet/Numerics/LinearAlgebra\" z:Id=\"10\" i:type=\"d6p1:DenseColumnMajorMatrixStorageOffloat\">\r\n            <d6p1:RowCount>1</d6p1:RowCount>\r\n            <d6p1:ColumnCount>10</d6p1:ColumnCount>\r\n            <d6p1:Data z:Id=\"11\" z:Size=\"10\">\r\n              <d3p1:float>0</d3p1:float>\r\n              <d3p1:float>0</d3p1:float>\r\n              <d3p1:float>0</d3p1:float>\r\n              <d3p1:float>0</d3p1:float>\r\n              <d3p1:float>0</d3p1:float>\r\n              <d3p1:float>0</d3p1:float>\r\n              <d3p1:float>0</d3p1:float>\r\n              <d3p1:float>0</d3p1:float>\r\n              <d3p1:float>0</d3p1:float>\r\n              <d3p1:float>0</d3p1:float>\r\n            </d6p1:Data>\r\n          </d5p1:_x003C_Storage_x003E_k__BackingField>\r\n          <_columnCount xmlns=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra.Single\">10</_columnCount>\r\n          <_rowCount xmlns=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra.Single\">1</_rowCount>\r\n          <_values z:Ref=\"11\" i:nil=\"true\" xmlns=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra.Single\" />\r\n        </OutputActivations>\r\n        <Weights xmlns:d5p1=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra\" z:Id=\"12\" xmlns:d5p2=\"MathNet.Numerics.LinearAlgebra.Single\" i:type=\"d5p2:DenseMatrix\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.Neural.Layers\">\r\n          <d5p1:_x003C_ColumnCount_x003E_k__BackingField>10</d5p1:_x003C_ColumnCount_x003E_k__BackingField>\r\n          <d5p1:_x003C_RowCount_x003E_k__BackingField>5</d5p1:_x003C_RowCount_x003E_k__BackingField>\r\n          <d5p1:_x003C_Storage_x003E_k__BackingField xmlns:d6p1=\"urn:MathNet/Numerics/LinearAlgebra\" z:Id=\"13\" i:type=\"d6p1:DenseColumnMajorMatrixStorageOffloat\">\r\n            <d6p1:RowCount>5</d6p1:RowCount>\r\n            <d6p1:ColumnCount>10</d6p1:ColumnCount>\r\n            <d6p1:Data z:Id=\"14\" z:Size=\"50\">\r\n              <d3p1:float>0.211666778</d3p1:float>\r\n              <d3p1:float>0.503424</d3p1:float>\r\n              <d3p1:float>0.0579257831</d3p1:float>\r\n              <d3p1:float>0.105672166</d3p1:float>\r\n              <d3p1:float>0.6359549</d3p1:float>\r\n              <d3p1:float>-0.26405862</d3p1:float>\r\n              <d3p1:float>0.09303447</d3p1:float>\r\n              <d3p1:float>-0.6748394</d3p1:float>\r\n              <d3p1:float>0.435685128</d3p1:float>\r\n              <d3p1:float>-0.316193521</d3p1:float>\r\n              <d3p1:float>0.0496498756</d3p1:float>\r\n              <d3p1:float>-0.620875061</d3p1:float>\r\n              <d3p1:float>0.490281552</d3p1:float>\r\n              <d3p1:float>0.141659319</d3p1:float>\r\n              <d3p1:float>0.199103966</d3p1:float>\r\n              <d3p1:float>-0.274011254</d3p1:float>\r\n              <d3p1:float>0.07792936</d3p1:float>\r\n              <d3p1:float>-0.573518336</d3p1:float>\r\n              <d3p1:float>0.24993518</d3p1:float>\r\n              <d3p1:float>0.07333441</d3p1:float>\r\n              <d3p1:float>0.268905669</d3p1:float>\r\n              <d3p1:float>-0.237347454</d3p1:float>\r\n              <d3p1:float>0.518560052</d3p1:float>\r\n              <d3p1:float>-0.0281783529</d3p1:float>\r\n              <d3p1:float>-0.1760884</d3p1:float>\r\n              <d3p1:float>-0.173181355</d3p1:float>\r\n              <d3p1:float>0.24288024</d3p1:float>\r\n              <d3p1:float>0.250365853</d3p1:float>\r\n              <d3p1:float>0.09046286</d3p1:float>\r\n              <d3p1:float>-0.4198636</d3p1:float>\r\n              <d3p1:float>-0.198026776</d3p1:float>\r\n              <d3p1:float>0.207603067</d3p1:float>\r\n              <d3p1:float>0.3292547</d3p1:float>\r\n              <d3p1:float>0.3825196</d3p1:float>\r\n              <d3p1:float>0.462806731</d3p1:float>\r\n              <d3p1:float>0.2868386</d3p1:float>\r\n              <d3p1:float>-0.337988526</d3p1:float>\r\n              <d3p1:float>-0.298222125</d3p1:float>\r\n              <d3p1:float>-0.3728843</d3p1:float>\r\n              <d3p1:float>-0.102399535</d3p1:float>\r\n              <d3p1:float>0.6806794</d3p1:float>\r\n              <d3p1:float>-0.07970294</d3p1:float>\r\n              <d3p1:float>-0.6313027</d3p1:float>\r\n              <d3p1:float>-0.118541352</d3p1:float>\r\n              <d3p1:float>-0.240650117</d3p1:float>\r\n              <d3p1:float>-0.220712453</d3p1:float>\r\n              <d3p1:float>-0.517995358</d3p1:float>\r\n              <d3p1:float>0.7483705</d3p1:float>\r\n              <d3p1:float>0.198054776</d3p1:float>\r\n              <d3p1:float>-0.0455305465</d3p1:float>\r\n            </d6p1:Data>\r\n          </d5p1:_x003C_Storage_x003E_k__BackingField>\r\n          <_columnCount xmlns=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra.Single\">10</_columnCount>\r\n          <_rowCount xmlns=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra.Single\">5</_rowCount>\r\n          <_values z:Ref=\"14\" i:nil=\"true\" xmlns=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra.Single\" />\r\n        </Weights>\r\n        <WeightsGradients xmlns:d5p1=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra\" i:nil=\"true\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.Neural.Layers\" />\r\n        <_x003C_ActivationFunc_x003E_k__BackingField xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.Neural.Layers\">Relu</_x003C_ActivationFunc_x003E_k__BackingField>\r\n        <_x003C_BatchNormalization_x003E_k__BackingField xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.Neural.Layers\">false</_x003C_BatchNormalization_x003E_k__BackingField>\r\n        <_x003C_Depth_x003E_k__BackingField xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.Neural.Layers\">10</_x003C_Depth_x003E_k__BackingField>\r\n        <_x003C_Height_x003E_k__BackingField xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.Neural.Layers\">1</_x003C_Height_x003E_k__BackingField>\r\n        <_x003C_Width_x003E_k__BackingField xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.Neural.Layers\">1</_x003C_Width_x003E_k__BackingField>\r\n        <m_delta xmlns:d5p1=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra\" i:nil=\"true\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.Neural.Layers\" />\r\n        <m_inputActivations xmlns:d5p1=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra\" i:nil=\"true\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.Neural.Layers\" />\r\n      </d3p1:anyType>\r\n      <d3p1:anyType z:Id=\"15\" xmlns:d4p1=\"SharpLearning.Neural.Layers\" i:type=\"d4p1:ActivationLayer\">\r\n        <ActivationDerivative xmlns:d5p1=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra\" z:Id=\"16\" xmlns:d5p2=\"MathNet.Numerics.LinearAlgebra.Single\" i:type=\"d5p2:DenseMatrix\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.Neural.Layers\">\r\n          <d5p1:_x003C_ColumnCount_x003E_k__BackingField>10</d5p1:_x003C_ColumnCount_x003E_k__BackingField>\r\n          <d5p1:_x003C_RowCount_x003E_k__BackingField>1</d5p1:_x003C_RowCount_x003E_k__BackingField>\r\n          <d5p1:_x003C_Storage_x003E_k__BackingField xmlns:d6p1=\"urn:MathNet/Numerics/LinearAlgebra\" z:Id=\"17\" i:type=\"d6p1:DenseColumnMajorMatrixStorageOffloat\">\r\n            <d6p1:RowCount>1</d6p1:RowCount>\r\n            <d6p1:ColumnCount>10</d6p1:ColumnCount>\r\n            <d6p1:Data z:Id=\"18\" z:Size=\"10\">\r\n              <d3p1:float>0</d3p1:float>\r\n              <d3p1:float>0</d3p1:float>\r\n              <d3p1:float>0</d3p1:float>\r\n              <d3p1:float>0</d3p1:float>\r\n              <d3p1:float>0</d3p1:float>\r\n              <d3p1:float>0</d3p1:float>\r\n              <d3p1:float>0</d3p1:float>\r\n              <d3p1:float>0</d3p1:float>\r\n              <d3p1:float>0</d3p1:float>\r\n              <d3p1:float>0</d3p1:float>\r\n            </d6p1:Data>\r\n          </d5p1:_x003C_Storage_x003E_k__BackingField>\r\n          <_columnCount xmlns=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra.Single\">10</_columnCount>\r\n          <_rowCount xmlns=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra.Single\">1</_rowCount>\r\n          <_values z:Ref=\"18\" i:nil=\"true\" xmlns=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra.Single\" />\r\n        </ActivationDerivative>\r\n        <OutputActivations xmlns:d5p1=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra\" z:Id=\"19\" xmlns:d5p2=\"MathNet.Numerics.LinearAlgebra.Single\" i:type=\"d5p2:DenseMatrix\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.Neural.Layers\">\r\n          <d5p1:_x003C_ColumnCount_x003E_k__BackingField>10</d5p1:_x003C_ColumnCount_x003E_k__BackingField>\r\n          <d5p1:_x003C_RowCount_x003E_k__BackingField>1</d5p1:_x003C_RowCount_x003E_k__BackingField>\r\n          <d5p1:_x003C_Storage_x003E_k__BackingField xmlns:d6p1=\"urn:MathNet/Numerics/LinearAlgebra\" z:Id=\"20\" i:type=\"d6p1:DenseColumnMajorMatrixStorageOffloat\">\r\n            <d6p1:RowCount>1</d6p1:RowCount>\r\n            <d6p1:ColumnCount>10</d6p1:ColumnCount>\r\n            <d6p1:Data z:Id=\"21\" z:Size=\"10\">\r\n              <d3p1:float>0</d3p1:float>\r\n              <d3p1:float>0</d3p1:float>\r\n              <d3p1:float>0</d3p1:float>\r\n              <d3p1:float>0</d3p1:float>\r\n              <d3p1:float>0</d3p1:float>\r\n              <d3p1:float>0</d3p1:float>\r\n              <d3p1:float>0</d3p1:float>\r\n              <d3p1:float>0</d3p1:float>\r\n              <d3p1:float>0</d3p1:float>\r\n              <d3p1:float>0</d3p1:float>\r\n            </d6p1:Data>\r\n          </d5p1:_x003C_Storage_x003E_k__BackingField>\r\n          <_columnCount xmlns=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra.Single\">10</_columnCount>\r\n          <_rowCount xmlns=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra.Single\">1</_rowCount>\r\n          <_values z:Ref=\"21\" i:nil=\"true\" xmlns=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra.Single\" />\r\n        </OutputActivations>\r\n        <_x003C_ActivationFunc_x003E_k__BackingField xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.Neural.Layers\">Relu</_x003C_ActivationFunc_x003E_k__BackingField>\r\n        <_x003C_Depth_x003E_k__BackingField xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.Neural.Layers\">10</_x003C_Depth_x003E_k__BackingField>\r\n        <_x003C_Height_x003E_k__BackingField xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.Neural.Layers\">1</_x003C_Height_x003E_k__BackingField>\r\n        <_x003C_Width_x003E_k__BackingField xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.Neural.Layers\">1</_x003C_Width_x003E_k__BackingField>\r\n        <m_activation z:Id=\"22\" xmlns:d5p1=\"SharpLearning.Neural.Activations\" i:type=\"d5p1:ReluActivation\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.Neural.Layers\" />\r\n        <m_delta xmlns:d5p1=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra\" i:nil=\"true\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.Neural.Layers\" />\r\n        <m_inputActivations xmlns:d5p1=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra\" i:nil=\"true\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.Neural.Layers\" />\r\n      </d3p1:anyType>\r\n      <d3p1:anyType z:Id=\"23\" xmlns:d4p1=\"SharpLearning.Neural.Layers\" i:type=\"d4p1:DenseLayer\">\r\n        <Bias xmlns:d5p1=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra\" z:Id=\"24\" xmlns:d5p2=\"MathNet.Numerics.LinearAlgebra.Single\" i:type=\"d5p2:DenseVector\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.Neural.Layers\">\r\n          <d5p1:_x003C_Count_x003E_k__BackingField>5</d5p1:_x003C_Count_x003E_k__BackingField>\r\n          <d5p1:_x003C_Storage_x003E_k__BackingField xmlns:d6p1=\"urn:MathNet/Numerics/LinearAlgebra\" z:Id=\"25\" i:type=\"d6p1:DenseVectorStorageOffloat\">\r\n            <d6p1:Length>5</d6p1:Length>\r\n            <d6p1:Data z:Id=\"26\" z:Size=\"5\">\r\n              <d3p1:float>0.0552292541</d3p1:float>\r\n              <d3p1:float>0.0550588854</d3p1:float>\r\n              <d3p1:float>-0.0321966745</d3p1:float>\r\n              <d3p1:float>-0.0767974257</d3p1:float>\r\n              <d3p1:float>-0.0029704913</d3p1:float>\r\n            </d6p1:Data>\r\n          </d5p1:_x003C_Storage_x003E_k__BackingField>\r\n          <_length xmlns=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra.Single\">5</_length>\r\n          <_values z:Ref=\"26\" i:nil=\"true\" xmlns=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra.Single\" />\r\n        </Bias>\r\n        <BiasGradients xmlns:d5p1=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra\" i:nil=\"true\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.Neural.Layers\" />\r\n        <OutputActivations xmlns:d5p1=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra\" z:Id=\"27\" xmlns:d5p2=\"MathNet.Numerics.LinearAlgebra.Single\" i:type=\"d5p2:DenseMatrix\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.Neural.Layers\">\r\n          <d5p1:_x003C_ColumnCount_x003E_k__BackingField>5</d5p1:_x003C_ColumnCount_x003E_k__BackingField>\r\n          <d5p1:_x003C_RowCount_x003E_k__BackingField>1</d5p1:_x003C_RowCount_x003E_k__BackingField>\r\n          <d5p1:_x003C_Storage_x003E_k__BackingField xmlns:d6p1=\"urn:MathNet/Numerics/LinearAlgebra\" z:Id=\"28\" i:type=\"d6p1:DenseColumnMajorMatrixStorageOffloat\">\r\n            <d6p1:RowCount>1</d6p1:RowCount>\r\n            <d6p1:ColumnCount>5</d6p1:ColumnCount>\r\n            <d6p1:Data z:Id=\"29\" z:Size=\"5\">\r\n              <d3p1:float>0</d3p1:float>\r\n              <d3p1:float>0</d3p1:float>\r\n              <d3p1:float>0</d3p1:float>\r\n              <d3p1:float>0</d3p1:float>\r\n              <d3p1:float>0</d3p1:float>\r\n            </d6p1:Data>\r\n          </d5p1:_x003C_Storage_x003E_k__BackingField>\r\n          <_columnCount xmlns=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra.Single\">5</_columnCount>\r\n          <_rowCount xmlns=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra.Single\">1</_rowCount>\r\n          <_values z:Ref=\"29\" i:nil=\"true\" xmlns=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra.Single\" />\r\n        </OutputActivations>\r\n        <Weights xmlns:d5p1=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra\" z:Id=\"30\" xmlns:d5p2=\"MathNet.Numerics.LinearAlgebra.Single\" i:type=\"d5p2:DenseMatrix\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.Neural.Layers\">\r\n          <d5p1:_x003C_ColumnCount_x003E_k__BackingField>5</d5p1:_x003C_ColumnCount_x003E_k__BackingField>\r\n          <d5p1:_x003C_RowCount_x003E_k__BackingField>10</d5p1:_x003C_RowCount_x003E_k__BackingField>\r\n          <d5p1:_x003C_Storage_x003E_k__BackingField xmlns:d6p1=\"urn:MathNet/Numerics/LinearAlgebra\" z:Id=\"31\" i:type=\"d6p1:DenseColumnMajorMatrixStorageOffloat\">\r\n            <d6p1:RowCount>10</d6p1:RowCount>\r\n            <d6p1:ColumnCount>5</d6p1:ColumnCount>\r\n            <d6p1:Data z:Id=\"32\" z:Size=\"50\">\r\n              <d3p1:float>0.239018068</d3p1:float>\r\n              <d3p1:float>0.3766493</d3p1:float>\r\n              <d3p1:float>0.119871616</d3p1:float>\r\n              <d3p1:float>0.368836731</d3p1:float>\r\n              <d3p1:float>-0.003968022</d3p1:float>\r\n              <d3p1:float>0.201241136</d3p1:float>\r\n              <d3p1:float>-0.4953054</d3p1:float>\r\n              <d3p1:float>-0.288781</d3p1:float>\r\n              <d3p1:float>0.644739151</d3p1:float>\r\n              <d3p1:float>0.269126058</d3p1:float>\r\n              <d3p1:float>0.3422774</d3p1:float>\r\n              <d3p1:float>0.202050343</d3p1:float>\r\n              <d3p1:float>-0.5275287</d3p1:float>\r\n              <d3p1:float>-0.07582889</d3p1:float>\r\n              <d3p1:float>0.515934646</d3p1:float>\r\n              <d3p1:float>0.215315819</d3p1:float>\r\n              <d3p1:float>0.598845243</d3p1:float>\r\n              <d3p1:float>-0.0262378529</d3p1:float>\r\n              <d3p1:float>0.251381546</d3p1:float>\r\n              <d3p1:float>-0.319025427</d3p1:float>\r\n              <d3p1:float>-0.07343854</d3p1:float>\r\n              <d3p1:float>0.0907833</d3p1:float>\r\n              <d3p1:float>-0.30306</d3p1:float>\r\n              <d3p1:float>0.583017</d3p1:float>\r\n              <d3p1:float>0.2896962</d3p1:float>\r\n              <d3p1:float>-0.18064934</d3p1:float>\r\n              <d3p1:float>0.0386406146</d3p1:float>\r\n              <d3p1:float>0.303400666</d3p1:float>\r\n              <d3p1:float>0.0941301361</d3p1:float>\r\n              <d3p1:float>0.125410527</d3p1:float>\r\n              <d3p1:float>0.168583438</d3p1:float>\r\n              <d3p1:float>0.273763448</d3p1:float>\r\n              <d3p1:float>-0.477898866</d3p1:float>\r\n              <d3p1:float>0.365937471</d3p1:float>\r\n              <d3p1:float>0.0709273</d3p1:float>\r\n              <d3p1:float>-0.316643238</d3p1:float>\r\n              <d3p1:float>-0.196966648</d3p1:float>\r\n              <d3p1:float>0.4224245</d3p1:float>\r\n              <d3p1:float>0.04616817</d3p1:float>\r\n              <d3p1:float>-0.00422706641</d3p1:float>\r\n              <d3p1:float>-0.0333151855</d3p1:float>\r\n              <d3p1:float>0.4916748</d3p1:float>\r\n              <d3p1:float>-0.390525</d3p1:float>\r\n              <d3p1:float>-0.04658397</d3p1:float>\r\n              <d3p1:float>-0.6587603</d3p1:float>\r\n              <d3p1:float>-0.245033592</d3p1:float>\r\n              <d3p1:float>0.260301679</d3p1:float>\r\n              <d3p1:float>-0.429512</d3p1:float>\r\n              <d3p1:float>-0.7081847</d3p1:float>\r\n              <d3p1:float>-0.4673957</d3p1:float>\r\n            </d6p1:Data>\r\n          </d5p1:_x003C_Storage_x003E_k__BackingField>\r\n          <_columnCount xmlns=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra.Single\">5</_columnCount>\r\n          <_rowCount xmlns=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra.Single\">10</_rowCount>\r\n          <_values z:Ref=\"32\" i:nil=\"true\" xmlns=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra.Single\" />\r\n        </Weights>\r\n        <WeightsGradients xmlns:d5p1=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra\" i:nil=\"true\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.Neural.Layers\" />\r\n        <_x003C_ActivationFunc_x003E_k__BackingField xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.Neural.Layers\">Undefined</_x003C_ActivationFunc_x003E_k__BackingField>\r\n        <_x003C_BatchNormalization_x003E_k__BackingField xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.Neural.Layers\">false</_x003C_BatchNormalization_x003E_k__BackingField>\r\n        <_x003C_Depth_x003E_k__BackingField xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.Neural.Layers\">5</_x003C_Depth_x003E_k__BackingField>\r\n        <_x003C_Height_x003E_k__BackingField xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.Neural.Layers\">1</_x003C_Height_x003E_k__BackingField>\r\n        <_x003C_Width_x003E_k__BackingField xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.Neural.Layers\">1</_x003C_Width_x003E_k__BackingField>\r\n        <m_delta xmlns:d5p1=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra\" i:nil=\"true\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.Neural.Layers\" />\r\n        <m_inputActivations xmlns:d5p1=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra\" i:nil=\"true\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.Neural.Layers\" />\r\n      </d3p1:anyType>\r\n      <d3p1:anyType z:Id=\"33\" xmlns:d4p1=\"SharpLearning.Neural.Layers\" i:type=\"d4p1:SvmLayer\">\r\n        <NumberOfClasses xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.Neural.Layers\">5</NumberOfClasses>\r\n        <OutputActivations xmlns:d5p1=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra\" z:Id=\"34\" xmlns:d5p2=\"MathNet.Numerics.LinearAlgebra.Single\" i:type=\"d5p2:DenseMatrix\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.Neural.Layers\">\r\n          <d5p1:_x003C_ColumnCount_x003E_k__BackingField>5</d5p1:_x003C_ColumnCount_x003E_k__BackingField>\r\n          <d5p1:_x003C_RowCount_x003E_k__BackingField>1</d5p1:_x003C_RowCount_x003E_k__BackingField>\r\n          <d5p1:_x003C_Storage_x003E_k__BackingField xmlns:d6p1=\"urn:MathNet/Numerics/LinearAlgebra\" z:Id=\"35\" i:type=\"d6p1:DenseColumnMajorMatrixStorageOffloat\">\r\n            <d6p1:RowCount>1</d6p1:RowCount>\r\n            <d6p1:ColumnCount>5</d6p1:ColumnCount>\r\n            <d6p1:Data z:Id=\"36\" z:Size=\"5\">\r\n              <d3p1:float>0</d3p1:float>\r\n              <d3p1:float>0</d3p1:float>\r\n              <d3p1:float>0</d3p1:float>\r\n              <d3p1:float>0</d3p1:float>\r\n              <d3p1:float>0</d3p1:float>\r\n            </d6p1:Data>\r\n          </d5p1:_x003C_Storage_x003E_k__BackingField>\r\n          <_columnCount xmlns=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra.Single\">5</_columnCount>\r\n          <_rowCount xmlns=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra.Single\">1</_rowCount>\r\n          <_values z:Ref=\"36\" i:nil=\"true\" xmlns=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra.Single\" />\r\n        </OutputActivations>\r\n        <_x003C_ActivationFunc_x003E_k__BackingField xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.Neural.Layers\">Undefined</_x003C_ActivationFunc_x003E_k__BackingField>\r\n        <_x003C_Depth_x003E_k__BackingField xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.Neural.Layers\">5</_x003C_Depth_x003E_k__BackingField>\r\n        <_x003C_Height_x003E_k__BackingField xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.Neural.Layers\">1</_x003C_Height_x003E_k__BackingField>\r\n        <_x003C_Width_x003E_k__BackingField xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.Neural.Layers\">1</_x003C_Width_x003E_k__BackingField>\r\n        <m_delta xmlns:d5p1=\"http://schemas.datacontract.org/2004/07/MathNet.Numerics.LinearAlgebra\" i:nil=\"true\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.Neural.Layers\" />\r\n      </d3p1:anyType>\r\n    </d2p1:Layers>\r\n    <d2p1:m_initialization>GlorotUniform</d2p1:m_initialization>\r\n  </m_neuralNet>\r\n  <m_targetNames xmlns:d2p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"37\" z:Size=\"5\">\r\n    <d2p1:double>0</d2p1:double>\r\n    <d2p1:double>1</d2p1:double>\r\n    <d2p1:double>2</d2p1:double>\r\n    <d2p1:double>3</d2p1:double>\r\n    <d2p1:double>4</d2p1:double>\r\n  </m_targetNames>\r\n</ClassificationNeuralNetModel>";
    }
}
