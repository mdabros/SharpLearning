using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using CNTK;
using SharpLearning.Common.Interfaces;
using SharpLearning.Containers.Extensions;
using SharpLearning.Containers.Matrices;

namespace SharpLearning.Neural.Cntk
{
    public class CntkNeuralNetLearner : ILearner<double>, IIndexedLearner<double>
    {
        readonly Function m_network;
        readonly DeviceDescriptor m_device;

        readonly double m_learningRate;
        readonly int m_epochs;
        readonly int m_batchSize;

        public CntkNeuralNetLearner(Function network,  DeviceDescriptor device,
            double learningRate = 0.001, int epochs = 100, int batchSize = 128)
        {
            if (network == null) { throw new ArgumentNullException("network"); }
            if (device == null) { throw new ArgumentNullException("device"); }
            if (learningRate <= 0) { throw new ArgumentNullException("learning rate must be larger than 0. Was: " + learningRate); }
            if (epochs <= 0) { throw new ArgumentNullException("Iterations must be larger than 0. Was: " + epochs); }
            if (batchSize <= 0) { throw new ArgumentNullException("batchSize must be larger than 0. Was: " + batchSize); }
            m_network = network;
            m_device = device;
            m_learningRate = learningRate;
            m_epochs = epochs;
            m_batchSize = batchSize;
        }

        public CntkNeuralNetModel Learn(F64Matrix observations, double[] targets)
        {
            return Learn(observations, targets, Enumerable.Range(0, targets.Length).ToArray());
        }

        public CntkNeuralNetModel Learn(F64Matrix observations, double[] targets, int[] indices)
        {
            var encodedTargets = Encode(targets);
            var learningIndices = indices.ToArray();

            // get from network or ctor
            var inputDimensions = m_network.Arguments[0].Shape.Dimensions.ToArray();
            int numberOfClasses = encodedTargets.ColumnCount;

            var featureVariable = Variable.InputVariable(inputDimensions, DataType.Float);
            var targetVariable = Variable.InputVariable(new int[] { numberOfClasses }, DataType.Float);

            // setup network
            var loss = CNTKLib.CrossEntropyWithSoftmax(m_network, targetVariable);
            var evalError = CNTKLib.ClassificationError(m_network, targetVariable);

            var device = CntkLayers.Device;
            Trace.WriteLine($"Using device: {device.Type}");

            // setup learner
            var learningRatePerSample = new CNTK.TrainingParameterScheduleDouble(m_learningRate, 1);
            var parameterLearners = new List<Learner>() { Learner.SGDLearner(m_network.Parameters(), learningRatePerSample) };
            var trainer = Trainer.CreateTrainer(m_network, loss, evalError, parameterLearners);

            // train the model
            var batchContainer = new Dictionary<Variable, Value>();
            var random = new Random(23);
            var numberOfBatches = targets.Length / m_batchSize;

            for (int epoch = 0; epoch < m_epochs; epoch++)
            {
                var accumulatedLoss = 0.0;
                learningIndices.Shuffle(random);

                for (int i = 0; i < numberOfBatches; i++)
                {
                    var workIndices = learningIndices
                        .Skip(i * m_batchSize)
                        .Take(m_batchSize).ToArray();

                    if (workIndices.Length != m_batchSize)
                    {
                        continue; // only train with full batch size
                    }

                    var features = CopyBatch(observations, workIndices);
                    var batchLabels = CopyBatch(encodedTargets, workIndices);

                    using (var batchObservations = Value.CreateBatch<float>(inputDimensions, features, device))
                    using (var batchTarget = Value.CreateBatch<float>(new int[] { numberOfClasses }, batchLabels, device))
                    {
                        batchContainer.Add(featureVariable, batchObservations);
                        batchContainer.Add(targetVariable, batchTarget);

                        trainer.TrainMinibatch(batchContainer, device);
                        batchContainer.Clear();

                        accumulatedLoss += trainer.PreviousMinibatchSampleCount() * trainer.PreviousMinibatchLossAverage();
                    }
                }

                var currentLoss = accumulatedLoss / (double)targets.Length;
                Trace.WriteLine($"Epoch: {epoch + 1}: Loss = {currentLoss}");
            }

            var orderedTargetNames = GetOrderedTargetNames(targets);
            return new CntkNeuralNetModel(m_network.Clone(ParameterCloningMethod.Clone), m_device, orderedTargetNames);
        }

        IPredictorModel<double> ILearner<double>.Learn(F64Matrix observations, double[] targets)
        {
            return Learn(observations, targets);
        }

        IPredictorModel<double> IIndexedLearner<double>.Learn(F64Matrix observations, double[] targets, int[] indices)
        {
            return Learn(observations, targets, indices);
        }

        static float[] CopyBatch(F64Matrix observations, int[] indices)
        {
            var batch = observations.Rows(indices);
            return batch.Data().Select(v => (float)v).ToArray();
        }

        static double[] GetOrderedTargetNames(double[] targets)
        {
            return targets.Distinct().OrderBy(v => v).ToArray();
        }

        static F64Matrix Encode(double[] targets)
        {
            var index = 0;
            var targetNameToTargetIndex = GetOrderedTargetNames(targets)
                .ToDictionary(v => v, v => index++);

            var oneOfN = new F64Matrix(targets.Length, targetNameToTargetIndex.Count);

            for (int i = 0; i < targets.Length; i++)
            {
                var target = targets[i];
                oneOfN[i, targetNameToTargetIndex[target]] = 1.0f;
            }

            return oneOfN;
        }
    }
}
