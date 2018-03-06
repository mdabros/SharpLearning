using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using CNTK;
using SharpLearning.Common.Interfaces;
using SharpLearning.Containers;
using SharpLearning.Containers.Extensions;
using SharpLearning.Containers.Matrices;

namespace SharpLearning.Neural.Cntk
{
    public class CntkNeuralNetLearner : ILearner<double>, IIndexedLearner<double>,
        ILearner<ProbabilityPrediction>, IIndexedLearner<ProbabilityPrediction>
    {
        readonly Function m_network;
        readonly DeviceDescriptor m_device;

        readonly double m_learningRate;
        readonly int m_epochs;
        readonly int m_batchSize;
        readonly double m_momentum;

        public CntkNeuralNetLearner(Function network, DeviceDescriptor device,
            double learningRate = 0.001, int epochs = 100, int batchSize = 128,
            double momentum = 0.9)
        {
            if (network == null) { throw new ArgumentNullException("network"); }
            if (device == null) { throw new ArgumentNullException("device"); }
            if (learningRate <= 0) { throw new ArgumentNullException("learning rate must be larger than 0. Was: " + learningRate); }
            if (epochs <= 0) { throw new ArgumentNullException("Iterations must be larger than 0. Was: " + epochs); }
            if (batchSize <= 0) { throw new ArgumentNullException("batchSize must be larger than 0. Was: " + batchSize); }
            if (momentum <= 0) { throw new ArgumentNullException("momentum must be larger than 0. Was: " + momentum); }
            m_network = network;
            m_device = device;
            m_learningRate = learningRate;
            m_epochs = epochs;
            m_batchSize = batchSize;
            m_momentum = momentum;
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

            var featureVariable = m_network.Arguments[0];
            var targetVariable = Variable.InputVariable(new int[] { numberOfClasses }, DataType.Float);

            // setup network
            // todo: add loss as input
            var loss = CNTKLib.CrossEntropyWithSoftmax(m_network, targetVariable);
            // todo: add error as input
            var evalError = CNTKLib.ClassificationError(m_network, targetVariable);

            var device = CntkLayers.Device;
            Trace.WriteLine($"Using device: {device.Type}");

            // setup learner
            // todo: add learner type as input
            var learningRatePerSample = new CNTK.TrainingParameterScheduleDouble(m_learningRate, 1);
            var momentumRatePerSample = new CNTK.TrainingParameterScheduleDouble(m_momentum, 1);
            //var parameterLearners = new List<Learner>() { Learner.SGDLearner(m_network.Parameters(), learningRatePerSample) };
            //var parameterLearners = new List<Learner>() { Learner.MomentumSGDLearner(m_network.Parameters(), 
            //    learningRatePerSample, momentumRatePerSample, false) };

            var parameterLearners = new List<Learner>() { CntkOptimizers.AdamLearner(m_network.Parameters(),
                new TrainingParameterScheduleDouble(m_learningRate, 1),
                new TrainingParameterScheduleDouble(0.9, 1),
                new TrainingParameterScheduleDouble(0.999, 1))
            };


            var trainer = Trainer.CreateTrainer(m_network, loss, evalError, parameterLearners);

            // train the model
            var batchContainer = new Dictionary<Variable, Value>();
            var random = new Random(23);
            var numberOfBatches = targets.Length / m_batchSize;

            var batchFeatures = new float[observations.ColumnCount * m_batchSize];
            var batchTargets = new float[encodedTargets.ColumnCount * m_batchSize];

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

                    CopyRows(observations, batchFeatures, workIndices);
                    CopyRows(encodedTargets, batchTargets, workIndices);

                    using (var batchObservations = Value.CreateBatch<float>(inputDimensions, batchFeatures, device))
                    using (var batchTarget = Value.CreateBatch<float>(new int[] { numberOfClasses }, batchTargets, device))
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

            // sometimes it seems the GC is too busy to collect.
            // Might be caused by native memory not visible for the managed gc
            GC.Collect();
            GC.WaitForPendingFinalizers();

            var orderedTargetNames = GetOrderedTargetNames(targets);
            return new CntkNeuralNetModel(m_network.Clone(ParameterCloningMethod.Clone), m_device, orderedTargetNames);
        }

        public CntkNeuralNetModel Learn(MinibatchSource minibatchSource,
            string observationsId, string targetsId, int numberOfClasses)
        {
            var inputDimensions = m_network.Arguments[0].Shape.Dimensions.ToArray();

            var imageStreamInfo = minibatchSource.StreamInfo(observationsId);
            var labelStreamInfo = minibatchSource.StreamInfo(targetsId);


            // build a model
            var imageInput = m_network.Arguments[0];//CNTKLib.InputVariable(inputDimensions, imageStreamInfo.m_elementType, "Images");
            var targetVariable = CNTKLib.InputVariable(new int[] { numberOfClasses }, labelStreamInfo.m_elementType);

            // prepare for training
            var loss = CNTKLib.CrossEntropyWithSoftmax(m_network, targetVariable, "lossFunction");
            var evalError = CNTKLib.ClassificationError(m_network, targetVariable, 5, "predictionError");

            //var learningRatePerSample = new TrainingParameterScheduleDouble(m_learningRate, 1);
            //var momentumRatePerSample = new TrainingParameterScheduleDouble(m_momentum, 1);
            //var parameterLearners = new List<Learner>() { Learner.SGDLearner(m_network.Parameters(), learningRatePerSample) };
            //var parameterLearners = new List<Learner>() { Learner.MomentumSGDLearner(m_network.Parameters(),
            //    learningRatePerSample, momentumRatePerSample, false) };

            //var trainer = Trainer.CreateTrainer(m_network, loss, evalError, parameterLearners);

            var learningRatePerSample = new TrainingParameterScheduleDouble(m_learningRate, 1);
            var trainer = Trainer.CreateTrainer(m_network, loss, evalError,
                new List<Learner> { Learner.SGDLearner(m_network.Parameters(), learningRatePerSample) });


            // Feed data to the trainer for number of epochs. 
            var uBatchSize = (uint)m_batchSize;
            var miniBatchCount = 0;
            var outputFrequencyInMinibatches = 1;

            uint numEpochs = (uint)m_epochs;
            while (numEpochs > 0)
            {
                var minibatchData = minibatchSource.GetNextMinibatch((uint)m_batchSize, m_device);

                // Stop training once max epochs is reached.
                if (minibatchData.empty())
                {
                    break;
                }

                var minibatchValues = new Dictionary<Variable, MinibatchData>()
                {
                    { imageInput, minibatchData[imageStreamInfo] },
                    { targetVariable, minibatchData[labelStreamInfo] }
                };

                trainer.TrainMinibatch(minibatchValues, m_device);

                var minibatchIdx = miniBatchCount++;
                if ((minibatchIdx % outputFrequencyInMinibatches) == 0 && trainer.PreviousMinibatchSampleCount() != 0)
                {
                    float trainLossValue = (float)trainer.PreviousMinibatchLossAverage();
                    float evaluationValue = (float)trainer.PreviousMinibatchEvaluationAverage();
                    Trace.WriteLine($"Minibatch: {minibatchIdx} CrossEntropyLoss = {trainLossValue}, EvaluationCriterion = {evaluationValue}");
                }

                // Because minibatchSource is created with MinibatchSource.InfinitelyRepeat, 
                // batching will not end. Each time minibatchSource completes an sweep (epoch),
                // the last minibatch data will be marked as end of a sweep. We use this flag
                // to count number of epochs.
                if (minibatchData.Values.Any(a => a.sweepEnd))
                {
                    Trace.WriteLine("Epoch: " + (m_epochs - numEpochs));
                    numEpochs--;
                }
            }

            return new CntkNeuralNetModel(m_network.Clone(ParameterCloningMethod.Clone), m_device,
                Enumerable.Range(0, numberOfClasses).Select(v => (double)v).ToArray()); // Hack. only works for zero-based consequtives class names.
        }

        IPredictorModel<double> ILearner<double>.Learn(F64Matrix observations, double[] targets)
        {
            return Learn(observations, targets);
        }

        IPredictorModel<double> IIndexedLearner<double>.Learn(F64Matrix observations, double[] targets, int[] indices)
        {
            return Learn(observations, targets, indices);
        }

        IPredictorModel<ProbabilityPrediction> ILearner<ProbabilityPrediction>.Learn(F64Matrix observations, double[] targets)
        {
            return Learn(observations, targets);
        }

        IPredictorModel<ProbabilityPrediction> IIndexedLearner<ProbabilityPrediction>.Learn(F64Matrix observations, double[] targets, int[] indices)
        {
            return Learn(observations, targets, indices);
        }

        void CopyRows(F64Matrix observations, float[] batch, int[] indices)
        {
            var columnCount = observations.ColumnCount;
            var rowCount = indices.Length;

            for (int i = 0; i < indices.Length; i++)
            {
                var rowOffSet = columnCount * indices[i];
                var subRowOffSet = columnCount * i;
                for (int j = 0; j < columnCount; j++)
                {
                    batch[subRowOffSet + j] = (float)observations.Data()[rowOffSet + j];
                }
            }
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
