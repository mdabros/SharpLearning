using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using CNTK;

namespace SharpLearning.Cntk.Test
{
    public class CntkLearner
    {
        readonly Learner m_optimizer;
        readonly Function m_loss;
        readonly Function m_metric;
        readonly DataType m_dataType;
        readonly DeviceDescriptor m_device;

        Variable m_inputVariable;
        Variable m_targetVariable;

        public Function Network;
        private int m_validationLossName;

        public CntkLearner(Function network,
            CntkOptimizerCreator optimizerCreator,
            CntkLossCreator lossCreator, 
            CntkMetricCreator metricCreator,
            DataType dataType,
            DeviceDescriptor device)
        {
            Network = network ?? throw new ArgumentNullException(nameof(network));
            m_device = device ?? throw new ArgumentNullException(nameof(device));
            m_dataType = dataType;

            // Get input and target variables from network.
            m_inputVariable = network.Arguments[0];
            var targetShape = network.Output.Shape;
            m_targetVariable = Variable.InputVariable(targetShape, dataType);

            // create loss and metric.
            m_loss = lossCreator(network.Output, m_targetVariable);
            m_metric = metricCreator(network.Output, m_targetVariable);

            // create learner.
            m_optimizer = optimizerCreator(network.Parameters());
        }


        public Function LearnFromSource(IMinibatchSource minibatchSource,
            int batchSize = 32, int epochs = 1)
        {
            //// setup minibatch source.
            //var minibatchSource = new MemoryMinibatchSource(x, y, seed: 5, randomize: true);

            // setup trainer.
            var trainer = CNTKLib.CreateTrainer(Network, m_loss, m_metric, new LearnerVector { m_optimizer });

            // variables for training loop.            
            var inputMap = new Dictionary<Variable, Value>();

            var lossSum = 0f;
            var metricSum = 0f;
            var totalSampleCount = 0;

            for (int epoch = 0; epoch < epochs;)
            {
                var minibatchData = minibatchSource.GetNextMinibatch((uint)batchSize, m_device);
                var isSweepEnd = minibatchData.Values.Any(a => a.sweepEnd);

                var obserationsStreamInfo = new StreamInformation { m_name = MemoryMinibatchSource<float>.ObservationsName };
                var targetsStreamInfo = new StreamInformation { m_name = MemoryMinibatchSource<float>.TargetsName };

                var observationsData = minibatchData[obserationsStreamInfo].data;
                var targetsData = minibatchData[targetsStreamInfo].data;

                inputMap.Add(m_inputVariable, observationsData);
                inputMap.Add(m_targetVariable, targetsData);

                trainer.TrainMinibatch(inputMap, false, m_device);

                var lossValue = (float)trainer.PreviousMinibatchLossAverage();
                var metricValue = (float)trainer.PreviousMinibatchEvaluationAverage();

                // Accumulate loss/metric.
                lossSum += lossValue * batchSize;
                metricSum += metricValue * batchSize;
                totalSampleCount += batchSize;

                if (isSweepEnd)
                {
                    var currentLoss = lossSum / totalSampleCount;
                    var currentMetric = metricSum / totalSampleCount;

                    var traceOutput = $"Epoch: {epoch + 1:000} Loss = {currentLoss:F8}, Metric = {currentMetric:F8}";

                    ++epoch;
                    lossSum = 0;
                    metricSum = 0;
                    totalSampleCount = 0;

                    //if (xValidation != null && yValidation != null)
                    //{
                    //    (var validationLoss, var validationMetric) = Evaluate(xValidation, yValidation, batchSize);
                    //    traceOutput += $" - ValidationLoss = {validationLoss:F8}, ValidationMetric = {validationMetric:F8}";

                    //    lossValidationHistory[m_validationLossName].Add(validationLoss);
                    //    lossValidationHistory[m_validationMetricName].Add(validationMetric);
                    //}

                    Trace.WriteLine(traceOutput);
                }

                // Ensure cleanup
                inputMap.Clear();
                observationsData.Dispose();
                targetsData.Dispose();
                minibatchData.Dispose();
            }

            return Network.Clone(ParameterCloningMethod.Clone);
        }
    }
}
