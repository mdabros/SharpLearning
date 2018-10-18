using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using CNTK;

namespace CntkCatalyst.Models
{
    public class Sequential
    {
        Learner m_learner;
        Function m_loss;
        Function m_metric;

        Variable m_inputVariable;
        Variable m_targetVariable;

        const string m_lossName = "Loss";
        const string m_metricName = "Metric";

        const string m_validationLossName = "Validation loss";
        const string m_validationMetricName = "Validation metric";

        readonly DataType m_dataType;
        readonly DeviceDescriptor m_device;

        public Sequential(Variable network,
            DataType dataType,
            DeviceDescriptor device)
        {
            Network = network ?? throw new ArgumentNullException(nameof(network));
            m_device = device ?? throw new ArgumentNullException(nameof(device));
            m_dataType = dataType;
        }

        public Function Network;

        public void Compile(Func<IList<Parameter>, Learner> learnerCreator,
            Func<Variable, Variable, Function> lossCreator,
            Func<Variable, Variable, Function> metricCreator)
        {
            // Get input and target variables from network.
            m_inputVariable = Network.Arguments[0];
            var targetShape = Network.Output.Shape;
            m_targetVariable = Variable.InputVariable(targetShape, m_dataType);

            // create loss and metric.
            m_loss = lossCreator(Network.Output, m_targetVariable);
            m_metric = metricCreator(Network.Output, m_targetVariable);

            // create learner.
            m_learner = learnerCreator(Network.Parameters());
        }

        public Dictionary<string, List<float>> Fit(MemoryMinibatchData x = null, MemoryMinibatchData y = null, int batchSize = 32, int epochs = 1, 
            MemoryMinibatchData xValidation = null, MemoryMinibatchData yValidation = null)
        {
            // setup minibatch source.
            var trainMinibatchSource = new MemoryMinibatchSource(x, y, seed: 5, randomize: true);

            MemoryMinibatchSource validationMinibatchSource = null;
            if(xValidation != null && yValidation != null)
            {
                validationMinibatchSource = new MemoryMinibatchSource(xValidation, yValidation, seed: 5, randomize: false);
            }
            
            return FitFromMinibatchSource(trainMinibatchSource, batchSize, epochs, validationMinibatchSource);
        }

        public Dictionary<string, List<float>> FitFromMinibatchSource(IMinibatchSource trainMinibatchSource = null, int batchSize = 32, int epochs = 1,
            IMinibatchSource validationMinibatchSource = null)

        {
            // setup trainer.
            var trainer = CNTKLib.CreateTrainer(Network, m_loss, m_metric, new LearnerVector { m_learner });

            // store epoch history
            var lossValidationHistory = new Dictionary<string, List<float>>
            {
                { m_lossName, new List<float>() },
                { m_metricName, new List<float>() },
                { m_validationLossName, new List<float>() },
                { m_validationMetricName, new List<float>() },
            };

            // variables for training loop.            
            var inputMap = new Dictionary<Variable, Value>();

            var lossSum = 0f;
            var metricSum = 0f;
            var totalSampleCount = 0;

            for (int epoch = 0; epoch < epochs; )
            {
                var minibatchData = trainMinibatchSource.GetNextMinibatch((uint)batchSize, m_device);
                var isSweepEnd = minibatchData.Values.Any(a => a.sweepEnd);

                var obserationsStreamInfo = trainMinibatchSource.StreamInfo(trainMinibatchSource.FeaturesName);
                var targetsStreamInfo = trainMinibatchSource.StreamInfo(trainMinibatchSource.LabelsName);

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

                if(isSweepEnd)
                {
                    var currentLoss = lossSum / totalSampleCount;
                    lossValidationHistory[m_lossName].Add(currentLoss);

                    var currentMetric = metricSum / totalSampleCount;
                    lossValidationHistory[m_metricName].Add(currentMetric);

                    var traceOutput = $"Epoch: {epoch + 1:000} Loss = {currentLoss:F8}, Metric = {currentMetric:F8}";

                    ++epoch;
                    lossSum = 0;
                    metricSum = 0;
                    totalSampleCount = 0;

                    if(validationMinibatchSource != null)
                    {
                        (var validationLoss, var validationMetric) = EvaluateFromMinibatchSource(validationMinibatchSource, batchSize);
                        traceOutput += $" - ValidationLoss = {validationLoss:F8}, ValidationMetric = {validationMetric:F8}";

                        lossValidationHistory[m_validationLossName].Add(validationLoss);
                        lossValidationHistory[m_validationMetricName].Add(validationMetric);
                    }

                    Trace.WriteLine(traceOutput);
                }

                // Ensure cleanup
                inputMap.Clear();
                observationsData.Dispose();
                targetsData.Dispose();
                minibatchData.Dispose();
            }

            return lossValidationHistory;
        }

        public (float loss, float metric) Evaluate(MemoryMinibatchData x = null, MemoryMinibatchData y = null, int batchSize = 32)
        {
            // setup minibatch source.
            var minibatchSource = new MemoryMinibatchSource(x, y, seed: 5, randomize: true);

            return EvaluateFromMinibatchSource(minibatchSource, batchSize);
        }

        public (float loss, float metric) EvaluateFromMinibatchSource(IMinibatchSource minibatchSource, int batchSize = 32)
        {
            // create loss and metric evaluators.
            var lossEvaluator = CNTKLib.CreateEvaluator(m_loss);
            var metricEvaluator = CNTKLib.CreateEvaluator(m_metric);

            // variables for training loop.            
            var inputMap = new UnorderedMapVariableMinibatchData();

            var lossSum = 0.0;
            var metricSum = 0.0;
            var totalSampleCount = 0;

            bool isSweepEnd = false;

            // TODO: Check if batchSize is larger than sample count.
            //var evaluationBatchSize = x.SampleCount < batchSize ? x.SampleCount : batchSize;
            var evaluationBatchSize = batchSize;

            while (!isSweepEnd)
            {
                using (var minibatchData = minibatchSource.GetNextMinibatch((uint)evaluationBatchSize, m_device))
                {
                    isSweepEnd = minibatchData.Values.Any(a => a.sweepEnd);

                    var obserationsStreamInfo = minibatchSource.StreamInfo(minibatchSource.FeaturesName);
                    var targetsStreamInfo = minibatchSource.StreamInfo(minibatchSource.LabelsName);

                    using (var observations = minibatchData[obserationsStreamInfo])
                    using (var targets = minibatchData[targetsStreamInfo])
                    {
                        inputMap.Add(m_inputVariable, observations);
                        inputMap.Add(m_targetVariable, targets);

                        var lossValue = lossEvaluator.TestMinibatch(inputMap);
                        var metricValue = metricEvaluator.TestMinibatch(inputMap);

                        // Accumulate loss/metric.
                        lossSum += lossValue * evaluationBatchSize;
                        metricSum += metricValue * evaluationBatchSize;
                        totalSampleCount += evaluationBatchSize;
                        
                        inputMap.Clear();
                    }
                }
            }

            // Ensure cleanup, call erase.
            inputMap.Dispose();
            lossEvaluator.Dispose();
            metricEvaluator.Dispose();

            var finalLoss = lossSum / totalSampleCount;
            var finalMetric = metricSum / totalSampleCount;

            return ((float)finalLoss, (float)finalMetric);
        }

        public MemoryMinibatchData Predict(MemoryMinibatchData x)
        {
            var sampleCount = x.SampleCount;
            var inputDimensions = Network.Arguments[0].Shape;

            var outputSize = Network.Output.Shape.Dimensions.Aggregate((d1, d2) => d1 * d1);
            var predictionData = new float[sampleCount * outputSize];

            for (int i = 0; i < sampleCount; i++)
            {
                var observation = x.GetSamples(i);
                var prediction = Predict(observation.Data);

                Array.Copy(prediction, 0, predictionData, i * outputSize, prediction.Length);
            }

            return new MemoryMinibatchData(predictionData, Network.Output.Shape.Dimensions.ToArray(), sampleCount);
        }

        float[] Predict(float[] observation)
        {
            var inputDimensions = Network.Arguments[0].Shape;

            using (var singleObservation = Value.CreateBatch<float>(inputDimensions, observation, m_device))
            {
                var inputDataMap = new Dictionary<Variable, Value>() { { Network.Arguments[0], singleObservation } };
                var outputVar = Network.Output;

                var outputDataMap = new Dictionary<Variable, Value>() { { outputVar, null } };
                Network.Evaluate(inputDataMap, outputDataMap, m_device);
                var outputVal = outputDataMap[outputVar];

                var predictions = outputVal.GetDenseData<float>(outputVar);
                var floatPrediction = predictions.Single().ToArray();

                // Ensure cleanup, call erase.
                inputDataMap.Clear();
                outputDataMap.Clear();
                singleObservation.Erase();

                return floatPrediction;
            }
        }

        /// <summary>
        /// Currently will only list all layers with empty name.
        /// </summary>
        /// <returns></returns>
        public string Summary()
        {
            var sb = new StringBuilder();
            sb.AppendLine("---------------------");
            sb.AppendLine("Model Summary");
            sb.AppendLine("Input Shape= " + Network.Arguments[0].Shape.AsString());
            sb.AppendLine("Output Shape= " + Network.Output.Shape.AsString());
            sb.AppendLine("=====================");
            sb.AppendLine("");

            var totalParameterCount = 0;

            // Finds all layers with empty name.
            // Figure out of to list all layers regardless of name.
            var layers = Network.FindAllWithName(string.Empty)
                .Reverse().ToList();

            foreach (var layer in layers)
            {                
                var outputShape = layer.Output.Shape;
                var layerParameterCount = 0;

                if (layer.Parameters().Any())
                {
                    layerParameterCount = layer.Parameters().First().Shape.TotalSize;
                }

                sb.AppendLine($"Layer Name='{layer.Name}' Output Shape={outputShape.AsString(),-30}" + 
                    $" Param #:{layerParameterCount}");

                totalParameterCount += layerParameterCount;
            }

            sb.AppendLine();
            sb.AppendLine("=====================");
            sb.AppendLine($"Total Number of Parameters: {totalParameterCount:N0}");
            sb.AppendLine("---------------------");

            return sb.ToString();
        }
    }
}
