using System;
using System.Linq;
using System.Collections.Generic;
using System.Diagnostics;
using CNTK;

namespace CntkExtensions.Models
{
    public class Sequential
    {
        Learner m_learner;
        Function m_loss;
        Function m_metric;

        Variable m_inputVariable;
        Variable m_targetVariable;

        public Sequential(Variable input)
        {
            Network = input;
        }

        public Function Network;

        public void Add(Func<Function, Function> layerCreator)
        {
            Network = layerCreator(Network);
        }

        public void Compile(Func<IList<Parameter>, Learner> learnerCreator,
            Func<Variable, Variable, Function> lossCreator,
            Func<Variable, Variable, Function> metricCreator)
        {
            // configuration
            var d = Layers.GlobalDevice;
            var dataType = Layers.GlobalDataType;

            // Get input and target variables from network.
            m_inputVariable = Network.Arguments[0];
            var targetShape = Network.Output.Shape;
            m_targetVariable = Variable.InputVariable(targetShape, dataType);

            // create loss and metric.
            m_loss = lossCreator(Network.Output, m_targetVariable);
            m_metric = metricCreator(Network.Output, m_targetVariable);

            // create learner.
            m_learner = learnerCreator(Network.Parameters());
        }

        public void Fit(Tensor x = null, Tensor y = null, int batchSize = 32, int epochs = 1, 
            Tensor xValidation = null, Tensor yValidation = null)
        {
            // configuration
            var d = Layers.GlobalDevice;

            // setup minibatch source.
            var minibatchSource = new MemoryMinibatchSource(x, y, seed: 5, randomize: true);
            
            // setup trainer.
            var trainer = CNTKLib.CreateTrainer(Network, m_loss, m_metric, new LearnerVector { m_learner });

            // variables for training loop.            
            var inputMap = new Dictionary<Variable, Value>();

            var lossSum = 0.0;
            var metricSum = 0.0;
            var totalSampleCount = 0;

            for (int epoch = 0; epoch < epochs; )
            {
                var minibatchData = minibatchSource.GetNextMinibatch(batchSize);

                var isSweepEnd = minibatchData.isSweepEnd;
                var observations = minibatchData.observations;
                var targets = minibatchData.targets;

                // Note that it is possible to create a batch using a data buffer array, to reduce allocations. 
                // However, unsure how to handle random shuffling in this case.
                using (Value batchObservations = Value.CreateBatch<float>(m_inputVariable.Shape, observations, d, true))
                using (Value batchTarget = Value.CreateBatch<float>(m_targetVariable.Shape, targets, d, true))
                {
                    inputMap.Add(m_inputVariable, batchObservations);
                    inputMap.Add(m_targetVariable, batchTarget);

                    trainer.TrainMinibatch(inputMap, false, d);

                    var lossValue = (float)trainer.PreviousMinibatchLossAverage();
                    var metricValue = (float)trainer.PreviousMinibatchEvaluationAverage();

                    // Accumulate loss/metric.
                    lossSum += lossValue * batchSize;
                    metricSum += metricValue * batchSize;
                    totalSampleCount += batchSize;

                    if(isSweepEnd)
                    {
                        var currentLoss = lossSum / totalSampleCount;
                        var currentMetric = metricSum / totalSampleCount;
                        var traceOutput = $"Epoch: {epoch + 1} Loss = {currentLoss:F16}, Metric = {currentMetric:F16}";

                        ++epoch;
                        lossSum = 0;
                        metricSum = 0;
                        totalSampleCount = 0;

                        if(xValidation != null && yValidation != null)
                        {
                            (var validationLoss, var validationMetric) = Evaluate(xValidation, yValidation, batchSize);
                            traceOutput += $" - ValidationLoss = {validationLoss:F16}, ValidationMetric = {validationMetric:F16}";
                        }

                        Trace.WriteLine(traceOutput);
                    }

                    // Ensure cleanup, call erase.
                    inputMap.Clear();
                    batchObservations.Erase();
                    batchTarget.Erase();
                }
            }
        }

        public (float loss, float metric) Evaluate(Tensor x = null, Tensor y = null, int batchSize = 32)
        {
            // configuration
            var d = Layers.GlobalDevice;

            // setup minibatch source.
            var minibatchSource = new MemoryMinibatchSource(x, y, seed: 5, randomize: false);

            // create learner and trainer.
            var lossEvaluator = CNTKLib.CreateEvaluator(m_loss);
            var metricEvaluator = CNTKLib.CreateEvaluator(m_metric);

            // variables for training loop.            
            var inputMap = new UnorderedMapVariableMinibatchData();

            var lossSum = 0.0;
            var metricSum = 0.0;
            var totalSampleCount = 0;

            bool isSweepEnd = false;
            
            // Check if batchSize is larger than sample count.
            var evaluationBatchSize = x.SampleCount < batchSize ? x.SampleCount : batchSize;

            while (!isSweepEnd)
            {
                var minibatchData = minibatchSource.GetNextMinibatch(evaluationBatchSize);

                isSweepEnd = minibatchData.isSweepEnd;
                var observations = minibatchData.observations;
                var targets = minibatchData.targets;

                // Note that it is possible to create a batch using a data buffer array, to reduce allocations. 
                // However, unsure how to handle random shuffling in this case.
                using (var batchObservationsValue = Value.CreateBatch<float>(m_inputVariable.Shape, observations, d, true))
                using (var batchTargetValue = Value.CreateBatch<float>(m_targetVariable.Shape, targets, d, true))
                using (var batchObservations = new MinibatchData(batchObservationsValue))
                using (var batchTarget = new MinibatchData(batchTargetValue))
                {
                    inputMap.Add(m_inputVariable, batchObservations);
                    inputMap.Add(m_targetVariable, batchTarget);

                    var lossValue = lossEvaluator.TestMinibatch(inputMap);
                    var metricValue = metricEvaluator.TestMinibatch(inputMap);

                    // Accumulate loss/metric.
                    lossSum += lossValue * evaluationBatchSize;
                    metricSum += metricValue * evaluationBatchSize;
                    totalSampleCount += evaluationBatchSize;

                    // Ensure cleanup, call erase.
                    inputMap.Clear();
                    batchObservationsValue.Erase();
                    batchTargetValue.Erase();
                }
            }

            var finalLoss = lossSum / totalSampleCount;
            var finalMetric = metricSum / totalSampleCount;

            return ((float)finalLoss, (float)finalMetric);
        }

        public Tensor Predict(Tensor x)
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

            return new Tensor(predictionData, Network.Output.Shape.Dimensions.ToArray(), sampleCount);
        }

        float[] Predict(float[] observation)
        {
            var d = Layers.GlobalDevice;
            var inputDimensions = Network.Arguments[0].Shape;

            using (var singleObservation = Value.CreateBatch<float>(inputDimensions, observation, d))
            {
                var inputDataMap = new Dictionary<Variable, Value>() { { Network.Arguments[0], singleObservation } };
                var outputVar = Network.Output;

                var outputDataMap = new Dictionary<Variable, Value>() { { outputVar, null } };
                Network.Evaluate(inputDataMap, outputDataMap, d);
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
    }
}
