using System;
using System.Linq;
using System.Collections.Generic;
using CNTK;
using System.Diagnostics;

namespace CntkExtensions.Models
{
    public class Sequential
    {
        // creators.
        List<Func<Function, Function>> m_layersCreators;
        Func<IList<Parameter>, Learner> LearnerCreator;
        Func<Variable, Variable, Function> LossCreator;
        Func<Variable, Variable, Function> MetricCreator;

        public Function Network;

        public Sequential(Variable input)
        {
            Network = input;
        }

        public void Add(Func<Function, Function> layerCreator)
        {
            Network = layerCreator(Network);
        }

        public void Compile(Func<IList<Parameter>, Learner> learnerCreator,
            Func<Variable, Variable, Function> lossCreator,
            Func<Variable, Variable, Function> metricCreator)
        {
            LearnerCreator = learnerCreator;
            LossCreator = lossCreator;
            MetricCreator = metricCreator;
        }

        public void Fit(float[][] x = null, float[][] y = null, int batchSize = 32, int epochs = 1)
        {
            // Get input and target variables from network.
            var inputVariable = Network.Arguments[0];
            var inputSize = inputVariable.Shape.Dimensions.Aggregate((d1, d2) => d1 * d2);

            var targetVariable = Network.Output;
            var targetSize = targetVariable.Shape.Dimensions.Aggregate((d1, d2) => d1 * d2);
            
            // setup minibatch source.
            var minibatchSource = new MemoryMinibatchSource(x, y, inputSize, targetSize, seed: 5);

            // Setup loss and metric.
            var loss = LossCreator(targetVariable, Network.Output); 
            var metric = MetricCreator(targetVariable, Network.Output);

            // create learner and trainer.
            var learner = LearnerCreator(Network.Parameters());
            var trainer = CNTKLib.CreateTrainer(Network, loss, new LearnerVector { learner });

            // variables for training loop.
            var d = Layers.GlobalDevice;
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
                using (Value batchObservations = Value.CreateBatch<float>(inputVariable.Shape, observations, d))
                using (Value batchTarget = Value.CreateBatch<float>(targetVariable.Shape, targets, d))
                {
                    inputMap.Add(inputVariable, batchObservations);
                    inputMap.Add(targetVariable, batchTarget);

                    trainer.TrainMinibatch(inputMap, false, d);

                    var lossValue = (float)trainer.PreviousMinibatchLossAverage();
                    var evaluationValue = (float)trainer.PreviousMinibatchEvaluationAverage();

                    // Accumulate loss/metric.
                    lossSum += lossValue * batchSize;
                    metricSum += evaluationValue * batchSize;
                    totalSampleCount += batchSize;

                    inputMap.Clear();

                    if(isSweepEnd)
                    {
                        var currentLoss = lossSum / totalSampleCount;
                        var currentMetric = metricSum / totalSampleCount;
                        Trace.WriteLine($"Epoch: {epoch + 1} Loss = {currentLoss:F16}, Metric = {currentMetric:F16}");

                        ++epoch;
                        lossSum = 0;
                        metricSum = 0;
                        totalSampleCount = 0;
                    }

                    // Ensure cleanup, call erase.
                    batchObservations.Erase();
                    batchTarget.Erase();
                }
            }
        }
    }
}
