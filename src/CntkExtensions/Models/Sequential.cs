using System;
using System.Collections.Generic;
using System.Diagnostics;
using CNTK;

namespace CntkExtensions.Models
{
    public class Sequential
    {
        // creators.
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

        public void Fit(Tensor x = null, Tensor y = null, int batchSize = 32, int epochs = 1)
        {
            // configuration
            var d = Layers.GlobalDevice;
            var dataType = Layers.GlobalDataType;

            // setup minibatch source.
            var minibatchSource = new MemoryMinibatchSource(x, y, seed: 5);

            // Get input and target variables from network.
            var inputVariable = Network.Arguments[0];
            var targetShape = Network.Output.Shape;
            var targetVariable = Variable.InputVariable(targetShape, dataType);

            // Setup loss and metric.
            var loss = LossCreator(targetVariable, Network.Output); 
            var metric = MetricCreator(targetVariable, Network.Output);

            // create learner and trainer.
            var learner = LearnerCreator(Network.Parameters());
            var trainer = CNTKLib.CreateTrainer(Network, loss, metric, new LearnerVector { learner });

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
                using (Value batchObservations = Value.CreateBatch<float>(inputVariable.Shape, observations, d))
                using (Value batchTarget = Value.CreateBatch<float>(targetVariable.Shape, targets, d))
                {
                    inputMap.Add(inputVariable, batchObservations);
                    inputMap.Add(targetVariable, batchTarget);

                    trainer.TrainMinibatch(inputMap, false, d);

                    var lossValue = (float)trainer.PreviousMinibatchLossAverage();
                    var metricValue = (float)trainer.PreviousMinibatchEvaluationAverage();

                    //Trace.WriteLine($"Loss: {lossValue}, Metric: {metricValue}");

                    // Accumulate loss/metric.
                    lossSum += lossValue * batchSize;
                    metricSum += metricValue * batchSize;
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
