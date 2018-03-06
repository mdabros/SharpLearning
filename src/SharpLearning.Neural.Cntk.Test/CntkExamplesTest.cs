using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using static CNTK.CNTKLib;
using CNTK;
using static SharpLearning.Neural.Cntk.CntkLayers;
using System.Collections.Generic;
using System.Diagnostics;

namespace SharpLearning.Neural.Cntk.Test
{
    [TestClass]
    public class CntkExamplesTest
    {
        [TestMethod]
        public void Cntk_Training_Learner()
        {
            var dataType = DataType.Float;
            var device = DeviceDescriptor.CPUDevice;

            var numberOfClasses = 10;
            var inputShape = new int[] { 28, 28, 1 };
            var inputDimensions = inputShape.Aggregate((d1, d2) => d1 * d2);

            // Input variables denoting the features and label data.
            var inputVariable = InputVariable(inputShape, dataType); ;
            var targetVariable = InputVariable(new int[] { numberOfClasses }, dataType);

            var scaledInput = ElementTimes(Constant.Scalar(dataType, 0.00390625, device), inputVariable);

            var net = Conv2D(scaledInput, 3, 3, 32);
            net = ReLU(net);
            net = Pool2D(net, 2, 2, PoolingType.Max);

            net = Conv2D(scaledInput, 3, 3, 32);
            net = ReLU(net);
            net = Pool2D(net, 2, 2, PoolingType.Max);

            net = Dense(net, 96);
            net = ReLU(net);
            net = Dropout(net, 0.5);
            net = Dense(net, numberOfClasses);

            var optimizer = CntkOptimizers.AdamLearner(net.Parameters());
            var loss = CrossEntropyWithSoftmax(targetVariable, net.Output);
            var metric = ClassificationError(targetVariable, net.Output);

            var epochSize = 60000;
            var epochs = 20;
            var iterations = epochSize * epochs;
            var minibatchSize = 64;

            var learner = new CntkNeuralNetLearner(net, device, iterations, minibatchSize,
                optimizer, loss, metric);

            var streamConfigurations = new StreamConfiguration[]
            {
                new StreamConfiguration("featureId", inputDimensions),
                new StreamConfiguration("labelId", numberOfClasses)
            };

            var dataFilePath = string.Empty;
            var source = MinibatchSource.TextFormatMinibatchSource(dataFilePath, streamConfigurations, (ulong)epochSize, true);
            var model = learner.Learn(source, targetVariable);
        }

        [TestMethod]
        public void Cntk_Training__Open_Loop()
        {
            var dataType = DataType.Float;
            var device = DeviceDescriptor.CPUDevice;

            var numberOfClasses = 10;
            var inputShape = new int[] { 28, 28, 1 };
            var inputDimensions = inputShape.Aggregate((d1, d2) => d1 * d2);

            // Input variables denoting the features and label data.
            var inputVariable = InputVariable(inputShape, dataType); ;
            var targetVariable = InputVariable(new int[] { numberOfClasses }, dataType);

            var scaledInput = ElementTimes(Constant.Scalar(dataType, 0.00390625, device), inputVariable);

            var net = Conv2D(scaledInput, 3, 3, 32);
            net = ReLU(net);
            net = Pool2D(net, 2, 2, PoolingType.Max);

            net = Conv2D(scaledInput, 3, 3, 32);
            net = ReLU(net);
            net = Pool2D(net, 2, 2, PoolingType.Max);

            net = Dense(net, 96);
            net = ReLU(net);
            net = Dropout(net, 0.5);
            net = Dense(net, numberOfClasses);

            var optimizer = CntkOptimizers.AdamLearner(net.Parameters());
            var loss = CrossEntropyWithSoftmax(targetVariable, net.Output);
            var metric = ClassificationError(targetVariable, net.Output);

            var epochSize = 60000;
            var epochs = 20;
            var iterations = epochSize * epochs;
            var minibatchSize = 64;

            var learner = new CntkNeuralNetLearner(net, device, iterations, minibatchSize,
                optimizer, loss, metric);

            var streamConfigurations = new StreamConfiguration[]
            {
                new StreamConfiguration("featureId", inputDimensions),
                new StreamConfiguration("labelId", numberOfClasses)
            };

            var dataFilePath = string.Empty;
            var source = MinibatchSource.TextFormatMinibatchSource(dataFilePath, streamConfigurations, (ulong)epochSize, true);

            // Instantiate progress writers.
            var trainingProgressOutputFreq = 100;

            var featureStreamInfo = source.StreamInfo(inputVariable);
            var labelStreamInfo = source.StreamInfo(targetVariable);

            // Train model.
            var trainer = Trainer.CreateTrainer(net, loss, metric, new List<Learner> { optimizer });

            for (int i = 0; i < iterations; i++)
            {
                var mb = source.GetNextMinibatch((uint)minibatchSize, device);
                var arguments = new Dictionary<Variable, MinibatchData>
                {
                    { inputVariable, mb[featureStreamInfo] },
                    { targetVariable, mb[labelStreamInfo] }
                };
                trainer.TrainMinibatch(arguments, device);

                if (((i + 1) % trainingProgressOutputFreq) == 0 && trainer.PreviousMinibatchSampleCount() != 0)
                {
                    var trainLossValue = trainer.PreviousMinibatchLossAverage();
                    var evaluationValue = trainer.PreviousMinibatchEvaluationAverage();
                    Trace.WriteLine($"Minibatch: {i + 1} Loss = {trainLossValue:F16}, Metric = {evaluationValue:F16}");
                }
            }
        }
    }
}
