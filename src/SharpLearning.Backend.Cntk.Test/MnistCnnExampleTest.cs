using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using CNTK;
using Microsoft.VisualStudio.TestTools.UnitTesting;
// To avoid name-clash with SharpLearning.Backend.DataType.
using CntkDataType = CNTK.DataType;

namespace SharpLearning.Backend.Cntk.Test
{
    [TestClass]
    public class MnistCnnExampleTest
    {
        const string DownloadPath = "MnistTest";
             
        [TestMethod]
        public void MnistCnnTest()
        {
            // Define data.
            var imageHeight = 28;
            var imageWidth = 28;
            var numChannels = 1;
            var input_dim = imageHeight * imageWidth * numChannels;
            var numOutputClasses = 10;
            var dataType = CntkDataType.Float;
            var inputDimensions = new int[] { numChannels, imageHeight, imageWidth };

            // Set device.
            var device = DeviceDescriptor.UseDefaultDevice();

            // Input variables denoting the features and label data.
            Variable inputVar = Variable.InputVariable(inputDimensions, dataType); ;
            Variable labelVar = Variable.InputVariable(new int[] { numOutputClasses }, dataType);

            // Instantiate the feedforward classification model
            //var scaledInput = CNTKLib.ElementTimes(Constant.Scalar(dataType, 0.00390625, device), inputVar);

            var layers = new CntkLayers(device, dataType);
            Function conv1 = layers.Convolution2D(inputVar, new int[] { 5, 5 }, 32, (v) => CNTKLib.ReLU(v));
            Function f4    = layers.Dense(conv1, 96, (v) => CNTKLib.ReLU(v));
            Function drop4 = layers.Dropout(f4, 0.5);
            Function z     = layers.Dense(drop4, numOutputClasses);

            // Define loss and error metric.
            Function loss = CNTKLib.CrossEntropyWithSoftmax(z, labelVar);
            Function errorMetric = CNTKLib.ClassificationError(z, labelVar);

            // Set learning parameters
            var lrSchedule = new TrainingParameterScheduleDouble(0.01, 1);
            var mmSchedule = new TrainingParameterScheduleDouble(0.9, 1);

            // Instantiate the trainer object to drive the model training
            var learner = new List<Learner>() { Learner.MomentumSGDLearner(z.Parameters(), lrSchedule, mmSchedule, false) };
            //var learner = new List<Learner>() { Learner.SGDLearner(z.Parameters(), lrSchedule) };
            var trainingProgressOutputFreq = 100;
            var trainer = Trainer.CreateTrainer(z, loss, errorMetric, learner);

            // Training config.
            var mnist = Mnist.Load(DownloadPath);
            var readerTrain = mnist.GetTrainReader();

            var minibatchSize = 64;
            var epochSize = mnist.TrainImages.Count();
            var maxEpochs = 40;
            var minibatchIterations = 2000;

            // Train model.
            var inputMap = new Dictionary<Variable, Value>();
            for (int i = 0; i < minibatchIterations; i++)
            {
                // Using mnist.GetTrainReader().NextBatchArray(batchSize), which return the data as a single dim array.
                // the batch images and targets arrays should be reused to reduce allocations.
                (var trainingImages, var trainingTargets) = readerTrain.NextBatch(minibatchSize);

                // Note that it is possible to create a batch using a data buffer array, to reduce allocations. 
                // However, unsure how to handle random shuffling in this case.
                using (Value batchImages = Value.CreateBatch<float>(inputDimensions, trainingImages, device))
                using (Value batchTarget = Value.CreateBatch<float>(new int[] { numOutputClasses }, trainingTargets, device))
                {
                    inputMap.Add(inputVar, batchImages);
                    inputMap.Add(labelVar, batchTarget);

                    // There seems to be a memory-leak of some kind, like the batches are not properly released.
                    // This is most noticable with GPU training, since batches are being processed quicker.
                    trainer.TrainMinibatch(inputMap, false, device);
                    inputMap.Clear();

                    if ((i % trainingProgressOutputFreq) == 0 && trainer.PreviousMinibatchSampleCount() != 0)
                    {
                        var batchLoss = (float)trainer.PreviousMinibatchLossAverage();
                        var batchError = (float)trainer.PreviousMinibatchEvaluationAverage();
                        Trace.WriteLine($"Minibatch: {i} CrossEntropyLoss = {batchLoss}, EvaluationCriterion = {batchError}");
                    }

                    var samplesSeenNextBatch = (i + 2) * minibatchSize;
                    if (samplesSeenNextBatch >= epochSize)
                    {
                        readerTrain.Reset();
                    }
                }
            }
        }
    }
}
