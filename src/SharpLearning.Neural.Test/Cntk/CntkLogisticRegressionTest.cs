using System.Collections.Generic;
using System.Diagnostics;
using CNTK;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace SharpLearning.Neural.Test.Cntk
{
    [TestClass]
    public class CntkLogisticRegressionTest
    { 

        [TestMethod]
        public void Run_Cntk_Logistic_Regression_Using()
        {
            // select execution device.
            var device = DeviceDescriptor.CPUDevice;

            // setup problem dimensions.
            var inputDim = 3;
            var numOutputClasses = 2;

            // setup variables for handling input to the model.
            Variable featureVariable = Variable.InputVariable(new int[] { inputDim }, DataType.Float);
            Variable targetVariable = Variable.InputVariable(new int[] { numOutputClasses }, DataType.Float);
            
            // create a logistic regression model.
            Function lr = CreateLogisticRegression(featureVariable, numOutputClasses, device);

            // setup loss and eval error.
            Function loss = CNTKLib.CrossEntropyWithSoftmax(lr, targetVariable);
            Function evalError = CNTKLib.ClassificationError(lr, targetVariable);

            // learning hyper-parameters.
            int minibatchSize = 64;
            int numMinibatchesToTrain = 10000;
            double learningRate = 0.01;

            // how often to report progress.
            int updatePerMinibatches = 50;

            // mini batch training loop.
            for (int minibatchCount = 0; minibatchCount < numMinibatchesToTrain; minibatchCount++)
            {
                // generate next minibatch.
                Dictionary<Variable, Value> miniBatch = GenerateBatchData(minibatchSize, inputDim, numOutputClasses,
                    featureVariable, targetVariable, device);

                // forward pass.
                var batchOutput = new Dictionary<Variable, Value>() { { lr.Output, null } };
                lr.Evaluate(miniBatch, batchOutput, device);

                // calculate loss.
                var lossOutput = new Dictionary<Variable, Value>() { { loss.Output, null } };
                loss.Evaluate(batchOutput, lossOutput, device);

                // backward pass. Calculate gradients.
                var parameterGradients = new Dictionary<Variable, Value>();
             
                
                // Update parameters.
            }
        }

        [TestMethod]
        public void Run_Cntk_Logistic_Regression_Using_Cntk_Learner()
        {
            // select execution device.
            var device = DeviceDescriptor.CPUDevice;

            // setup problem dimensions.
            var inputDim = 3;
            var numOutputClasses = 2;

            // setup variables for handling input to the model.
            Variable featureVariable = Variable.InputVariable(new int[] { inputDim }, DataType.Float);
            Variable targetVariable = Variable.InputVariable(new int[] { numOutputClasses }, DataType.Float);

            // create a logistic regression model.
            Function lr = CreateLogisticRegression(featureVariable, numOutputClasses, device);

            // setup loss and eval error.
            Function loss = CNTKLib.CrossEntropyWithSoftmax(lr, targetVariable);
            Function evalError = CNTKLib.ClassificationError(lr, targetVariable);

            // setup learner.
            TrainingParameterScheduleDouble learningRatePerSample = new CNTK.TrainingParameterScheduleDouble(0.02, 1);
            var parameterLearners = new List<Learner>() { Learner.SGDLearner(lr.Parameters(), learningRatePerSample) };
            var trainer = Trainer.CreateTrainer(lr, loss, evalError, parameterLearners);

            // learning hyper-parameters.
            int minibatchSize = 64;
            int numMinibatchesToTrain = 1000;

            // how often to report progress.
            int updatePerMinibatches = 50;

            // mini batch training loop.
            for (int minibatchCount = 0; minibatchCount < numMinibatchesToTrain; minibatchCount++)
            {
                // generate next minibatch.
                Dictionary<Variable, Value> miniBatch = GenerateBatchData(minibatchSize, inputDim, numOutputClasses,
                    featureVariable, targetVariable, device);

                // train model with mini batch
                trainer.TrainMinibatch(miniBatch, false, device);

                // print progress
                TraceProgress(trainer, updatePerMinibatches, minibatchCount);
            }
        }

        static Function CreateLogisticRegression(Variable input, int outputDim, DeviceDescriptor device)
        {
            int inputDim = input.Shape[0];
            var weightParam = new Parameter(new int[] { outputDim, inputDim }, DataType.Float, 1, device, "w");
            var biasParam = new Parameter(new int[] { outputDim }, DataType.Float, 0, device, "b");
            
            var linearModel = CNTKLib.Times(weightParam, input) + biasParam;

            return CNTKLib.Sigmoid(linearModel);
        }

        static Dictionary<Variable, Value> GenerateBatchData(int sampleSize, int inputDim, int numOutputClasses,
            Variable featureVariable, Variable targetVariable, DeviceDescriptor device)
        {
            float[] features;
            float[] oneHottargets;
            DataGenerator.GenerateRawDataSamples(sampleSize, inputDim, numOutputClasses, out features, out oneHottargets);

            // these should be disposed.
            var featureValue = Value.CreateBatch<float>(new int[] { inputDim }, features, device);
            var targetValue = Value.CreateBatch<float>(new int[] { numOutputClasses }, oneHottargets, device);

            return new Dictionary<Variable, Value>() { { featureVariable, featureValue }, { targetVariable, targetValue } };
        }

        static void TraceProgress(Trainer trainer, int updatePerMinibatches, int minibatchCount)
        {
            if ((minibatchCount % updatePerMinibatches) == 0 && trainer.PreviousMinibatchSampleCount() != 0)
            {
                float trainLossValue = (float)trainer.PreviousMinibatchLossAverage();
                float evaluationValue = (float)trainer.PreviousMinibatchEvaluationAverage();

                Trace.WriteLine($"Minibatch: {minibatchCount} CrossEntropyLoss = {trainLossValue}, EvaluationCriterion = {evaluationValue}");
            }
        }
    }
}
