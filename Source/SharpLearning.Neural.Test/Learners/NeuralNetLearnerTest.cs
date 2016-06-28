using System;
using System.Text;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Neural.Learners;
using SharpLearning.InputOutput.Csv;
using System.IO;
using SharpLearning.Neural.Test.Properties;
using System.Diagnostics;
using SharpLearning.Neural.Activations;
using SharpLearning.Neural.Loss;
using SharpLearning.Neural.TargetEncoders;
using SharpLearning.Neural.Optimizers;
using SharpLearning.Neural.Layers;

namespace SharpLearning.Neural.Test.Learners
{
    /// <summary>
    /// Summary description for NeuralNetLearnerTest
    /// </summary>
    [TestClass]
    public class NeuralNetLearnerTest
    {
        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void NeuralNetLearner_Constructor_HiddenLayers_Null()
        {
            new NeuralNetLearner(null, new TanhActivation(), new LogLoss(),
                new CopyTargetEncoder(), new IdentityActivation(), new SGDNeuralNetOptimizer(0.1, LearningRateSchedule.Adaptive));
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void NeuralNetLearner_Constructor_InputActiviation_Null()
        {
            new NeuralNetLearner(new HiddenLayer[] { HiddenLayer.New(10) }, null, new LogLoss(),
                new CopyTargetEncoder(), new IdentityActivation(), new SGDNeuralNetOptimizer(0.1, LearningRateSchedule.Adaptive));
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void NeuralNetLearner_Constructor_Loss_Null()
        {
            new NeuralNetLearner(new HiddenLayer[] { HiddenLayer.New(10) }, new TanhActivation(), null,
                new CopyTargetEncoder(), new IdentityActivation(), new SGDNeuralNetOptimizer(0.1, LearningRateSchedule.Adaptive));
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void NeuralNetLearner_Constructor_TargetEncoder_Null()
        {
            new NeuralNetLearner(new HiddenLayer[] { HiddenLayer.New(10) }, new TanhActivation(), new LogLoss(),
                null, new IdentityActivation(), new SGDNeuralNetOptimizer(0.1, LearningRateSchedule.Adaptive));
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void NeuralNetLearner_Constructor_OutputActivation_Null()
        {
            new NeuralNetLearner(new HiddenLayer[] { HiddenLayer.New(10) }, new TanhActivation(), new LogLoss(),
                new CopyTargetEncoder(), null, new SGDNeuralNetOptimizer(0.1, LearningRateSchedule.Adaptive));
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void NeuralNetLearner_Constructor_Optimizer_Null()
        {
            new NeuralNetLearner(new HiddenLayer[] { HiddenLayer.New(10) }, new TanhActivation(), new LogLoss(),
                new CopyTargetEncoder(), new IdentityActivation(), null);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void NeuralNetLearner_Constructor_Optimizer_MaxIterations_Below_One()
        {
            new NeuralNetLearner(new HiddenLayer[] { HiddenLayer.New(10) }, new TanhActivation(), new LogLoss(),
                new CopyTargetEncoder(), new IdentityActivation(), new SGDNeuralNetOptimizer(0.1, LearningRateSchedule.Adaptive), 0);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void NeuralNetLearner_Constructor_Optimizer_BatchSize_Below_One()
        {
            new NeuralNetLearner(new HiddenLayer[] { HiddenLayer.New(10) }, new TanhActivation(), new LogLoss(),
                new CopyTargetEncoder(), new IdentityActivation(), new SGDNeuralNetOptimizer(0.1, LearningRateSchedule.Adaptive), 1, 0);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void NeuralNetLearner_Constructor_Optimizer_L2Reguralization_Below_Zero()
        {
            new NeuralNetLearner(new HiddenLayer[] { HiddenLayer.New(10) }, new TanhActivation(), new LogLoss(),
                new CopyTargetEncoder(), new IdentityActivation(), new SGDNeuralNetOptimizer(0.1, LearningRateSchedule.Adaptive), 1, 1, -0.1);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void NeuralNetLearner_Constructor_Optimizer_Input_Dropout_Below_Zero()
        {
            new NeuralNetLearner(new HiddenLayer[] { HiddenLayer.New(10) }, new TanhActivation(), new LogLoss(),
                new CopyTargetEncoder(), new IdentityActivation(), new SGDNeuralNetOptimizer(0.1, LearningRateSchedule.Adaptive), 1, 1, 0.1, -0.1);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void NeuralNetLearner_Constructor_Optimizer_Input_Dropout_Equals_One()
        {
            new NeuralNetLearner(new HiddenLayer[] { HiddenLayer.New(10) }, new TanhActivation(), new LogLoss(),
                new CopyTargetEncoder(), new IdentityActivation(), new SGDNeuralNetOptimizer(0.1, LearningRateSchedule.Adaptive), 1, 1, 0.1, 1.0);
        }
    }

}
