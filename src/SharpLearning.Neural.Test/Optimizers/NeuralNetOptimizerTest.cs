using System;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Neural.Optimizers;

namespace SharpLearning.Neural.Test.Optimizers
{
    [TestClass]
    public class NeuralNetOptimizerTest
    {
        [TestMethod]
        public void NeuralNetOptimizer_Reset_Does_Not_Throw()
        {
            var parametersAndGradients = new List<ParametersAndGradients>
            {
                new(new float[10], new float[10]),
                new(new float[10], new float[10]),
            };

            foreach (OptimizerMethod optimizer in Enum.GetValues(typeof(OptimizerMethod)))
            {
                var sut = new NeuralNetOptimizer(0.001, 10, optimizerMethod: optimizer);
                sut.UpdateParameters(parametersAndGradients);
                sut.Reset();
            }
        }
    }
}
