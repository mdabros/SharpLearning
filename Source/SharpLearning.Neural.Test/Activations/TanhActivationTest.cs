using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MathNet.Numerics.LinearAlgebra;
using SharpLearning.Neural.Activations;
using System.Diagnostics;

namespace SharpLearning.Neural.Test.Activations
{
    [TestClass]
    public class TanhActivationTest
    {
        [TestMethod]
        public void TanhActivation_Activiation()
        {
            var actual = Matrix<float>.Build.Random(5, 5, 23);
            var sut = new TanhActivation();
            sut.Activation(actual);
            var expected = Matrix<float>.Build.Dense(5, 5, new float[] { 0.9220566f, 0.4109178f, 0.8357052f, -0.1137989f, 0.7587373f, 0.9408718f, 0.599718f, 0.4980977f, -0.9032097f, -0.4064444f, -0.9402503f, -0.9646701f, -0.631319f, -0.1855543f, 0.2058925f, -0.9313608f, -0.6119131f, -0.7593536f, 0.7939757f, -0.02670056f, 0.2745694f, -0.1115566f, 0.1453609f, 0.2984324f, 0.688675f });

            Assert.AreEqual(expected.ToString(), actual.ToString());
        }

        [TestMethod]
        public void TanhActivation_Derivative()
        {
            var actual = Matrix<float>.Build.Random(5, 5, 23);
            var sut = new TanhActivation();
            sut.Derivative(actual, actual);
            var expected = Matrix<float>.Build.Dense(5, 5, new float[] { -1.568276f, 0.8092801f, -0.4562715f, 0.9869369f, 0.01348946f, -2.047092f, 0.5201573f, 0.7010393f, -1.218238f, 0.8139418f, -2.028309f, -3.036783f, 0.4470497f, 0.9647579f, 0.9563699f, -1.784088f, 0.4930931f, 0.0105988f, -0.1709353f, 0.9992868f, 0.9205893f, 0.9874509f, 0.9785677f, 0.9052604f, 0.2852462f });
                        
            Assert.AreEqual(expected.ToString(), actual.ToString());
        }

    }
}
