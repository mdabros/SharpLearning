using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Collections.Generic;
using SharpLearning.CrossValidation.BiasVarianceAnalysis;
using SharpLearning.Containers.Matrices;
using System.IO;

namespace SharpLearning.CrossValidation.Test.BiasVarianceAnalysis
{
    [TestClass]
    public class BiasVarianceLearningCurvePointExtensionsTest
    {
        [TestMethod]
        public void BiasVarianceLearningCurvePointExtensions_ToF64Matrix()
        {
            var sut = new List<BiasVarianceLearningCurvePoint> { new BiasVarianceLearningCurvePoint(10, 0.0, 1.0), 
                new BiasVarianceLearningCurvePoint(100, 3.0, 8.0), new BiasVarianceLearningCurvePoint(1000, 4.0, 4.0) };

            var actual = sut.ToF64Matrix();
            var expected = new F64Matrix(new double[] { 10, 0.0, 1.0,
                100, 3.0, 8.0, 
                1000, 4.0, 4.0 },
                3, 3);

            Assert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void BiasVarianceLearningCurvePointExtensions_Write()
        {
            var sut = new List<BiasVarianceLearningCurvePoint> { new BiasVarianceLearningCurvePoint(10, 0.0, 1.0), 
                new BiasVarianceLearningCurvePoint(100, 3.0, 8.0), new BiasVarianceLearningCurvePoint(1000, 4.0, 4.0) };

            var writer = new StringWriter();
            sut.Write(() => writer);

            var expected = "SampleCount;TrainingError;TestError\r\n10;0;1\r\n100;3;8\r\n1000;4;4";
            Assert.AreEqual(expected, writer.ToString());
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void BiasVarianceLearningCurvePointExtensions_ToF64Matrix_No_Elements()
        {
            new List<BiasVarianceLearningCurvePoint>().ToF64Matrix();
        }
    }
}
