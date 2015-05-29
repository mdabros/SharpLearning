using System;
using System.Text;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Extensions;
using SharpLearning.GradientBoost.GBM;
using System.Linq;
using System.Diagnostics;

namespace SharpLearning.GradientBoost.Test.GBM
{
    [TestClass]
    public class GBMRunningMedianTest
    {
        [TestMethod]
        public void GBMMedianEstimator_Median_NoSamplesSample()
        {
            var sut = new GBMRunningMedian();
            Assert.AreEqual(0, sut.Median(), 0.001);
        }

        [TestMethod]
        public void GBMMedianEstimator_Median_Reset_MediumSample()
        {
            var data = new double[] { 1.231, 4.123, 2.232, 232.45, 0.23, 23.423, 2355.12, 2.231, 324.2 };

            var sut = new GBMRunningMedian();
            
            data.ForEach(v => sut.AddSample(v));
            Assert.AreEqual(4.123, sut.Median(), 0.001);

            sut.Reset();
            Assert.AreEqual(0.0, sut.Median(), 0.001);

            data.ForEach(v => sut.AddSample(v));
            Assert.AreEqual(4.123, sut.Median(), 0.001);
        }


        [TestMethod]
        public void GBMMedianEstimator_Median_SmallSample()
        {
            var data = new double[] { 1.231, 4.123, 2.232 };

            var sut = new GBMRunningMedian();
            
            data.ForEach(v => sut.AddSample(v));
            Assert.AreEqual(2.232, sut.Median(), 0.001);
        }

        [TestMethod]
        public void GBMMedianEstimator_Median_LargeSample()
        {
            var random = new Random(42);
            var data = Enumerable.Range(0, 1000).Select(t => (double)random.Next()).ToArray();

            var sut = new GBMRunningMedian();
            
            data.ForEach(v => sut.AddSample(v));
            Assert.AreEqual(1068747866.5, sut.Median(), 0.001);
        }
    }
}
