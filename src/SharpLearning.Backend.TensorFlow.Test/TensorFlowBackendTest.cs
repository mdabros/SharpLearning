using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace SharpLearning.Backend.TensorFlow.Test
{
    [TestClass]
    public class TensorFlowBackendTest
    {
        [TestInitialize]
        public void Check64Bit()
        {
            Assert.IsTrue(Environment.Is64BitProcess, "Must execute as 64-bit");
        }

        [TestMethod]
        public void TensorFlowBackendTest_CtorAndDispose()
        {
            using (var b = new TensorFlowBackend(DeviceType.Automatic))
            {
            }
        }

        [TestMethod]
        public void TensorFlowBackendTest_CreateGraph()
        {
            var deviceType = DeviceType.Automatic;
            using (var b = new TensorFlowBackend(deviceType))
            using (var g = b.CreateGraph())
            {
                Assert.AreEqual(g.DefaultDeviceType, deviceType);
            }
        }

        [TestMethod]
        public void TensorFlowBackendTest_CreateGraph_Placeholder()
        {
            var deviceType = DeviceType.Automatic;
            using (var b = new TensorFlowBackend(deviceType))
            using (var g = b.CreateGraph())
            {
                g.Placeholder(DataType.Single, new int[] { 1, 2, 3 }, "p");
            }
        }
    }
}
