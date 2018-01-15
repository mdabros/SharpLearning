using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace SharpLearning.Backend.TensorFlow.Test
{
    [TestClass]
    public class TensorFlowBackendTest
    {
        // Only works for 64-bit

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
    }
}
