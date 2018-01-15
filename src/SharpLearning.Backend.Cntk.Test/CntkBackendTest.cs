using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace SharpLearning.Backend.Cntk.Test
{
    [TestClass]
    public class CntkBackendTest
    {
        [TestMethod]
        public void CntkBackendTest_CtorAndDispose()
        {
            // Doesn't really do anything here
            using (var b = new CntkBackend(DeviceType.Automatic))
            {
            }
        }

        [TestMethod]
        public void CntkBackendTest_CreateGraph()
        {
            var deviceType = DeviceType.Automatic;
            using (var b = new CntkBackend(deviceType))
            using (var g = b.CreateGraph())
            {
                Assert.AreEqual(g.DefaultDeviceType, deviceType);
            }
        }
    }
}
