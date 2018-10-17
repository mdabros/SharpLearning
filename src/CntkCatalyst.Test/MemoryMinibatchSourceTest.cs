using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace CntkCatalyst.Test
{
    [TestClass]
    public class MemoryMinibatchSourceTest
    {
        float[] m_observationsData = new float[]
        {
            1, 1, 1, 1, 1,
            2, 2, 2, 2, 2,
            3, 3, 3, 3, 3,
            4, 4, 4, 4, 4,
            5, 5, 5, 5 ,5,
            6, 6, 6, 6, 6,
            7, 7, 7, 7, 7,
            8, 8, 8, 8, 8,
            9, 9, 9, 9, 9,
        };

        float[] m_targetData = new float[]
        {
            1, 1, 1, 2, 2, 2, 3, 3, 3
        };

        [TestMethod]
        public void MemoryMinibatchSource()
        {
            var observations = new MemoryMinibatchData(m_observationsData, new int[] { 5 }, 9);
            var targets = new MemoryMinibatchData(m_targetData, new int[] { 1 }, 9);

            var sut = new MemoryMinibatchSource(observations, targets, 5, false);

            for (int i = 0; i < 30; i++)
            {
                var minibatch = sut.GetNextMinibatch(3);
                var obs = minibatch.observations;
                var tar = minibatch.targets;
            }
        }
    }
}
