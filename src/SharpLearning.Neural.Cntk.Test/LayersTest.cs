using System.Linq;
using System.Collections.Generic;
using CNTK;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace SharpLearning.Neural.Cntk.Test
{
    [TestClass]
    public class LayersTest
    {
        [TestMethod]
        public void Dense()
        {
            var dataType = DataType.Float;
            var device = DeviceDescriptor.CPUDevice;
            
            var inputVariable = CNTKLib.InputVariable(new int[] { 2 }, dataType);
            var data = new float[] { 1, 1 };

            var sut = Layers.Dense(inputVariable, 2);

            var input = new Dictionary<Variable, Value> { { inputVariable, Value.CreateBatch(inputVariable.Shape, data, 0, data.Length, device)} };
            var output = new Dictionary<Variable, Value> { { sut.Output, null } };
            sut.Evaluate(input, output, true, device);

            var actual = output[sut.Output].GetDenseData<float>(sut.Output).Single();

            Assert.AreEqual(1.2, actual);
        }
    }
}
