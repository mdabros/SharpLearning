using System.Collections.Generic;
using System.Linq;
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

            var sut = Layers.Dense(inputVariable, 2, weightInitializer: Initializer.Ones, biasInitializer: Initializer.Ones);

            var input = new Dictionary<Variable, Value>
            {
                { inputVariable, Value.CreateBatch(inputVariable.Shape, data, 0, data.Length, device)}
            };
            var output = new Dictionary<Variable, Value> { { sut.Output, null } };

            sut.Evaluate(input, output, true, device);

            var actual = output[sut.Output].GetDenseData<float>(sut.Output)
                .Single().ToArray();

            var expected = new float[] { 3, 3 };
            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void Dense_No_Bias()
        {
            var dataType = DataType.Float;
            var device = DeviceDescriptor.CPUDevice;

            var inputVariable = CNTKLib.InputVariable(new int[] { 2 }, dataType);
            var data = new float[] { 1, 1 };

            var sut = Layers.Dense(inputVariable, 2,
                weightInitializer: Initializer.Ones,
                bias: false);

            var actual = Evaluate(device, inputVariable, data, sut);

            var expected = new float[] { 2, 2 };
            CollectionAssert.AreEqual(expected, actual);
        }

        static float[] Evaluate(DeviceDescriptor device, Variable inputVariable, float[] data, Function sut)
        {
            var input = new Dictionary<Variable, Value>
            {
                { inputVariable, Value.CreateBatch(inputVariable.Shape, data, 0, data.Length, device) }
            };
            var output = new Dictionary<Variable, Value> { { sut.Output, null } };

            sut.Evaluate(input, output, true, device);

            var actual = output[sut.Output].GetDenseData<float>(sut.Output)
                .Single()
                .ToArray();

            return actual;
        }
    }
}
