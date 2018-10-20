using System.Collections.Generic;
using System.Linq;
using CNTK;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace CntkCatalyst.Test
{
    [TestClass]
    public class LossesTest
    {
        readonly DataType m_dataType = DataType.Float;
        readonly DeviceDescriptor m_device = DeviceDescriptor.CPUDevice;

        [TestMethod]
        public void MeanSquareError()
        {
            var targetsData = new float[] { 1.0f, 2.3f, 3.1f, 4.4f, 5.8f };
            var targetsVariable = CNTKLib.InputVariable(new int[] { targetsData.Length }, m_dataType);

            var predictionsData = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
            var predictionsVariable = CNTKLib.InputVariable(new int[] { predictionsData.Length }, m_dataType);

            var sut = Losses.MeanSquaredError(predictionsVariable, targetsVariable);
            var actual = Evaluate(sut, targetsVariable, targetsData,
                predictionsVariable, predictionsData);

            Assert.AreEqual(0.18f, actual, 0.00001);
        }

        [TestMethod]
        public void MeanSquareError_Zero_Error()
        {
            var targetsData = new float[] { 0, 0, 0, 0, 0, 0 };
            var targetsVariable = CNTKLib.InputVariable(new int[] { targetsData.Length }, m_dataType);

            var predictionsData = new float[] { 0, 0, 0, 0, 0, 0 };
            var predictionsVariable = CNTKLib.InputVariable(new int[] { predictionsData.Length }, m_dataType);

            var sut = Losses.MeanSquaredError(predictionsVariable, targetsVariable);
            var actual = Evaluate(sut, targetsVariable, targetsData,
                predictionsVariable, predictionsData);

            Assert.AreEqual(0.0f, actual);
        }

        float Evaluate(Function loss, Variable targetsVariable, float[] targetsData, 
            Variable predictionsVariable, float[] predictionsData)
        {
            var input = new Dictionary<Variable, Value>
            {
                { targetsVariable, Value.CreateBatch(targetsVariable.Shape, targetsData, 0, targetsData.Length, m_device) },
                { predictionsVariable, Value.CreateBatch(predictionsVariable.Shape, predictionsData, 0, predictionsData.Length, m_device) }
            };
            var output = new Dictionary<Variable, Value> { { loss.Output, null } };

            loss.Evaluate(input, output, true, m_device);

            var actual = output[loss.Output].GetDenseData<float>(loss.Output)
                .Single()
                .Single();

            return actual;
        }
    }
}
