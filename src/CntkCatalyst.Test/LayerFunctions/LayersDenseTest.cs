using System;
using System.Collections.Generic;
using System.Linq;
using CNTK;
using CntkCatalyst.LayerFunctions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace CntkCatalyst.Test.LayerFunctions
{
    [TestClass]
    public class LayersDenseTest
    {
        readonly DataType m_dataType = DataType.Float;
        readonly DeviceDescriptor m_device = DeviceDescriptor.CPUDevice;

        [TestMethod]
        public void Dense()
        {           
            var data = new float[] { 1, 1 };
            var inputVariable = CNTKLib.InputVariable(new int[] { data.Length }, m_dataType);

            var sut = Layers.Dense(inputVariable, 2,
                m_dataType, m_device,
                weightInitializer: Initializer.Ones, 
                biasInitializer: Initializer.Ones);

            var actual = Evaluate(sut, inputVariable, data);

            var expected = new float[] { 3, 3 };
            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void Dense_No_Bias()
        {
            var data = new float[] { 1, 1 };
            var inputVariable = CNTKLib.InputVariable(new int[] { data.Length }, m_dataType);

            var sut = Layers.Dense(inputVariable, 2,
                m_dataType, m_device,
                weightInitializer: Initializer.Ones,
                biasInitializer: Initializer.None);

            var actual = Evaluate(sut, inputVariable, data);

            var expected = new float[] { 2, 2 };
            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void Dense_Stack_Of_Two()
        {
            var data = new float[] { 1, 1 };
            var inputVariable = CNTKLib.InputVariable(new int[] { data.Length }, m_dataType);

            var sut1 = Layers.Dense(inputVariable, 2,
                m_dataType, m_device,
                weightInitializer: Initializer.Ones,
                biasInitializer: Initializer.Ones);

            var sut2 = Layers.Dense(sut1, 2,
                m_dataType, m_device,
                weightInitializer: Initializer.Ones,
                biasInitializer: Initializer.Ones);

            var actual = Evaluate(sut2, inputVariable, data);

            var expected = new float[] { 7, 7 };
            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void Dense_InputRank_And_MapRank_At_The_Same_Time_Throws()
        {
            var inputVariable = CNTKLib.InputVariable(new int[] { 2 }, m_dataType);
            Layers.Dense(inputVariable, 2, m_dataType, m_device, inputRank: 1, mapRank: 1);
        }

        float[] Evaluate(Function layer, Variable inputVariable, float[] inputData)
        {
            var input = new Dictionary<Variable, Value>
            {
                { inputVariable, Value.CreateBatch(inputVariable.Shape, inputData, 0, inputData.Length, m_device) }
            };
            var output = new Dictionary<Variable, Value> { { layer.Output, null } };

            layer.Evaluate(input, output, true, m_device);

            var actual = output[layer.Output].GetDenseData<float>(layer.Output)
                .Single()
                .ToArray();

            return actual;
        }
    }
}
