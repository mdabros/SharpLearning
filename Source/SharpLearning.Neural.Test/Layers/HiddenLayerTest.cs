using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Neural.Layers;

namespace SharpLearning.Neural.Test.Layers
{
    [TestClass]
    public class HiddenLayerTest
    {
        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void HiddenLayer_Constructor_Units_Below_One()
        {
            new HiddenLayer(0);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void HiddenLayer_Constructor_Dropout_Below_Zero()
        {
            new HiddenLayer(1, -0.1);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void HiddenLayer_Constructor_Dropout_Equals_One()
        {
            new HiddenLayer(1, 1.0);
        }
    }
}
