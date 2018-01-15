using System;
using System.Collections.Generic;
using System.Diagnostics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorFlow;

namespace SharpLearning.Backend.TensorFlow.Test
{
    [TestClass]
    public class TensorFlowRawTest
    {
        [TestMethod]
        public void TensorFlowRawTest_Simple()
        {
            using (var g = new TFGraph())
            {
                var one = g.Const(new TFTensor(1f), "one");
                var two = g.Const(new TFTensor(2f), "two");
                var add = g.Add(one, two, "add");

                using (var s = new TFSession(g))
                {
                    var runner = s.GetRunner();
                    var r = runner.Run(add);

                    Assert.AreEqual(TFDataType.Float, r.TensorType);
                    Assert.AreEqual(3f, r.GetValue());

                    Trace.WriteLine(r);
                }
            }
        }
    }
}
