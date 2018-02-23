using System;
using System.Collections.Generic;
using System.Diagnostics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Backend.Testing;
using TensorFlow;

namespace SharpLearning.Backend.TensorFlow.Test
{
    [TestClass]
    public class TensorFlowRawTruncatedNormalTest
    {
        const int GlobalSeed = 42;

        [TestInitialize]
        public void Check64Bit()
        {
            Assert.IsTrue(Environment.Is64BitProcess, "Must execute as 64-bit");
        }

        [TestMethod]
        public void TensorFlowRawTruncatedNormalTest_2x3()
        {
            using (var g = new TFGraph())
            {
                var variables = WeightVariable(g, new TFShape(2, 3));

                TFOutput[] variablesOutputs = new TFOutput[] { variables.variable };
                TFOperation[] variablesAssignOps = new TFOperation[] { variables.variable.Operation };

                using (var session = new TFSession(g))
                {
                    TFOperation[] targets = null; // TODO: It is possible to create a single operation target, instead of using outputs... how?
                    TFBuffer runMetaData = null;
                    TFBuffer runOptions = null;
                    TFStatus trainStatus = new TFStatus();

                    // Initialize variables
                    session.GetRunner().AddTarget(variablesAssignOps).Run();

                    // Dump initial assigns to see random initializes are identical
                    TFTensor[] variablesInitialized = session.Run(new TFOutput[] { }, new TFTensor[] { }, variablesOutputs,
                        targets, runMetaData, runOptions, trainStatus);

                    trainStatus.Raise();


                    foreach (var v in variablesInitialized)
                    {
                        var vText = v.ToString();
                        var array = (Array)v.GetValue();
                        Trace.WriteLine(vText);
                        Trace.WriteLine(array.ToDebugText());
                    }
                    // Can be used for debugging initialization of all variables
                    //using (var w = new StreamWriter("MnistDeepVariablesInitial.txt"))
                    //{
                    //    foreach (var v in variablesInitialized)
                    //    {
                    //        var vText = v.ToString();
                    //        var array = (Array)v.GetValue();
                    //        w.WriteLine(vText);
                    //        w.WriteLine(array.ToDebugText());
                    //        Log(vText + " Initialize Logged");
                    //    }
                    //}
                }
            }
        }

        public static (TFOutput assign, TFOutput variable) WeightVariable(TFGraph g, TFShape shape)
        {
            const float mean = 0.0f;
            const float stddev = 0.1f;
            TFOutput shape_output = g.Const(shape.AsTensor());
            TFOutput rnd = g.TruncatedNormal(shape_output, TFDataType.Float, seed: GlobalSeed);
            TFTensor mean_tensor = new TFTensor(mean);
            TFTensor stddev_tensor = new TFTensor(stddev);
            TFOutput mean_output = g.Const(mean_tensor);
            TFOutput stddev_output = g.Const(stddev_tensor);
            TFOutput mul = g.Mul(rnd, stddev_output);
            TFOutput value = g.Add(mul, mean_output);
            TFOutput initial = value;
            //TFOutput initial = g.ParameterizedTruncatedNormal(shape_output, TFDataType.Float, seed: GlobalSeed);
            //with ops.name_scope(name, "truncated_normal", [shape, mean, stddev]) as name:
            //shape_tensor = _ShapeTensor(shape)
            //mean_tensor = ops.convert_to_tensor(mean, dtype = dtype, name = "mean")
            //stddev_tensor = ops.convert_to_tensor(stddev, dtype = dtype, name = "stddev")
            //seed1, seed2 = random_seed.get_seed(seed)
            //rnd = gen_random_ops._truncated_normal(
            //    shape_tensor, dtype, seed = seed1, seed2 = seed2)
            //mul = rnd * stddev_tensor
            //value = math_ops.add(mul, mean_tensor, name = name)
            //return value

            // What about std dev???
            //Variable variable = g.Variable(initial);

            TFOutput w = g.VariableV2(shape, TFDataType.Float, operName: "W");
            TFOutput w_init = g.Assign(w, initial);

            return (w_init, w);
        }

    }
}
