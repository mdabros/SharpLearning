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
        const int GraphGlobalSeed = 17;
        const int OpGlobalSeed = 42;

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
                var shape = new TFShape(2, 3);
                TFOutput shape_output = g.Const(shape.AsTensor());
                TFOutput truncatedNormalDirect = g.TruncatedNormal(shape_output, TFDataType.Float, seed: GraphGlobalSeed, seed2: OpGlobalSeed);
                (TFOutput wAssign, TFOutput wVariable) = WeightVariable(g, shape);

                TFOperation[] variablesAssignOps = new TFOperation[] { wAssign.Operation };
                TFOutput[] variablesOutputs = new TFOutput[] { wVariable, truncatedNormalDirect };

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
                    {
                        var expectedDirect = new float[,]{ 
                            { -0.8194038f,  -0.54961103f, 1.2118553f },
                            {  0.43519333f,   -0.9206078f,  -1.0796201f },
                        };

                        var actualDirect = (float[,])variablesInitialized[1].GetValue();
                        // Direct is same as python "gen_random_ops._truncated_normal"
                        CollectionAssert.AreEqual(expectedDirect, actualDirect, $"\nExpected\n{expectedDirect.ToDebugText()} Actual\n{actualDirect.ToDebugText()}");

                        // Simply scaled by std dev
                        var expected = new float[,]{
                            {-0.08194038f, -0.054961103f, 0.12118553f },
                            { 0.043519333f, -0.09206078f, -0.10796201f }
                        };

                        var actual = (float[,])variablesInitialized[0].GetValue();
                        CollectionAssert.AreEqual(expected, actual, $"\nExpected\n{expected.ToDebugText()} Actual\n{actual.ToDebugText()}");

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
            TFOutput rnd = g.TruncatedNormal(shape_output, TFDataType.Float, seed: GraphGlobalSeed, seed2: OpGlobalSeed);
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
