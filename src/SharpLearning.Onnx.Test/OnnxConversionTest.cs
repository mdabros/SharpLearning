using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace SharpLearning.Onnx
{
    /// <summary>
    /// Resources:
    /// ONNX Schema: https://github.com/onnx/onnx/blob/da13be2d8494b99807bd1f071b40c40cc6b6d1cd/onnx/defs/traditionalml/defs.cc#L872
    /// ONNX Conversion ML.Net: https://github.com/dotnet/machinelearning/blob/master/src/Microsoft.ML.FastTree/FastTree.cs#L3030
    /// ONNX Conversion SKLearn: https://github.com/onnx/sklearn-onnx/blob/master/skl2onnx/common/tree_ensemble.py
    /// ONNX Conversion SKLearn: https://github.com/onnx/sklearn-onnx/blob/master/skl2onnx/operator_converters/decision_tree.py 
    /// ONNX Conversion SKLearn: https://github.com/onnx/sklearn-onnx/blob/master/skl2onnx/operator_converters/random_forest.py
    /// </summary>
    [TestClass]
    public class OnnxConversionTest
    {
        [TestMethod]
        public void MyTestMethod()
        {

        }
    }
}
