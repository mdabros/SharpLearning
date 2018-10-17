using CNTK;

namespace CntkCatalyst
{
    public static class Metrics
    {
        public static Function Accuracy(Variable predictions, Variable targets)
        {
            // assumes float datatype.
            return CNTKLib.Minus(Constant.Scalar(DataType.Float, 1), CNTKLib.ClassificationError(predictions, targets));
        }

        public static Function BinaryAccuracy(Variable prediction, Variable targets)
        {
            var round_predictions = CNTK.CNTKLib.Round(prediction);
            var equal_elements = CNTK.CNTKLib.Equal(round_predictions, targets);
            var result = CNTK.CNTKLib.ReduceMean(equal_elements, CNTK.Axis.AllStaticAxes());
            return result;
        }
    }
}
