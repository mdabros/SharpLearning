using CNTK;

namespace CntkExtensions
{
    public static class Metrics
    {
        public static Function Accuracy(Variable targets, Variable predictions)
        {
            // assumes float datatype.
            return CNTKLib.Minus(Constant.Scalar(DataType.Float, 1), CNTKLib.ClassificationError(predictions, targets));
        }

        public static Function BinaryAccuracy(Variable targets, Variable prediction)
        {
            var round_predictions = CNTK.CNTKLib.Round(prediction);
            var equal_elements = CNTK.CNTKLib.Equal(round_predictions, targets);
            var result = CNTK.CNTKLib.ReduceMean(equal_elements, CNTK.Axis.AllStaticAxes());
            return result;
        }
    }
}
