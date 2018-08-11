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
    }
}
