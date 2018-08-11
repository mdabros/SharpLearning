using CNTK;

namespace CntkExtensions
{
    public static class Losses
    {
        public static Function MeanAbsoluteError(Variable targets, Variable predictions)
        {
            return CNTKLib.ReduceMean(CNTKLib.Abs(CNTKLib.Minus(predictions, targets)), new Axis(-1));
        }

        public static Function MeanSquaredError(Variable targets, Variable predictions)
        {
            return CNTKLib.ReduceMean(CNTKLib.Square(CNTKLib.Minus(predictions, targets)), new Axis(-1));
        }

        public static Function CategoricalCrossEntropy(Variable targets, Variable predictions)
        {
            return CNTKLib.CrossEntropyWithSoftmax(targets, predictions);
        }
    }
}
