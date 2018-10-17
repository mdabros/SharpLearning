using CNTK;

namespace CntkCatalyst
{
    public static class Losses
    {
        public static Function MeanAbsoluteError(Variable predictions, Variable targets)
        {
            return CNTKLib.ReduceMean(CNTKLib.Abs(CNTKLib.Minus(predictions, targets)), new Axis(-1));
        }

        public static Function MeanSquaredError(Variable predictions, Variable targets)
        {
            return CNTKLib.ReduceMean(CNTKLib.Square(CNTKLib.Minus(predictions, targets)), new Axis(-1));
        }

        public static Function CategoricalCrossEntropy(Variable predictions, Variable targets)
        {
            return CNTKLib.CrossEntropyWithSoftmax(predictions, targets);
        }

        public static Function BinaryCrossEntropy(Variable predictions, Variable targets)
        {
            return CNTKLib.BinaryCrossEntropy(predictions, targets);
        }
    }
}
