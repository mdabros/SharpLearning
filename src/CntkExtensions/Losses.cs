using CNTK;

namespace CntkExtensions
{
    public static class Losses
    {
        internal static Function MeanAbsoluteError(Variable labels, Variable predictions)
        {
            return CNTKLib.ReduceMean(CNTKLib.Abs(CNTKLib.Minus(predictions, labels)), new Axis(-1));
        }

        public static Function MeanSquaredError(Variable labels, Variable predictions)
        {
            return CNTKLib.ReduceMean(CNTKLib.Square(CNTKLib.Minus(predictions, labels)), new Axis(-1));
        }
    }
}
