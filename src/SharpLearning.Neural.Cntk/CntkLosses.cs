using CNTK;

namespace SharpLearning.Neural.Cntk
{
    public static class CntkLosses
    {
        internal static Function MeanAbsError(Variable labels, Variable predictions)
        {
            return CNTKLib.ReduceMean(CNTKLib.Abs(CNTKLib.Minus(predictions, labels)), new Axis(-1));
        }

        public static Function MeanSquaredError(Variable labels, Variable predictions)
        {
            return CNTKLib.ReduceMean(CNTKLib.Square(CNTKLib.Minus(predictions, labels)), new Axis(-1));
        }
    }
}
