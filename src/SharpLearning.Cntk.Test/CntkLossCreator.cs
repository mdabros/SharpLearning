using CNTK;

namespace SharpLearning.Cntk.Test
{
    public delegate Function CntkLossCreator(Variable predictions, Variable Targets);
}
