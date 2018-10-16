using CNTK;

namespace SharpLearning.Cntk.Test
{
    public delegate Function CntkMetricCreator(Variable predictions, Variable Targets);
}
