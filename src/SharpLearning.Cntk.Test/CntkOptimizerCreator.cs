using System.Collections.Generic;
using CNTK;

namespace SharpLearning.Cntk.Test
{
    public delegate Learner CntkOptimizerCreator(IList<Parameter> parameters);
}
