using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpLearning.GradientBoost.GBM
{
    public class GBMSplitResult
    {
        public GBMSplit BestSplit;
        public GBMSplitInfo Left;
        public GBMSplitInfo Right;
    }
}
