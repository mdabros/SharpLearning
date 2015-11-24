using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpLearning.GradientBoost.GBMDecisionTree
{
    /// <summary>
    /// Split Results. Contains the best split 
    /// and the left and right split information
    /// </summary>
    public struct GBMSplitResult
    {
        /// <summary>
        /// Best split found
        /// </summary>
        public GBMSplit BestSplit;
        
        /// <summary>
        /// Left values corresponding to best split
        /// </summary>
        public GBMSplitInfo Left;
        
        /// <summary>
        /// Right values corresponding to best split
        /// </summary>
        public GBMSplitInfo Right;
    }
}
