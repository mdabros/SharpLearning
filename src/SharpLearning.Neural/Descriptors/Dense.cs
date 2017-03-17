using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpLearning.Neural.Descriptors
{
    /// <summary>
    /// 
    /// </summary>
    public struct Dense
    {
        /// <summary>
        /// 
        /// </summary>
        public readonly int Units;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="units"></param>
        public Dense(int units)
        {
            Units = units;
        }
    }
}
