using System.Threading;
using System.Threading.Tasks;

namespace SharpLearning.Optimization.Test.StagedOptimization
{
    public interface IControlFlowStep
    {
        /// <summary>
        /// contains code to synchronously execute step
        /// </summary>
        void Execute(); 
    }
}