using SharpLearning.Containers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpLearning.Ensemble.Strategies
{
    /// <summary>
    /// Interface for classification ensemble strategies
    /// </summary>
    public interface IClassificationEnsembleStrategy
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="ensemblePredictions"></param>
        /// <returns></returns>
        ProbabilityPrediction Combine(ProbabilityPrediction[] ensemblePredictions);
    }
}
