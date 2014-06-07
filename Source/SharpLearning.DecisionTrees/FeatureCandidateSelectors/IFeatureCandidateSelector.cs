using System.Collections.Generic;

namespace SharpLearning.DecisionTrees.Learners
{
    /// <summary>
    /// Provides feature candidates/indices to select from
    /// </summary>
    public interface IFeatureCandidateSelector
    {
        void Select(int count, List<int> candidates);
    }
}
