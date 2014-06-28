using System.Collections.Generic;

namespace SharpLearning.DecisionTrees.FeatureCandidateSelectors
{
    /// <summary>
    /// Provides feature candidates/indices to select from
    /// </summary>
    public interface IFeatureCandidateSelector
    {
        void Select(int selectCount, int totalCount, List<int> candidates);
    }
}
