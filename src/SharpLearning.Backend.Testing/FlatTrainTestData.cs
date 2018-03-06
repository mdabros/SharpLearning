namespace SharpLearning.Backend.Testing
{
    public class FlatTrainTestData<TFeature, TTarget>
    {
        // TODO: Create type for FlatFeaturesTargetsData
        public FlatTrainTestData(
            FlatData<TFeature> trainFeatures, 
            FlatData<TTarget> trainTargets, 
            FlatData<TFeature> testFeatures, 
            FlatData<TTarget> testTargets)
        {
            // TODO: Perhaps assert shapes are compatible for features/targets
            TrainFeatures = trainFeatures;
            TrainTargets = trainTargets;
            TestFeatures = testFeatures;
            TestTargets = testTargets;
        }

        public FlatData<TFeature> TrainFeatures { get; }
        public FlatData<TTarget> TrainTargets { get; }
        public FlatData<TFeature> TestFeatures { get; }
        public FlatData<TTarget> TestTargets { get; }

        public FlatBatchFeaturesTargetEnumerator<TFeature, TTarget> CreateTrainBatchEnumerator(int batchSize)
        {
            return new FlatBatchFeaturesTargetEnumerator<TFeature, TTarget>(TrainFeatures, TrainTargets, batchSize);
        }

        public FlatBatchFeaturesTargetEnumerator<TFeature, TTarget> CreateTestBatchEnumerator(int batchSize)
        {
            return new FlatBatchFeaturesTargetEnumerator<TFeature, TTarget>(TestFeatures, TestTargets, batchSize);
        }
    }
}
