using System;

namespace SharpLearning.Backend.Testing
{
    public interface IFunc<T, TResult> { TResult Invoke(T value); }

    public struct DelegateFunc<T, TResult> : IFunc<T, TResult>
    {
        readonly Func<T, TResult> m_func;

        public DelegateFunc(Func<T, TResult> func)
        {
            m_func = func ?? throw new ArgumentNullException(nameof(func));
        }

        public TResult Invoke(T value) => m_func(value);
    }

    public static class Transforms
    {
        public static void Transform<T, TResult, TFunc>(Span<T> src, TFunc func, Span<TResult> dst)
            where TFunc : IFunc<T, TResult>
        {
            if (src.Length != dst.Length) { throw new ArgumentException("lengths not equal"); }
            for (int i = 0; i < src.Length; i++)
            {
                dst[i] = func.Invoke(src[i]);
            }
        }
    }

    public class FeatureConvertFlatBatchFeaturesTargetEnumerator<TFeature, TTarget, TFeatureResult, TFeatureConverter>
        : IFlatBatchFeaturesTargetEnumerator<TFeatureResult, TTarget>
        where TFeatureConverter : IFunc<TFeature, TFeatureResult>
    {
        readonly IFlatBatchFeaturesTargetEnumerator<TFeature, TTarget> m_enumerator;
        readonly TFeatureConverter m_featureConverter;
        // NOTE: This reuses these!
        readonly TFeatureResult[] m_batchFeatures;

        public FeatureConvertFlatBatchFeaturesTargetEnumerator(
            IFlatBatchFeaturesTargetEnumerator<TFeature, TTarget> enumerator,
            TFeatureConverter featureConverter)
        {
            m_enumerator = enumerator ?? throw new ArgumentNullException(nameof(enumerator));
            m_featureConverter = featureConverter;
            m_batchFeatures = new TFeatureResult[enumerator.TotalBatchSize];
        }

        public int TotalBatchSize => m_enumerator.TotalBatchSize;

        public bool MoveNext() => m_enumerator.MoveNext();

        public (TFeatureResult[] batchFeatures, TTarget[] batchTargets) CurrentBatch()
        {
            var (fs, ts) = m_enumerator.CurrentBatch();
            Transforms.Transform<TFeature, TFeatureResult, TFeatureConverter>(fs, m_featureConverter, m_batchFeatures);
            return (m_batchFeatures, ts);
        }

        public void Reset() => m_enumerator.Reset();
    }

    public class TargetConvertFlatBatchFeaturesTargetEnumerator<TFeature, TTarget, TTargetResult, TTargetConverter>
        : IFlatBatchFeaturesTargetEnumerator<TFeature, TTargetResult>
        where TTargetConverter : IFunc<TTarget, TTargetResult>
    {
        readonly IFlatBatchFeaturesTargetEnumerator<TFeature, TTarget> m_enumerator;
        readonly TTargetConverter m_targetConverter;
        // NOTE: This reuses these!
        readonly TTargetResult[] m_batchTargets;

        public TargetConvertFlatBatchFeaturesTargetEnumerator(
            IFlatBatchFeaturesTargetEnumerator<TFeature, TTarget> enumerator,
            TTargetConverter targetConverter)
        {
            m_enumerator = enumerator ?? throw new ArgumentNullException(nameof(enumerator));
            m_targetConverter = targetConverter;
            m_batchTargets = new TTargetResult[enumerator.TotalBatchSize];
        }

        public int TotalBatchSize => m_enumerator.TotalBatchSize;

        public bool MoveNext() => m_enumerator.MoveNext();

        public (TFeature[] batchFeatures, TTargetResult[] batchTargets) CurrentBatch()
        {
            var (fs, ts) = m_enumerator.CurrentBatch();
            Transforms.Transform<TTarget, TTargetResult, TTargetConverter>(ts, m_targetConverter, m_batchTargets);
            return (fs, m_batchTargets);
        }

        public void Reset() => m_enumerator.Reset();
    }

    public class OneHotTargetConvertFlatBatchFeaturesTargetEnumerator<TFeature, TTarget, TTargetResult, TTargetIndexer>
        : IFlatBatchFeaturesTargetEnumerator<TFeature, TTargetResult>
        where TTargetIndexer : IFunc<TTarget, int>
    {
        readonly IFlatBatchFeaturesTargetEnumerator<TFeature, TTarget> m_enumerator;
        readonly TTargetIndexer m_indexer;
        readonly TTargetResult m_one;
        readonly int m_targetCount;
        // NOTE: This reuses these!
        readonly TTargetResult[] m_batchTargets;

        public OneHotTargetConvertFlatBatchFeaturesTargetEnumerator(
            IFlatBatchFeaturesTargetEnumerator<TFeature, TTarget> enumerator,
            TTargetIndexer indexer,
            TTargetResult one,
            int targetCount)
        {
            m_enumerator = enumerator ?? throw new ArgumentNullException(nameof(enumerator));
            m_indexer = indexer;
            m_one = one;
            m_targetCount = targetCount;
            m_batchTargets = new TTargetResult[enumerator.TotalBatchSize * m_targetCount];
        }

        public int TotalBatchSize => m_enumerator.TotalBatchSize;

        public bool MoveNext() => m_enumerator.MoveNext();

        public (TFeature[] batchFeatures, TTargetResult[] batchTargets) CurrentBatch()
        {
            var (fs, ts) = m_enumerator.CurrentBatch();
            Array.Clear(m_batchTargets, 0, m_batchTargets.Length);
            for (int i = 0; i < ts.Length; i++)
            {
                var index = m_indexer.Invoke(ts[i]);
                m_batchTargets[i * m_targetCount + index] = m_one;
            }
            return (fs, m_batchTargets);
        }

        public void Reset() => m_enumerator.Reset();
    }


    public class ConvertFlatBatchFeaturesTargetEnumerator<TFeature, TTarget, TFeatureResult, TTargetResult, TFeatureConverter, TTargetConverter>
        : IFlatBatchFeaturesTargetEnumerator<TFeatureResult, TTargetResult>
        where TFeatureConverter : IFunc<TFeature, TFeatureResult>
        where TTargetConverter : IFunc<TTarget, TTargetResult>
    {
        readonly IFlatBatchFeaturesTargetEnumerator<TFeature, TTarget> m_enumerator;
        readonly TFeatureConverter m_featureConverter;
        readonly TTargetConverter m_targetConverter;
        // NOTE: This reuses these!
        readonly TFeatureResult[] m_batchFeatures;
        readonly TTargetResult[] m_batchTargets;

        public ConvertFlatBatchFeaturesTargetEnumerator(
            IFlatBatchFeaturesTargetEnumerator<TFeature, TTarget> enumerator,
            TFeatureConverter featureConverter,
            TTargetConverter targetConverter)
        {
            m_enumerator = enumerator ?? throw new ArgumentNullException(nameof(enumerator));
            m_featureConverter = featureConverter;
            m_targetConverter = targetConverter;
            m_batchFeatures = new TFeatureResult[enumerator.TotalBatchSize];
            m_batchTargets = new TTargetResult[enumerator.TotalBatchSize];
        }

        public int TotalBatchSize => m_enumerator.TotalBatchSize;

        public bool MoveNext() => m_enumerator.MoveNext();

        public (TFeatureResult[] batchFeatures, TTargetResult[] batchTargets) CurrentBatch()
        {
            var (fs, ts) = m_enumerator.CurrentBatch();
            Transforms.Transform<TFeature, TFeatureResult, TFeatureConverter>(fs, m_featureConverter, m_batchFeatures);
            Transforms.Transform<TTarget, TTargetResult, TTargetConverter>(ts, m_targetConverter, m_batchTargets);
            return (m_batchFeatures, m_batchTargets);
        }

        public void Reset() => m_enumerator.Reset();
    }

    // NOTE: This is specifically NOT an IEnumerator<T> for now, it could be perhaps
    public class FlatBatchFeaturesTargetEnumerator<TFeature, TTarget> : IFlatBatchFeaturesTargetEnumerator<TFeature, TTarget>
    {
        readonly FlatData<TFeature> m_features;
        readonly FlatData<TTarget> m_targets;
        readonly int m_count;
        readonly int m_batchSize;
        // NOTE: This reuses these!
        readonly TFeature[] m_batchFeatures;
        readonly TTarget[] m_batchTargets;
        int m_index = -1;

        // Can we always assume first dimension of shape is SampleCount??
        public FlatBatchFeaturesTargetEnumerator(FlatData<TFeature> features, FlatData<TTarget> targets, int batchSize)
        {
            m_features = features;
            m_targets = targets;
            m_batchSize = batchSize;
            var featuresSampleCount = m_features.Shape.SampleCount();
            var targetsSampleCount = m_targets.Shape.SampleCount();
            if (featuresSampleCount != targetsSampleCount)
            {
                throw new ArgumentException($"Features sample count {featuresSampleCount} not equal to target sample count {targetsSampleCount}");
            }
            m_count = featuresSampleCount;
            m_batchFeatures = new TFeature[m_features.Shape.FeatureSize() * batchSize];
            m_batchTargets = new TTarget[m_targets.Shape.FeatureSize() * batchSize];
        }

        public int TotalBatchSize => m_batchFeatures.Length;

        public bool MoveNext() => ++m_index < (m_count - m_batchSize + 1);

        // FUTURE: Make Span<T> version without memory copying, unfortunately no Backend supports span yet...
        public (TFeature[] batchFeatures, TTarget[] batchTargets) CurrentBatch()
        {
            Buffer.BlockCopy(m_features.Data, m_batchFeatures.Length * m_index, m_batchFeatures, 0, m_batchFeatures.Length);
            Buffer.BlockCopy(m_targets.Data, m_batchTargets.Length * m_index, m_batchTargets, 0, m_batchTargets.Length);
            return (m_batchFeatures, m_batchTargets);
        }
        
        public void Reset()
        {
            m_index = -1;
        }
    }

    // TODO: The dual generic type params of all this gets really annoying and long...
    //       create api where converting one at a time, easier...
    //       one about one hot encodings etc.?
    public static class FlatBatchFeaturesTargetEnumeratorExtensions
    {
        public struct Capture<TFeature, TTarget>
        {
            public Capture(IFlatBatchFeaturesTargetEnumerator<TFeature, TTarget> enumerator)
            {
                Enumerator = enumerator;
            }
            IFlatBatchFeaturesTargetEnumerator<TFeature, TTarget> Enumerator { get; }

            public IFlatBatchFeaturesTargetEnumerator<TFeatureResult, TTargetResult> To<TFeatureResult, TTargetResult, TFeatureConverter, TTargetConverter>(
                TFeatureConverter featureConverter, TTargetConverter targetConverter)
                where TFeatureConverter : IFunc<TFeature, TFeatureResult>
                where TTargetConverter : IFunc<TTarget, TTargetResult>
            {
                return new ConvertFlatBatchFeaturesTargetEnumerator<TFeature, TTarget, TFeatureResult, TTargetResult, TFeatureConverter, TTargetConverter>(
                    Enumerator, featureConverter, targetConverter);
            }
            public IFlatBatchFeaturesTargetEnumerator<TFeatureResult, TTargetResult> To<TFeatureResult, TTargetResult>(
                Func<TFeature, TFeatureResult> featureConverter, Func<TTarget, TTargetResult> targetConverter)
            {
                return To<TFeatureResult, TTargetResult, DelegateFunc<TFeature, TFeatureResult> , DelegateFunc<TTarget, TTargetResult>>(
                    new DelegateFunc<TFeature, TFeatureResult>(featureConverter), new DelegateFunc<TTarget, TTargetResult>(targetConverter));
            }
        }

        public struct FeatureCapture<TFeature, TTarget>
        {
            public FeatureCapture(IFlatBatchFeaturesTargetEnumerator<TFeature, TTarget> enumerator)
            {
                Enumerator = enumerator;
            }
            IFlatBatchFeaturesTargetEnumerator<TFeature, TTarget> Enumerator { get; }

            public IFlatBatchFeaturesTargetEnumerator<TFeatureResult, TTarget> To<TFeatureResult, TFeatureConverter>(
                TFeatureConverter featureConverter)
                where TFeatureConverter : IFunc<TFeature, TFeatureResult>
            {
                return new FeatureConvertFlatBatchFeaturesTargetEnumerator<TFeature, TTarget, TFeatureResult, TFeatureConverter>(
                    Enumerator, featureConverter);
            }
            public IFlatBatchFeaturesTargetEnumerator<TFeatureResult, TTarget> To<TFeatureResult>(
                Func<TFeature, TFeatureResult> featureConverter)
            {
                return To<TFeatureResult, DelegateFunc<TFeature, TFeatureResult>>(
                    new DelegateFunc<TFeature, TFeatureResult>(featureConverter));
            }
        }

        public struct TargetCapture<TFeature, TTarget>
        {
            public TargetCapture(IFlatBatchFeaturesTargetEnumerator<TFeature, TTarget> enumerator)
            {
                Enumerator = enumerator;
            }
            IFlatBatchFeaturesTargetEnumerator<TFeature, TTarget> Enumerator { get; }

            public IFlatBatchFeaturesTargetEnumerator<TFeature, TTargetResult> To<TTargetResult, TTargetConverter>(
                TTargetConverter targetConverter)
                where TTargetConverter : IFunc<TTarget, TTargetResult>
            {
                return new TargetConvertFlatBatchFeaturesTargetEnumerator<TFeature, TTarget, TTargetResult, TTargetConverter>(
                    Enumerator, targetConverter);
            }
            public IFlatBatchFeaturesTargetEnumerator<TFeature, TTargetResult> To<TTargetResult>(
                Func<TTarget, TTargetResult> targetConverter)
            {
                return To<TTargetResult, DelegateFunc<TTarget, TTargetResult>>(
                    new DelegateFunc<TTarget, TTargetResult>(targetConverter));
            }

            public IFlatBatchFeaturesTargetEnumerator<TFeature, TTargetResult> ToOneHot<TTargetResult, TTargetIndexer>(
                TTargetIndexer indexer, TTargetResult one, int targetCount)
                where TTargetIndexer : IFunc<TTarget, int>
            {
                return new OneHotTargetConvertFlatBatchFeaturesTargetEnumerator<TFeature, TTarget, TTargetResult, TTargetIndexer>(
                    Enumerator, indexer, one, targetCount);
            }
            public IFlatBatchFeaturesTargetEnumerator<TFeature, TTargetResult> ToOneHot<TTargetResult>(
                Func<TTarget, int> indexer, TTargetResult one, int targetCount)
            {
                return new OneHotTargetConvertFlatBatchFeaturesTargetEnumerator<TFeature, TTarget, TTargetResult, DelegateFunc<TTarget, int>>(
                    Enumerator, new DelegateFunc<TTarget,int>(indexer), one, targetCount);
            }
        }

        public static Capture<TFeature,TTarget> From<TFeature, TTarget>(this IFlatBatchFeaturesTargetEnumerator<TFeature, TTarget> enumerator)
        {
            return new Capture<TFeature, TTarget>(enumerator);
        }
        public static FeatureCapture<TFeature, TTarget> Feature<TFeature, TTarget>(this IFlatBatchFeaturesTargetEnumerator<TFeature, TTarget> enumerator)
        {
            return new FeatureCapture<TFeature, TTarget>(enumerator);
        }
        public static TargetCapture<TFeature, TTarget> Target<TFeature, TTarget>(this IFlatBatchFeaturesTargetEnumerator<TFeature, TTarget> enumerator)
        {
            return new TargetCapture<TFeature, TTarget>(enumerator);
        }
    }
}
