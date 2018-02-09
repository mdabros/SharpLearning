namespace SharpLearning.Backend.Testing
{
    // TODO: Revise whether Shape type is needed then move to Backend etc.
    public static class ShapeExtensions
    {
        public static int Product(this int[] shape)
        {
            int product = 1;
            foreach (var s in shape)
            {
                product *= s;
            }
            return product;
        }
        // Default is index 0 is number of samples/observations, revise if necessary
        public static int SampleCount(this int[] shape)
        {
            return shape[0];
        }
        public static int FeatureSize(this int[] shape)
        {
            int product = 1;
            for (int i = 1; i < shape.Length; i++)
            {
                product *= i;

            }
            return product;
        }
    }
}
