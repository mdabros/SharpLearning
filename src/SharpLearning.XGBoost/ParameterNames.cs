namespace SharpLearning.XGBoost
{
    internal static class ParameterNames
    {
        /// <summary>
        /// Maximum tree depth for base learners
        /// </summary>
        public const string MaxDepth = "max_depth";

        /// <summary>
        /// Boosting learning rate (xgb's "eta")
        /// </summary>
        public const string LearningRate = "learning_rate";

        /// <summary>
        /// Number of boosted trees to fit
        /// </summary>
        public const string Estimators = "n_estimators";

        /// <summary>
        /// Whether to print messages while running boosting
        /// </summary>
        public const string Silent = "silent";

        /// <summary>
        /// Specify the learning task and the corresponding learning objective or
        /// a custom objective function to be used(see note below)
        /// </summary>
        public const string objective = "objective";

        /// <summary>
        /// Number of parallel threads used to run xgboost
        /// </summary>
        public const string Threads = "nthread";

        /// <summary>
        /// Minimum loss reduction required to make a further partition on a leaf node of the tree
        /// </summary>
        public const string Gamma = "gamma";

        /// <summary>
        /// Minimum sum of instance weight(hessian) needed in a child
        /// </summary>
        public const string MinChildWeight = "min_child_weight";

        /// <summary>
        /// Maximum delta step we allow each tree's weight estimation to be
        /// </summary>
        public const string MaxDeltaStep = "max_delta_step";

        /// <summary>
        /// Subsample ratio of the training instance
        /// </summary>
        public const string SubSample = "subsample";

        /// <summary>
        /// Subsample ratio of columns when constructing each tree TODO prevent error for bigger range of vals
        /// </summary>
        public const string ColSampleByTree = "colsample_bytree";

        /// <summary>
        /// Subsample ratio of columns for each split, in each level TODO prevent error for bigger range of vals
        /// </summary>
        public const string ColSampleByLevel = "colsample_bylevel";

        /// <summary>
        /// L1 regularization term on weights
        /// </summary>
        public const string RegAlpha = "reg_alpha";

        /// <summary>
        /// L2 regularization term on weights
        /// </summary>
        public const string RegLambda = "reg_lambda";

        /// <summary>
        /// Balancing of positive and negative weights
        /// </summary>
        public const string ScalePosWeight = "scale_pos_weight";

        /// <summary>
        /// The initial prediction score of all instances, global bias
        /// </summary>
        public const string BaseScore = "base_score";

        /// <summary>
        /// Random number seed
        /// </summary>
        public const string Seed = "seed";

        /// <summary>
        /// Value in the data which needs to be present as a missing value
        /// </summary>
        public const string Missing = "missing";

        /// <summary>
        /// Existing booster
        /// </summary>
        public const string ExistingBooster = "_Booster";

        /// <summary>
        /// which booster to use, can be gbtree, gblinear or dart. gbtree and dart use tree based model while gblinear uses linear function.
        /// </summary>
        public const string Booster = "booster";

        /// <summary>
        /// The tree construction algorithm used in XGBoost. See reference paper: https://arxiv.org/abs/1603.02754
        /// Distributed and external memory version only support approximate algorithm.
        /// Choices: {'auto', 'exact', 'approx', 'hist', 'gpu_exact', 'gpu_hist'}.
        /// 'auto': Use heuristic to choose faster one.
        /// - For small to medium dataset, exact greedy will be used.
        /// - For very large-dataset, approximate algorithm will be chosen.
        /// - Because old behavior is always use exact greedy in single machine, 
        /// user will get a message when approximate algorithm is chosen to notify this choice.
        /// 
        /// 'exact': Exact greedy algorithm.
        /// 'approx': Approximate greedy algorithm using sketching and histogram.
        /// 'hist': Fast histogram optimized approximate greedy algorithm. It uses some performance improvements such as bins caching.
        /// 'gpu_exact': GPU implementation of exact algorithm.
        /// 'gpu_hist': GPU implementation of hist algorithm.
        /// </summary>
        public const string TreeMethod = "tree_method";

        /// <summary>
        /// The number of classes in a classification problem.
        /// </summary>
        public const string NumberOfClasses = "num_class";
    }
}
